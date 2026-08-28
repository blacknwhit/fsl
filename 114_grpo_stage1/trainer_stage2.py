from __future__ import annotations

from dataclasses import dataclass
import time

import torch
import torch.nn.functional as F

from .policy import scale_task_weights
from .train_utils import (
    assign_grads,
    capture_param_grads,
    combine_mixed_grads,
    ddp_allreduce_float_buffers,
    ddp_allreduce_param_grads,
    is_ddp_wrapped,
    maybe_no_sync,
    run_stage2_validation,
    state_dict_cpu_clone,
    to_device_cnt,
    to_device_det,
    to_device_seg,
)
from .utils import save_multitask_checkpoint


@dataclass
class Stage2Artifacts:
    best_metric: float
    best_epoch: int | None
    best_loss_weights: tuple[float, float, float]


def run_stage2(
    *,
    args,
    model,
    model_for_state,
    policy,
    state_tracker,
    optimizer_theta,
    theta_params,
    base_loss_weights,
    shared_param_mask,
    train_loaders,
    val_loaders,
    primary_task,
    device: torch.device,
    use_ddp: bool,
    world_size: int,
    manual_theta_grad_sync: bool,
    det_num_classes: int,
    save_dir,
    is_main_process: bool,
    metric_logger=None,
) -> Stage2Artifacts:
    for param in policy.parameters():
        param.requires_grad = False
    policy.eval()
    for param in state_tracker.parameters():
        param.requires_grad = False
    state_tracker.eval()

    other_tasks = [name for name in train_loaders.keys() if name != primary_task]
    best_metric = float("-inf")
    best_state = None
    best_policy_state = None
    best_state_feature_state = None
    best_epoch = None
    base_loss_weights = tuple(float(x) for x in base_loss_weights)
    shared_param_mask = tuple(bool(flag) for flag in shared_param_mask)
    best_loss_weights = base_loss_weights
    current_weights = best_loss_weights
    debug_timing = bool(getattr(args, "debug_step_timing", False))
    timing_interval = max(int(getattr(args, "debug_step_timing_interval", 1)), 1)

    theta_index_by_id = {id(param): idx for idx, param in enumerate(theta_params)}
    state_a_idx = theta_index_by_id.get(id(state_tracker.last_moe.lora_A_shared))
    state_b_idx = theta_index_by_id.get(id(state_tracker.last_moe.lora_B_shared))
    global_step = 0

    def now_ts() -> float:
        if debug_timing and device.type == "cuda":
            torch.cuda.synchronize(device)
        return time.perf_counter()

    def compute_task_loss_and_grads(task_name: str, batch):
        optimizer_theta.zero_grad(set_to_none=True)
        with maybe_no_sync(model):
            if task_name == "det":
                images, targets = to_device_det(batch, device)
                loss_dict = model("det", images, targets)
                loss = sum(loss_dict.values())
            elif task_name == "seg":
                imgs, masks = to_device_seg(batch, device)
                logits = model("seg", imgs)
                loss = F.cross_entropy(logits, masks)
            elif task_name == "cnt":
                imgs, dens = to_device_cnt(batch, device)
                gt_counts = dens.flatten(2).sum(dim=2)
                pred_dens, pred_counts = model("cnt", imgs, cnt_backbone_grad_mult=float(args.cnt_backbone_grad_mult))
                dens_loss = F.mse_loss(pred_dens, dens, reduction="sum") / imgs.size(0)
                cnt_l1 = F.l1_loss(pred_counts, gt_counts)
                loss = dens_loss + float(args.cnt_count_loss_weight) * cnt_l1
            else:
                raise ValueError(f"Unknown task: {task_name}")
            loss.backward()
            grads = capture_param_grads(theta_params)
        optimizer_theta.zero_grad(set_to_none=True)
        grad_a = grads[state_a_idx] if state_a_idx is not None else None
        grad_b = grads[state_b_idx] if state_b_idx is not None else None
        task_state = state_tracker.encode_shared_grad_state(
            grad_a,
            grad_b,
            device=device,
            use_ddp=use_ddp,
            world_size=world_size,
        )
        return loss.detach(), grads, task_state

    for epoch in range(1, int(args.stage2_epochs) + 1):
        if use_ddp:
            for loader in train_loaders.values():
                sampler = getattr(loader, "sampler", None)
                if isinstance(sampler, torch.utils.data.distributed.DistributedSampler):
                    sampler.set_epoch(epoch + int(args.stage1_epochs) + int(getattr(args, "warmup_epochs", 0)))

        other_iters = {name: iter(train_loaders[name]) for name in other_tasks}
        model.train()
        total_loss = 0.0
        steps = 0

        for step, primary_batch in enumerate(train_loaders[primary_task], start=1):
            global_step += 1
            step_t0 = now_ts()
            if debug_timing and step % timing_interval == 0:
                print(f"[stage2][timing] epoch {epoch}/{int(args.stage2_epochs)} step {step} start")

            batches = {primary_task: primary_batch}
            for name in other_tasks:
                try:
                    batches[name] = next(other_iters[name])
                except StopIteration:
                    other_iters[name] = iter(train_loaders[name])
                    batches[name] = next(other_iters[name])
            t_after_batch = now_ts()

            model.train()
            optimizer_theta.zero_grad(set_to_none=True)

            # Low-peak mode: compute each task separately so only one task graph is alive at a time.
            det_loss, det_grads, det_state = compute_task_loss_and_grads("det", batches["det"])
            seg_loss, seg_grads, seg_state = compute_task_loss_and_grads("seg", batches["seg"])
            cnt_loss, cnt_grads, cnt_state = compute_task_loss_and_grads("cnt", batches["cnt"])
            t_after_loss = now_ts()

            state = torch.cat([det_state, seg_state, cnt_state], dim=0)
            t_after_state = now_ts()

            with torch.no_grad():
                policy_output = policy.build_dirichlet(state)
                weights = scale_task_weights(policy_output.mu.detach(), policy.weight_scale.detach())

            task_grads = {
                "det": det_grads,
                "seg": seg_grads,
                "cnt": cnt_grads,
            }
            t_after_task_grads = now_ts()

            final_grads = combine_mixed_grads(
                task_grads,
                weights,
                base_loss_weights,
                shared_param_mask,
            )
            assign_grads(theta_params, final_grads)
            if manual_theta_grad_sync:
                ddp_allreduce_param_grads(theta_params, world_size)
            if float(args.grad_clip_norm) > 0:
                torch.nn.utils.clip_grad_norm_(theta_params, max_norm=float(args.grad_clip_norm))
            optimizer_theta.step()
            if use_ddp and not is_ddp_wrapped(model):
                ddp_allreduce_float_buffers(model_for_state, world_size)
            t_after_theta = now_ts()

            current_weights = tuple(float(x) for x in weights.detach().cpu().tolist())
            step_det_loss = float(det_loss.detach().item())
            step_seg_loss = float(seg_loss.detach().item())
            step_cnt_loss = float(cnt_loss.detach().item())
            step_loss = (
                base_loss_weights[0] * step_det_loss
                + base_loss_weights[1] * step_seg_loss
                + base_loss_weights[2] * step_cnt_loss
            )
            total_loss += step_loss
            steps += 1

            if args.log_interval and step % int(args.log_interval) == 0:
                print(
                    f"[stage2] epoch {epoch}/{int(args.stage2_epochs)} step {step} | "
                    f"loss {step_loss:.4f} | "
                    f"train det {step_det_loss:.4f} seg {step_seg_loss:.4f} cnt {step_cnt_loss:.4f} | "
                    f"shared_w [{current_weights[0]:.3f}, {current_weights[1]:.3f}, {current_weights[2]:.3f}] | "
                    f"fixed_w [{base_loss_weights[0]:.3f}, {base_loss_weights[1]:.3f}, {base_loss_weights[2]:.3f}]"
                )
                if metric_logger is not None:
                    metric_logger.log_metrics(
                        {
                            "stage2/epoch": int(epoch),
                            "stage2/step_in_epoch": int(step),
                            "stage2/step_loss": float(step_loss),
                            "stage2/step_train_det_loss": float(step_det_loss),
                            "stage2/step_train_seg_loss": float(step_seg_loss),
                            "stage2/step_train_cnt_loss": float(step_cnt_loss),
                            "stage2/step_weight_det": float(current_weights[0]),
                            "stage2/step_weight_seg": float(current_weights[1]),
                            "stage2/step_weight_cnt": float(current_weights[2]),
                            "stage2/step_shared_weight_det": float(current_weights[0]),
                            "stage2/step_shared_weight_seg": float(current_weights[1]),
                            "stage2/step_shared_weight_cnt": float(current_weights[2]),
                            "stage2/step_fixed_weight_det": float(base_loss_weights[0]),
                            "stage2/step_fixed_weight_seg": float(base_loss_weights[1]),
                            "stage2/step_fixed_weight_cnt": float(base_loss_weights[2]),
                        },
                        step=global_step,
                    )
            if debug_timing and step % timing_interval == 0:
                print(
                    f"[stage2][timing] epoch {epoch}/{int(args.stage2_epochs)} step {step} | "
                    f"batch {t_after_batch-step_t0:.3f}s | "
                    f"task_loss {t_after_loss-t_after_batch:.3f}s | "
                    f"state {t_after_state-t_after_loss:.3f}s | "
                    f"task_grads {t_after_task_grads-t_after_state:.3f}s | "
                    f"theta_update {t_after_theta-t_after_task_grads:.3f}s | "
                    f"step_total {t_after_theta-step_t0:.3f}s"
                )

            if int(args.max_train_steps) and step >= int(args.max_train_steps):
                break

        print(
            f"[stage2] epoch {epoch}/{int(args.stage2_epochs)} | "
            f"train {total_loss/max(steps,1):.4f} | "
            f"shared_w [{current_weights[0]:.3f}, {current_weights[1]:.3f}, {current_weights[2]:.3f}] | "
            f"fixed_w [{base_loss_weights[0]:.3f}, {base_loss_weights[1]:.3f}, {base_loss_weights[2]:.3f}]"
        )
        if metric_logger is not None:
            metric_logger.log_metrics(
                {
                    "stage2/epoch": int(epoch),
                    "stage2/epoch_train_loss": float(total_loss / max(steps, 1)),
                    "stage2/epoch_weight_det": float(current_weights[0]),
                    "stage2/epoch_weight_seg": float(current_weights[1]),
                    "stage2/epoch_weight_cnt": float(current_weights[2]),
                    "stage2/epoch_shared_weight_det": float(current_weights[0]),
                    "stage2/epoch_shared_weight_seg": float(current_weights[1]),
                    "stage2/epoch_shared_weight_cnt": float(current_weights[2]),
                    "stage2/epoch_fixed_weight_det": float(base_loss_weights[0]),
                    "stage2/epoch_fixed_weight_seg": float(base_loss_weights[1]),
                    "stage2/epoch_fixed_weight_cnt": float(base_loss_weights[2]),
                },
                step=int(epoch),
            )

        if not bool(args.skip_validation):
            result = run_stage2_validation(
                model,
                val_loaders,
                device=device,
                seg_num_classes=int(args.seg_num_classes),
                det_num_classes=det_num_classes,
                det_ap_score_thr=float(args.det_ap_score_thr),
                cnt_count_loss_weight=float(args.cnt_count_loss_weight),
                max_val_steps=int(args.max_val_steps),
                epoch=epoch,
                stage_name="stage2",
                metric_logger=metric_logger,
            )
            if result.combo_metric > best_metric:
                best_metric = float(result.combo_metric)
                best_epoch = epoch
                if is_main_process:
                    best_state = state_dict_cpu_clone(model_for_state.state_dict())
                    best_policy_state = state_dict_cpu_clone(policy.state_dict())
                    best_state_feature_state = state_tracker.state_dict()
                    best_loss_weights = current_weights
                    print(f"[ckpt] new best cached (stage2 epoch {epoch}, combo {best_metric:.6f})")

    best_path = save_dir / "best_combo.pt"
    if is_main_process and best_state is None:
        best_state = state_dict_cpu_clone(model_for_state.state_dict())
        best_policy_state = state_dict_cpu_clone(policy.state_dict())
        best_state_feature_state = state_tracker.state_dict()
        best_epoch = int(args.stage2_epochs)
        best_metric = float("nan")
        best_loss_weights = current_weights

    if is_main_process and best_state is not None:
        model_for_state.load_state_dict(best_state)
        save_multitask_checkpoint(
            str(best_path),
            model=model_for_state,
            optimizer=optimizer_theta,
            epoch=int(best_epoch or int(args.stage2_epochs)),
            best_by="combo",
            metrics={
                "best_metric": float(best_metric),
                "best_stage": "stage2",
                "best_epoch": int(best_epoch or int(args.stage2_epochs)),
            },
            loss_weights=best_loss_weights,
            phi_state=best_policy_state,
            config={
                "use_lora_moe": bool(args.use_lora_moe),
                "unfreeze_backbone": bool(args.unfreeze_backbone),
                "backbone_lr": float(args.backbone_lr) if args.backbone_lr is not None else float(args.lr) * float(args.backbone_lr_mult),
                "policy_weight_prior": str(getattr(args, "policy_weight_prior", "15,8,1")),
                "fixed_loss_weights": tuple(float(x) for x in base_loss_weights),
                "loss_weight_scope": "shared_experts_only",
                "lora_rank": int(args.lora_rank),
                "num_experts_private": int(args.num_experts_private),
                "num_experts_shared": int(args.num_experts_shared),
                "moe_k_private": int(args.moe_k_private),
                "moe_k_shared": int(args.moe_k_shared),
            },
            state_feature_state=best_state_feature_state,
        )
        print(f"[ckpt] saved best -> {best_path} (combo {best_metric:.6f})")

    return Stage2Artifacts(
        best_metric=best_metric,
        best_epoch=best_epoch,
        best_loss_weights=best_loss_weights,
    )
