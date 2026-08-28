from __future__ import annotations

from dataclasses import dataclass
import time

import torch
import torch.nn.functional as F

from .block_selection_logger import BlockSelectionRecorder
from .train_utils import (
    SharedExpertMatrixIndex,
    assign_grads,
    capture_param_grads,
    combine_matrixwise_top_pair_grads,
    combine_weighted_grads,
    ddp_allreduce_float_buffers,
    ddp_allreduce_param_grads,
    is_ddp_wrapped,
    maybe_no_sync,
    run_stage2_validation,
    state_dict_cpu_clone,
    sync_grad_tuple,
    to_device_cnt,
    to_device_det,
    to_device_seg,
)
from .utils import save_multitask_checkpoint


@dataclass
class StageArtifacts:
    best_metric: float
    best_epoch: int | None
    checkpoint_path: str


def _compute_task_loss_and_grads(
    *,
    args,
    model,
    optimizer_theta,
    theta_params,
    task_name: str,
    batch,
    device: torch.device,
    use_ddp: bool,
    world_size: int,
):
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
        if use_ddp:
            grads = sync_grad_tuple(theta_params, grads, world_size)
    optimizer_theta.zero_grad(set_to_none=True)
    return loss.detach(), grads


def _maybe_sync_after_step(*, model, model_for_state, use_ddp: bool, world_size: int) -> None:
    if use_ddp and not is_ddp_wrapped(model):
        ddp_allreduce_float_buffers(model_for_state, world_size)


def _build_batches(primary_task: str, primary_batch, other_tasks: list[str], other_iters: dict[str, object], train_loaders):
    batches = {primary_task: primary_batch}
    for name in other_tasks:
        try:
            batches[name] = next(other_iters[name])
        except StopIteration:
            other_iters[name] = iter(train_loaders[name])
            batches[name] = next(other_iters[name])
    return batches


def run_stage1_plain(
    *,
    args,
    model,
    model_for_state,
    optimizer_theta,
    theta_params,
    base_loss_weights,
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
) -> StageArtifacts:
    other_tasks = [name for name in train_loaders.keys() if name != primary_task]
    best_metric = float("-inf")
    best_state = None
    best_epoch = None
    checkpoint_path = str(save_dir / "stage1_best.pt")
    base_loss_weights = tuple(float(x) for x in base_loss_weights)
    debug_timing = bool(getattr(args, "debug_step_timing", False))
    timing_interval = max(int(getattr(args, "debug_step_timing_interval", 1)), 1)
    global_step = 0
    val_last_k_epochs = min(max(int(getattr(args, "stage1_val_last_k_epochs", 50)), 1), int(args.stage1_epochs))
    val_start_epoch = max(1, int(args.stage1_epochs) - val_last_k_epochs + 1)

    def now_ts() -> float:
        if debug_timing and device.type == "cuda":
            torch.cuda.synchronize(device)
        return time.perf_counter()

    for epoch in range(1, int(args.stage1_epochs) + 1):
        if use_ddp:
            for loader in train_loaders.values():
                sampler = getattr(loader, "sampler", None)
                if isinstance(sampler, torch.utils.data.distributed.DistributedSampler):
                    sampler.set_epoch(epoch)

        other_iters = {name: iter(train_loaders[name]) for name in other_tasks}
        model.train()
        total_loss = 0.0
        det_loss_sum = 0.0
        seg_loss_sum = 0.0
        cnt_loss_sum = 0.0
        steps = 0

        for step, primary_batch in enumerate(train_loaders[primary_task], start=1):
            global_step += 1
            step_t0 = now_ts()
            if debug_timing and step % timing_interval == 0:
                print(f"[stage1][timing] epoch {epoch}/{int(args.stage1_epochs)} step {step} start")

            batches = _build_batches(primary_task, primary_batch, other_tasks, other_iters, train_loaders)
            t_after_batch = now_ts()

            model.train()
            optimizer_theta.zero_grad(set_to_none=True)

            det_loss, det_grads = _compute_task_loss_and_grads(
                args=args,
                model=model,
                optimizer_theta=optimizer_theta,
                theta_params=theta_params,
                task_name="det",
                batch=batches["det"],
                device=device,
                use_ddp=use_ddp,
                world_size=world_size,
            )
            seg_loss, seg_grads = _compute_task_loss_and_grads(
                args=args,
                model=model,
                optimizer_theta=optimizer_theta,
                theta_params=theta_params,
                task_name="seg",
                batch=batches["seg"],
                device=device,
                use_ddp=use_ddp,
                world_size=world_size,
            )
            cnt_loss, cnt_grads = _compute_task_loss_and_grads(
                args=args,
                model=model,
                optimizer_theta=optimizer_theta,
                theta_params=theta_params,
                task_name="cnt",
                batch=batches["cnt"],
                device=device,
                use_ddp=use_ddp,
                world_size=world_size,
            )
            t_after_loss = now_ts()

            task_grads = {
                "det": det_grads,
                "seg": seg_grads,
                "cnt": cnt_grads,
            }
            final_grads = combine_weighted_grads(task_grads, base_loss_weights)
            t_after_mix = now_ts()

            assign_grads(theta_params, final_grads)
            if manual_theta_grad_sync:
                ddp_allreduce_param_grads(theta_params, world_size)
            if float(args.grad_clip_norm) > 0:
                torch.nn.utils.clip_grad_norm_(theta_params, max_norm=float(args.grad_clip_norm))
            optimizer_theta.step()
            _maybe_sync_after_step(model=model, model_for_state=model_for_state, use_ddp=use_ddp, world_size=world_size)
            t_after_step = now_ts()

            step_det_loss = float(det_loss.item())
            step_seg_loss = float(seg_loss.item())
            step_cnt_loss = float(cnt_loss.item())
            step_loss = (
                base_loss_weights[0] * step_det_loss
                + base_loss_weights[1] * step_seg_loss
                + base_loss_weights[2] * step_cnt_loss
            )
            total_loss += step_loss
            det_loss_sum += step_det_loss
            seg_loss_sum += step_seg_loss
            cnt_loss_sum += step_cnt_loss
            steps += 1

            should_log_step_metrics = (not args.log_interval) or (step % int(args.log_interval) == 0)
            if should_log_step_metrics:
                print(
                    f"[stage1] epoch {epoch}/{int(args.stage1_epochs)} step {step} | "
                    f"loss {step_loss:.4f} | "
                    f"w [{base_loss_weights[0]:.1f}, {base_loss_weights[1]:.1f}, {base_loss_weights[2]:.1f}] | "
                    f"train det {step_det_loss:.4f} seg {step_seg_loss:.4f} cnt {step_cnt_loss:.4f}"
                )
            if should_log_step_metrics and metric_logger is not None:
                metric_logger.log_metrics(
                    {
                        "stage1/epoch": int(epoch),
                        "stage1/step_in_epoch": int(step),
                        "stage1/step_loss": float(step_loss),
                        "stage1/step_train_det_loss": float(step_det_loss),
                        "stage1/step_train_seg_loss": float(step_seg_loss),
                        "stage1/step_train_cnt_loss": float(step_cnt_loss),
                        "stage1/step_weight_det": float(base_loss_weights[0]),
                        "stage1/step_weight_seg": float(base_loss_weights[1]),
                        "stage1/step_weight_cnt": float(base_loss_weights[2]),
                    },
                    step=global_step,
                )
            if debug_timing and step % timing_interval == 0:
                print(
                    f"[stage1][timing] epoch {epoch}/{int(args.stage1_epochs)} step {step} | "
                    f"batch {t_after_batch-step_t0:.3f}s | "
                    f"task_loss {t_after_loss-t_after_batch:.3f}s | "
                    f"grad_mix {t_after_mix-t_after_loss:.3f}s | "
                    f"optimizer {t_after_step-t_after_mix:.3f}s | "
                    f"step_total {t_after_step-step_t0:.3f}s"
                )

            if int(args.max_train_steps) and step >= int(args.max_train_steps):
                break

        print(
            f"[stage1] epoch {epoch}/{int(args.stage1_epochs)} | "
            f"train {total_loss/max(steps,1):.4f} | "
            f"det {det_loss_sum/max(steps,1):.4f} seg {seg_loss_sum/max(steps,1):.4f} cnt {cnt_loss_sum/max(steps,1):.4f}"
        )
        if metric_logger is not None:
            metric_logger.log_metrics(
                {
                    "stage1/epoch": int(epoch),
                    "stage1/epoch_train_loss": float(total_loss / max(steps, 1)),
                    "stage1/epoch_train_det_loss": float(det_loss_sum / max(steps, 1)),
                    "stage1/epoch_train_seg_loss": float(seg_loss_sum / max(steps, 1)),
                    "stage1/epoch_train_cnt_loss": float(cnt_loss_sum / max(steps, 1)),
                    "stage1/epoch_weight_det": float(base_loss_weights[0]),
                    "stage1/epoch_weight_seg": float(base_loss_weights[1]),
                    "stage1/epoch_weight_cnt": float(base_loss_weights[2]),
                },
                step=int(epoch),
            )

        if (not bool(args.skip_validation)) and epoch >= val_start_epoch:
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
                stage_name="stage1",
                metric_logger=metric_logger,
            )
            if result.combo_metric > best_metric:
                best_metric = float(result.combo_metric)
                best_epoch = epoch
                if is_main_process:
                    best_state = state_dict_cpu_clone(model_for_state.state_dict())
                    print(f"[ckpt] new best cached (stage1 epoch {epoch}, combo {best_metric:.6f})")

    if is_main_process and best_state is None:
        best_state = state_dict_cpu_clone(model_for_state.state_dict())
        best_epoch = int(args.stage1_epochs)
        best_metric = float("nan")

    if is_main_process and best_state is not None:
        model_for_state.load_state_dict(best_state)
        save_multitask_checkpoint(
            checkpoint_path,
            model=model_for_state,
            optimizer=optimizer_theta,
            epoch=int(best_epoch or int(args.stage1_epochs)),
            best_by="combo",
            metrics={
                "best_metric": float(best_metric),
                "best_stage": "stage1",
                "best_epoch": int(best_epoch or int(args.stage1_epochs)),
            },
            loss_weights=base_loss_weights,
            config={
                "training_mode": "stage1_plain",
                "stage1_epochs": int(args.stage1_epochs),
                "stage1_val_last_k_epochs": int(val_last_k_epochs),
                "loss_weights": tuple(float(x) for x in base_loss_weights),
                "use_lora_moe": bool(args.use_lora_moe),
                "unfreeze_backbone": bool(args.unfreeze_backbone),
                "backbone_lr": float(args.backbone_lr)
                if args.backbone_lr is not None
                else float(args.lr) * float(args.backbone_lr_mult),
                "lora_rank": int(args.lora_rank),
                "num_experts_private": int(args.num_experts_private),
                "num_experts_shared": int(args.num_experts_shared),
                "moe_k_private": int(args.moe_k_private),
                "moe_k_shared": int(args.moe_k_shared),
            },
        )
        print(f"[ckpt] saved stage1 best -> {checkpoint_path} (combo {best_metric:.6f})")

    return StageArtifacts(
        best_metric=best_metric,
        best_epoch=best_epoch,
        checkpoint_path=checkpoint_path,
    )


def run_stage2_matrix_pair(
    *,
    args,
    model,
    model_for_state,
    optimizer_theta,
    theta_params,
    base_loss_weights,
    shared_matrix_index: SharedExpertMatrixIndex,
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
    block_selection_recorder: BlockSelectionRecorder | None = None,
) -> StageArtifacts:
    other_tasks = [name for name in train_loaders.keys() if name != primary_task]
    best_metric = float("-inf")
    best_state = None
    best_epoch = None
    checkpoint_path = str(save_dir / "best_combo.pt")
    base_loss_weights = tuple(float(x) for x in base_loss_weights)
    debug_timing = bool(getattr(args, "debug_step_timing", False))
    timing_interval = max(int(getattr(args, "debug_step_timing_interval", 1)), 1)
    global_step = 0
    total_matrix_units = max(int(len(shared_matrix_index.matrix_units)), 1)

    def now_ts() -> float:
        if debug_timing and device.type == "cuda":
            torch.cuda.synchronize(device)
        return time.perf_counter()

    if not bool(args.skip_validation):
        init_result = run_stage2_validation(
            model,
            val_loaders,
            device=device,
            seg_num_classes=int(args.seg_num_classes),
            det_num_classes=det_num_classes,
            det_ap_score_thr=float(args.det_ap_score_thr),
            cnt_count_loss_weight=float(args.cnt_count_loss_weight),
            max_val_steps=int(args.max_val_steps),
            epoch=0,
            stage_name="stage2_init",
            metric_logger=metric_logger,
        )
        best_metric = float(init_result.combo_metric)
        best_epoch = 0
        if is_main_process:
            best_state = state_dict_cpu_clone(model_for_state.state_dict())
            print(f"[ckpt] cached loaded checkpoint as initial stage2 best (epoch 0, combo {best_metric:.6f})")

    for epoch in range(1, int(args.stage2_epochs) + 1):
        if use_ddp:
            for loader in train_loaders.values():
                sampler = getattr(loader, "sampler", None)
                if isinstance(sampler, torch.utils.data.distributed.DistributedSampler):
                    sampler.set_epoch(epoch + int(args.stage1_epochs))

        other_iters = {name: iter(train_loaders[name]) for name in other_tasks}
        model.train()
        total_loss = 0.0
        det_loss_sum = 0.0
        seg_loss_sum = 0.0
        cnt_loss_sum = 0.0
        task_update_counts = {"det": 0, "seg": 0, "cnt": 0}
        positive_pair_counts = {"det_seg": 0, "det_cnt": 0, "seg_cnt": 0}
        update_mode_counts = {"all": 0, "shared": 0, "none": 0}
        epoch_block_selection_counts = {
            block_id: {"all": 0, "det": 0, "seg": 0, "cnt": 0, "none": 0}
            for block_id in shared_matrix_index.block_ids
        }
        steps = 0

        for step, primary_batch in enumerate(train_loaders[primary_task], start=1):
            global_step += 1
            step_t0 = now_ts()
            if debug_timing and step % timing_interval == 0:
                print(f"[stage2][timing] epoch {epoch}/{int(args.stage2_epochs)} step {step} start")

            batches = _build_batches(primary_task, primary_batch, other_tasks, other_iters, train_loaders)
            t_after_batch = now_ts()

            model.train()
            optimizer_theta.zero_grad(set_to_none=True)

            det_loss, det_grads = _compute_task_loss_and_grads(
                args=args,
                model=model,
                optimizer_theta=optimizer_theta,
                theta_params=theta_params,
                task_name="det",
                batch=batches["det"],
                device=device,
                use_ddp=use_ddp,
                world_size=world_size,
            )
            seg_loss, seg_grads = _compute_task_loss_and_grads(
                args=args,
                model=model,
                optimizer_theta=optimizer_theta,
                theta_params=theta_params,
                task_name="seg",
                batch=batches["seg"],
                device=device,
                use_ddp=use_ddp,
                world_size=world_size,
            )
            cnt_loss, cnt_grads = _compute_task_loss_and_grads(
                args=args,
                model=model,
                optimizer_theta=optimizer_theta,
                theta_params=theta_params,
                task_name="cnt",
                batch=batches["cnt"],
                device=device,
                use_ddp=use_ddp,
                world_size=world_size,
            )
            t_after_loss = now_ts()

            task_grads = {
                "det": det_grads,
                "seg": seg_grads,
                "cnt": cnt_grads,
            }
            final_grads, mix_stats = combine_matrixwise_top_pair_grads(
                task_grads,
                base_loss_weights,
                shared_matrix_index,
            )
            for task_name, count in mix_stats.task_update_counts.items():
                task_update_counts[task_name] += int(count)
            for pair_name, count in mix_stats.positive_pair_counts.items():
                positive_pair_counts[pair_name] += int(count)
            for mode_name, count in mix_stats.update_mode_counts.items():
                update_mode_counts[mode_name] += int(count)
            for block_id, block_counts in mix_stats.block_selection_counts.items():
                for label_name, count in block_counts.items():
                    epoch_block_selection_counts[block_id][label_name] += int(count)
            t_after_mix = now_ts()

            assign_grads(theta_params, final_grads)
            if manual_theta_grad_sync:
                ddp_allreduce_param_grads(theta_params, world_size)
            if float(args.grad_clip_norm) > 0:
                torch.nn.utils.clip_grad_norm_(theta_params, max_norm=float(args.grad_clip_norm))
            optimizer_theta.step()
            _maybe_sync_after_step(model=model, model_for_state=model_for_state, use_ddp=use_ddp, world_size=world_size)
            t_after_step = now_ts()

            step_det_loss = float(det_loss.item())
            step_seg_loss = float(seg_loss.item())
            step_cnt_loss = float(cnt_loss.item())
            step_loss = (
                base_loss_weights[0] * step_det_loss
                + base_loss_weights[1] * step_seg_loss
                + base_loss_weights[2] * step_cnt_loss
            )
            total_loss += step_loss
            det_loss_sum += step_det_loss
            seg_loss_sum += step_seg_loss
            cnt_loss_sum += step_cnt_loss
            steps += 1

            if block_selection_recorder is not None:
                block_selection_recorder.log_step(
                    epoch=int(epoch),
                    global_step=int(global_step),
                    step_in_epoch=int(step),
                    selected_pairs=mix_stats.selected_labels,
                    pair_counts=mix_stats.selection_counts,
                    block_pair_counts=mix_stats.block_selection_counts,
                    pair_scores=mix_stats.pair_scores,
                )

            if metric_logger is not None:
                metric_logger.log_metrics(
                    {
                        "stage2/epoch": int(epoch),
                        "stage2/step_in_epoch": int(step),
                        "stage2/step_pair_score_det_seg": float(mix_stats.pair_scores["det_seg"]),
                        "stage2/step_pair_score_det_cnt": float(mix_stats.pair_scores["det_cnt"]),
                        "stage2/step_pair_score_seg_cnt": float(mix_stats.pair_scores["seg_cnt"]),
                        "stage2/step_positive_pair_count_det_seg": int(mix_stats.positive_pair_counts["det_seg"]),
                        "stage2/step_positive_pair_count_det_cnt": int(mix_stats.positive_pair_counts["det_cnt"]),
                        "stage2/step_positive_pair_count_seg_cnt": int(mix_stats.positive_pair_counts["seg_cnt"]),
                        "stage2/step_task_update_count_det": int(mix_stats.task_update_counts["det"]),
                        "stage2/step_task_update_count_seg": int(mix_stats.task_update_counts["seg"]),
                        "stage2/step_task_update_count_cnt": int(mix_stats.task_update_counts["cnt"]),
                        "stage2/step_task_update_ratio_det": float(mix_stats.task_update_counts["det"] / total_matrix_units),
                        "stage2/step_task_update_ratio_seg": float(mix_stats.task_update_counts["seg"] / total_matrix_units),
                        "stage2/step_task_update_ratio_cnt": float(mix_stats.task_update_counts["cnt"] / total_matrix_units),
                        "stage2/step_update_mode_count_all": int(mix_stats.update_mode_counts["all"]),
                        "stage2/step_update_mode_count_shared": int(mix_stats.update_mode_counts["shared"]),
                        "stage2/step_update_mode_count_none": int(mix_stats.update_mode_counts["none"]),
                        "stage2/step_update_mode_ratio_all": float(mix_stats.update_mode_counts["all"] / total_matrix_units),
                        "stage2/step_update_mode_ratio_shared": float(mix_stats.update_mode_counts["shared"] / total_matrix_units),
                        "stage2/step_update_mode_ratio_none": float(mix_stats.update_mode_counts["none"] / total_matrix_units),
                    },
                    step=global_step,
                )

            should_log_step_metrics = (not args.log_interval) or (step % int(args.log_interval) == 0)
            if should_log_step_metrics:
                print(
                    f"[stage2] epoch {epoch}/{int(args.stage2_epochs)} step {step} | "
                    f"loss {step_loss:.4f} | "
                    f"train det {step_det_loss:.4f} seg {step_seg_loss:.4f} cnt {step_cnt_loss:.4f} | "
                    f"shared_update det {mix_stats.task_update_counts['det']}/{total_matrix_units} "
                    f"seg {mix_stats.task_update_counts['seg']}/{total_matrix_units} "
                    f"cnt {mix_stats.task_update_counts['cnt']}/{total_matrix_units} | "
                    f"modes all {mix_stats.update_mode_counts['all']}/{total_matrix_units} "
                    f"shared {mix_stats.update_mode_counts['shared']}/{total_matrix_units} "
                    f"none {mix_stats.update_mode_counts['none']}/{total_matrix_units} | "
                    f"positive_pairs det_seg {mix_stats.positive_pair_counts['det_seg']}/{total_matrix_units} "
                    f"det_cnt {mix_stats.positive_pair_counts['det_cnt']}/{total_matrix_units} "
                    f"seg_cnt {mix_stats.positive_pair_counts['seg_cnt']}/{total_matrix_units} | "
                    f"scores det_seg {mix_stats.pair_scores['det_seg']:.4f} "
                    f"det_cnt {mix_stats.pair_scores['det_cnt']:.4f} "
                    f"seg_cnt {mix_stats.pair_scores['seg_cnt']:.4f}"
                )
            if should_log_step_metrics and metric_logger is not None:
                metric_logger.log_metrics(
                    {
                        "stage2/epoch": int(epoch),
                        "stage2/step_in_epoch": int(step),
                        "stage2/step_loss": float(step_loss),
                        "stage2/step_train_det_loss": float(step_det_loss),
                        "stage2/step_train_seg_loss": float(step_seg_loss),
                        "stage2/step_train_cnt_loss": float(step_cnt_loss),
                        "stage2/step_weight_det": float(base_loss_weights[0]),
                        "stage2/step_weight_seg": float(base_loss_weights[1]),
                        "stage2/step_weight_cnt": float(base_loss_weights[2]),
                    },
                    step=global_step,
                )
            if debug_timing and step % timing_interval == 0:
                print(
                    f"[stage2][timing] epoch {epoch}/{int(args.stage2_epochs)} step {step} | "
                    f"batch {t_after_batch-step_t0:.3f}s | "
                    f"task_loss {t_after_loss-t_after_batch:.3f}s | "
                    f"grad_mix {t_after_mix-t_after_loss:.3f}s | "
                    f"optimizer {t_after_step-t_after_mix:.3f}s | "
                    f"step_total {t_after_step-step_t0:.3f}s"
                )

            if int(args.max_train_steps) and step >= int(args.max_train_steps):
                break

        if block_selection_recorder is not None:
            block_selection_recorder.log_epoch(
                epoch=int(epoch),
                steps=int(steps),
                block_pair_counts=epoch_block_selection_counts,
            )
        print(
            f"[stage2] epoch {epoch}/{int(args.stage2_epochs)} | "
            f"train {total_loss/max(steps,1):.4f} | "
            f"det {det_loss_sum/max(steps,1):.4f} seg {seg_loss_sum/max(steps,1):.4f} cnt {cnt_loss_sum/max(steps,1):.4f} | "
            f"task_updates det {task_update_counts['det']} seg {task_update_counts['seg']} cnt {task_update_counts['cnt']} | "
            f"modes all {update_mode_counts['all']} shared {update_mode_counts['shared']} none {update_mode_counts['none']} | "
            f"positive_pairs det_seg {positive_pair_counts['det_seg']} "
            f"det_cnt {positive_pair_counts['det_cnt']} seg_cnt {positive_pair_counts['seg_cnt']}"
        )
        if metric_logger is not None:
            epoch_matrix_unit_total = max(steps * total_matrix_units, 1)
            metric_logger.log_metrics(
                {
                    "stage2/epoch": int(epoch),
                    "stage2/epoch_train_loss": float(total_loss / max(steps, 1)),
                    "stage2/epoch_train_det_loss": float(det_loss_sum / max(steps, 1)),
                    "stage2/epoch_train_seg_loss": float(seg_loss_sum / max(steps, 1)),
                    "stage2/epoch_train_cnt_loss": float(cnt_loss_sum / max(steps, 1)),
                    "stage2/epoch_positive_pair_count_det_seg": int(positive_pair_counts["det_seg"]),
                    "stage2/epoch_positive_pair_count_det_cnt": int(positive_pair_counts["det_cnt"]),
                    "stage2/epoch_positive_pair_count_seg_cnt": int(positive_pair_counts["seg_cnt"]),
                    "stage2/epoch_positive_pair_ratio_det_seg": float(positive_pair_counts["det_seg"] / epoch_matrix_unit_total),
                    "stage2/epoch_positive_pair_ratio_det_cnt": float(positive_pair_counts["det_cnt"] / epoch_matrix_unit_total),
                    "stage2/epoch_positive_pair_ratio_seg_cnt": float(positive_pair_counts["seg_cnt"] / epoch_matrix_unit_total),
                    "stage2/epoch_task_update_count_det": int(task_update_counts["det"]),
                    "stage2/epoch_task_update_count_seg": int(task_update_counts["seg"]),
                    "stage2/epoch_task_update_count_cnt": int(task_update_counts["cnt"]),
                    "stage2/epoch_task_update_ratio_det": float(task_update_counts["det"] / epoch_matrix_unit_total),
                    "stage2/epoch_task_update_ratio_seg": float(task_update_counts["seg"] / epoch_matrix_unit_total),
                    "stage2/epoch_task_update_ratio_cnt": float(task_update_counts["cnt"] / epoch_matrix_unit_total),
                    "stage2/epoch_update_mode_count_all": int(update_mode_counts["all"]),
                    "stage2/epoch_update_mode_count_shared": int(update_mode_counts["shared"]),
                    "stage2/epoch_update_mode_count_none": int(update_mode_counts["none"]),
                    "stage2/epoch_update_mode_ratio_all": float(update_mode_counts["all"] / epoch_matrix_unit_total),
                    "stage2/epoch_update_mode_ratio_shared": float(update_mode_counts["shared"] / epoch_matrix_unit_total),
                    "stage2/epoch_update_mode_ratio_none": float(update_mode_counts["none"] / epoch_matrix_unit_total),
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
                    print(f"[ckpt] new best cached (stage2 epoch {epoch}, combo {best_metric:.6f})")

    if is_main_process and best_state is None:
        best_state = state_dict_cpu_clone(model_for_state.state_dict())
        best_epoch = int(args.stage2_epochs)
        best_metric = float("nan")

    if is_main_process and best_state is not None:
        save_epoch = int(args.stage2_epochs) if best_epoch is None else int(best_epoch)
        model_for_state.load_state_dict(best_state)
        save_multitask_checkpoint(
            checkpoint_path,
            model=model_for_state,
            optimizer=optimizer_theta,
            epoch=save_epoch,
            best_by="combo",
            metrics={
                "best_metric": float(best_metric),
                "best_stage": "stage2",
                "best_epoch": save_epoch,
            },
            loss_weights=base_loss_weights,
            config={
                "training_mode": "stage2_matrixwise_pair",
                "stage1_epochs": int(args.stage1_epochs),
                "stage2_epochs": int(args.stage2_epochs),
                "loss_weights": tuple(float(x) for x in base_loss_weights),
                "shared_update_rule": "matrixwise_positive_cosine_sign_rule",
                "use_lora_moe": bool(args.use_lora_moe),
                "unfreeze_backbone": bool(args.unfreeze_backbone),
                "backbone_lr": float(args.backbone_lr)
                if args.backbone_lr is not None
                else float(args.lr) * float(args.backbone_lr_mult),
                "lora_rank": int(args.lora_rank),
                "num_experts_private": int(args.num_experts_private),
                "num_experts_shared": int(args.num_experts_shared),
                "moe_k_private": int(args.moe_k_private),
                "moe_k_shared": int(args.moe_k_shared),
                "stage1_checkpoint": str(save_dir / "stage1_best.pt"),
            },
        )
        print(f"[ckpt] saved stage2 best -> {checkpoint_path} (combo {best_metric:.6f})")

    return StageArtifacts(
        best_metric=best_metric,
        best_epoch=best_epoch,
        checkpoint_path=checkpoint_path,
    )
