from __future__ import annotations

from dataclasses import dataclass
import time

import torch

from .train_utils import (
    compute_task_losses,
    ddp_allreduce_float_buffers,
    ddp_allreduce_param_grads,
    run_stage2_validation,
    state_dict_cpu_clone,
)
from .utils import parse_loss_weights


@dataclass
class WarmupArtifacts:
    # warmup 结束后返回最佳基模信息。
    best_metric: float
    best_epoch: int
    selected_by: str


def run_warmup(
    *,
    args,
    model,
    model_for_state,
    optimizer_theta,
    theta_params,
    train_loaders,
    val_loaders,
    primary_task,
    device: torch.device,
    use_ddp: bool,
    world_size: int,
    manual_theta_grad_sync: bool,
    det_num_classes: int,
    is_main_process: bool,
) -> WarmupArtifacts:
    # warmup 只做固定权重监督训练，但后 20 个 epoch 用正式 val 选最佳基模。
    other_tasks = [name for name in train_loaders.keys() if name != primary_task]
    warmup_weights = parse_loss_weights(getattr(args, "warmup_loss_weights", "15,8,1"))
    debug_timing = bool(getattr(args, "debug_step_timing", False))
    timing_interval = max(int(getattr(args, "debug_step_timing_interval", 1)), 1)
    val_start_epoch = max(1, int(args.warmup_epochs) - 20 + 1)
    best_metric = float("-inf")
    best_epoch = int(args.warmup_epochs)
    best_state = None

    def now_ts() -> float:
        if debug_timing and device.type == "cuda":
            torch.cuda.synchronize(device)
        return time.perf_counter()

    for epoch in range(1, int(args.warmup_epochs) + 1):
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
            step_t0 = now_ts()
            if debug_timing and step % timing_interval == 0:
                print(f"[warmup][timing] epoch {epoch}/{int(args.warmup_epochs)} step {step} start")

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
            det_loss, seg_loss, cnt_loss = compute_task_losses(
                model,
                batches["det"],
                batches["seg"],
                batches["cnt"],
                device=device,
                cnt_count_loss_weight=float(args.cnt_count_loss_weight),
                cnt_backbone_grad_mult=float(args.cnt_backbone_grad_mult),
            )
            total = (
                float(warmup_weights[0]) * det_loss
                + float(warmup_weights[1]) * seg_loss
                + float(warmup_weights[2]) * cnt_loss
            )
            t_after_loss = now_ts()

            total.backward()
            if manual_theta_grad_sync:
                ddp_allreduce_param_grads(theta_params, world_size)
            if float(args.grad_clip_norm) > 0:
                torch.nn.utils.clip_grad_norm_(theta_params, max_norm=float(args.grad_clip_norm))
            optimizer_theta.step()
            if use_ddp:
                ddp_allreduce_float_buffers(model, world_size)
            t_after_step = now_ts()

            total_loss += float(total.detach().item())
            step_det_loss = float(det_loss.detach().item())
            step_seg_loss = float(seg_loss.detach().item())
            step_cnt_loss = float(cnt_loss.detach().item())
            step_loss = float(total.detach().item())
            det_loss_sum += step_det_loss
            seg_loss_sum += step_seg_loss
            cnt_loss_sum += step_cnt_loss
            steps += 1

            if args.log_interval and step % int(args.log_interval) == 0:
                print(
                    f"[warmup] epoch {epoch}/{int(args.warmup_epochs)} step {step} | "
                    f"loss {step_loss:.4f} | "
                    f"w [{warmup_weights[0]:.1f}, {warmup_weights[1]:.1f}, {warmup_weights[2]:.1f}] | "
                    f"train det {step_det_loss:.4f} seg {step_seg_loss:.4f} cnt {step_cnt_loss:.4f}"
                )
            if debug_timing and step % timing_interval == 0:
                print(
                    f"[warmup][timing] epoch {epoch}/{int(args.warmup_epochs)} step {step} | "
                    f"batch {t_after_batch-step_t0:.3f}s | "
                    f"task_loss {t_after_loss-t_after_batch:.3f}s | "
                    f"backward_step {t_after_step-t_after_loss:.3f}s | "
                    f"step_total {t_after_step-step_t0:.3f}s"
                )

            if int(args.max_train_steps) and step >= int(args.max_train_steps):
                break

        print(
            f"[warmup] epoch {epoch}/{int(args.warmup_epochs)} | "
            f"loss {total_loss/max(steps,1):.4f} | "
            f"w [{warmup_weights[0]:.1f}, {warmup_weights[1]:.1f}, {warmup_weights[2]:.1f}] | "
            f"det {det_loss_sum/max(steps,1):.4f} seg {seg_loss_sum/max(steps,1):.4f} "
            f"cnt {cnt_loss_sum/max(steps,1):.4f}"
        )

        if epoch >= val_start_epoch:
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
                stage_name="warmup",
            )
            if result.combo_metric > best_metric:
                best_metric = float(result.combo_metric)
                best_epoch = epoch
                if is_main_process:
                    best_state = state_dict_cpu_clone(model_for_state.state_dict())
                    print(f"[ckpt] new warmup best cached (epoch {epoch}, combo {best_metric:.6f})")

    if best_state is not None:
        model_for_state.load_state_dict(best_state)
        selected_by = "warmup_combo"
    else:
        selected_by = "warmup_last"

    return WarmupArtifacts(
        best_metric=best_metric if best_state is not None else float("nan"),
        best_epoch=best_epoch,
        selected_by=selected_by,
    )
