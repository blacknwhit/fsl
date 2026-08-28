from __future__ import annotations

from dataclasses import dataclass
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .policy import (
    build_dirichlet,
    compute_clipped_grpo_loss,
    dirichlet_log_prob,
    sample_candidates,
    scale_task_weights,
    smooth_task_weights,
)
from .train_utils import (
    assign_grads,
    capture_param_grads,
    combine_mixed_grads,
    ddp_allreduce_float_buffers,
    ddp_allreduce_param_grads,
    is_ddp_wrapped,
    maybe_no_sync,
    sync_grad_tuple,
    temporary_param_updates,
    to_device_cnt,
    to_device_det,
    to_device_seg,
    validation_total_loss,
)


@dataclass
class Stage1Artifacts:
    # Stage1 结束后要传给 Stage2 的状态。
    generator_state: dict
    state_feature_state: dict
    last_weights: tuple[float, float, float]


def run_stage1(
    *,
    args,
    model,
    model_for_state,
    policy,
    state_tracker,
    optimizer_theta,
    optimizer_policy,
    theta_params,
    theta_param_names,
    base_loss_weights,
    shared_param_mask,
    train_loaders,
    val_loaders,
    primary_task,
    device: torch.device,
    use_ddp: bool,
    world_size: int,
    manual_theta_grad_sync: bool,
    metric_logger=None,
) -> Stage1Artifacts:
    # Stage1 先更新生成器侧，再更新主模型。
    other_tasks = [name for name in train_loaders.keys() if name != primary_task]
    reward_eps = 1e-8
    policy_kl_beta = float(getattr(args, "policy_kl_beta", 0.0))
    base_loss_weights = tuple(float(x) for x in base_loss_weights)
    shared_param_mask = tuple(bool(flag) for flag in shared_param_mask)
    policy_kl_prior = torch.tensor(base_loss_weights, device=device, dtype=torch.float32)
    last_weights = base_loss_weights
    debug_timing = bool(getattr(args, "debug_step_timing", False))
    timing_interval = max(int(getattr(args, "debug_step_timing_interval", 1)), 1)
    debug_reward = bool(getattr(args, "debug_reward_details", False))
    reward_interval = max(int(getattr(args, "debug_reward_details_interval", 1)), 1)
    policy_side_params = list(policy.parameters()) + list(state_tracker.parameters())
    theta_index_by_id = {id(param): idx for idx, param in enumerate(theta_params)}
    state_a_idx = theta_index_by_id.get(id(state_tracker.last_moe.lora_A_shared))
    state_b_idx = theta_index_by_id.get(id(state_tracker.last_moe.lora_B_shared))
    global_step = 0

    def now_ts() -> float:
        # timing 日志前同步 CUDA，避免异步误差。
        if debug_timing and device.type == "cuda":
            torch.cuda.synchronize(device)
        return time.perf_counter()

    def grad_l2_norm(grads) -> float:
        # 统计一组梯度的 L2 范数。
        total_sq = 0.0
        for grad in grads:
            if grad is None:
                continue
            total_sq += float(grad.detach().float().pow(2).sum().item())
        return total_sq ** 0.5

    def encode_task_state(grads: tuple[torch.Tensor | None, ...], *, grads_are_synced: bool) -> torch.Tensor:
        grad_a = grads[state_a_idx] if state_a_idx is not None else None
        grad_b = grads[state_b_idx] if state_b_idx is not None else None
        return state_tracker.encode_shared_grad_state(
            grad_a,
            grad_b,
            device=device,
            use_ddp=use_ddp and (not grads_are_synced),
            world_size=world_size,
        )

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
            if use_ddp:
                grads = sync_grad_tuple(theta_params, grads, world_size)
        optimizer_theta.zero_grad(set_to_none=True)
        task_state = encode_task_state(grads, grads_are_synced=use_ddp)
        return loss.detach(), grads, task_state

    def update_l2_norm(grads: tuple[torch.Tensor | None, ...], step_size: float) -> float:
        # 统计一次虚拟更新的位移大小。
        total_sq = 0.0
        scale = float(step_size)
        for grad in grads:
            if grad is None:
                continue
            delta = grad.detach().float() * scale
            total_sq += float(delta.pow(2).sum().item())
        return total_sq ** 0.5

    def relative_improvement(old_loss: float, new_loss: float) -> float:
        # Δ_i = (L_i^old - L_i^new) / (L_i^old + ε)
        return (float(old_loss) - float(new_loss)) / (float(old_loss) + reward_eps)

    def harmonic_reward(delta_det: float, delta_seg: float, delta_cnt: float) -> float:
        # r_k = 3 / (1/(Δ_d+ε) + 1/(Δ_s+ε) + 1/(Δ_c+ε))
        denom = (1.0 / (delta_det + reward_eps)) + (1.0 / (delta_seg + reward_eps)) + (1.0 / (delta_cnt + reward_eps))
        return 3.0 / denom

    for epoch in range(1, int(args.stage1_epochs) + 1):
        if use_ddp:
            for loader in train_loaders.values():
                sampler = getattr(loader, "sampler", None)
                if isinstance(sampler, torch.utils.data.distributed.DistributedSampler):
                    sampler.set_epoch(epoch + int(getattr(args, "warmup_epochs", 0)))

        other_iters = {name: iter(train_loaders[name]) for name in other_tasks}
        model.train()
        policy.train()
        state_tracker.train()
        total_loss = 0.0
        reward_mean_sum = 0.0
        reward_std_sum = 0.0
        steps = 0
        last_val_breakdown = {"det": float("nan"), "seg": float("nan"), "cnt": float("nan")}

        for step, primary_batch in enumerate(train_loaders[primary_task], start=1):
            global_step += 1
            step_t0 = now_ts()
            if debug_timing and step % timing_interval == 0:
                print(f"[stage1][timing] epoch {epoch}/{int(args.stage1_epochs)} step {step} start")

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
            optimizer_policy.zero_grad(set_to_none=True)

            det_loss, det_grads, det_state = compute_task_loss_and_grads("det", batches["det"])
            seg_loss, seg_grads, seg_state = compute_task_loss_and_grads("seg", batches["seg"])
            cnt_loss, cnt_grads, cnt_state = compute_task_loss_and_grads("cnt", batches["cnt"])
            t_after_loss = now_ts()

            state = torch.cat([det_state, seg_state, cnt_state], dim=0)
            t_after_state = now_ts()

            task_grads = {"det": det_grads, "seg": seg_grads, "cnt": cnt_grads}
            t_after_task_grads = now_ts()

            if debug_reward and step % reward_interval == 0:
                print(
                    f"[stage1][reward] epoch {epoch}/{int(args.stage1_epochs)} step {step} | "
                    f"raw_loss det {float(det_loss.detach().item()):.6e} "
                    f"seg {float(seg_loss.detach().item()):.6e} "
                    f"cnt {float(cnt_loss.detach().item()):.6e} | "
                    f"grad_norm det {grad_l2_norm(task_grads['det']):.6e} "
                    f"seg {grad_l2_norm(task_grads['seg']):.6e} "
                    f"cnt {grad_l2_norm(task_grads['cnt']):.6e}"
                )

            with torch.no_grad():
                if debug_timing and step % timing_interval == 0:
                    print(f"[stage1][timing] epoch {epoch}/{int(args.stage1_epochs)} step {step} reward_eval start")

                _old_dist, old_output = build_dirichlet(policy, state)
                raw_samples = sample_candidates(old_output.alpha, int(args.num_candidates))
                if use_ddp and dist.is_initialized():
                    dist.broadcast(raw_samples, src=0)
                smoothed_samples = smooth_task_weights(raw_samples, float(args.candidate_smoothing_gamma))
                scaled_samples = scale_task_weights(smoothed_samples, policy.weight_scale.detach())

                if debug_reward and step % reward_interval == 0:
                    state_cpu = state.detach().cpu()
                    alpha_cpu = old_output.alpha.detach().cpu()
                    mu_cpu = old_output.mu.detach().cpu()
                    delta_cpu = old_output.delta.detach().cpu()
                    print(
                        f"[stage1][reward] state mean {float(state_cpu.mean().item()):.6e} "
                        f"std {float(state_cpu.std(unbiased=False).item()):.6e} "
                        f"min {float(state_cpu.min().item()):.6e} max {float(state_cpu.max().item()):.6e}"
                    )
                    print(
                        f"[stage1][reward] old_policy mu "
                        f"[{mu_cpu[0]:.6f}, {mu_cpu[1]:.6f}, {mu_cpu[2]:.6f}] | "
                        f"delta [{delta_cpu[0]:.6f}, {delta_cpu[1]:.6f}, {delta_cpu[2]:.6f}] | "
                        f"alpha [{alpha_cpu[0]:.6e}, {alpha_cpu[1]:.6e}, {alpha_cpu[2]:.6e}] | "
                        f"concentration {float(old_output.concentration.detach().cpu().item()):.6e}"
                    )

                base_val_loss, base_val_breakdown = validation_total_loss(
                    model,
                    val_loaders,
                    device=device,
                    cnt_count_loss_weight=float(args.cnt_count_loss_weight),
                    cnt_backbone_grad_mult=float(args.cnt_backbone_grad_mult),
                    max_val_steps=int(args.max_val_steps),
                    return_breakdown=True,
                )
                last_val_breakdown = base_val_breakdown
                t_after_base_val = now_ts()

                if debug_timing and step % timing_interval == 0:
                    print(
                        f"[stage1][timing] epoch {epoch}/{int(args.stage1_epochs)} step {step} "
                        f"base_val done ({t_after_base_val-t_after_task_grads:.3f}s), "
                        f"candidates {int(args.num_candidates)} start"
                    )
                if debug_reward and step % reward_interval == 0:
                    print(f"[stage1][reward] base_val_loss {base_val_loss:.6e}")

                rewards = []
                candidate_val_losses = []
                candidate_val_breakdowns = []
                candidate_deltas = []
                candidate_update_norms = []
                candidate_grad_norms = []
                for sample in scaled_samples:
                    candidate_grads = combine_mixed_grads(
                        task_grads,
                        sample,
                        base_loss_weights,
                        shared_param_mask,
                    )
                    candidate_grad_norms.append(grad_l2_norm(candidate_grads))
                    candidate_update_norms.append(update_l2_norm(candidate_grads, float(args.meta_alpha)))
                    with temporary_param_updates(theta_params, candidate_grads, float(args.meta_alpha)):
                        candidate_val_loss, candidate_val_breakdown = validation_total_loss(
                            model,
                            val_loaders,
                            device=device,
                            cnt_count_loss_weight=float(args.cnt_count_loss_weight),
                            cnt_backbone_grad_mult=float(args.cnt_backbone_grad_mult),
                            max_val_steps=int(args.max_val_steps),
                            return_breakdown=True,
                        )
                    candidate_val_losses.append(candidate_val_loss)
                    candidate_val_breakdowns.append(candidate_val_breakdown)
                    delta_det = relative_improvement(base_val_breakdown["det"], candidate_val_breakdown["det"])
                    delta_seg = relative_improvement(base_val_breakdown["seg"], candidate_val_breakdown["seg"])
                    delta_cnt = relative_improvement(base_val_breakdown["cnt"], candidate_val_breakdown["cnt"])
                    candidate_deltas.append((delta_det, delta_seg, delta_cnt))
                    rewards.append(harmonic_reward(delta_det, delta_seg, delta_cnt))

                t_after_candidates = now_ts()
                rewards_tensor = torch.tensor(rewards, device=device, dtype=torch.float64)
                reward_mean = rewards_tensor.mean()
                reward_std = rewards_tensor.std(unbiased=False)
                advantages = (rewards_tensor - reward_mean) / (reward_std + 1e-8)

            if debug_reward and step % reward_interval == 0:
                raw_cpu = raw_samples.detach().cpu()
                smooth_cpu = smoothed_samples.detach().cpu()
                scaled_cpu = scaled_samples.detach().cpu()
                rewards_cpu = rewards_tensor.detach().cpu()
                adv_cpu = advantages.detach().cpu()
                for idx in range(int(args.num_candidates)):
                    delta_det, delta_seg, delta_cnt = candidate_deltas[idx]
                    print(
                        f"[stage1][reward][cand{idx}] "
                        f"raw [{raw_cpu[idx,0]:.6f}, {raw_cpu[idx,1]:.6f}, {raw_cpu[idx,2]:.6f}] | "
                        f"smooth [{smooth_cpu[idx,0]:.6f}, {smooth_cpu[idx,1]:.6f}, {smooth_cpu[idx,2]:.6f}] | "
                        f"shared [{scaled_cpu[idx,0]:.6f}, {scaled_cpu[idx,1]:.6f}, {scaled_cpu[idx,2]:.6f}] | "
                        f"grad_norm {candidate_grad_norms[idx]:.6e} | "
                        f"update_norm {candidate_update_norms[idx]:.6e} | "
                        f"val_total {candidate_val_losses[idx]:.6e} | "
                        f"val_det {candidate_val_breakdowns[idx]['det']:.6e} "
                        f"val_seg {candidate_val_breakdowns[idx]['seg']:.6e} "
                        f"val_cnt {candidate_val_breakdowns[idx]['cnt']:.6e} | "
                        f"delta [{delta_det:.6e}, {delta_seg:.6e}, {delta_cnt:.6e}] | "
                        f"reward {float(rewards_cpu[idx].item()):.6e} | "
                        f"adv {float(adv_cpu[idx].item()):.6e}"
                    )
                print(
                    f"[stage1][reward] summary | "
                    f"reward_min {float(rewards_cpu.min().item()):.6e} "
                    f"reward_max {float(rewards_cpu.max().item()):.6e} "
                    f"reward_std {float(reward_std.detach().item()):.6e}"
                )

            old_logp = dirichlet_log_prob(old_output.alpha.detach(), raw_samples)
            new_output = policy.build_dirichlet(state)
            new_logp = dirichlet_log_prob(new_output.alpha, raw_samples)
            policy_loss, _ratio = compute_clipped_grpo_loss(
                new_logp,
                old_logp.detach(),
                advantages.detach(),
                float(args.grpo_clip_eps),
            )
            prior_alpha = policy_kl_prior.to(device=new_output.alpha.device, dtype=new_output.alpha.dtype)
            policy_kl = torch.distributions.kl.kl_divergence(
                torch.distributions.Dirichlet(new_output.alpha),
                torch.distributions.Dirichlet(prior_alpha),
            )
            policy_total_loss = policy_loss + policy_kl_beta * policy_kl.mean()
            policy_total_loss.backward()
            if use_ddp:
                ddp_allreduce_param_grads(policy_side_params, world_size)
            if float(args.phi_grad_clip_norm) > 0:
                torch.nn.utils.clip_grad_norm_(policy_side_params, max_norm=float(args.phi_grad_clip_norm))
            optimizer_policy.step()
            t_after_policy = now_ts()

            optimizer_theta.zero_grad(set_to_none=True)
            final_output = policy.build_dirichlet(state)
            final_weights = scale_task_weights(final_output.mu.detach(), policy.weight_scale.detach())
            final_grads = combine_mixed_grads(
                task_grads,
                final_weights,
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

            last_weights = tuple(float(x) for x in final_weights.detach().cpu().tolist())
            step_det_loss = float(det_loss.detach().item())
            step_seg_loss = float(seg_loss.detach().item())
            step_cnt_loss = float(cnt_loss.detach().item())
            step_loss = (
                base_loss_weights[0] * step_det_loss
                + base_loss_weights[1] * step_seg_loss
                + base_loss_weights[2] * step_cnt_loss
            )
            total_loss += step_loss
            reward_mean_sum += float(reward_mean.detach().item())
            reward_std_sum += float(reward_std.detach().item())
            steps += 1

            if args.log_interval and step % int(args.log_interval) == 0:
                print(
                    f"[stage1] epoch {epoch}/{int(args.stage1_epochs)} step {step} | "
                    f"loss {step_loss:.4f} | "
                    f"train det {step_det_loss:.4f} seg {step_seg_loss:.4f} cnt {step_cnt_loss:.4f} | "
                    f"val det {last_val_breakdown['det']:.4f} seg {last_val_breakdown['seg']:.4f} "
                    f"cnt {last_val_breakdown['cnt']:.4f} total {base_val_loss:.4f} | "
                    f"shared_w [{last_weights[0]:.3f}, {last_weights[1]:.3f}, {last_weights[2]:.3f}] | "
                    f"fixed_w [{base_loss_weights[0]:.3f}, {base_loss_weights[1]:.3f}, {base_loss_weights[2]:.3f}] | "
                    f"reward_mean {float(reward_mean.detach().item()):.6f} "
                    f"reward_std {float(reward_std.detach().item()):.6f}"
                )
                if metric_logger is not None:
                    metric_logger.log_metrics(
                        {
                            "stage1/epoch": int(epoch),
                            "stage1/step_in_epoch": int(step),
                            "stage1/step_loss": float(step_loss),
                            "stage1/step_train_det_loss": float(step_det_loss),
                            "stage1/step_train_seg_loss": float(step_seg_loss),
                            "stage1/step_train_cnt_loss": float(step_cnt_loss),
                            "stage1/step_val_det_loss": float(last_val_breakdown["det"]),
                            "stage1/step_val_seg_loss": float(last_val_breakdown["seg"]),
                            "stage1/step_val_cnt_loss": float(last_val_breakdown["cnt"]),
                            "stage1/step_val_total_loss": float(base_val_loss),
                            "stage1/step_weight_det": float(last_weights[0]),
                            "stage1/step_weight_seg": float(last_weights[1]),
                            "stage1/step_weight_cnt": float(last_weights[2]),
                            "stage1/step_shared_weight_det": float(last_weights[0]),
                            "stage1/step_shared_weight_seg": float(last_weights[1]),
                            "stage1/step_shared_weight_cnt": float(last_weights[2]),
                            "stage1/step_fixed_weight_det": float(base_loss_weights[0]),
                            "stage1/step_fixed_weight_seg": float(base_loss_weights[1]),
                            "stage1/step_fixed_weight_cnt": float(base_loss_weights[2]),
                            "stage1/step_reward_mean": float(reward_mean.detach().item()),
                            "stage1/step_reward_std": float(reward_std.detach().item()),
                        },
                        step=global_step,
                    )
            if debug_timing and step % timing_interval == 0:
                print(
                    f"[stage1][timing] epoch {epoch}/{int(args.stage1_epochs)} step {step} | "
                    f"batch {t_after_batch-step_t0:.3f}s | "
                    f"task_loss {t_after_loss-t_after_batch:.3f}s | "
                    f"state {t_after_state-t_after_loss:.3f}s | "
                    f"task_grads {t_after_task_grads-t_after_state:.3f}s | "
                    f"base_val {t_after_base_val-t_after_task_grads:.3f}s | "
                    f"candidates_total {t_after_candidates-t_after_base_val:.3f}s | "
                    f"policy_update {t_after_policy-t_after_candidates:.3f}s | "
                    f"theta_update {t_after_theta-t_after_policy:.3f}s | "
                    f"step_total {t_after_theta-step_t0:.3f}s"
                )

            if int(args.max_train_steps) and step >= int(args.max_train_steps):
                break

        print(
            f"[stage1] epoch {epoch}/{int(args.stage1_epochs)} | "
            f"train {total_loss/max(steps,1):.4f} | "
            f"reward_mean {reward_mean_sum/max(steps,1):.6f} "
            f"reward_std {reward_std_sum/max(steps,1):.6f} | "
            f"shared_w [{last_weights[0]:.3f}, {last_weights[1]:.3f}, {last_weights[2]:.3f}] | "
            f"fixed_w [{base_loss_weights[0]:.3f}, {base_loss_weights[1]:.3f}, {base_loss_weights[2]:.3f}]"
        )
        if metric_logger is not None:
            metric_logger.log_metrics(
                {
                    "stage1/epoch": int(epoch),
                    "stage1/epoch_train_loss": float(total_loss / max(steps, 1)),
                    "stage1/epoch_reward_mean": float(reward_mean_sum / max(steps, 1)),
                    "stage1/epoch_reward_std": float(reward_std_sum / max(steps, 1)),
                    "stage1/epoch_weight_det": float(last_weights[0]),
                    "stage1/epoch_weight_seg": float(last_weights[1]),
                    "stage1/epoch_weight_cnt": float(last_weights[2]),
                    "stage1/epoch_shared_weight_det": float(last_weights[0]),
                    "stage1/epoch_shared_weight_seg": float(last_weights[1]),
                    "stage1/epoch_shared_weight_cnt": float(last_weights[2]),
                    "stage1/epoch_fixed_weight_det": float(base_loss_weights[0]),
                    "stage1/epoch_fixed_weight_seg": float(base_loss_weights[1]),
                    "stage1/epoch_fixed_weight_cnt": float(base_loss_weights[2]),
                },
                step=int(epoch),
            )

    return Stage1Artifacts(
        generator_state=policy.state_dict(),
        state_feature_state=state_tracker.state_dict(),
        last_weights=last_weights,
    )
