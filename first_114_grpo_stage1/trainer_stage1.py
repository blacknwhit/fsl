from __future__ import annotations

from dataclasses import dataclass
import time

import torch

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
    build_functional_state,
    build_virtual_updates,
    combine_weighted_grads,
    compute_task_losses,
    ddp_allreduce_float_buffers,
    ddp_allreduce_param_grads,
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
    train_loaders,
    val_loaders,
    primary_task,
    device: torch.device,
    use_ddp: bool,
    world_size: int,
    manual_theta_grad_sync: bool,
) -> Stage1Artifacts:
    # Stage1 先更新生成器侧，再更新主模型。
    other_tasks = [name for name in train_loaders.keys() if name != primary_task]
    last_weights = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    debug_timing = bool(getattr(args, "debug_step_timing", False))
    timing_interval = max(int(getattr(args, "debug_step_timing_interval", 1)), 1)
    debug_reward = bool(getattr(args, "debug_reward_details", False))
    reward_interval = max(int(getattr(args, "debug_reward_details_interval", 1)), 1)
    policy_side_params = list(policy.parameters()) + list(state_tracker.parameters())

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

    def update_l2_norm(params, updates) -> float:
        # 统计一次虚拟更新的位移大小。
        total_sq = 0.0
        for name, param in zip(theta_param_names, params):
            updated = updates[name]
            delta = updated.detach().float() - param.detach().float()
            total_sq += float(delta.pow(2).sum().item())
        return total_sq ** 0.5

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

            det_loss, seg_loss, cnt_loss = compute_task_losses(
                model,
                batches["det"],
                batches["seg"],
                batches["cnt"],
                device=device,
                cnt_count_loss_weight=float(args.cnt_count_loss_weight),
                cnt_backbone_grad_mult=float(args.cnt_backbone_grad_mult),
            )
            t_after_loss = now_ts()

            state = state_tracker.extract_features(
                det_loss=det_loss,
                seg_loss=seg_loss,
                cnt_loss=cnt_loss,
                device=device,
                use_ddp=use_ddp,
                world_size=world_size,
            )
            t_after_state = now_ts()

            task_grads = {
                "det": torch.autograd.grad(det_loss, theta_params, retain_graph=True, allow_unused=True),
                "seg": torch.autograd.grad(seg_loss, theta_params, retain_graph=True, allow_unused=True),
                "cnt": torch.autograd.grad(cnt_loss, theta_params, retain_graph=False, allow_unused=True),
            }
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
                    model_for_state,
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
                candidate_update_norms = []
                candidate_grad_norms = []
                for sample in scaled_samples:
                    candidate_grads = combine_weighted_grads(task_grads, sample)
                    updates = build_virtual_updates(
                        theta_param_names,
                        theta_params,
                        candidate_grads,
                        float(args.meta_alpha),
                    )
                    candidate_grad_norms.append(grad_l2_norm(candidate_grads))
                    candidate_update_norms.append(update_l2_norm(theta_params, updates))
                    params = build_functional_state(model_for_state, updates)
                    candidate_val_loss = validation_total_loss(
                        model_for_state,
                        val_loaders,
                        device=device,
                        cnt_count_loss_weight=float(args.cnt_count_loss_weight),
                        cnt_backbone_grad_mult=float(args.cnt_backbone_grad_mult),
                        max_val_steps=int(args.max_val_steps),
                        params=params,
                    )
                    candidate_val_losses.append(candidate_val_loss)
                    rewards.append(base_val_loss - candidate_val_loss)

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
                    print(
                        f"[stage1][reward][cand{idx}] "
                        f"raw [{raw_cpu[idx,0]:.6f}, {raw_cpu[idx,1]:.6f}, {raw_cpu[idx,2]:.6f}] | "
                        f"smooth [{smooth_cpu[idx,0]:.6f}, {smooth_cpu[idx,1]:.6f}, {smooth_cpu[idx,2]:.6f}] | "
                        f"scaled [{scaled_cpu[idx,0]:.6f}, {scaled_cpu[idx,1]:.6f}, {scaled_cpu[idx,2]:.6f}] | "
                        f"grad_norm {candidate_grad_norms[idx]:.6e} | "
                        f"update_norm {candidate_update_norms[idx]:.6e} | "
                        f"val {candidate_val_losses[idx]:.6e} | "
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
            policy_loss.backward()
            if use_ddp:
                ddp_allreduce_param_grads(policy_side_params, world_size)
            if float(args.phi_grad_clip_norm) > 0:
                torch.nn.utils.clip_grad_norm_(policy_side_params, max_norm=float(args.phi_grad_clip_norm))
            optimizer_policy.step()
            t_after_policy = now_ts()

            optimizer_theta.zero_grad(set_to_none=True)
            final_output = policy.build_dirichlet(state)
            final_weights = scale_task_weights(final_output.mu.detach(), policy.weight_scale.detach())
            final_grads = combine_weighted_grads(task_grads, final_weights)
            assign_grads(theta_params, final_grads)
            if manual_theta_grad_sync:
                ddp_allreduce_param_grads(theta_params, world_size)
            if float(args.grad_clip_norm) > 0:
                torch.nn.utils.clip_grad_norm_(theta_params, max_norm=float(args.grad_clip_norm))
            optimizer_theta.step()
            if use_ddp:
                ddp_allreduce_float_buffers(model_for_state, world_size)
            t_after_theta = now_ts()

            last_weights = tuple(float(x) for x in final_weights.detach().cpu().tolist())
            step_det_loss = float(det_loss.detach().item())
            step_seg_loss = float(seg_loss.detach().item())
            step_cnt_loss = float(cnt_loss.detach().item())
            step_loss = (
                last_weights[0] * step_det_loss
                + last_weights[1] * step_seg_loss
                + last_weights[2] * step_cnt_loss
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
                    f"w [{last_weights[0]:.3f}, {last_weights[1]:.3f}, {last_weights[2]:.3f}] | "
                    f"reward_mean {float(reward_mean.detach().item()):.6f} "
                    f"reward_std {float(reward_std.detach().item()):.6f}"
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
            f"w [{last_weights[0]:.3f}, {last_weights[1]:.3f}, {last_weights[2]:.3f}]"
        )

    return Stage1Artifacts(
        generator_state=policy.state_dict(),
        state_feature_state=state_tracker.state_dict(),
        last_weights=last_weights,
    )
