from __future__ import annotations

from contextlib import contextmanager, nullcontext
import math
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.parallel import DistributedDataParallel
try:
    from torch.func import functional_call
except Exception:
    from torch.nn.utils.stateless import functional_call

from .models import MultiTaskModel
from segmentation.utils import per_class_iou_from_confusion, update_confusion_matrix


def to_device_det(batch, device: torch.device):
    # 检测 batch 的张量搬运。
    images, targets = batch
    images = [img.to(device, non_blocking=True) for img in images]
    targets = [{k: v.to(device, non_blocking=True) for k, v in tgt.items()} for tgt in targets]
    return images, targets


def to_device_seg(batch, device: torch.device):
    # 分割 batch 的张量搬运。
    imgs, masks = batch
    return imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)


def to_device_cnt(batch, device: torch.device):
    # 计数 batch 的张量搬运，并统一为 float。
    imgs, dens = batch
    return imgs.to(device, non_blocking=True).float(), dens.to(device, non_blocking=True).float()


def ddp_allreduce_param_grads(params: list[torch.nn.Parameter], world_size: int) -> None:
    # 手动做参数梯度平均，供未包 DDP 的训练路径使用。
    if not dist.is_initialized():
        return
    for param in params:
        if param.grad is None:
            continue
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        param.grad.div_(float(world_size))


def ddp_allreduce_float_buffers(module: torch.nn.Module, world_size: int) -> None:
    if not dist.is_initialized():
        return
    for buffer in module.buffers():
        if not torch.is_tensor(buffer) or not buffer.dtype.is_floating_point:
            continue
        dist.all_reduce(buffer, op=dist.ReduceOp.SUM)
        buffer.div_(float(world_size))


def is_ddp_wrapped(module: torch.nn.Module) -> bool:
    return isinstance(module, DistributedDataParallel)


def maybe_no_sync(module: torch.nn.Module):
    if is_ddp_wrapped(module):
        return module.no_sync()
    return nullcontext()


def capture_param_grads(params: list[torch.nn.Parameter]) -> tuple[torch.Tensor | None, ...]:
    return tuple(None if param.grad is None else param.grad.detach() for param in params)


def sync_grad_tuple(
    params: list[torch.nn.Parameter],
    grads: tuple[torch.Tensor | None, ...],
    world_size: int,
) -> tuple[torch.Tensor | None, ...]:
    if not dist.is_initialized():
        return grads

    synced = []
    for param, grad in zip(params, grads):
        has_grad = torch.tensor(
            1 if grad is not None else 0,
            device=param.device,
            dtype=torch.int32,
        )
        dist.all_reduce(has_grad, op=dist.ReduceOp.SUM)
        if int(has_grad.item()) == 0:
            synced.append(None)
            continue
        buf = grad if grad is not None else torch.zeros_like(param, memory_format=torch.preserve_format)
        dist.all_reduce(buf, op=dist.ReduceOp.SUM)
        buf.div_(float(world_size))
        synced.append(buf)
    return tuple(synced)


@contextmanager
def freeze_batchnorm_stats(module: torch.nn.Module):
    bn_modules = [(submodule, bool(submodule.training)) for submodule in module.modules() if isinstance(submodule, _BatchNorm)]
    try:
        for submodule, _was_training in bn_modules:
            submodule.eval()
        yield
    finally:
        for submodule, was_training in bn_modules:
            submodule.train(was_training)


def param_delta_l2_norm(params: list[torch.nn.Parameter], refs: list[torch.Tensor]) -> float:
    # 计算一组参数相对参考值的整体变化量。
    total_sq = 0.0
    for param, ref in zip(params, refs):
        delta = param.detach().float() - ref
        total_sq += float(delta.pow(2).sum().item())
    return math.sqrt(total_sq)


def state_dict_cpu_clone(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    # 把状态拷到 CPU，便于缓存最佳权重。
    out: Dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if torch.is_tensor(value):
            out[key] = value.detach().cpu().clone()
        else:
            out[key] = deepcopy(value)
    return out


def build_functional_state(model: torch.nn.Module, updates: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    # functional_call 既要参数也要 buffer，因此统一组装成完整状态。
    state = {name: param for name, param in model.named_parameters()}
    for name, buffer in model.named_buffers():
        state[name] = buffer
    state.update(updates)
    return state


def _normalize_task_weights(weights: torch.Tensor | tuple[float, float, float] | list[float]) -> tuple[float, float, float]:
    if isinstance(weights, torch.Tensor):
        values = weights.detach().view(-1).tolist()
    else:
        values = list(weights)
    if len(values) != 3:
        raise ValueError(f"Expected 3 task weights, got {len(values)}")
    return float(values[0]), float(values[1]), float(values[2])


def _combine_param_grads(
    grads: tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None],
    weights: tuple[float, float, float],
) -> torch.Tensor | None:
    cur = None
    for task_idx, grad in enumerate(grads):
        if grad is None:
            continue
        contrib = float(weights[task_idx]) * grad.detach()
        cur = contrib if cur is None else cur + contrib
    return cur


def combine_weighted_grads(
    task_grads: dict[str, tuple[torch.Tensor | None, ...]],
    weights: torch.Tensor | tuple[float, float, float] | list[float],
) -> tuple[torch.Tensor | None, ...]:
    # 把 det/seg/cnt 三路梯度按权重线性组合。
    out = []
    normalized = _normalize_task_weights(weights)
    for grads in zip(task_grads["det"], task_grads["seg"], task_grads["cnt"]):
        out.append(_combine_param_grads(grads, normalized))
    return tuple(out)


def combine_mixed_grads(
    task_grads: dict[str, tuple[torch.Tensor | None, ...]],
    shared_weights: torch.Tensor | tuple[float, float, float] | list[float],
    base_weights: torch.Tensor | tuple[float, float, float] | list[float],
    shared_param_mask: tuple[bool, ...] | list[bool],
) -> tuple[torch.Tensor | None, ...]:
    shared_normalized = _normalize_task_weights(shared_weights)
    base_normalized = _normalize_task_weights(base_weights)
    mask = tuple(bool(flag) for flag in shared_param_mask)

    num_params = len(task_grads["det"])
    if len(task_grads["seg"]) != num_params or len(task_grads["cnt"]) != num_params:
        raise ValueError("Task gradient tuples must have identical lengths")
    if len(mask) != num_params:
        raise ValueError(f"shared_param_mask length {len(mask)} does not match parameter count {num_params}")

    out = []
    for use_shared_weights, grads in zip(mask, zip(task_grads["det"], task_grads["seg"], task_grads["cnt"])):
        weights = shared_normalized if use_shared_weights else base_normalized
        out.append(_combine_param_grads(grads, weights))
    return tuple(out)


@dataclass(frozen=True)
class SharedExpertBlockIndex:
    block_param_indices: Dict[int, Dict[str, int]]
    param_to_block: Dict[int, int]


@dataclass(frozen=True)
class SharedExpertMatrixUnit:
    block_id: int
    param_index: int
    expert_index: int
    matrix_key: str


@dataclass(frozen=True)
class SharedExpertMatrixIndex:
    matrix_units: tuple[SharedExpertMatrixUnit, ...]
    shared_param_indices: tuple[int, ...]
    block_ids: tuple[int, ...]


_SHARED_EXPERT_PARAM_RE = re.compile(
    r"^shared\.(?:wrapped_blocks\.(\d+)\.lora_moe|lora_moes\.(\d+))\.(lora_A_shared|lora_B_shared)$"
)
_PAIR_ORDER = (("det", "seg"), ("det", "cnt"), ("seg", "cnt"))
_PAIR_NAME_ORDER = tuple(f"{task_a}_{task_b}" for task_a, task_b in _PAIR_ORDER)
_PAIR_NAME_TO_TASKS = {
    f"{task_a}_{task_b}": (task_a, task_b)
    for task_a, task_b in _PAIR_ORDER
}
_NO_PAIR_SELECTION = "none"


@dataclass(frozen=True)
class MatrixwiseTopPairCombineStats:
    pair_scores: Dict[str, float]
    pair_counts: Dict[str, int]
    selected_pairs: Dict[int, str]
    block_pair_counts: Dict[int, Dict[str, int]]


def build_shared_expert_block_index(theta_param_names: list[str]) -> SharedExpertBlockIndex:
    block_param_indices: Dict[int, Dict[str, int]] = {}
    param_to_block: Dict[int, int] = {}
    for idx, name in enumerate(theta_param_names):
        match = _SHARED_EXPERT_PARAM_RE.match(name)
        if match is None:
            continue
        block_id_text = match.group(1) or match.group(2)
        if block_id_text is None:
            continue
        block_id = int(block_id_text)
        key = "a" if match.group(3) == "lora_A_shared" else "b"
        block_param_indices.setdefault(block_id, {})[key] = idx
        param_to_block[idx] = block_id
    return SharedExpertBlockIndex(
        block_param_indices=block_param_indices,
        param_to_block=param_to_block,
    )


def build_shared_expert_matrix_index(
    theta_param_names: list[str],
    theta_params: list[torch.nn.Parameter],
) -> SharedExpertMatrixIndex:
    matrix_units: list[SharedExpertMatrixUnit] = []
    shared_param_indices: set[int] = set()
    block_ids: set[int] = set()

    for idx, (name, param) in enumerate(zip(theta_param_names, theta_params)):
        match = _SHARED_EXPERT_PARAM_RE.match(name)
        if match is None:
            continue
        block_id_text = match.group(1) or match.group(2)
        if block_id_text is None:
            continue
        if param.dim() < 1:
            continue
        block_id = int(block_id_text)
        matrix_key = "a" if match.group(3) == "lora_A_shared" else "b"
        num_experts = int(param.shape[0])
        shared_param_indices.add(idx)
        block_ids.add(block_id)
        for expert_index in range(num_experts):
            matrix_units.append(
                SharedExpertMatrixUnit(
                    block_id=block_id,
                    param_index=idx,
                    expert_index=expert_index,
                    matrix_key=matrix_key,
                )
            )

    return SharedExpertMatrixIndex(
        matrix_units=tuple(matrix_units),
        shared_param_indices=tuple(sorted(shared_param_indices)),
        block_ids=tuple(sorted(block_ids)),
    )


def _safe_cosine_similarity(a: torch.Tensor | None, b: torch.Tensor | None) -> float:
    if a is None or b is None:
        return 0.0
    denom = float(a.norm().item()) * float(b.norm().item())
    if denom <= 0.0:
        return 0.0
    return float(torch.dot(a, b).item() / denom)


def _block_task_grad_vector(
    grad_a: torch.Tensor | None,
    grad_b: torch.Tensor | None,
    *,
    template_a: torch.Tensor | None,
    template_b: torch.Tensor | None,
) -> torch.Tensor | None:
    if template_a is None and template_b is None:
        return None

    parts = []
    if template_a is not None:
        cur_a = grad_a.detach().float() if grad_a is not None else torch.zeros_like(template_a, dtype=torch.float32)
        parts.append(cur_a.reshape(-1))
    if template_b is not None:
        cur_b = grad_b.detach().float() if grad_b is not None else torch.zeros_like(template_b, dtype=torch.float32)
        parts.append(cur_b.reshape(-1))
    return torch.cat(parts, dim=0) if parts else None


def _matrix_grad_vector(grad: torch.Tensor | None, expert_index: int) -> torch.Tensor | None:
    if grad is None:
        return None
    return grad[expert_index].detach().float().reshape(-1)


def _empty_pair_scores() -> Dict[str, float]:
    return {pair_name: 0.0 for pair_name in _PAIR_NAME_ORDER}


def _empty_pair_counts() -> Dict[str, int]:
    return {pair_name: 0 for pair_name in _PAIR_NAME_ORDER}


def _select_best_pair_from_scores(pair_scores: Dict[str, float]) -> str:
    best_pair = _PAIR_NAME_ORDER[0]
    best_score = float("-inf")
    for pair_name in _PAIR_NAME_ORDER:
        score = float(pair_scores[pair_name])
        if score > best_score:
            best_score = score
            best_pair = pair_name
    return best_pair


def _max_pair_score(pair_scores: Dict[str, float]) -> float:
    return max(float(pair_scores[pair_name]) for pair_name in _PAIR_NAME_ORDER)


def _select_dominant_pair_for_block(
    pair_counts: Dict[str, int],
    pair_scores: Dict[str, float],
) -> str:
    if sum(int(pair_counts[pair_name]) for pair_name in _PAIR_NAME_ORDER) <= 0:
        return _NO_PAIR_SELECTION
    best_pair = _PAIR_NAME_ORDER[0]
    best_key = (int(pair_counts[best_pair]), float(pair_scores[best_pair]))
    for pair_name in _PAIR_NAME_ORDER[1:]:
        cur_key = (int(pair_counts[pair_name]), float(pair_scores[pair_name]))
        if cur_key > best_key:
            best_key = cur_key
            best_pair = pair_name
    return best_pair


def _pair_weights_from_name(
    pair_name: str,
    base_normalized: tuple[float, float, float],
) -> tuple[float, float, float]:
    pair_tasks = set(_PAIR_NAME_TO_TASKS[pair_name])
    return (
        base_normalized[0] if "det" in pair_tasks else 0.0,
        base_normalized[1] if "seg" in pair_tasks else 0.0,
        base_normalized[2] if "cnt" in pair_tasks else 0.0,
    )


def combine_blockwise_top_pair_grads(
    task_grads: dict[str, tuple[torch.Tensor | None, ...]],
    base_weights: torch.Tensor | tuple[float, float, float] | list[float],
    shared_block_index: SharedExpertBlockIndex,
) -> tuple[tuple[torch.Tensor | None, ...], Dict[int, str]]:
    base_normalized = _normalize_task_weights(base_weights)
    num_params = len(task_grads["det"])
    if len(task_grads["seg"]) != num_params or len(task_grads["cnt"]) != num_params:
        raise ValueError("Task gradient tuples must have identical lengths")

    out: list[torch.Tensor | None] = [None] * num_params
    selected_pairs: Dict[int, str] = {}
    shared_param_indices = set(shared_block_index.param_to_block.keys())

    for idx, grads in enumerate(zip(task_grads["det"], task_grads["seg"], task_grads["cnt"])):
        if idx in shared_param_indices:
            continue
        out[idx] = _combine_param_grads(grads, base_normalized)

    for block_id, param_indices in shared_block_index.block_param_indices.items():
        per_task_grads = {
            "det": {
                "a": task_grads["det"][param_indices["a"]] if "a" in param_indices else None,
                "b": task_grads["det"][param_indices["b"]] if "b" in param_indices else None,
            },
            "seg": {
                "a": task_grads["seg"][param_indices["a"]] if "a" in param_indices else None,
                "b": task_grads["seg"][param_indices["b"]] if "b" in param_indices else None,
            },
            "cnt": {
                "a": task_grads["cnt"][param_indices["a"]] if "a" in param_indices else None,
                "b": task_grads["cnt"][param_indices["b"]] if "b" in param_indices else None,
            },
        }

        template_a = next(
            (
                per_task_grads[task_name]["a"].detach().float()
                for task_name in ("det", "seg", "cnt")
                if per_task_grads[task_name]["a"] is not None
            ),
            None,
        )
        template_b = next(
            (
                per_task_grads[task_name]["b"].detach().float()
                for task_name in ("det", "seg", "cnt")
                if per_task_grads[task_name]["b"] is not None
            ),
            None,
        )

        task_vectors = {
            task_name: _block_task_grad_vector(
                per_task_grads[task_name]["a"],
                per_task_grads[task_name]["b"],
                template_a=template_a,
                template_b=template_b,
            )
            for task_name in ("det", "seg", "cnt")
        }

        local_pair_scores = {
            "det_seg": _safe_cosine_similarity(task_vectors["det"], task_vectors["seg"]),
            "det_cnt": _safe_cosine_similarity(task_vectors["det"], task_vectors["cnt"]),
            "seg_cnt": _safe_cosine_similarity(task_vectors["seg"], task_vectors["cnt"]),
        }
        if _max_pair_score(local_pair_scores) < 0.0:
            selected_pairs[block_id] = _NO_PAIR_SELECTION
            pair_weights = (0.0, 0.0, 0.0)
        else:
            best_pair_name = _select_best_pair_from_scores(local_pair_scores)
            selected_pairs[block_id] = best_pair_name
            pair_weights = _pair_weights_from_name(best_pair_name, base_normalized)

        if "a" in param_indices:
            idx_a = param_indices["a"]
            out[idx_a] = _combine_param_grads(
                (task_grads["det"][idx_a], task_grads["seg"][idx_a], task_grads["cnt"][idx_a]),
                pair_weights,
            )
        if "b" in param_indices:
            idx_b = param_indices["b"]
            out[idx_b] = _combine_param_grads(
                (task_grads["det"][idx_b], task_grads["seg"][idx_b], task_grads["cnt"][idx_b]),
                pair_weights,
            )

    return tuple(out), selected_pairs


def combine_matrixwise_top_pair_grads(
    task_grads: dict[str, tuple[torch.Tensor | None, ...]],
    base_weights: torch.Tensor | tuple[float, float, float] | list[float],
    shared_matrix_index: SharedExpertMatrixIndex,
) -> tuple[tuple[torch.Tensor | None, ...], MatrixwiseTopPairCombineStats]:
    base_normalized = _normalize_task_weights(base_weights)
    num_params = len(task_grads["det"])
    if len(task_grads["seg"]) != num_params or len(task_grads["cnt"]) != num_params:
        raise ValueError("Task gradient tuples must have identical lengths")

    out: list[torch.Tensor | None] = [None] * num_params
    shared_param_indices = set(shared_matrix_index.shared_param_indices)
    for idx, grads in enumerate(zip(task_grads["det"], task_grads["seg"], task_grads["cnt"])):
        if idx in shared_param_indices:
            continue
        out[idx] = _combine_param_grads(grads, base_normalized)

    pair_scores = _empty_pair_scores()
    pair_counts = _empty_pair_counts()
    block_pair_counts = {
        block_id: _empty_pair_counts()
        for block_id in shared_matrix_index.block_ids
    }
    block_pair_scores = {
        block_id: _empty_pair_scores()
        for block_id in shared_matrix_index.block_ids
    }

    for matrix_unit in shared_matrix_index.matrix_units:
        task_matrix_grads = {
            task_name: task_grads[task_name][matrix_unit.param_index]
            for task_name in ("det", "seg", "cnt")
        }
        det_vec = _matrix_grad_vector(task_matrix_grads["det"], matrix_unit.expert_index)
        seg_vec = _matrix_grad_vector(task_matrix_grads["seg"], matrix_unit.expert_index)
        cnt_vec = _matrix_grad_vector(task_matrix_grads["cnt"], matrix_unit.expert_index)
        local_pair_scores = {
            "det_seg": _safe_cosine_similarity(det_vec, seg_vec),
            "det_cnt": _safe_cosine_similarity(det_vec, cnt_vec),
            "seg_cnt": _safe_cosine_similarity(seg_vec, cnt_vec),
        }
        for pair_name, score in local_pair_scores.items():
            pair_scores[pair_name] += float(score)
            block_pair_scores[matrix_unit.block_id][pair_name] += float(score)

        best_pair = _select_best_pair_from_scores(local_pair_scores)
        pair_counts[best_pair] += 1
        block_pair_counts[matrix_unit.block_id][best_pair] += 1
        local_pair_weights = _pair_weights_from_name(best_pair, base_normalized)
        combined_slice = _combine_param_grads(
            (
                None if task_matrix_grads["det"] is None else task_matrix_grads["det"][matrix_unit.expert_index],
                None if task_matrix_grads["seg"] is None else task_matrix_grads["seg"][matrix_unit.expert_index],
                None if task_matrix_grads["cnt"] is None else task_matrix_grads["cnt"][matrix_unit.expert_index],
            ),
            local_pair_weights,
        )
        if combined_slice is None:
            continue

        param_index = matrix_unit.param_index
        if out[param_index] is None:
            template_grad = next(
                (
                    grad_tensor
                    for grad_tensor in task_matrix_grads.values()
                    if grad_tensor is not None
                ),
                None,
            )
            if template_grad is None:
                continue
            out[param_index] = torch.zeros_like(template_grad.detach())
        out[param_index][matrix_unit.expert_index].copy_(combined_slice.detach())

    selected_pairs = {
        block_id: _select_dominant_pair_for_block(block_pair_counts[block_id], block_pair_scores[block_id])
        for block_id in shared_matrix_index.block_ids
    }
    stats = MatrixwiseTopPairCombineStats(
        pair_scores=pair_scores,
        pair_counts=pair_counts,
        selected_pairs=selected_pairs,
        block_pair_counts=block_pair_counts,
    )
    return tuple(out), stats


def assign_grads(
    params: list[torch.nn.Parameter],
    grads: tuple[torch.Tensor | None, ...],
) -> None:
    # 直接把组合后的梯度写回真实参数。
    for param, grad in zip(params, grads):
        param.grad = None if grad is None else grad.detach()


@contextmanager
def temporary_param_updates(
    params: list[torch.nn.Parameter],
    grads: tuple[torch.Tensor | None, ...],
    step_size: float,
):
    # 在真实模型参数上临时施加一次虚拟 SGD，退出上下文后原样回滚。
    step_size = float(step_size)
    with torch.no_grad():
        for param, grad in zip(params, grads):
            if grad is None:
                continue
            param.add_(grad, alpha=-step_size)
    try:
        yield
    finally:
        with torch.no_grad():
            for param, grad in zip(params, grads):
                if grad is None:
                    continue
                param.add_(grad, alpha=step_size)


def build_virtual_updates(
    param_names: list[str],
    params: list[torch.nn.Parameter],
    grads: tuple[torch.Tensor | None, ...],
    step_size: float,
) -> Dict[str, torch.Tensor]:
    # 生成“虚拟更新后”的参数字典，不改真实模型。
    updates = {}
    for name, param, grad in zip(param_names, params, grads):
        updates[name] = param if grad is None else (param - float(step_size) * grad.detach())
    return updates


@torch.no_grad()
def eval_det_loss(model: MultiTaskModel, loader, device: torch.device, *, max_steps: int) -> float:
    # 评估检测任务平均损失。
    model.train()
    total = 0.0
    samples = 0
    steps = 0
    with freeze_batchnorm_stats(model):
        for images, targets in loader:
            images, targets = to_device_det((images, targets), device)
            loss_dict = model("det", images, targets)
            loss = sum(loss_dict.values())
            total += float(loss.item()) * len(images)
            samples += len(images)
            steps += 1
            if max_steps and steps >= max_steps:
                break
    if dist.is_initialized():
        stats = torch.tensor([total, float(samples)], device=device, dtype=torch.float64)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total = float(stats[0].item())
        samples = int(stats[1].item())
    return total / max(samples, 1)


@torch.no_grad()
def eval_seg_loss(
    model: MultiTaskModel,
    loader,
    device: torch.device,
    *,
    max_steps: int,
    num_classes: int,
) -> tuple[float, float]:
    # 评估分割损失和 mIoU。
    model.eval()
    total = 0.0
    samples = 0
    steps = 0
    conf = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for imgs, masks in loader:
        imgs, masks = to_device_seg((imgs, masks), device)
        logits = model("seg", imgs)
        loss = F.cross_entropy(logits, masks)
        total += float(loss.item()) * imgs.size(0)
        samples += imgs.size(0)
        steps += 1
        update_confusion_matrix(
            conf=conf,
            logits_or_preds=logits.detach(),
            target=masks.detach(),
            num_classes=num_classes,
            ignore_indices=(255, 11),
        )
        if max_steps and steps >= max_steps:
            break
    if dist.is_initialized():
        stats = torch.tensor([total, float(samples)], device=device, dtype=torch.float64)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total = float(stats[0].item())
        samples = int(stats[1].item())
        conf_dev = conf.to(device=device)
        dist.all_reduce(conf_dev, op=dist.ReduceOp.SUM)
        conf = conf_dev.cpu()
    _, miou = per_class_iou_from_confusion(conf)
    return total / max(samples, 1), float(miou.item())


@torch.no_grad()
def eval_cnt_loss(
    model: MultiTaskModel,
    loader,
    device: torch.device,
    *,
    max_steps: int,
    count_loss_weight: float,
) -> tuple[float, float, float, float]:
    # 评估计数任务的总损失、密度损失和计数误差。
    model.eval()
    total = 0.0
    total_density = 0.0
    total_count_mae = 0.0
    total_total_mae = 0.0
    samples = 0
    steps = 0
    for imgs, dens in loader:
        imgs, dens = to_device_cnt((imgs, dens), device)
        gt_counts = dens.flatten(2).sum(dim=2)
        pred_dens, pred_counts = model("cnt", imgs)
        dens_loss = F.mse_loss(pred_dens, dens, reduction="sum") / imgs.size(0)
        cnt_l1 = F.l1_loss(pred_counts, gt_counts)
        loss = dens_loss + float(count_loss_weight) * cnt_l1
        count_mae = (pred_counts - gt_counts).abs().mean()
        total_mae = (pred_counts.sum(dim=1) - gt_counts.sum(dim=1)).abs().mean()
        total += float(loss.item()) * imgs.size(0)
        total_density += float(dens_loss.item()) * imgs.size(0)
        total_count_mae += float(count_mae.item()) * imgs.size(0)
        total_total_mae += float(total_mae.item()) * imgs.size(0)
        samples += imgs.size(0)
        steps += 1
        if max_steps and steps >= max_steps:
            break
    if dist.is_initialized():
        stats = torch.tensor(
            [total, total_density, total_count_mae, total_total_mae, float(samples)],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total = float(stats[0].item())
        total_density = float(stats[1].item())
        total_count_mae = float(stats[2].item())
        total_total_mae = float(stats[3].item())
        samples = int(stats[4].item())
    denom = max(samples, 1)
    return total / denom, total_density / denom, total_count_mae / denom, total_total_mae / denom


def _box_iou_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # Numpy 版 IoU，供快速 AP50 计算。
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.maximum(0.0, br - tl)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-12)


def _ap_from_pr(rec: np.ndarray, prec: np.ndarray) -> float:
    # 101 点插值 AP。
    ap = 0.0
    for threshold in np.linspace(0, 1, 101):
        precision = prec[rec >= threshold].max() if np.any(rec >= threshold) else 0.0
        ap += precision
    return ap / 101.0


@torch.no_grad()
def eval_det_ap50_fast(
    model: MultiTaskModel,
    loader,
    device: torch.device,
    *,
    num_classes: int,
    score_thresh: float,
) -> float:
    # 训练期快速 AP50 评估，逻辑保持和旧项目一致。
    model.eval()
    preds_by_cls = {cls_id: [] for cls_id in range(1, num_classes + 1)}
    gts_by_cls = {cls_id: {} for cls_id in range(1, num_classes + 1)}
    rank = dist.get_rank() if dist.is_initialized() else 0
    image_counter = 0

    for images, targets in loader:
        images = [img.to(device, non_blocking=True) for img in images]
        outputs = model("det", images)
        for output, target in zip(outputs, targets):
            default_id = rank * 10_000_000_000 + image_counter
            img_id = int(target.get("image_id", torch.tensor([default_id])).item())
            image_counter += 1
            gt_boxes = target["boxes"].detach().cpu().numpy()
            gt_labels = target["labels"].detach().cpu().numpy().astype(int)
            for box, cls_id in zip(gt_boxes, gt_labels):
                gts_by_cls[cls_id].setdefault(img_id, []).append(box)
            pred_boxes = output["boxes"].detach().cpu().numpy()
            pred_labels = output["labels"].detach().cpu().numpy().astype(int)
            pred_scores = output["scores"].detach().cpu().numpy()
            keep = pred_scores >= score_thresh
            for box, cls_id, score in zip(pred_boxes[keep], pred_labels[keep], pred_scores[keep]):
                preds_by_cls[cls_id].append((img_id, float(score), box))

    if dist.is_initialized():
        # 多卡下先汇总预测，再统一算 AP。
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, (preds_by_cls, gts_by_cls))
        merged_preds = {cls_id: [] for cls_id in range(1, num_classes + 1)}
        merged_gts = {cls_id: {} for cls_id in range(1, num_classes + 1)}
        for rank_preds, rank_gts in gathered:
            for cls_id in range(1, num_classes + 1):
                merged_preds[cls_id].extend(rank_preds.get(cls_id, []))
                for img_id, boxes in rank_gts.get(cls_id, {}).items():
                    merged_gts[cls_id].setdefault(img_id, []).extend(boxes)
        preds_by_cls = merged_preds
        gts_by_cls = merged_gts

    ap_list = []
    for cls_id in range(1, num_classes + 1):
        preds = preds_by_cls[cls_id]
        gts = gts_by_cls[cls_id]
        num_gt = sum(len(v) for v in gts.values())
        if num_gt == 0:
            continue
        preds.sort(key=lambda item: item[1], reverse=True)
        matched = {img_id: [False] * len(boxes) for img_id, boxes in gts.items()}
        tp = np.zeros(len(preds), dtype=np.float32)
        fp = np.zeros(len(preds), dtype=np.float32)
        for idx, (img_id, _score, pred_box) in enumerate(preds):
            if img_id not in gts:
                fp[idx] = 1.0
                continue
            gt_boxes = np.array(gts[img_id], dtype=np.float32)
            ious = _box_iou_np(np.array([pred_box], dtype=np.float32), gt_boxes)[0]
            best_idx = int(np.argmax(ious)) if ious.size > 0 else -1
            if best_idx >= 0 and ious[best_idx] >= 0.5 and not matched[img_id][best_idx]:
                tp[idx] = 1.0
                matched[img_id][best_idx] = True
            else:
                fp[idx] = 1.0
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        rec = tp_cum / max(num_gt, 1)
        prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
        ap_list.append(_ap_from_pr(rec, prec))
    return float(np.mean(ap_list)) if ap_list else 0.0


def compute_task_losses(
    model,
    det_batch,
    seg_batch,
    cnt_batch,
    *,
    device: torch.device,
    cnt_count_loss_weight: float,
    cnt_backbone_grad_mult: float,
    params: Dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # 一次同时计算 det/seg/cnt 三个原始损失。
    model.train()
    det_images, det_targets = to_device_det(det_batch, device)
    seg_imgs, seg_masks = to_device_seg(seg_batch, device)
    cnt_imgs, cnt_dens = to_device_cnt(cnt_batch, device)
    cnt_gt_counts = cnt_dens.flatten(2).sum(dim=2)
    if params is None:
        # 常规前向：直接用当前模型参数。
        det_loss_dict = model("det", det_images, det_targets)
        seg_logits = model("seg", seg_imgs)
        pred_dens, pred_counts = model("cnt", cnt_imgs, cnt_backbone_grad_mult=float(cnt_backbone_grad_mult))
    else:
        # 虚拟前向：用 functional_call 替换成候选参数。
        det_loss_dict = functional_call(model, params, args=("det", det_images, det_targets), kwargs={})
        seg_logits = functional_call(model, params, args=("seg", seg_imgs), kwargs={})
        pred_dens, pred_counts = functional_call(
            model,
            params,
            args=("cnt", cnt_imgs),
            kwargs={"cnt_backbone_grad_mult": float(cnt_backbone_grad_mult)},
        )
    det_loss = sum(det_loss_dict.values())
    seg_loss = F.cross_entropy(seg_logits, seg_masks)
    dens_loss = F.mse_loss(pred_dens, cnt_dens, reduction="sum") / cnt_imgs.size(0)
    cnt_l1 = F.l1_loss(pred_counts, cnt_gt_counts)
    cnt_loss = dens_loss + float(cnt_count_loss_weight) * cnt_l1
    return det_loss, seg_loss, cnt_loss


@torch.no_grad()
def validation_total_loss(
    model,
    val_loaders: dict[str, object],
    *,
    device: torch.device,
    cnt_count_loss_weight: float,
    cnt_backbone_grad_mult: float,
    max_val_steps: int,
    params: Dict[str, torch.Tensor] | None = None,
    return_breakdown: bool = False,
) -> float | tuple[float, dict[str, float]]:
    # 返回 det/seg/cnt 三任务验证损失均值及其总和。
    was_training = model.training
    model.train()
    det_total = 0.0
    det_samples = 0
    det_steps = 0
    for det_batch in val_loaders["det"]:
        # 检测验证仍需 train 模式，才能从 FasterRCNN 拿到 loss。
        det_images, det_targets = to_device_det(det_batch, device)
        if params is None:
            with freeze_batchnorm_stats(model):
                det_loss_dict = model("det", det_images, det_targets)
        else:
            with freeze_batchnorm_stats(model):
                det_loss_dict = functional_call(model, params, args=("det", det_images, det_targets), kwargs={})
        det_total += float(sum(det_loss_dict.values()).item()) * len(det_images)
        det_samples += len(det_images)
        det_steps += 1
        if max_val_steps and det_steps >= max_val_steps:
            break

    model.eval()
    seg_total = 0.0
    seg_samples = 0
    seg_steps = 0
    for seg_batch in val_loaders["seg"]:
        seg_imgs, seg_masks = to_device_seg(seg_batch, device)
        if params is None:
            seg_logits = model("seg", seg_imgs)
        else:
            seg_logits = functional_call(model, params, args=("seg", seg_imgs), kwargs={})
        seg_total += float(F.cross_entropy(seg_logits, seg_masks).item()) * seg_imgs.size(0)
        seg_samples += seg_imgs.size(0)
        seg_steps += 1
        if max_val_steps and seg_steps >= max_val_steps:
            break

    cnt_total = 0.0
    cnt_samples = 0
    cnt_steps = 0
    for cnt_batch in val_loaders["cnt"]:
        cnt_imgs, cnt_dens = to_device_cnt(cnt_batch, device)
        cnt_gt_counts = cnt_dens.flatten(2).sum(dim=2)
        if params is None:
            pred_dens, pred_counts = model("cnt", cnt_imgs, cnt_backbone_grad_mult=float(cnt_backbone_grad_mult))
        else:
            pred_dens, pred_counts = functional_call(
                model,
                params,
                args=("cnt", cnt_imgs),
                kwargs={"cnt_backbone_grad_mult": float(cnt_backbone_grad_mult)},
            )
        dens_loss = F.mse_loss(pred_dens, cnt_dens, reduction="sum") / cnt_imgs.size(0)
        cnt_l1 = F.l1_loss(pred_counts, cnt_gt_counts)
        cnt_total += float((dens_loss + float(cnt_count_loss_weight) * cnt_l1).item()) * cnt_imgs.size(0)
        cnt_samples += cnt_imgs.size(0)
        cnt_steps += 1
        if max_val_steps and cnt_steps >= max_val_steps:
            break

    if dist.is_initialized():
        # 各任务先各自求平均，再按文档定义直接相加。
        stats = torch.tensor(
            [det_total, float(det_samples), seg_total, float(seg_samples), cnt_total, float(cnt_samples)],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        det_total = float(stats[0].item())
        det_samples = int(stats[1].item())
        seg_total = float(stats[2].item())
        seg_samples = int(stats[3].item())
        cnt_total = float(stats[4].item())
        cnt_samples = int(stats[5].item())
    det_avg = det_total / max(det_samples, 1)
    seg_avg = seg_total / max(seg_samples, 1)
    cnt_avg = cnt_total / max(cnt_samples, 1)
    total = det_avg + seg_avg + cnt_avg
    if was_training:
        model.train()
    else:
        model.eval()
    if return_breakdown:
        return total, {
            "det": float(det_avg),
            "seg": float(seg_avg),
            "cnt": float(cnt_avg),
        }
    return total


@dataclass
class ValidationResult:
    # Stage2 一次验证后的组合指标和明细指标。
    combo_metric: float
    metrics: Dict[str, float]


def run_stage2_validation(
    model,
    val_loaders: dict[str, object],
    *,
    device: torch.device,
    seg_num_classes: int,
    det_num_classes: int,
    det_ap_score_thr: float,
    cnt_count_loss_weight: float,
    max_val_steps: int,
    epoch: int,
    stage_name: str,
    metric_logger=None,
) -> ValidationResult:
    # Stage2 选模入口：复用旧项目的组合指标定义。
    val_det = eval_det_loss(model, val_loaders["det"], device, max_steps=max_val_steps)
    val_seg, val_seg_miou = eval_seg_loss(
        model,
        val_loaders["seg"],
        device,
        max_steps=max_val_steps,
        num_classes=seg_num_classes,
    )
    val_cnt, val_cnt_density, val_cnt_mae, val_cnt_total_mae = eval_cnt_loss(
        model,
        val_loaders["cnt"],
        device,
        max_steps=max_val_steps,
        count_loss_weight=cnt_count_loss_weight,
    )
    val_ap50 = eval_det_ap50_fast(
        model,
        val_loaders["det"],
        device,
        num_classes=det_num_classes,
        score_thresh=det_ap_score_thr,
    )
    val_total_loss = float(val_det) + float(val_seg) + float(val_cnt)
    combo_metric = float(val_ap50) + float(val_seg_miou) + 1.0 / max(float(val_cnt_mae), 1e-8)
    print(
        f"[{stage_name}] epoch {epoch} | "
        f"val det {val_det:.4f} seg {val_seg:.4f} miou {val_seg_miou:.4f} "
        f"cnt {val_cnt:.4f} dens {val_cnt_density:.6e} mae {val_cnt_mae:.4f} total_mae {val_cnt_total_mae:.4f} | "
        f"ap50 {val_ap50:.4f} | combo {combo_metric:.6f}"
    )
    if metric_logger is not None:
        metric_logger.log_metrics(
            {
                f"{stage_name}/epoch": int(epoch),
                f"{stage_name}/val_det_loss": float(val_det),
                f"{stage_name}/val_seg_loss": float(val_seg),
                f"{stage_name}/val_seg_miou": float(val_seg_miou),
                f"{stage_name}/val_cnt_loss": float(val_cnt),
                f"{stage_name}/val_cnt_density_mse": float(val_cnt_density),
                f"{stage_name}/val_cnt_mae": float(val_cnt_mae),
                f"{stage_name}/val_cnt_total_mae": float(val_cnt_total_mae),
                f"{stage_name}/val_total_loss": float(val_total_loss),
                f"{stage_name}/val_ap50": float(val_ap50),
                f"{stage_name}/selected_metric": float(combo_metric),
            },
            step=int(epoch),
        )
    return ValidationResult(
        combo_metric=combo_metric,
        metrics={
            "val_det_loss": float(val_det),
            "val_seg_loss": float(val_seg),
            "val_seg_miou": float(val_seg_miou),
            "val_cnt_loss": float(val_cnt),
            "val_cnt_density_mse": float(val_cnt_density),
            "val_cnt_mae": float(val_cnt_mae),
            "val_cnt_total_mae": float(val_cnt_total_mae),
            "val_total_loss": float(val_total_loss),
            "val_ap50": float(val_ap50),
            "selected_metric": float(combo_metric),
        },
    )
