from __future__ import annotations

from pathlib import Path
import random
from typing import Dict, Iterable, Iterator, Sequence, Tuple

import numpy as np
import torch

try:
    from sklearn.cluster import AffinityPropagation
except Exception:
    AffinityPropagation = None


def parse_int_list(text: str | Sequence[int], *, expected_len: int | None = None) -> tuple[int, ...]:
    if isinstance(text, (list, tuple)):
        values = tuple(int(value) for value in text)
    else:
        raw = str(text or "").strip()
        if not raw:
            values = tuple()
        else:
            values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if expected_len is not None and len(values) != expected_len:
        raise ValueError(f"Expected {expected_len} integers, got {len(values)} from {text!r}")
    return values


def parse_loss_weights(text: str | Sequence[float]) -> Tuple[float, float, float]:
    if isinstance(text, (list, tuple)):
        values = tuple(float(value) for value in text)
    else:
        raw = str(text or "").strip()
        if not raw:
            values = (15.0, 8.0, 1.0)
        else:
            for sep in (",", ":", "/"):
                if sep in raw:
                    parts = [part.strip() for part in raw.split(sep) if part.strip()]
                    break
            else:
                parts = [raw]
            values = tuple(float(part) for part in parts)
    if len(values) != 3:
        raise ValueError("Expected exactly 3 loss weights in det,seg,cnt order.")
    return float(values[0]), float(values[1]), float(values[2])


def infinite_loader(loader: Iterable) -> Iterator:
    while True:
        for batch in loader:
            yield batch


def choose_primary(lengths: Dict[str, int], override: str | None) -> str:
    if override:
        key = str(override).lower()
        if key not in lengths:
            raise ValueError(f"--primary-task must be one of {sorted(lengths.keys())}")
        return key
    return max(lengths.items(), key=lambda item: item[1])[0]


def _filter_det_head_state(det_state: dict) -> dict:
    drop_prefix = "backbone.shared."
    return {key: value for key, value in det_state.items() if not key.startswith(drop_prefix)}


def save_multitask_checkpoint(
    path: str,
    *,
    model,
    optimizer,
    epoch: int,
    metrics: Dict[str, float],
    loss_weights: Tuple[float, float, float],
    model_config: Dict | None = None,
    sel_config: Dict | None = None,
    train_strategy: str | None = None,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "version": 1,
        "epoch": int(epoch),
        "metrics": dict(metrics),
        "loss_weights": tuple(float(weight) for weight in loss_weights),
        "model_config": dict(model_config or {}),
        "sel_config": dict(sel_config or {}),
        "train_strategy": str(train_strategy or ""),
        "backbone": model.shared.state_dict(),
        "det_head": _filter_det_head_state(model.detector.state_dict()),
        "seg_head": model.seg_head.state_dict(),
        "cnt_head": model.cnt_head.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None and hasattr(optimizer, "state_dict") else None,
    }
    torch.save(checkpoint, target)


class TaskAffinityController:
    def __init__(
        self,
        tasks: Sequence[str],
        *,
        initial_groups: Dict[str, Sequence[str]] | None = None,
        warmup_steps: int = 100,
        affin_decay: float = 1e-3,
        preference: float | None = None,
        convergence_iter: int = 50,
        max_cluster_retries: int = 10,
    ) -> None:
        self.tasks = [str(task) for task in tasks]
        self.warmup_steps = int(warmup_steps)
        self.affin_decay = float(affin_decay)
        self.preference = preference
        self.convergence_iter = int(convergence_iter)
        self.max_cluster_retries = int(max_cluster_retries)
        self.affinity_map = torch.zeros((len(self.tasks), len(self.tasks)), dtype=torch.float32)
        self.pre_loss = {task: -1.0 for task in self.tasks}
        if initial_groups:
            self.group_map = self._normalize_groups(initial_groups)
        else:
            self.group_map = {f"group{i + 1}": [task] for i, task in enumerate(self.tasks)}

    def _normalize_groups(self, groups: Dict[str, Sequence[str]]) -> Dict[str, list[str]]:
        normalized: Dict[str, list[str]] = {}
        seen: set[str] = set()
        for name, members in groups.items():
            valid_members = [str(member) for member in members if str(member) in self.tasks and str(member) not in seen]
            if valid_members:
                normalized[str(name)] = valid_members
                seen.update(valid_members)
        for task in self.tasks:
            if task not in seen:
                normalized[f"group{len(normalized) + 1}"] = [task]
        return normalized

    def init_pre_loss(self) -> None:
        for task in self.tasks:
            self.pre_loss[task] = -1.0

    def current_group_names(self) -> list[str]:
        return list(self.group_map.keys())

    def shuffled_group_names(self) -> list[str]:
        names = self.current_group_names()
        random.shuffle(names)
        return names

    def update(self, group_name: str, loss_values: Dict[str, float]) -> None:
        members = self.group_map.get(group_name, [])
        if not members:
            return

        for task_s in members:
            for task_t in self.tasks:
                prev_t = float(self.pre_loss.get(task_t, -1.0))
                if prev_t <= 0:
                    continue
                if task_t in members:
                    prev_s = float(self.pre_loss.get(task_s, -1.0))
                    if prev_s <= 0:
                        continue
                    affin_t = 1.0 - float(loss_values[task_t]) / prev_t
                    affin_t /= max(len(members), 1)
                    affin_s = 1.0 - float(loss_values[task_s]) / prev_s
                    affin_s /= max(len(members), 1)
                    if task_t == task_s:
                        if affin_t >= 0:
                            self._affin_update(affin_t, task_s, task_t)
                    elif affin_t * affin_s >= 0:
                        self._affin_update(affin_t, task_s, task_t)
                    else:
                        self._affin_update(-max(affin_t, affin_s), task_s, task_t)
                else:
                    affin = 1.0 - float(loss_values[task_t]) / prev_t
                    affin /= max(len(members), 1)
                    self._affin_update(affin, task_s, task_t)

        for task in self.tasks:
            if task in loss_values:
                self.pre_loss[task] = float(loss_values[task])

    def _affin_update(self, affin: float, task_s: str, task_t: str) -> None:
        idx_s = self.tasks.index(task_s)
        idx_t = self.tasks.index(task_t)
        current = float(self.affinity_map[idx_s, idx_t].item())
        updated = (1.0 - self.affin_decay) * current + self.affin_decay * float(affin)
        self.affinity_map[idx_s, idx_t] = updated

    def maybe_recluster(self, step_index: int) -> None:
        if len(self.tasks) < 2 or int(step_index) <= self.warmup_steps:
            return

        matrix = self.affinity_map.detach().clone()
        if matrix.numel() == 0:
            return

        diag = torch.diag(matrix).clone()
        for index in range(len(self.tasks)):
            denom = float(diag[index].item())
            if abs(denom) > 1e-12:
                matrix[:, index] = matrix[:, index] / denom
            else:
                matrix[:, index] = 0.0
        matrix = (matrix + matrix.T) / 2.0
        matrix_np = np.nan_to_num(matrix.cpu().numpy(), nan=0.0, posinf=0.0, neginf=0.0)

        if AffinityPropagation is None:
            candidate = self._fallback_recluster(matrix_np)
            if candidate:
                self.group_map = candidate
            return

        convergence_iter = self.convergence_iter
        candidate: Dict[str, list[str]] = {}
        for _ in range(max(self.max_cluster_retries, 1)):
            cluster = AffinityPropagation(
                preference=self.preference,
                affinity="precomputed",
                convergence_iter=convergence_iter,
                random_state=0,
            )
            labels = cluster.fit_predict(matrix_np)
            centers = cluster.cluster_centers_indices_
            if centers is None or len(centers) == 0:
                convergence_iter += 100
                continue
            unique_labels = []
            for label in labels.tolist():
                if label not in unique_labels:
                    unique_labels.append(label)
            grouped: Dict[str, list[str]] = {}
            for group_index, label in enumerate(unique_labels, start=1):
                grouped[f"group{group_index}"] = [task for task, task_label in zip(self.tasks, labels.tolist()) if task_label == label]
            candidate = self._normalize_groups(grouped)
            if candidate:
                break
            convergence_iter += 100

        if candidate:
            self.group_map = candidate

    def _fallback_recluster(self, matrix_np: np.ndarray) -> Dict[str, list[str]]:
        if len(self.tasks) <= 1:
            return dict(self.group_map)

        # Greedy positive-affinity connected components fallback for environments
        # without scikit-learn. This preserves dynamic grouping behavior, albeit
        # with a simpler clustering rule than AffinityPropagation.
        threshold = float(self.preference) if self.preference is not None else 0.0
        parent = list(range(len(self.tasks)))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(lhs: int, rhs: int) -> None:
            root_l = find(lhs)
            root_r = find(rhs)
            if root_l != root_r:
                parent[root_r] = root_l

        for i in range(len(self.tasks)):
            for j in range(i + 1, len(self.tasks)):
                pair_affinity = 0.5 * (float(matrix_np[i, j]) + float(matrix_np[j, i]))
                if pair_affinity > threshold:
                    union(i, j)

        grouped: Dict[int, list[str]] = {}
        for index, task in enumerate(self.tasks):
            grouped.setdefault(find(index), []).append(task)
        groups = {f"group{group_index + 1}": members for group_index, members in enumerate(grouped.values())}
        return self._normalize_groups(groups)
