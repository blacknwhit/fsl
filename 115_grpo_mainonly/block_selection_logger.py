from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping


_PAIR_NAMES = ("det_seg", "det_cnt", "seg_cnt")


class BlockSelectionRecorder:
    def __init__(
        self,
        *,
        save_dir: Path,
        enabled: bool,
        is_main_process: bool,
    ) -> None:
        self.enabled = bool(enabled) and bool(is_main_process)
        self.history_path = save_dir / "block_task_selection_history.jsonl"
        self.epoch_summary_path = save_dir / "block_task_selection_epoch_summary.json"
        self._epoch_summaries: list[dict[str, object]] = []

        if not self.enabled:
            return

        self.history_path.write_text("", encoding="utf-8")
        self.epoch_summary_path.write_text("[]\n", encoding="utf-8")

    @staticmethod
    def _empty_pair_counts() -> dict[str, int]:
        return {name: 0 for name in _PAIR_NAMES}

    @staticmethod
    def _normalize_selected_pairs(selected_pairs: Mapping[int, str]) -> Dict[str, str]:
        return {str(block_id): str(selected_pairs[block_id]) for block_id in sorted(selected_pairs)}

    @staticmethod
    def _normalize_pair_counts(pair_counts: Mapping[str, int]) -> Dict[str, int]:
        counts = BlockSelectionRecorder._empty_pair_counts()
        for pair_name, value in pair_counts.items():
            counts[str(pair_name)] = int(value)
        return counts

    @staticmethod
    def _normalize_block_pair_counts(block_pair_counts: Mapping[int, Mapping[str, int]]) -> Dict[str, Dict[str, int]]:
        return {
            str(block_id): BlockSelectionRecorder._normalize_pair_counts(block_pair_counts[block_id])
            for block_id in sorted(block_pair_counts)
        }

    def log_step(
        self,
        *,
        epoch: int,
        global_step: int,
        step_in_epoch: int,
        selected_pairs: Mapping[int, str],
        pair_counts: Mapping[str, int] | None = None,
        block_pair_counts: Mapping[int, Mapping[str, int]] | None = None,
        pair_scores: Mapping[str, float] | None = None,
    ) -> None:
        if not self.enabled:
            return

        normalized_pair_counts = (
            self._normalize_pair_counts(pair_counts)
            if pair_counts is not None
            else self._empty_pair_counts()
        )
        if pair_counts is None:
            for pair_name in selected_pairs.values():
                normalized_pair_counts[pair_name] = normalized_pair_counts.get(pair_name, 0) + 1

        payload = {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "step_in_epoch": int(step_in_epoch),
            "pair_counts": normalized_pair_counts,
            "selected_pairs": self._normalize_selected_pairs(selected_pairs),
        }
        if block_pair_counts is not None:
            payload["block_pair_counts"] = self._normalize_block_pair_counts(block_pair_counts)
        if pair_scores is not None:
            payload["pair_scores"] = {name: float(pair_scores.get(name, 0.0)) for name in _PAIR_NAMES}
        with self.history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def log_epoch(
        self,
        *,
        epoch: int,
        steps: int,
        block_pair_counts: Mapping[int, Mapping[str, int]],
    ) -> None:
        if not self.enabled:
            return

        per_block_summary: dict[str, dict[str, object]] = {}
        for block_id in sorted(block_pair_counts):
            counts = self._empty_pair_counts()
            for pair_name, value in block_pair_counts[block_id].items():
                counts[pair_name] = int(value)
            if sum(counts.values()) <= 0:
                dominant_pair = "none"
            else:
                dominant_pair = max(_PAIR_NAMES, key=lambda pair_name: (counts[pair_name], -_PAIR_NAMES.index(pair_name)))
            per_block_summary[str(block_id)] = {
                "counts": counts,
                "dominant_pair": dominant_pair,
            }

        payload = {
            "epoch": int(epoch),
            "steps": int(steps),
            "blocks": per_block_summary,
        }
        self._epoch_summaries.append(payload)
        self.epoch_summary_path.write_text(
            json.dumps(self._epoch_summaries, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
