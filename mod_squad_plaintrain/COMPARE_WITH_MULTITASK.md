# Comparison with multitask

This document compares this project against the baseline `multitask/` directory.
File lists below ignore `.git` and `__pycache__`.

## Summary
- Purpose: Dual-pool LoRA-MoE adapters (task-private + shared experts) with optional automatic loss weighting.
- Core change: backbone is frozen when LoRA-MoE is enabled; routing uses per-task routers for private/shared pools.

## File changes
- Added (5):
  - `auto_weighted_loss.py`
  - `dinov3_moe_wrapper.py`
  - `eval_train_model.py`
  - `lora_moe.py`
  - `test_lora_moe.py`
- Removed (0): none
- Modified (7):
  - `eval.py`
  - `models.py`
  - `test.sh`
  - `train.py`
  - `train.sh`
  - `train_10per.sh`
  - `utils.py`

## Key code and behavior differences
- `models.py`
  - LoRA-MoE layers are wrapped around each DINOv3 block with private + shared experts per task.
  - Requires `task_id` for routed forward passes when LoRA-MoE is enabled.
- `train.py`
  - New CLI flags for MoE control: `--use-lora-moe`, `--lora-rank`, `--num-experts-private`, `--num-experts-shared`, `--moe-k-private`, `--moe-k-shared`.
  - Optional uncertainty-based weighting for task losses: `--use-auto-weighted-loss`.
  - Adds `--val-every` to control validation cadence.
- `utils.py`
  - Saves the full shared module to preserve LoRA-MoE parameters.
- `eval.py`
  - Can verify full-model loading and evaluate the multitask model in-process (no subprocess single-task evals).
  - Infers LoRA-MoE config from checkpoints and reports load coverage.
- `eval_train_model.py` and `test_lora_moe.py`
  - Additional utilities to load/evaluate the training-time model and sanity check LoRA-MoE wrappers.
- `train.sh` and `train_10per.sh`
  - Expose MoE flags via environment variables.
