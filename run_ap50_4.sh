#!/bin/bash
PY_BIN="/data/xiangyuyue/ULLM-zf/fsl-20260209/miniconda3/envs/fsl/bin/python"
SCRIPT="/data/xiangyuyue/ULLM-zf/fsl-20260209/runs/115_mainonly_1_10per/compute_per_category_ap50.py"

FILES=(
  "/data/xiangyuyue/ULLM-zf/fsl-20260209/runs/mod_squad_plaintrain_sharedonly_mi_10per/stats_eval_best_combo_20260316_211610/predictions_det_instances_test.json"
  "/data/xiangyuyue/ULLM-zf/fsl-20260209/runs/113_mtlora_small_10per/stats_eval_best_combo_20260319_180019/predictions_det_instances_test.json"
  "/data/xiangyuyue/ULLM-zf/fsl-20260209/runs/113_pivrg_10per/stats_eval_best_combo_20260320_122311/predictions_det_instances_test.json"
  "/data/xiangyuyue/ULLM-zf/fsl-20260209/runs/113_mtlora_vision_10per/stats_eval_best_combo_20260323_120003/predictions_det_instances_test.json"
)

for PRED_FILE in "${FILES[@]}"; do
  echo "Processing: $PRED_FILE"
  $PY_BIN $SCRIPT "$PRED_FILE"
  echo "----------------------------------------"
done
