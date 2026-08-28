import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

anno_file = "/data/xiangyuyue/ULLM-zf/fsl-20260209/data/det/coco/annotations/instances_test.json"
res_file = "/data/xiangyuyue/ULLM-zf/fsl-20260209/runs/ours_lora_loramoe_100per/stats_eval_best_combo_20260424_142447/predictions_det_instances_test.json"

cocoGt = COCO(anno_file)
# Patch info missing issue
if 'info' not in cocoGt.dataset:
    cocoGt.dataset['info'] = {}

cocoDt = cocoGt.loadRes(res_file)

cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()

print("\nPer-category AP@0.50:")
for catId in cocoGt.getCatIds():
    cat = cocoGt.loadCats(catId)[0]
    cat_idx = cocoEval.params.catIds.index(catId)
    precisions = cocoEval.eval['precision'][0, :, cat_idx, 0, 2] # AP@0.50, all area, max det=100
    valid_precisions = precisions[precisions > -1]
    if len(valid_precisions) > 0:
        ap = sum(valid_precisions) / len(valid_precisions)
    else:
        ap = -1
    print(f"[{cat['name']}] AP@0.50: {ap:.4f}")
