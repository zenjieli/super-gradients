# Notes: commercially usable YOLO-NAS-S person detector

Official YOLO-NAS weights aren't licensed for commercial use, so we train our own and compare against the official `yolo_nas_s` on our person data.

## Setup

- Env: `uv sync` (Python 3.11, torch 2.9.0+cu128). The constraints are explained in `pyproject.toml`.
- Data: `~/data/person_detection_data/` (YOLO format, 1 class, manually corrected; 13.7k train / 2k val images).
  The val split mixes 1,580 Milestone images and 385 images from COCO **val2017**, which the official model never trained on.
- Full COCO 2017: `~/data/coco`.

## Train / evaluate

```
uv run python src/super_gradients/train_from_recipe.py --config-name=person_yolo_nas_s_finetune   # or person_yolo_nas_s (from scratch)
uv run python scripts/eval_person_detection.py --official | --checkpoint <ckpt> [--num-classes 80]
```

## Results (person mAP@0.50:0.95 on val; JSON in `results/`)

| model | all | COCO | Milestone |
|---|---|---|---|
| official yolo_nas_s (not commercial) | 0.688 | **0.718** | 0.683 |
| ours, from scratch (300 epochs) | 0.703 | 0.567 | **0.741** |
| own COCO model (Jan 2026), no fine-tune | 0.571 | 0.613 | 0.566 |
| **ours, fine-tuned from own COCO model (100 epochs)** | **0.704** | 0.616 | 0.727 |

## Checkpoints

| what | path |
|---|---|
| **best: fine-tuned person model** | `checkpoints/person_yolo_nas_s_finetune/RUN_20261126_094354_659371/ckpt_best.pth` |
| from-scratch person model | `checkpoints/person_yolo_nas_s/RUN_20261125_151345_787192/ckpt_best.pth` |
| own COCO yolo_nas_s (from scratch, 500 epochs, COCO mAP 0.356) | `~/workspace/mine/MapNavigation/output/coco/coco2017_yolo_nas_s/RUN_20260116_160401_243410/average_model.pth` |

## Learnings

- Fine-tuning from our own COCO model gives the same overall score as training from scratch, but about 5 points more on COCO scenes, in a third of the time.
- The remaining COCO gap (0.616 vs 0.718) comes from the starting model (0.356 vs 0.475 COCO mAP), not the fine-tune. More fine-tuning epochs won't fix it.
- Under a cosine LR schedule, mAP stays flat while the LR is high and rises at the end. Resuming with more epochs restarts the schedule and barely helps (the COCO run went 300→500 epochs for +0.5 mAP).

## Next

- Better COCO pre-training: scale the LR for 1 GPU (the recipe assumes 8×25 batch) and add mosaic augmentation. Then fine-tune again (update `checkpoint_path` in `person_yolo_nas_s_finetune.yaml`).
- Check COCO image licences with legal before commercial use.
