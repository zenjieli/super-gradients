"""Evaluate a YOLO-NAS model on person_detection_data (YOLO-format, single class "person").

    # official COCO weights (person = COCO class 0, other 79 classes are dropped before NMS)
    uv run python scripts/eval_person_detection.py --official
    # our checkpoint (add --num-classes 80 for a self-trained COCO model)
    uv run python scripts/eval_person_detection.py --checkpoint checkpoints/<exp>/<run>/ckpt_best.pth

Reports mAP on the whole val split and on the COCO (numeric file names) / Milestone (everything else) subsets.
Preprocessing and NMS settings mirror the coco2017_yolo_nas_s recipe's validation.
"""
import argparse
import json
import os
import re

import torch
from torch.utils.data import DataLoader

from super_gradients.training import models
from super_gradients.training.datasets.detection_datasets.yolo_format_detection import YoloDarknetFormatDetectionDataset
from super_gradients.training.metrics import DetectionMetrics
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
from super_gradients.training.transforms.transforms import (
    DetectionImagePermute,
    DetectionPadToSize,
    DetectionRGB2BGR,
    DetectionStandardize,
    DetectionTargetsFormatTransform,
)
from super_gradients.training.utils.collate_fn import DetectionCollateFN

SUBSETS = {
    "all": lambda name: True,
    "coco": lambda name: re.fullmatch(r"\d+\.jpg", name) is not None,
    "milestone": lambda name: re.fullmatch(r"\d+\.jpg", name) is None,
}


class SubsetYoloDataset(YoloDarknetFormatDetectionDataset):
    def __init__(self, *args, keep=SUBSETS["all"], **kwargs):
        self.keep = keep
        super().__init__(*args, **kwargs)

    def _setup_data_source(self) -> int:
        super()._setup_data_source()
        pairs = [(img, lbl) for img, lbl in zip(self.images_file_names, self.labels_file_names) if self.keep(img)]
        self.images_file_names, self.labels_file_names = map(list, zip(*pairs))
        return len(self.images_file_names)


class PersonOnlyPostPredictionCallback(PPYoloEPostPredictionCallback):
    """Keep only the class-0 (person) score column before NMS, i.e. treat an 80-class COCO model as a person detector."""

    def _get_decoded_predictions_from_model_output(self, outputs):
        bboxes, scores = super()._get_decoded_predictions_from_model_output(outputs)
        return bboxes, scores[..., :1]


def evaluate(model, data_dir: str, split: str, subset: str, batch_size: int, person_only: bool) -> dict:
    dataset = SubsetYoloDataset(
        data_dir=data_dir,
        images_dir=split,
        labels_dir=split,
        classes=["person"],
        keep=SUBSETS[subset],
        input_dim=[636, 636],
        ignore_empty_annotations=False,  # images without persons are real negatives
        transforms=[
            DetectionRGB2BGR(prob=1),
            DetectionPadToSize(output_size=(640, 640), pad_value=114),
            DetectionStandardize(max_value=255.0),
            DetectionImagePermute(),
            DetectionTargetsFormatTransform(input_dim=(640, 640), output_format="LABEL_CXCYWH"),
        ],
    )
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=8, collate_fn=DetectionCollateFN())
    callback_cls = PersonOnlyPostPredictionCallback if person_only else PPYoloEPostPredictionCallback
    metric = DetectionMetrics(
        num_cls=1,
        normalize_targets=True,
        score_thres=0.1,
        top_k_predictions=300,
        post_prediction_callback=callback_cls(score_threshold=0.01, nms_top_k=1000, max_predictions=300, nms_threshold=0.7),
    )
    device = next(model.parameters()).device
    metric.to(device)
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            metric.update(model(images), targets.to(device), device=str(device), inputs=images)
    result = {k: float(v) for k, v in metric.compute().items()}
    result["num_images"] = len(dataset)
    return result


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--official", action="store_true", help="official yolo_nas_s COCO weights")
    source.add_argument("--checkpoint", help="SG checkpoint of a yolo_nas_s")
    parser.add_argument("--num-classes", type=int, default=1, help="classes in --checkpoint; 80 = COCO model, person (class 0) only is scored")
    parser.add_argument("--arch", default="yolo_nas_s")
    parser.add_argument("--data-dir", default=os.path.expanduser("~/data/person_detection_data"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out", help="write results as JSON here")
    args = parser.parse_args()

    if args.official:
        model = models.get(args.arch, pretrained_weights="coco")
    else:
        model = models.get(args.arch, num_classes=args.num_classes, checkpoint_path=args.checkpoint)
    model = model.cuda().eval()

    results = {s: evaluate(model, args.data_dir, args.split, s, args.batch_size, person_only=args.official or args.num_classes > 1) for s in SUBSETS}
    text = json.dumps(results, indent=2)
    print(text, flush=True)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)


if __name__ == "__main__":
    main()
