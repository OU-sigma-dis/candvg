"""Multi-Target Visual Grounding Task for gRefCOCO dataset."""

import json
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from candvg.utils import BoundingBox, GroundingSample

from .sampling import (
    load_bin_quota,
    sample_records,
    sample_records_by_bin_quota,
)
from .utils import (
    VISUAL_GROUNDING_MULTI_DATASET_NAMES,
    resolve_dataset_path,
)

MAX_GREFCOCO_TARGETS = 15
IMAGE_ROOT = Path("data/coco/train2014")


class VisualGroundingMultiDataset(Dataset[GroundingSample]):
    def __init__(self, dataset_name: str, max_samples: int | None = None) -> None:

        self.splits = VISUAL_GROUNDING_MULTI_DATASET_NAMES[dataset_name]
        grefs_path = resolve_dataset_path(dataset_name)
        instances_path = grefs_path.parent / "instances.json"

        with grefs_path.open("r", encoding="utf-8") as file:
            raw_annotations: list[dict[str, Any]] = json.load(file)
        with instances_path.open("r", encoding="utf-8") as file:
            instances: dict[str, Any] = json.load(file)

        annotations = {
            int(annotation["id"]): annotation for annotation in instances["annotations"]
        }
        images = {int(image["id"]): image for image in instances["images"]}

        records: list[GroundingSample] = []
        self.sid2record: dict[str, GroundingSample] = {}
        for raw_annotation in raw_annotations:
            split = str(raw_annotation["split"])
            if split not in self.splits or raw_annotation["no_target"]:
                continue

            ann_ids = [
                int(ann_id) for ann_id in raw_annotation["ann_id"] if int(ann_id) != -1
            ]
            if len(ann_ids) < 2:
                continue
            if len(ann_ids) > MAX_GREFCOCO_TARGETS:
                continue

            gt_bboxes = []
            for ann_id in ann_ids:
                x, y, w, h = (float(value) for value in annotations[ann_id]["bbox"])
                box_xyxy = BoundingBox.from_xywh(top_left_x=x, top_left_y=y, w=w, h=h)
                gt_bboxes.append(box_xyxy)

            for sentence in raw_annotation["sentences"]:
                text = str(sentence["sent"]).strip()
                if not text:
                    continue

                sample_id = f"{split}_{raw_annotation['ref_id']}_{sentence['sent_id']}"
                record = GroundingSample(
                    sample_id=sample_id,
                    sentence=text,
                    positive_token_span=(0, len(text)),
                    image_path=IMAGE_ROOT / str(raw_annotation["file_name"]),
                    gt_bboxes=gt_bboxes,
                    metadata={
                        "split": split,
                        "annotation_ids": ann_ids,
                        "bin_size": len(ann_ids),
                        "bin_size_scope": "sample",
                    },
                )
                if sample_id in self.sid2record:
                    raise ValueError(f"Duplicate gRefCOCO sample_id: {sample_id}")
                self.sid2record[sample_id] = record
                records.append(record)

        # NOTE: Determine sampling strategy based on target_mode and max_samples
        quota = load_bin_quota() if max_samples is None else None
        if quota is not None:
            self.records = sample_records_by_bin_quota(records, quota=quota)
        else:
            self.records = sample_records(records, max_samples=max_samples)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> GroundingSample:
        return self.records[idx]

    def get_sample_from_id(self, sample_id: str) -> GroundingSample:
        """Return the original sample by stable sample ID."""
        return self.sid2record[sample_id]
