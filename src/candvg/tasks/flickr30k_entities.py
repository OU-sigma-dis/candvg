"""Phrase Grounding task for Flickr30k Entities dataset."""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from pycocotools.coco import COCO
from torch.utils.data import Dataset

from candvg.utils import BoundingBox, GroundingSample

from .sampling import (
    load_bin_quota,
    sample_records,
    sample_records_by_bin_quota,
)
from .utils import resolve_dataset_path


@dataclass(frozen=True)
class ImageRecord:
    id: int
    file_name: str
    height: int
    width: int
    caption: str


@dataclass(frozen=True)
class AnnotationRecord:
    id: int
    image_id: int
    phrase_ids: int
    tokens_positive: list[list[int]]
    bbox_xyxy: BoundingBox


def _normalize_positive_token_span(
    sentence: str, token_span: list[int]
) -> tuple[int, int] | None:
    """Validate and normalize one positive token span."""
    if len(token_span) != 2:
        return None
    start, end = int(token_span[0]), int(token_span[1])
    if start < 0 or end <= start or end > len(sentence):
        return None
    if not sentence[start:end].strip():
        return None
    return start, end


class PhraseGroundingDataset(Dataset[GroundingSample]):
    """Task dataset wrapper for Flickr30k Entities COCO-style JSON files."""

    def __init__(
        self,
        dataset_name: str,
        max_samples: int | None = None,
        target_mode: str = "single",
    ) -> None:
        if target_mode not in {"single", "candidates"}:
            raise ValueError(f"Unknown target_mode '{target_mode}'")

        annotation_path = resolve_dataset_path(dataset_name)
        coco = COCO(str(annotation_path))

        records: list[GroundingSample] = []
        self.sid2record: dict[str, GroundingSample] = {}
        for image_id in sorted(coco.imgs.keys()):
            coco_image = coco.loadImgs(image_id)[0]
            image = ImageRecord(
                id=int(coco_image["id"]),
                file_name=str(coco_image["file_name"]),
                height=int(coco_image["height"]),
                width=int(coco_image["width"]),
                caption=str(coco_image.get("caption", "")),
            )

            ann_ids = coco.getAnnIds(imgIds=[image_id], iscrowd=None)
            raw_annotations = coco.loadAnns(ann_ids)

            annotations: list[AnnotationRecord] = []
            for idx, raw_annotation in enumerate(raw_annotations):
                x, y, w, h = (float(value) for value in raw_annotation["bbox"])
                box_xyxy = BoundingBox.from_xywh(top_left_x=x, top_left_y=y, w=w, h=h)
                annotations.append(
                    AnnotationRecord(
                        id=int(raw_annotation.get("id", idx)),
                        image_id=int(raw_annotation["image_id"]),
                        phrase_ids=int(raw_annotation.get("phrase_ids", -1)),
                        tokens_positive=raw_annotation.get("tokens_positive", []),
                        bbox_xyxy=box_xyxy,
                    )
                )

            phrase_groups: dict[tuple[int, tuple[int, int]], list[AnnotationRecord]] = (
                defaultdict(list)
            )
            for annotation in annotations:
                for token_span in annotation.tokens_positive:
                    positive_token_span = _normalize_positive_token_span(
                        sentence=image.caption,
                        token_span=token_span,
                    )
                    if positive_token_span is None:
                        continue

                    key = (annotation.phrase_ids, positive_token_span)
                    phrase_groups[key].append(annotation)

            image_path = annotation_path.parent / "flickr30k_images" / image.file_name
            for (
                phrase_id,
                positive_token_span,
            ), group_annotations in phrase_groups.items():
                gt_bboxes = [annotation.bbox_xyxy for annotation in group_annotations]
                target_count = len(gt_bboxes)
                if target_mode == "single" and target_count != 1:
                    continue
                if target_mode == "candidates" and target_count < 2:
                    continue

                start, end = positive_token_span
                sample_id = f"{image.id}_{phrase_id}_{start}_{end}"

                assert sample_id not in self.sid2record
                self.sid2record[sample_id] = GroundingSample(
                    sample_id=sample_id,
                    sentence=image.caption,
                    positive_token_span=positive_token_span,
                    image_path=Path(image_path),
                    gt_bboxes=gt_bboxes,
                    metadata={
                        "image_id": image.id,
                        "phrase_id": phrase_id,
                        "annotation_ids": [
                            annotation.id for annotation in group_annotations
                        ],
                        "bin_size": target_count,
                        "bin_size_scope": "sample",
                    },
                )

        records = list(self.sid2record.values())

        # NOTE: Determine sampling strategy based on target_mode and max_samples
        quota = (
            load_bin_quota()
            if target_mode == "candidates" and max_samples is None
            else None
        )
        if quota is not None:
            self.records = sample_records_by_bin_quota(records, quota=quota)
        else:
            self.records = sample_records(records, max_samples=max_samples)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> GroundingSample:
        return self.records[idx]

    def get_sample_from_id(self, sample_id: str) -> GroundingSample:
        return self.sid2record[sample_id]
