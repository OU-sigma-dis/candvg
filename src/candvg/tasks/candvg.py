""" "Context-Aware & Candidate Set Visual Grounding Task for CandVG dataset."""

import json
from collections import defaultdict
from typing import Any

from torch.utils.data import Dataset

from candvg.utils import BoundingBox, GroundingSample

from .sampling import sample_records
from .utils import resolve_dataset_path


def _bbox_key(
    bboxes: list[list[float]],
) -> tuple[tuple[float, float, float, float], ...]:
    """Return a stable key for comparing candidate bbox sets."""
    return tuple(sorted(tuple(float(value) for value in bbox) for bbox in bboxes))


def _ordered_unique_bboxes(samples: list[dict[str, Any]]) -> list[BoundingBox]:
    """Collect dialogue-level unique candidate bboxes while preserving first appearance."""
    bboxes: list[BoundingBox] = []
    seen: set[BoundingBox] = set()
    for sample in samples:
        for bbox in sample["annotation"]["candidate_bboxes"]:
            key = BoundingBox.from_sequence([float(value) for value in bbox])
            if key in seen:
                continue
            seen.add(key)
            bboxes.append(key)
    return bboxes


class ContextVisualGroundingDataset(Dataset[GroundingSample]):
    """Task dataset wrapper for Context-aware Visual Grounding."""

    def __init__(
        self,
        dataset_name: str,
        max_samples: int | None = None,
        target_mode: str = "single",
    ) -> None:
        jsonl_path = resolve_dataset_path(dataset_name)
        image_root = jsonl_path.parent / "images"

        self.sid2record: dict[str, GroundingSample] = {}
        raw_samples: list[dict[str, Any]] = []
        with jsonl_path.open("r", encoding="utf-8") as file:
            for line in file:
                content = line.strip()
                if not content:
                    continue
                raw_samples.append(json.loads(content))

        dialogue_bboxes: dict[str, set[tuple[float, float, float, float]]] = (
            defaultdict(set)
        )
        samples_by_dialogue: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for sample in raw_samples:
            dialogue_id = str(sample["metadata"]["global_id"])
            samples_by_dialogue[dialogue_id].append(sample)
            for bbox in sample["annotation"]["candidate_bboxes"]:
                dialogue_bboxes[dialogue_id].add(tuple(float(value) for value in bbox))

        dialogue_bin_sizes = {
            dialogue_id: len(unique_bboxes)
            for dialogue_id, unique_bboxes in dialogue_bboxes.items()
        }
        candidate_bboxes_by_dialogue: dict[str, list[BoundingBox]] = {}
        for dialogue_id, dialogue_samples in samples_by_dialogue.items():
            ordered_samples = sorted(
                dialogue_samples, key=lambda item: int(item["metadata"]["turn_index"])
            )
            candidate_bboxes_by_dialogue[dialogue_id] = _ordered_unique_bboxes(
                ordered_samples
            )

        for sample in raw_samples:
            sample_id = sample["sample_id"]

            if target_mode == "single" and sample["metadata"]["num_points"] > 1:
                continue
            if target_mode == "candidates" and sample["metadata"]["num_points"] == 1:
                continue

            image_path = image_root / str(sample["filename"])
            annotation = sample["annotation"]
            target_utterance = annotation["target_utterance"]
            dialogue = annotation["dialogue"]
            target_start = dialogue.find(target_utterance)
            if target_start < 0:
                raise ValueError(
                    f"Target utterance not found in dialogue for sample_id={sample_id}"
                )

            gt_bboxes = [
                BoundingBox.from_sequence(bbox)
                for bbox in annotation["candidate_bboxes"]
            ]
            dialogue_id = str(sample["metadata"]["global_id"])
            candidate_bboxes = [
                bbox.to_relative_from_pixel(
                    width=float(sample["width"]),
                    height=float(sample["height"]),
                    scale=1000.0,
                )
                for bbox in candidate_bboxes_by_dialogue[dialogue_id]
            ]
            num_points = int(sample["metadata"]["num_points"])
            bin_size = (
                dialogue_bin_sizes[dialogue_id]
                if target_mode == "single"
                else num_points
            )

            assert sample_id not in self.sid2record
            self.sid2record[sample_id] = GroundingSample(
                sample_id=sample_id,
                sentence=dialogue,
                positive_token_span=(
                    target_start,
                    target_start + len(target_utterance),
                ),
                image_path=image_path,
                gt_bboxes=gt_bboxes,
                metadata={
                    "bin_size": bin_size,
                    "bin_size_scope": "dialogue_unique"
                    if target_mode == "single"
                    else "sample",
                    "dialogue_id": dialogue_id,
                    "turn_index": int(sample["metadata"]["turn_index"]),
                    "num_turns": int(sample["metadata"]["num_turns"]),
                    "candidate_bboxes": candidate_bboxes,
                },
            )

        records = list(self.sid2record.values())
        self.records = sample_records(records, max_samples=max_samples)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> GroundingSample:
        return self.records[idx]

    def get_sample_from_id(self, sample_id: str) -> GroundingSample:
        return self.sid2record[sample_id]
