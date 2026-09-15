"""Single Target Visual Grounding Task for RefCOCO, RefCOCO+, and RefCOCOg datasets."""

import json
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from candvg.utils import BoundingBox, GroundingSample

from .sampling import sample_records
from .utils import resolve_dataset_path


class VisualGroundingDataset(Dataset[GroundingSample]):
    """Task dataset wrapper for REC JSONL files."""

    def __init__(self, dataset_name: str, max_samples: int | None = None) -> None:
        jsonl_path = resolve_dataset_path(dataset_name)
        self.sid2record: list[str, GroundingSample] = {}

        with jsonl_path.open("r", encoding="utf-8") as file:
            for idx, line in enumerate(file):
                content = line.strip()
                if not content:
                    continue
                sample: dict[str, Any] = json.loads(content)
                sample_id = str(idx)

                assert sample_id not in self.sid2record
                self.sid2record[sample_id] = GroundingSample(
                    sample_id=sample_id,
                    sentence=sample["sent"],
                    positive_token_span=(0, len(sample["sent"])),
                    image_path=Path(sample["image"]),
                    gt_bboxes=[BoundingBox.from_sequence(sample["bbox"])],
                )

        records = list(self.sid2record.values())
        self.records = sample_records(records, max_samples=max_samples)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> GroundingSample:
        return self.records[idx]

    def get_sample_from_id(self, sample_id: str) -> GroundingSample:
        return self.sid2record[sample_id]
