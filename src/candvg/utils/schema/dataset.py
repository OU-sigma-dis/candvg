from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..bbox import BoundingBox


@dataclass(frozen=True)
class GroundingSample:
    """One grounding sample normalized across dataset formats."""

    sample_id: str
    sentence: str
    positive_token_span: tuple[int, int]
    image_path: Path
    gt_bboxes: list[BoundingBox]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GroundingBatch:
    """Batched grounding samples for model inference."""

    sample_ids: list[str]
    image_paths: list[Path]
    sentences: list[str]
    positive_token_spans: list[tuple[int, int]]
    input_texts: list[str]  # Model inputs
    gt_bboxes: list[BoundingBox]
    metadata: list[dict[str, Any]]
