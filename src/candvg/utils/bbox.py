"""Utility functions for bounding box operations."""

from dataclasses import dataclass
from typing import Literal

import torch
from dataclasses_json import DataClassJsonMixin, LetterCase, config
from torchvision.ops.boxes import box_area

number = int | float
_COORDMODE = Literal["relative_1000", "pixel"]


def box_iou(
    boxes1: torch.Tensor, boxes2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute IoU between two box tensors."""
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter
    iou = inter / union
    return iou, union


def normalize_pred_bbox(
    pred_bboxes: list["BoundingBox"],
    hw: tuple[int, int],
    coord_mode: _COORDMODE = "pixel",
    coord_scale: float | None = None,
) -> list["BoundingBox"]:
    """Normalize one predicted bbox to pixel coordinates."""
    height, width = hw
    if coord_mode == "relative_1000":
        assert coord_scale is not None
        return [
            pred_bbox.to_pixel_from_relative(
                width=width, height=height, scale=coord_scale
            )
            for pred_bbox in pred_bboxes
        ]
    return pred_bboxes


class CamelCaseDataClassJsonMixin(DataClassJsonMixin):
    dataclass_json_config = config(letter_case=LetterCase.CAMEL)["dataclasses_json"]  # type: ignore


@dataclass(frozen=True)
class BoundingBox(CamelCaseDataClassJsonMixin):
    """Axis-aligned bbox in xyxy format."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def left(self) -> float:
        return min(self.x1, self.x2)

    @property
    def top(self) -> float:
        return min(self.y1, self.y2)

    @property
    def width(self) -> float:
        return abs(self.x2 - self.x1)

    @property
    def height(self) -> float:
        return abs(self.y2 - self.y1)

    @property
    def center(self) -> tuple[float, float]:
        return (self.left + self.width / 2.0, self.top + self.height / 2.0)

    @property
    def area(self) -> float:
        return self.width * self.height

    @classmethod
    def from_xyxy(cls, x1: number, y1: number, x2: number, y2: number) -> "BoundingBox":
        return cls(float(x1), float(y1), float(x2), float(y2))

    def to_xyxy(self) -> tuple[float, float, float, float]:
        return (
            min(self.x1, self.x2),
            min(self.y1, self.y2),
            max(self.x1, self.x2),
            max(self.y1, self.y2),
        )

    @classmethod
    def from_sequence(
        cls, values: list[float] | tuple[float, float, float, float]
    ) -> "BoundingBox":
        """Build a bbox from 4 coordinates in xyxy order."""
        if len(values) != 4:
            raise ValueError(f"Expected 4 values for bbox, got {len(values)}")
        return cls(
            float(values[0]), float(values[1]), float(values[2]), float(values[3])
        )

    @classmethod
    def from_xywh(
        cls, top_left_x: number, top_left_y: number, w: number, h: number
    ) -> "BoundingBox":
        if w < 0 or h < 0:
            raise ValueError("w and h must be positive")
        return cls.from_xyxy(top_left_x, top_left_y, top_left_x + w, top_left_y + h)

    def to_xywh(self) -> tuple[float, float, float, float]:
        """Return bbox as xywh tuple."""
        return (self.left, self.top, self.width, self.height)

    def to_pixel_from_relative(
        self, width: number, height: number, scale: float = 1000.0
    ) -> "BoundingBox":
        """Convert relative coords in [0, scale] to pixel-space bbox."""
        return BoundingBox(
            x1=float(self.x1 / scale * width),
            y1=float(self.y1 / scale * height),
            x2=float(self.x2 / scale * width),
            y2=float(self.y2 / scale * height),
        )

    def to_relative_from_pixel(
        self, width: number, height: number, scale: float = 1000.0
    ) -> "BoundingBox":
        """Convert pixel-space bbox to relative coords in [0, scale]."""
        return BoundingBox(
            x1=float(self.x1 / width * scale),
            y1=float(self.y1 / height * scale),
            x2=float(self.x2 / width * scale),
            y2=float(self.y2 / height * scale),
        )
