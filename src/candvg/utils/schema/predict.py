from dataclasses import dataclass

from ..bbox import BoundingBox, CamelCaseDataClassJsonMixin


@dataclass(frozen=True)
class BoundingBoxPrediction(CamelCaseDataClassJsonMixin):
    image_id: str
    rect: BoundingBox
    confidence: float


@dataclass(frozen=True)
class PhrasePrediction(CamelCaseDataClassJsonMixin):
    index: int
    text: str
    bounding_boxes: list[BoundingBoxPrediction]


@dataclass(frozen=True)
class SentencePrediction(CamelCaseDataClassJsonMixin):
    doc_id: str
    image_id: str
    phrases: list[PhrasePrediction]
