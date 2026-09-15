"""Common prompt and parsing helpers for VLM backends."""

import re
from typing import Literal

from candvg.utils import BoundingBox

SelectionCandidateFormat = Literal["qwen", "rynnbrain", "locate_anything", "internvl"]


NUMBER_REGEX = r"[-+]?\d*\.?\d+"
CAP = r"(" + NUMBER_REGEX + r")"
BBOX_2D_PATTERN = re.compile(
    r'"?bbox_2d"?\s*:\s*\[\s*'
    + CAP
    + r"\s*,\s*"
    + CAP
    + r"\s*,\s*"
    + CAP
    + r"\s*,\s*"
    + CAP
    + r"\s*(?:\]|$)"
)
BARE_BBOX_LIST_PATTERN = re.compile(
    r"\[\s*"
    + CAP
    + r"\s*,\s*"
    + CAP
    + r"\s*,\s*"
    + CAP
    + r"\s*,\s*"
    + CAP
    + r"\s*(?:\]|$)"
)
LOCATE_ANYTHING_BBOX_PATTERN = re.compile(
    r"<box>\s*<\s*"
    + CAP
    + r"\s*>\s*<\s*"
    + CAP
    + r"\s*>\s*<\s*"
    + CAP
    + r"\s*>\s*<\s*"
    + CAP
    + r"\s*>\s*</box>"
)
RYNNBRAIN_BBOX_PATTERN = re.compile(
    r"<object>\s*\(\s*"
    + CAP
    + r"\s*,\s*"
    + CAP
    + r"\s*\)\s*,\s*\(\s*"
    + CAP
    + r"\s*,\s*"
    + CAP
    + r"\s*\)\s*</object>"
)
SPECIAL_TOKEN_PATTERN = re.compile(r"<\|[^|>]+(?:\|[^|>]+)?\|>")


def _format_candidate_bbox(
    bbox: BoundingBox, candidate_format: SelectionCandidateFormat
) -> str:
    """Format one candidate bbox using a model-family-specific output form."""

    # NOTE: BBoxes are rounded to int to reduce I/O tokens
    values = bbox.to_xyxy()
    x1, y1, x2, y2 = (f"{int(round(float(value)))}" for value in values)

    if candidate_format == "qwen":
        return f'{{"bbox_2d": [{x1}, {y1}, {x2}, {y2}]}}'
    if candidate_format == "rynnbrain":
        return f"<object> ({x1}, {y1}), ({x2}, {y2}) </object>"
    if candidate_format == "locate_anything":
        return f"<box><{x1}><{y1}><{x2}><{y2}></box>"
    if candidate_format == "internvl":
        return f"[{x1}, {y1}, {x2}, {y2}]"
    raise ValueError(f"Unknown selection candidate format: {candidate_format}")


def format_selection_candidate_section(
    candidate_bboxes: list[BoundingBox],
    candidate_format: SelectionCandidateFormat,
) -> str:
    """Build a candidate bbox section for selection-mode prompts."""
    lines = [
        "Candidate boxes (select from these boxes; coordinates are normalized to 0-1000):",
    ]
    if candidate_format == "qwen":
        lines.append("[")
        for index, bbox in enumerate(candidate_bboxes):
            suffix = "," if index < len(candidate_bboxes) - 1 else ""
            lines.append(f"  {_format_candidate_bbox(bbox, candidate_format)}{suffix}")
        lines.append("]")
    elif candidate_format == "internvl":
        bbox_strs = [
            _format_candidate_bbox(bbox, candidate_format) for bbox in candidate_bboxes
        ]
        lines.append(f"<box>[{', '.join(bbox_strs)}]</box>")
    elif candidate_format in {"rynnbrain", "locate_anything"}:
        lines.extend(
            _format_candidate_bbox(bbox, candidate_format) for bbox in candidate_bboxes
        )
    else:
        raise ValueError(f"Unknown selection candidate format: {candidate_format}")
    return "\n".join(lines)


def _clean_answer_text(answer: str) -> str:
    """Drop markdown fences and model special tokens."""
    text = answer.replace("```json", "```").replace("```JSON", "```")
    text = text.replace("```", "\n")
    return SPECIAL_TOKEN_PATTERN.sub("", text)


def _dedupe_bboxes(bboxes: list[BoundingBox]) -> list[BoundingBox]:
    """Deduplicate bboxes while preserving order."""
    unique: list[BoundingBox] = []
    seen: set[tuple[float, float, float, float]] = set()
    for bbox in bboxes:
        key = tuple(round(v, 6) for v in bbox.to_xyxy())
        if key in seen:
            continue
        seen.add(key)
        unique.append(bbox)
    return unique


def parse_bbox_2d_json(answer: str) -> list[BoundingBox]:
    """Parse all bbox coordinates from text output."""
    cleaned_answer = _clean_answer_text(answer)
    bboxes: list[BoundingBox] = []

    for pattern in (
        BBOX_2D_PATTERN,
        BARE_BBOX_LIST_PATTERN,
        LOCATE_ANYTHING_BBOX_PATTERN,
        RYNNBRAIN_BBOX_PATTERN,
    ):
        for match in pattern.finditer(cleaned_answer):
            bboxes.append(
                BoundingBox.from_sequence([float(match.group(i)) for i in range(1, 5)])
            )

    bboxes = _dedupe_bboxes(bboxes)
    if bboxes:
        return bboxes

    return [BoundingBox(0.0, 0.0, 0.0, 0.0)]
