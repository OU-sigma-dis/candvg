"""ACC@threshold metrics for single-object visual grounding datasets."""

from typing import Any

import torch

from candvg.utils.bbox import box_iou


def _bin_label(bin_size: int) -> str:
    """Return the public bin label for grouped CandVG metrics."""
    return "7+" if bin_size >= 7 else str(bin_size)


def _bin_sort_key(label: str) -> int:
    """Sort numeric bin labels with 7+ after 6."""
    return 7 if label == "7+" else int(label)


def compute_acc(
    outputs: list[dict[str, Any]],
    threshold: float = 0.5,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compute ACC@threshold plus optional bin metrics and enrich outputs."""
    correct = 0
    bin_counts: dict[str, int] = {}
    bin_correct: dict[str, int] = {}
    condition_counts: dict[str, int] = {}
    condition_correct: dict[str, int] = {}
    enriched_outputs: list[dict[str, Any]] = []

    for output in outputs:
        gt_bboxes = output["gt_bboxes"]
        pred_bboxes = output["pred_bboxes"]
        assert gt_bboxes is not None and pred_bboxes is not None

        best_iou = 0.0
        pred_bbox_xyxy = []

        # NOTE: Consider the top-1 predicted bbox for single-object grounding evaluation.
        if pred_bboxes:
            pred_bbox = pred_bboxes[0]
            pred_bbox_xyxy = [pred_bbox.to_xyxy()]
            pred_bbox_tensor = torch.tensor(
                pred_bbox.to_xyxy(), dtype=torch.float32
            ).view(-1, 4)

            # NOTE: Consider the prediction correct if its IoU is at least the threshold for any GT bbox.
            for gt_bbox in gt_bboxes:
                gt_bbox_tensor = torch.tensor(
                    gt_bbox.to_xyxy(), dtype=torch.float32
                ).view(-1, 4)
                iou, _ = box_iou(pred_bbox_tensor, gt_bbox_tensor)
                iou_value = float(iou.item())
                best_iou = max(best_iou, iou_value)

        is_correct = best_iou >= threshold
        if best_iou >= threshold:
            correct += 1

        bin_size = output.get("bin_size")
        if bin_size is not None:
            bin_label = _bin_label(int(bin_size))
            bin_counts[bin_label] = bin_counts.get(bin_label, 0) + 1
            if is_correct:
                bin_correct[bin_label] = bin_correct.get(bin_label, 0) + 1
        condition = output.get("condition")
        if condition is not None:
            condition = str(condition)
            condition_counts[condition] = condition_counts.get(condition, 0) + 1
            if is_correct:
                condition_correct[condition] = condition_correct.get(condition, 0) + 1

        enriched_outputs.append(
            {
                **output,
                "gt_bboxes": [gt_bbox.to_xyxy() for gt_bbox in gt_bboxes],
                "pred_bbox": pred_bbox_xyxy,
                "iou": best_iou,
                f"is_correct_at_{threshold}": is_correct,
            }
        )

    total = len(outputs)
    accuracy = (correct / total) if total else 0.0
    bin_metrics = {
        bin_label: {
            "total": count,
            "accuracy": bin_correct.get(bin_label, 0) / count,
        }
        for bin_label, count in sorted(
            bin_counts.items(), key=lambda item: _bin_sort_key(item[0])
        )
    }
    condition_metrics = {
        condition: {
            "total": count,
            "accuracy": condition_correct.get(condition, 0) / count,
        }
        for condition, count in sorted(condition_counts.items())
    }
    return {
        "accuracy": accuracy,
        "bin_metrics": bin_metrics,
        "condition_metrics": condition_metrics,
    }, enriched_outputs
