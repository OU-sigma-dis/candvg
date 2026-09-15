"""Precision/Recall/F1 metrics for multiple-object visual grounding datasets."""

from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from candvg.utils.bbox import box_iou


def _hungarian_maximize(weights: list[list[float]]) -> list[tuple[int, int]]:
    """Solve max-weight bipartite matching."""
    if not weights or not weights[0]:
        return []
    weight_matrix = np.asarray(weights, dtype=np.float64)
    row_ind, col_ind = linear_sum_assignment(-weight_matrix)
    return [
        (int(row), int(col)) for row, col in zip(row_ind.tolist(), col_ind.tolist())
    ]


def _bin_label(bin_size: int) -> str:
    """Return the public bin label for grouped CandVG metrics."""
    return "7+" if bin_size >= 7 else str(bin_size)


def _bin_sort_key(label: str) -> int:
    """Sort numeric bin labels with 7+ after 6."""
    return 7 if label == "7+" else int(label)


def compute_prec_rec_f1_metrics(
    outputs: list[dict[str, Any]],
    threshold: float = 0.5,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compute per-sample Precision/Recall/F1 plus optional bin metrics."""
    sample_precisions: list[float] = []
    sample_recalls: list[float] = []
    sample_f1s: list[float] = []
    bin_sample_metrics: dict[str, list[tuple[float, float, float]]] = {}
    condition_sample_metrics: dict[str, list[tuple[float, float, float]]] = {}
    enriched_outputs: list[dict[str, Any]] = []

    for output in outputs:
        gt_bboxes = output["gt_bboxes"]
        pred_bboxes = output["pred_bboxes"]
        assert gt_bboxes is not None and pred_bboxes is not None

        num_gt = len(gt_bboxes)
        num_pred = len(pred_bboxes)

        iou_matrix: list[list[float]] = []
        for pred_bbox in pred_bboxes:
            pred_tensor = torch.tensor(pred_bbox.to_xyxy(), dtype=torch.float32).view(
                -1, 4
            )
            row: list[float] = []
            for gt_bbox in gt_bboxes:
                gt_tensor = torch.tensor(gt_bbox.to_xyxy(), dtype=torch.float32).view(
                    -1, 4
                )
                iou, _ = box_iou(pred_tensor, gt_tensor)
                row.append(float(iou.item()))
            iou_matrix.append(row)

        # NOTE: Maximizing TP count at the given threshold.
        binary_iou_matrix = [
            [1.0 if iou >= threshold else 0.0 for iou in row] for row in iou_matrix
        ]
        matches = (
            _hungarian_maximize(binary_iou_matrix)
            if (num_pred > 0 and num_gt > 0)
            else []
        )
        tp = sum(
            1
            for pred_idx, gt_idx in matches
            if iou_matrix[pred_idx][gt_idx] >= threshold
        )
        fp = max(0, num_pred - tp)
        fn = max(0, num_gt - tp)

        precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (
            (2.0 * precision * recall / (precision + recall))
            if (precision + recall) > 0
            else 0.0
        )

        sample_precisions.append(precision)
        sample_recalls.append(recall)
        sample_f1s.append(f1)

        bin_size = output.get("bin_size")
        if bin_size is not None:
            bin_label = _bin_label(int(bin_size))
            bin_sample_metrics.setdefault(bin_label, []).append((precision, recall, f1))
        condition = output.get("condition")
        if condition is not None:
            condition_sample_metrics.setdefault(str(condition), []).append(
                (precision, recall, f1)
            )

        enriched_outputs.append(
            {
                **output,
                "gt_bboxes": [gt_bbox.to_xyxy() for gt_bbox in gt_bboxes],
                "pred_bboxes": [pred_bbox.to_xyxy() for pred_bbox in pred_bboxes],
                "match_pairs": [
                    {"pred_idx": p, "gt_idx": g, "iou": iou_matrix[p][g]}
                    for p, g in matches
                ],
                f"tp_at_{threshold}": tp,
                f"fp_at_{threshold}": fp,
                f"fn_at_{threshold}": fn,
                f"precision_at_{threshold}": precision,
                f"recall_at_{threshold}": recall,
                f"f1_at_{threshold}": f1,
            }
        )

    total = len(outputs)
    metrics = {
        "precision": (sum(sample_precisions) / total) if total else 0.0,
        "recall": (sum(sample_recalls) / total) if total else 0.0,
        "f1": (sum(sample_f1s) / total) if total else 0.0,
        "bin_metrics": {
            bin_label: {
                "total": len(values),
                "precision": sum(value[0] for value in values) / len(values),
                "recall": sum(value[1] for value in values) / len(values),
                "f1": sum(value[2] for value in values) / len(values),
            }
            for bin_label, values in sorted(
                bin_sample_metrics.items(), key=lambda item: _bin_sort_key(item[0])
            )
        },
        "condition_metrics": {
            condition: {
                "total": len(values),
                "precision": sum(value[0] for value in values) / len(values),
                "recall": sum(value[1] for value in values) / len(values),
                "f1": sum(value[2] for value in values) / len(values),
            }
            for condition, values in sorted(condition_sample_metrics.items())
        },
    }
    return metrics, enriched_outputs
