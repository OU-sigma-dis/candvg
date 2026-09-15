"""Plot average CandVG bbox counts by dialogue turn."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

os.environ.setdefault("MPLCONFIGDIR", "/tmp/context_vlm_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from candvg.utils.bbox import box_iou  # noqa: E402

BBox = tuple[float, float, float, float]
DISPLAY_TURNS = (1, 2, 3)


@dataclass(frozen=True)
class CandVGRow:
    """One CandVG row with fields needed for turn-level aggregation."""

    sample_id: str
    dialogue_id: str
    turn_index: int
    gold_bboxes: list[BBox]


@dataclass(frozen=True)
class PredictionRow:
    """One inference output row."""

    sample_id: str
    pred_bboxes: list[BBox]


@dataclass(frozen=True)
class TurnSummary:
    """Aggregated candidate counts for one turn index."""

    turn_index: int
    n_samples: int

    mean_gold_count: float
    mean_pred_count: float
    mean_correct_count: float
    mean_precision: float
    mean_recall: float


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exp-dir",
        type=Path,
        required=True,
        help="Experiment directory containing inference_outputs.jsonl.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/candvg/candvg.jsonl"),
        help="CandVG JSONL path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <exp-dir>/<target-mode>_bbox_count_transitions.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold used to count correct predicted bboxes.",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="pdf",
        choices=["png", "pdf", "svg"],
        help="Plot output format.",
    )
    return parser.parse_args()


def _as_bbox(raw_bbox: list[Any]) -> BBox:
    """Convert a raw sequence to an xyxy bbox tuple."""
    if len(raw_bbox) != 4:
        raise ValueError(f"Expected bbox with 4 values, got {len(raw_bbox)}")
    x1, y1, x2, y2 = (float(value) for value in raw_bbox)
    return x1, y1, x2, y2


def _count_correct_bboxes(
    gold_bboxes: list[BBox], pred_bboxes: list[BBox], threshold: float
) -> int:
    """Count one-to-one matched predicted bboxes at the IoU threshold."""
    if not gold_bboxes or not pred_bboxes:
        return 0

    pred_tensor = torch.tensor(pred_bboxes, dtype=torch.float32)
    gold_tensor = torch.tensor(gold_bboxes, dtype=torch.float32)
    iou_tensor, _ = box_iou(pred_tensor, gold_tensor)
    iou_matrix = iou_tensor.cpu().numpy().astype(np.float64)
    binary_matrix = (iou_matrix >= threshold).astype(np.float64)
    pred_indices, gold_indices = linear_sum_assignment(-binary_matrix)
    return int(
        sum(
            1
            for pred_idx, gold_idx in zip(
                pred_indices.tolist(), gold_indices.tolist(), strict=True
            )
            if iou_matrix[pred_idx, gold_idx] >= threshold
        )
    )


def load_candvg_rows(data_path: Path) -> dict[str, CandVGRow]:
    """Load CandVG rows indexed by sample id."""
    sample_rows: dict[str, CandVGRow] = {}
    with data_path.open("r", encoding="utf-8") as file:
        for line in file:
            content = line.strip()
            if not content:
                continue
            payload = json.loads(content)
            annotation = payload["annotation"]
            metadata = payload["metadata"]
            sample_id = str(payload["sample_id"])
            sample_rows[sample_id] = CandVGRow(
                sample_id=sample_id,
                dialogue_id=str(metadata["global_id"]),
                turn_index=int(metadata["turn_index"]),
                gold_bboxes=[_as_bbox(bbox) for bbox in annotation["candidate_bboxes"]],
            )
    return sample_rows


def load_predictions(exp_dir: Path) -> dict[str, PredictionRow]:
    """Load inference outputs indexed by sample id."""
    inference_path = exp_dir / "inference_outputs.jsonl"
    if not inference_path.exists():
        raise FileNotFoundError(f"inference_outputs.jsonl not found: {inference_path}")

    predictions: dict[str, PredictionRow] = {}
    with inference_path.open("r", encoding="utf-8") as file:
        for line in file:
            content = line.strip()
            if not content:
                continue
            payload = json.loads(content)
            sample_id = str(payload["sample_id"])
            predictions[sample_id] = PredictionRow(
                sample_id=sample_id,
                pred_bboxes=[_as_bbox(bbox) for bbox in payload.get("pred_bboxes", [])],
            )
    return predictions


def summarize_by_turn(
    *,
    sample_rows: dict[str, CandVGRow],
    predictions: dict[str, PredictionRow],
    iou_threshold: float,
) -> list[TurnSummary]:
    """Aggregate gold, predicted, and correct bbox counts by turn index."""
    turn_values: dict[int, list[dict[str, float]]] = defaultdict(list)
    for sample_id, prediction in predictions.items():
        sample = sample_rows.get(sample_id)
        if sample is None:
            continue

        gold_count = len(sample.gold_bboxes)
        pred_count = len(prediction.pred_bboxes)

        correct_count = _count_correct_bboxes(
            gold_bboxes=sample.gold_bboxes,
            pred_bboxes=prediction.pred_bboxes,
            threshold=iou_threshold,
        )
        precision = correct_count / pred_count if pred_count > 0 else 0.0
        recall = correct_count / gold_count if gold_count > 0 else 0.0

        display_turn = sample.turn_index + 1
        if display_turn not in DISPLAY_TURNS:
            continue
        turn_values[display_turn].append(
            {
                "gold_count": float(gold_count),
                "pred_count": float(pred_count),
                "correct_count": float(correct_count),
                "precision": precision,
                "recall": recall,
            }
        )

    summaries: list[TurnSummary] = []
    for turn_index, values in sorted(turn_values.items()):
        n_samples = len(values)
        summaries.append(
            TurnSummary(
                turn_index=turn_index,
                n_samples=n_samples,
                mean_gold_count=sum(value["gold_count"] for value in values)
                / n_samples,
                mean_pred_count=sum(value["pred_count"] for value in values)
                / n_samples,
                mean_correct_count=sum(value["correct_count"] for value in values)
                / n_samples,
                mean_precision=sum(value["precision"] for value in values) / n_samples,
                mean_recall=sum(value["recall"] for value in values) / n_samples,
            )
        )
    return summaries


def write_summary_csv(csv_path: Path, summaries: list[TurnSummary]) -> None:
    """Write turn-level count summary."""
    fieldnames = [
        "turn_index",
        "n_samples",
        "mean_gold_count",
        "mean_pred_count",
        "mean_correct_count",
        "mean_precision",
        "mean_recall",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(
                {
                    "turn_index": summary.turn_index,
                    "n_samples": summary.n_samples,
                    "mean_gold_count": summary.mean_gold_count,
                    "mean_pred_count": summary.mean_pred_count,
                    "mean_correct_count": summary.mean_correct_count,
                    "mean_precision": summary.mean_precision,
                    "mean_recall": summary.mean_recall,
                }
            )


def plot_summary(
    plot_path: Path, summaries: list[TurnSummary], iou_threshold: float
) -> None:
    """Plot average bbox counts by turn."""
    if not summaries:
        raise ValueError("No summaries to plot.")

    turns = [summary.turn_index for summary in summaries]
    gold_counts = [summary.mean_gold_count for summary in summaries]
    pred_counts = [summary.mean_pred_count for summary in summaries]
    correct_counts = [summary.mean_correct_count for summary in summaries]
    sample_counts = [summary.n_samples for summary in summaries]

    fig, count_ax = plt.subplots(figsize=(8, 6))
    sample_ax = count_ax.twinx()

    sample_ax.set_zorder(0)
    count_ax.set_zorder(1)
    count_ax.patch.set_visible(False)

    count_ax.plot(
        turns,
        gold_counts,
        color="#6aa94b",
        marker="o",
        linewidth=2,
        markersize=12,
        label="Ground-truth BBoxes",
    )
    count_ax.plot(
        turns,
        pred_counts,
        color="#ff9705",
        marker="x",
        linewidth=2,
        markersize=12,
        label="Predicted BBoxes",
    )
    count_ax.plot(
        turns,
        correct_counts,
        color="#7570b3",
        marker="s",
        linewidth=2,
        markersize=12,
        label=f"Correct BBoxes (IoU>={iou_threshold:g})",
    )
    count_ax.set_xlabel("w/ Turns", fontsize=20)
    count_ax.set_ylabel("Average BBox count", fontsize=20)
    count_ax.tick_params(axis="both", labelsize=18)
    count_ax.set_xticks(turns)
    count_ax.grid(True, alpha=0.25)
    count_ax.set_ylim(0, 6)

    sample_ax.bar(
        turns, sample_counts, color="0.85", alpha=0.6, width=0.4, label="Samples"
    )
    sample_ax.set_ylabel("Samples", fontsize=20)
    sample_ax.tick_params(axis="y", labelsize=18)

    line_handles, line_labels = count_ax.get_legend_handles_labels()
    bar_handles, bar_labels = sample_ax.get_legend_handles_labels()
    count_ax.legend(
        line_handles + bar_handles,
        line_labels + bar_labels,
        loc="upper right",
        fontsize=18,
    )

    # fig.suptitle(f"CandVG {target_label} average bbox counts by turn")
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)


def main() -> int:
    """Entry point."""
    args = parse_args()

    output_dir = args.output_dir or args.exp_dir / "bbox_count_transitions"

    sample_rows = load_candvg_rows(args.data_path)
    predictions = load_predictions(args.exp_dir)
    summaries = summarize_by_turn(
        sample_rows=sample_rows,
        predictions=predictions,
        iou_threshold=args.iou_threshold,
    )
    if not summaries:
        raise ValueError("No prediction rows matched CandVG sample ids.")

    output_stem = "bbox_count_transitions"
    write_summary_csv(output_dir / f"{output_stem}.csv", summaries)
    plot_summary(
        plot_path=output_dir / f"{output_stem}.{args.format}",
        summaries=summaries,
        iou_threshold=args.iou_threshold,
    )

    print(f"Done. turns={len(summaries)}, output={output_dir}")
    return 0


if __name__ == "__main__":
    main()
