import argparse
import json
from pathlib import Path
from typing import Any

from candvg.metrics import compute_acc, compute_prec_rec_f1_metrics
from candvg.tasks import build_task
from candvg.utils import SentencePrediction, write_evaluation_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="candvg")
    parser.add_argument("--exp-dir", type=Path, help="Path to an experiment directory")
    parser.add_argument(
        "--target-mode",
        type=str,
        default="single",
        choices=["single", "candidates"],
        help="Prediction mode for context-aware visual grounding. single: predict single bbox; candidates: predict all candidate bboxes.",
    )
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    return parser.parse_args()


def _build_eval_inputs(
    dataset_name: str, predictions: list[SentencePrediction], target_mode: str
) -> list[dict[str, Any]]:
    """Build metric inputs by matching dataset samples with saved predictions."""
    dataset = build_task(dataset_name=dataset_name, target_mode=target_mode)

    eval_inputs: list[dict[str, Any]] = []
    for prediction in predictions:
        sample = dataset.get_sample_from_id(prediction.doc_id)
        pred_bboxes = [
            bbox_prediction.rect
            for phrase in prediction.phrases
            for bbox_prediction in phrase.bounding_boxes
        ]
        eval_inputs.append(
            {
                "sample_id": sample.sample_id,
                "image_path": sample.image_path,
                "sentence": sample.sentence,
                "positive_token_span": sample.positive_token_span,
                "gt_bboxes": sample.gt_bboxes,
                "pred_bboxes": pred_bboxes,
                **sample.metadata,
            }
        )
    return eval_inputs


def main() -> int:
    args = parse_args()
    predictions_dir = args.exp_dir / "predictions"
    assert predictions_dir.exists(), "Prediction directory not found: {predictions_dir}"

    predictions: list[SentencePrediction] = []
    for prediction_file in sorted(predictions_dir.glob("*.json")):
        rows = json.loads(prediction_file.read_text(encoding="utf-8"))
        predictions.extend(SentencePrediction.from_dict(row) for row in rows)

    eval_inputs = _build_eval_inputs(
        dataset_name=args.dataset,
        predictions=predictions,
        target_mode=args.target_mode,
    )

    if args.target_mode == "single":
        # Calculate ACC@0.5
        metrics, eval_outputs = compute_acc(eval_inputs, threshold=args.iou_threshold)
        metric_name = "single_object_accuracy"
    else:
        # Calculate Precision, Recall, F1
        metrics, eval_outputs = compute_prec_rec_f1_metrics(
            eval_inputs, threshold=args.iou_threshold
        )
        metric_name = "multi_object_precision_recall_f1"

    write_evaluation_outputs(
        run_dir=args.exp_dir,
        dataset_name=args.dataset,
        metric_name=metric_name,
        metrics=metrics,
        iou_threshold=args.iou_threshold,
        eval_outputs=eval_outputs,
    )
    print(f"Evaluated {len(eval_inputs)} samples from {predictions_dir}")
    print(f"Metrics ({metric_name}):")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"- {key}: {value:.4f}")
        else:
            print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
