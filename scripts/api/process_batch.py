"""Convert and optionally evaluate OpenAI Batch API grounding output."""

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from PIL import Image

from candvg.metrics import compute_acc, compute_prec_rec_f1_metrics
from candvg.models.openai import OpenAIGPTModel
from candvg.models.utils import parse_bbox_2d_json
from candvg.tasks import build_task
from candvg.utils import (
    GroundingSample,
    SentencePrediction,
    build_sentence_predictions,
    normalize_pred_bbox,
    write_evaluation_outputs,
    write_prediction_artifacts,
)


def parse_args() -> argparse.Namespace:
    """Parse conversion and evaluation arguments."""
    parser = argparse.ArgumentParser(
        description="Convert and evaluate OpenAI Batch grounding output."
    )
    parser.add_argument("--dataset", type=str, default="candvg")
    parser.add_argument(
        "--exp-dir", type=Path, required=True, help="Path to an experiment directory"
    )
    parser.add_argument(
        "--target-mode",
        type=str,
        default="single",
        choices=["single", "candidates"],
        help="Prediction mode for context-aware visual grounding. single: predict single bbox; candidates: predict all candidate bboxes.",
    )
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Create prediction artifacts without running evaluation.",
    )
    return parser.parse_args()


def resolve_batch_output_paths(
    run_dir: Path, run_config: dict[str, Any]
) -> list[Path]:
    """Resolve legacy or split Batch output paths for one experiment run."""
    if "num_batches" not in run_config:
        batch_output_path = run_dir / "batches" / "batch_output.jsonl"
        if not batch_output_path.is_file():
            raise FileNotFoundError(f"Batch output not found: {batch_output_path}")
        return [batch_output_path]

    num_batches = int(run_config["num_batches"])
    if num_batches <= 0:
        raise ValueError("run_config num_batches must be positive")
    batch_output_paths = [
        run_dir / f"batch{index}" / "batch_output.jsonl"
        for index in range(1, num_batches + 1)
    ]
    missing_paths = [path for path in batch_output_paths if not path.is_file()]
    if missing_paths:
        missing = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Batch output not found: {missing}")
    return batch_output_paths

def load_batch_answers(batch_output_paths: Iterable[Path]) -> dict[str, str]:
    """Load successful Batch rows and map custom IDs to parsed bbox JSON."""
    answers: dict[str, str] = {}
    for batch_output_path in batch_output_paths:
        with batch_output_path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON on line {line_number}: {error}"
                    ) from error

                custom_id = row.get("custom_id")
                if not isinstance(custom_id, str) or not custom_id:
                    raise ValueError(f"Missing custom_id on line {line_number}")
                if custom_id in answers:
                    raise ValueError(f"Duplicate custom_id: {custom_id}")

                response = row.get("response")
                if not isinstance(response, dict) or response.get("status_code") != 200:
                    detail = row.get("error") or response
                    raise ValueError(
                        f"Batch request failed for {custom_id}: {detail}"
                    )
                response_body = response.get("body")
                if not isinstance(response_body, dict):
                    raise ValueError(f"Missing response body for {custom_id}")
                answers[custom_id] = OpenAIGPTModel.parse_response_body(response_body)
    return answers


def build_inference_outputs(
    samples: Iterable[GroundingSample],
    answers: dict[str, str],
    selection_mode: bool,
) -> list[dict[str, object]]:
    """Match Batch answers to samples and build synchronous-style output rows."""
    sample_list = list(samples)
    expected_ids = {sample.sample_id for sample in sample_list}
    answer_ids = set(answers)
    if expected_ids != answer_ids:
        missing = sorted(expected_ids - answer_ids)
        unexpected = sorted(answer_ids - expected_ids)
        raise ValueError(
            f"Batch/sample ID mismatch: missing={missing}, unexpected={unexpected}"
        )

    outputs: list[dict[str, object]] = []
    for sample in sample_list:
        answer = answers[sample.sample_id]
        with Image.open(sample.image_path) as image:
            width, height = image.size
        pred_bboxes = normalize_pred_bbox(
            pred_bboxes=parse_bbox_2d_json(answer),
            hw=(height, width),
            coord_mode="relative_1000" if selection_mode else "pixel",
            coord_scale=1000 if selection_mode else None,
        )
        outputs.append(
            {
                "sample_id": sample.sample_id,
                "image_path": sample.image_path,
                "sentence": sample.sentence,
                "positive_token_span": sample.positive_token_span,
                "answer": answer,
                "pred_bboxes": pred_bboxes,
            }
        )
    return outputs


def _validate_run_config(
    run_config: dict[str, Any], dataset: str, target_mode: str
) -> None:
    config_dataset = str(run_config["dataset"])
    config_target_mode = str(run_config["target_mode"])
    if dataset != config_dataset or target_mode != config_target_mode:
        raise ValueError(
            "Arguments do not match run_config.json: "
            f"dataset={dataset!r} (expected {config_dataset!r}), "
            f"target_mode={target_mode!r} (expected {config_target_mode!r})"
        )


def _build_eval_inputs(
    dataset_name: str,
    predictions: list[SentencePrediction],
    target_mode: str,
) -> list[dict[str, Any]]:
    """Build metric inputs by matching saved predictions to task samples."""
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


def evaluate_predictions(
    run_dir: Path,
    dataset_name: str,
    target_mode: str,
    iou_threshold: float,
) -> None:
    """Evaluate saved predictions and write standard evaluation artifacts."""
    predictions_dir = run_dir / "predictions"
    if not predictions_dir.is_dir():
        raise FileNotFoundError(f"Prediction directory not found: {predictions_dir}")

    predictions: list[SentencePrediction] = []
    for prediction_file in sorted(predictions_dir.glob("*.json")):
        rows = json.loads(prediction_file.read_text(encoding="utf-8"))
        predictions.extend(SentencePrediction.from_dict(row) for row in rows)

    eval_inputs = _build_eval_inputs(
        dataset_name=dataset_name,
        predictions=predictions,
        target_mode=target_mode,
    )
    if target_mode == "single":
        metrics, eval_outputs = compute_acc(eval_inputs, threshold=iou_threshold)
        metric_name = "single_object_accuracy"
    else:
        metrics, eval_outputs = compute_prec_rec_f1_metrics(
            eval_inputs, threshold=iou_threshold
        )
        metric_name = "multi_object_precision_recall_f1"

    write_evaluation_outputs(
        run_dir=run_dir,
        dataset_name=dataset_name,
        metric_name=metric_name,
        metrics=metrics,
        iou_threshold=iou_threshold,
        eval_outputs=eval_outputs,
    )
    print(f"Evaluated {len(eval_inputs)} samples from {predictions_dir}")
    print(f"Metrics ({metric_name}):")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"- {key}: {value:.4f}")
        else:
            print(f"- {key}: {value}")


def main(args: argparse.Namespace) -> Path:
    """Convert one completed Batch run and optionally evaluate predictions."""
    run_dir = args.exp_dir
    run_config_path = run_dir / "run_config.json"
    if not run_config_path.is_file():
        raise FileNotFoundError(f"Run config not found: {run_config_path}")

    run_config: dict[str, Any] = json.loads(run_config_path.read_text(encoding="utf-8"))
    batch_output_paths = resolve_batch_output_paths(run_dir, run_config)
    _validate_run_config(run_config, args.dataset, args.target_mode)
    dataset = build_task(
        dataset_name=args.dataset,
        max_samples=run_config.get("max_samples"),
        target_mode=args.target_mode,
    )
    samples = [dataset[index] for index in range(len(dataset))]
    answers = load_batch_answers(batch_output_paths)
    inference_outputs = build_inference_outputs(
        samples=samples,
        answers=answers,
        selection_mode=bool(run_config.get("selection_mode", False)),
    )
    predictions = build_sentence_predictions(inference_outputs)
    write_prediction_artifacts(
        run_dir=run_dir,
        predictions=predictions,
        inference_outputs=inference_outputs,
    )
    print(f"Converted {len(inference_outputs)} Batch responses")
    print(f"Outputs: {run_dir}")

    if not args.skip_evaluation:
        evaluate_predictions(
            run_dir=run_dir,
            dataset_name=args.dataset,
            target_mode=args.target_mode,
            iou_threshold=args.iou_threshold,
        )
    return run_dir


if __name__ == "__main__":
    main(parse_args())
