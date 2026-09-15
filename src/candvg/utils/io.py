"""I/O utilities for evaluation outputs and visualization."""

import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .bbox import BoundingBox
from .schema.predict import BoundingBoxPrediction, PhrasePrediction, SentencePrediction


def _json_default(obj: Any) -> Any:
    """Fallback serializer for JSON dumping."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, BoundingBox):
        return list(obj.to_xyxy())
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write JSONL rows with project-specific serialization."""
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(
                json.dumps(row, ensure_ascii=False, default=_json_default) + "\n"
            )


def _model_name_from_checkpoint(model_checkpoint: str) -> str:
    """Derive output folder-safe model name from checkpoint identifier or path."""
    trimmed = model_checkpoint.rstrip("/")
    candidate = Path(trimmed).name if "/" in trimmed or "\\" in trimmed else trimmed
    return re.sub(r"[^0-9A-Za-z._-]+", "_", candidate)


def build_sentence_predictions(
    samples: list[dict[str, Any]],
) -> list[SentencePrediction]:
    """Convert sample-level outputs into prediction objects.

    `SentencePrediction.doc_id` is the stable sample identifier used by
    evaluation scripts to match predictions back to dataset samples.
    """
    sentence_predictions: list[SentencePrediction] = []
    for sample in sorted(samples, key=lambda item: str(item["sample_id"])):
        sample_id = str(sample["sample_id"])
        image_path = str(sample["image_path"])
        sentence = str(sample["sentence"])
        image_id = Path(image_path).stem
        start, end = (
            int(sample["positive_token_span"][0]),
            int(sample["positive_token_span"][1]),
        )
        phrase_text = sentence[start:end]
        pred_bboxes = sample["pred_bboxes"]

        sentence_predictions.append(
            SentencePrediction(
                doc_id=sample_id,
                image_id=image_id,
                phrases=[
                    PhrasePrediction(
                        index=0,
                        text=phrase_text,
                        bounding_boxes=[
                            BoundingBoxPrediction(
                                image_id=image_id,
                                rect=pred_bbox,
                                confidence=1.0,  # VLM backends do not expose calibrated scores.
                            )
                            for pred_bbox in pred_bboxes
                        ],
                    )
                ],
            )
        )

    return sentence_predictions


def write_prediction_outputs(
    model: str,
    exp_name: str,
    runtime_mode: str,
    world_size: int,
    export_dir: Path,
    predictions: list[SentencePrediction],
    inference_outputs: list[dict[str, Any]] | None = None,
) -> Path:
    """Write inference artifacts without computing or storing metrics."""
    model_name = _model_name_from_checkpoint(model)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = export_dir / exp_name / model_name / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    write_prediction_artifacts(
        run_dir=run_dir,
        predictions=predictions,
        inference_outputs=inference_outputs,
    )

    run_config = {
        "exp_name": exp_name,
        "model": model,
        "runtime_mode": runtime_mode,
        "world_size": world_size,
        "created_at_utc": timestamp,
    }

    with (run_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, ensure_ascii=False)

    return run_dir


def write_prediction_artifacts(
    run_dir: Path,
    predictions: list[SentencePrediction],
    inference_outputs: list[dict[str, Any]] | None,
) -> Path:
    """Write predictions and inference rows into an existing run directory."""
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir = run_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    grouped_predictions: dict[str, list[SentencePrediction]] = defaultdict(list)
    for prediction in predictions:
        grouped_predictions[prediction.image_id].append(prediction)

    for image_id, image_predictions in grouped_predictions.items():
        prediction_path = predictions_dir / f"{image_id}.json"
        with prediction_path.open("w", encoding="utf-8") as f:
            json.dump(
                [item.to_dict() for item in image_predictions],
                f,
                ensure_ascii=False,
                indent=2,
            )

    if inference_outputs is not None:
        _write_jsonl(run_dir / "inference_outputs.jsonl", inference_outputs)
    return run_dir


def write_evaluation_outputs(
    run_dir: Path,
    dataset_name: str,
    metric_name: str,
    metrics: dict[str, Any],
    iou_threshold: float,
    eval_outputs: list[dict[str, Any]],
) -> Path:
    """Write evaluation metrics and per-sample evaluation details."""
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(run_dir / "evaluation_outputs.jsonl", eval_outputs)

    payload = {
        "dataset": dataset_name,
        "metric": metric_name,
        "total_outputs": len(eval_outputs),
        "iou_threshold": iou_threshold,
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
        **metrics,
    }
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return run_dir
