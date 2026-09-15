"""Visualize grounding predictions in one experiment directory."""

import argparse
import json
from pathlib import Path

from PIL import Image

from candvg.utils import PhrasePrediction, SentencePrediction, plot_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-dir",
        type=Path,
        required=True,
        help="Experiment directory path, e.g. outputs/<dataset>/<model>/<timestamp>",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        nargs="*",
        default=[],
        help="Optional image root paths used when image path cannot be found from inference outputs",
    )
    return parser.parse_args()


def _load_image_path_map(exp_dir: Path) -> dict[str, Path]:
    """Load image_id -> image_path map from available JSONL artifacts."""
    candidate_paths = [
        exp_dir / "inference_outputs.jsonl",
        exp_dir / "model_enriched_outputs.jsonl",
        exp_dir / "evaluation_outputs.jsonl",
    ]
    artifact_path = next((path for path in candidate_paths if path.exists()), None)
    if artifact_path is None:
        return {}

    mapping: dict[str, Path] = {}
    with artifact_path.open("r", encoding="utf-8") as file:
        for line in file:
            row = line.strip()
            if not row:
                continue
            payload = json.loads(row)
            image_path = payload.get("image_path")
            if not image_path:
                continue
            path = Path(str(image_path))
            mapping[path.stem] = path
    return mapping


def _resolve_image_path(
    image_id: str, image_map: dict[str, Path], image_roots: list[Path]
) -> Path | None:
    """Resolve image path from known map or optional root directories."""
    mapped = image_map.get(image_id)
    if mapped is not None and mapped.exists():
        return mapped

    for root in image_roots:
        for matched in root.glob(f"{image_id}.*"):
            if matched.is_file():
                return matched
    return None


def _load_predictions(prediction_file: Path) -> list[SentencePrediction]:
    """Load one predictions json file into SentencePrediction list."""
    payload = json.loads(prediction_file.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = [payload]
    return [SentencePrediction.from_dict(item) for item in payload]


def _merge_predictions(
    image_id: str, predictions: list[SentencePrediction]
) -> SentencePrediction:
    """Merge sentence-level predictions into one object for one image plot."""
    merged_phrases: list[PhrasePrediction] = []
    for prediction in predictions:
        for phrase in prediction.phrases:
            merged_phrases.append(
                PhrasePrediction(
                    index=len(merged_phrases),
                    text=phrase.text,
                    bounding_boxes=phrase.bounding_boxes,
                )
            )
    return SentencePrediction(
        doc_id=f"{image_id}:merged", image_id=image_id, phrases=merged_phrases
    )


def main() -> int:
    args = parse_args()
    exp_dir = args.exp_dir
    predictions_dir = exp_dir / "predictions"
    visualize_dir = exp_dir / "visualize"

    if not predictions_dir.exists():
        raise FileNotFoundError(f"predictions dir not found: {predictions_dir}")

    image_map = _load_image_path_map(exp_dir)
    prediction_files = sorted(predictions_dir.glob("*.json"))
    print(f"Loaded {len(prediction_files)} prediction files from {predictions_dir}")

    rendered = 0
    skipped = 0
    for prediction_file in prediction_files:
        image_id = prediction_file.stem
        output_file = visualize_dir / f"{image_id}.png"
        if output_file.exists():
            skipped += 1
            continue

        predictions = _load_predictions(prediction_file)
        if not predictions:
            skipped += 1
            continue

        image_path = _resolve_image_path(
            image_id=image_id, image_map=image_map, image_roots=args.image_root
        )
        if image_path is None:
            print(f"[SKIP] image path not found for image_id={image_id}")
            skipped += 1
            continue

        merged_prediction = _merge_predictions(
            image_id=image_id, predictions=predictions
        )
        with Image.open(image_path) as image:
            plot_results(
                image=image.convert("RGB"),
                prediction=merged_prediction,
                output_file=output_file,
            )
        rendered += 1

    print(f"Done. rendered={rendered}, skipped={skipped}, output={visualize_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
