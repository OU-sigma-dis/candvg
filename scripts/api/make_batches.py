"""Build OpenAI Batch API input files for grounding inference."""

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from candvg.models import build_model_backend
from candvg.models.openai import OpenAIGPTModel
from candvg.tasks import build_task
from candvg.tasks.utils import build_grounding_text
from candvg.utils import GroundingSample

BATCH_ENDPOINT = "/v1/responses"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build OpenAI Batch API input JSONL files."
    )
    parser.add_argument(
        "--model", type=str, required=True, help="OpenAI model ID, e.g. gpt-5."
    )
    parser.add_argument("--dataset", type=str, default="candvg")
    parser.add_argument(
        "--target-mode",
        type=str,
        default="single",
        choices=["single", "candidates"],
    )
    parser.add_argument("--export-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--num-batches",
        type=int,
        default=1,
        help="Split requests evenly across this many Batch API input files.",
    )
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default="high",
        choices=["none", "low", "medium", "high"],
    )
    parser.add_argument(
        "--selection-mode",
        action="store_true",
        help="Append oracle candidate bboxes from task metadata.",
    )
    return parser.parse_args()


def model_card_name(model: str) -> str:
    """Derive the output directory name used for a model identifier."""
    trimmed = model.rstrip("/")
    candidate = Path(trimmed).name if "/" in trimmed or "\\" in trimmed else trimmed
    return re.sub(r"[^0-9A-Za-z._-]+", "_", candidate)


def experiment_name(dataset: str, target_mode: str, selection_mode: bool) -> str:
    """Build the experiment directory name used by task inference."""
    name = f"{dataset}_{target_mode}"
    return f"{name}_selection" if selection_mode else name


def prepare_input_text(sample: GroundingSample) -> str:
    """Build one grounding user prompt."""
    return build_grounding_text(
        sentence=sample.sentence,
        positive_token_span=sample.positive_token_span,
    )


def build_batch_request(
    sample: GroundingSample,
    model_backend: OpenAIGPTModel,
    max_output_tokens: int,
    reasoning_effort: str,
) -> dict[str, Any]:
    """Build one Responses API request in Batch API JSONL format."""
    body = model_backend.build_request_body(
        image_path=sample.image_path,
        input_text=prepare_input_text(sample),
        max_output_tokens=max_output_tokens,
        reasoning_effort=reasoning_effort,
        metadata=sample.metadata,
    )
    return {
        "custom_id": sample.sample_id,
        "method": "POST",
        "url": BATCH_ENDPOINT,
        "body": body,
    }


def write_batch_input(requests: Iterable[dict[str, Any]], output_path: Path) -> int:
    """Write requests as JSONL and return the number of rows written."""
    output_path.parent.mkdir(parents=True, exist_ok=False)
    count = 0
    seen_custom_ids: set[str] = set()
    with output_path.open("x", encoding="utf-8") as file:
        for request in requests:
            custom_id = str(request["custom_id"])
            if custom_id in seen_custom_ids:
                raise ValueError(f"Duplicate custom_id: {custom_id}")
            seen_custom_ids.add(custom_id)
            file.write(json.dumps(request, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_split_batch_inputs(
    requests: Iterable[dict[str, Any]],
    run_dir: Path,
    request_count: int,
    num_batches: int,
) -> list[tuple[Path, int]]:
    """Write requests evenly across numbered Batch API input directories."""
    if num_batches <= 0:
        raise ValueError("num_batches must be positive")
    if num_batches > request_count:
        raise ValueError("num_batches cannot exceed the number of requests")

    base_size, remainder = divmod(request_count, num_batches)
    batch_sizes = [
        base_size + (1 if index < remainder else 0)
        for index in range(num_batches)
    ]
    request_iterator = iter(requests)
    seen_custom_ids: set[str] = set()
    outputs: list[tuple[Path, int]] = []
    assert sum(batch_sizes) == request_count, "Batch sizes do not sum to request count"

    for batch_index, batch_size in enumerate(batch_sizes, start=1):
        output_path = run_dir / f"batch{batch_index}" / "batch_input.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=False)
        with output_path.open("x", encoding="utf-8") as file:
            for _ in range(batch_size):
                request = next(request_iterator)
                custom_id = str(request["custom_id"])
                if custom_id in seen_custom_ids:
                    raise ValueError(f"Duplicate custom_id: {custom_id}")
                seen_custom_ids.add(custom_id)
                file.write(json.dumps(request, ensure_ascii=False) + "\n")
        outputs.append((output_path, batch_size))

    return outputs


def main(args: argparse.Namespace) -> Path:
    """Generate timestamped Batch API input files and run configuration."""
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("max_samples must be positive")
    if args.max_output_tokens <= 0:
        raise ValueError("max_output_tokens must be positive")
    if args.num_batches <= 0:
        raise ValueError("num_batches must be positive")

    dataset = build_task(
        dataset_name=args.dataset,
        max_samples=args.max_samples,
        target_mode=args.target_mode,
    )
    model_backend = build_model_backend(
        checkpoint=args.model,
        device=None,
        selection_mode=args.selection_mode,
    )
    if not isinstance(model_backend, OpenAIGPTModel):
        raise TypeError("Batch API generation requires an OpenAI model backend")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    exp_name = experiment_name(
        dataset=args.dataset,
        target_mode=args.target_mode,
        selection_mode=args.selection_mode,
    )
    run_dir = args.export_dir / exp_name / f"{model_card_name(args.model)}_{args.reasoning_effort}" / timestamp
    request_count = len(dataset)
    if args.num_batches > request_count:
        raise ValueError("num_batches cannot exceed the number of requests")

    requests = (
        build_batch_request(
            sample=dataset[index],
            model_backend=model_backend,
            max_output_tokens=args.max_output_tokens,
            reasoning_effort=args.reasoning_effort,
        )
        for index in range(len(dataset))
    )
    batch_outputs = write_split_batch_inputs(
        requests=requests,
        run_dir=run_dir,
        request_count=request_count,
        num_batches=args.num_batches,
    )

    run_config = {
        "exp_name": exp_name,
        "model": args.model,
        "dataset": args.dataset,
        "target_mode": args.target_mode,
        "selection_mode": args.selection_mode,
        "max_samples": args.max_samples,
        "num_batches": args.num_batches,
        "batch_request_counts": [count for _, count in batch_outputs],
        "max_output_tokens": args.max_output_tokens,
        "reasoning_effort": args.reasoning_effort,
        "batch_endpoint": BATCH_ENDPOINT,
        "request_count": request_count,
        "created_at_utc": timestamp,
    }
    with (run_dir / "run_config.json").open("x", encoding="utf-8") as file:
        json.dump(run_config, file, ensure_ascii=False, indent=2)

    print(f"Requests: {request_count}")
    for batch_path, batch_size in batch_outputs:
        print(f"Batch input ({batch_size} requests): {batch_path}")
    return run_dir


if __name__ == "__main__":
    main(parse_args())
