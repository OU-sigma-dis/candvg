import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from huggingface_hub import get_token

MODEL_GROUPS: dict[str, list[str]] = {
    "qwen3_vl": [
        "Qwen/Qwen3-VL-2B-Instruct",
        "Qwen/Qwen3-VL-4B-Instruct",
        "Qwen/Qwen3-VL-8B-Instruct",
        "Qwen/Qwen3-VL-32B-Instruct",
    ],
    "qwen3_5": [
        "Qwen/Qwen3.5-2B",
        "Qwen/Qwen3.5-4B",
        "Qwen/Qwen3.5-9B",
        "Qwen/Qwen3.5-27B",
    ],
    "intern_vl_3_5": [
        "OpenGVLab/InternVL3_5-2B-HF",          # InternViT-300M + Qwen3-1.7B
        "OpenGVLab/InternVL3_5-4B-HF",          # InternViT-300M + Qwen3-4B
        "OpenGVLab/InternVL3_5-8B-HF",          # InternViT-300M + Qwen3-8B
        "OpenGVLab/InternVL3_5-38B-HF",         # InternViT-300M + Qwen3-32B
    ],
    "cosmos_reason_2": [
        "nvidia/Cosmos-Reason2-2B",
        "nvidia/Cosmos-Reason2-8B",
    ],
    "rynnbrain": [
        "Alibaba-DAMO-Academy/RynnBrain-2B",
        "Alibaba-DAMO-Academy/RynnBrain-8B",
    ],
}


@dataclass(frozen=True)
class DownloadResult:
    """Status for one model download attempt."""

    model_id: str
    success: bool
    output_path: Path | None
    error: str | None


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Download model checkpoints from Hugging Face Hub.",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=["all"],
        choices=["all", *MODEL_GROUPS.keys()],
        help="Model groups to download. Use 'all' for all preset groups.",
    )
    parser.add_argument(
        "--model-id",
        action="append",
        default=[],
        help=(
            "Hugging Face model ID to download. Repeatable; when specified, "
            "preset groups are ignored."
        ),
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Root directory for downloaded checkpoints.",
    )
    return parser.parse_args()


def ordered_unique(items: Iterable[str]) -> list[str]:
    """Return items in insertion order while dropping duplicates."""
    return list(dict.fromkeys(items))


def resolve_model_ids(groups: list[str], extra_model_ids: list[str]) -> list[str]:
    """Resolve explicit model IDs or selected groups into a model list."""
    if extra_model_ids:
        return ordered_unique(extra_model_ids)

    if "all" in groups:
        model_ids = [
            model_id
            for group_name in MODEL_GROUPS
            for model_id in MODEL_GROUPS[group_name]
        ]
    else:
        model_ids = [
            model_id for group in groups for model_id in MODEL_GROUPS[group]
        ]

    return ordered_unique(model_ids)


def download_model(
    model_id: str,
    checkpoints_dir: Path,
    token: str | None,
) -> DownloadResult:
    """Download one model snapshot into checkpoints_dir/model_id."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        return DownloadResult(
            model_id=model_id,
            success=False,
            output_path=None,
            error=(
                "huggingface_hub is not installed. Run `uv sync` before downloading. "
                f"Original error: {exc}"
            ),
        )

    local_dir = checkpoints_dir / model_id
    local_dir.parent.mkdir(parents=True, exist_ok=True)

    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=str(local_dir),
            token=token,
        )
        return DownloadResult(
            model_id=model_id,
            success=True,
            output_path=local_dir,
            error=None,
        )
    except Exception as exc:  # noqa: BLE001
        return DownloadResult(
            model_id=model_id,
            success=False,
            output_path=None,
            error=str(exc),
        )


def main() -> int:
    """CLI entry point."""
    args = parse_args()
    model_ids = resolve_model_ids(args.groups, args.model_id)

    if not model_ids:
        print("No model IDs resolved. Use --groups and/or --model-id.")
        return 1

    print("Resolved download list:")
    for model_id in model_ids:
        print(f"- {model_id}")

    checkpoints_dir: Path = args.checkpoints_dir
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    token = get_token()  # Use cached token if available, otherwise None

    results: list[DownloadResult] = []
    for model_id in model_ids:
        print(f"\nDownloading: {model_id}")
        result = download_model(
            model_id=model_id,
            checkpoints_dir=checkpoints_dir,
            token=token,
        )
        results.append(result)
        if result.success:
            print(f"  OK -> {result.output_path}")
        else:
            print(f"  FAIL -> {result.error}")

    succeeded = sum(1 for result in results if result.success)
    failed = len(results) - succeeded

    print("\nSummary")
    print(f"- total: {len(results)}")
    print(f"- success: {succeeded}")
    print(f"- failed: {failed}")

    if failed:
        print("\nFailed models:")
        for result in results:
            if not result.success:
                print(f"- {result.model_id}: {result.error}")
        return 1

    return 0


if __name__ == "__main__":
    main()
