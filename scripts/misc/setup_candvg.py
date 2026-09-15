"""Reconstruct complete CandVG annotations with original InViG dialogue text."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Restore InViG dialogue text in distributable CandVG annotations."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("data/candvg/candvg_original.jsonl"),
        help="Distributable CandVG JSONL path.",
    )
    parser.add_argument(
        "--valid-path",
        type=Path,
        default=Path("path/to/invig21k_valid_anns.jsonl"),
        help="Original InViG validation annotation path.",
    )
    parser.add_argument(
        "--test-path",
        type=Path,
        default=Path("path/to/invig21k_test_anns.jsonl"),
        help="Original InViG test annotation path.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/candvg/candvg.jsonl"),
        help="Reconstructed complete CandVG JSONL path.",
    )
    return parser.parse_args()


def load_invig_annotations(
    split_paths: dict[str, Path],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Load original InViG annotations keyed by split and global ID."""
    annotations: dict[tuple[str, str], dict[str, Any]] = {}
    for split, path in split_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"InViG annotation not found: {path}")
        with path.open("r", encoding="utf-8") as src:
            for line_no, line in enumerate(src, 1):
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                global_id = sample.get("invig_label_id", sample.get("id"))
                key = (split, str(global_id))
                if key in annotations:
                    raise ValueError(f"Duplicate InViG ID {key} at {path}:{line_no}")
                annotations[key] = sample["ann"]
    return annotations


def build_dialogue(annotation: dict[str, Any], turn_index: int) -> str:
    """Build an InViG dialogue through the requested turn index."""
    questions = annotation["questions"]
    answers = annotation["answers"]
    if turn_index < 0 or turn_index > len(questions):
        raise ValueError(
            f"Invalid turn_index={turn_index} for {len(questions)} questions"
        )

    parts: list[str] = [str(annotation["ref_exp"])]
    for index in range(turn_index):
        if index < len(questions):
            parts.append(str(questions[index]))
        if index < len(answers):
            parts.append(str(answers[index]))
    return " ".join(part.strip() for part in parts if part and part.strip())


def main() -> None:
    """Restore dialogue fields and write the complete CandVG JSONL."""
    args = parse_args()
    if not args.input_path.is_file():
        raise FileNotFoundError(f"CandVG annotation not found: {args.input_path}")

    invig_annotations = load_invig_annotations(
        {"valid": args.valid_path, "test": args.test_path}
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    sample_count = 0
    with (
        args.input_path.open("r", encoding="utf-8") as src,
        args.output_path.open("w", encoding="utf-8") as dst,
    ):
        for line_no, line in enumerate(src, 1):
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            metadata = sample["metadata"]
            key = (str(metadata["split"]), str(metadata["global_id"]))
            if key not in invig_annotations:
                raise KeyError(f"Missing InViG annotation for {key} at line {line_no}")

            invig_annotation = invig_annotations[key]
            target_utterance = str(invig_annotation["ref_exp"])
            dialogue = build_dialogue(
                annotation=invig_annotation,
                turn_index=int(metadata["turn_index"]),
            )
            sample["annotation"] = {
                "dialogue": dialogue,
                "target_utterance": target_utterance,
                **sample["annotation"],
            }
            if sample_count > 0:
                dst.write("\n")
            dst.write(json.dumps(sample, ensure_ascii=False))
            sample_count += 1

    print(f"input_path: {args.input_path}")
    print(f"valid_path: {args.valid_path}")
    print(f"test_path: {args.test_path}")
    print(f"output_path: {args.output_path}")
    print(f"samples: {sample_count}")


if __name__ == "__main__":
    main()
