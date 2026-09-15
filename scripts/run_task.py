import argparse
import itertools
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from candvg.models import build_model_backend
from candvg.models.utils import parse_bbox_2d_json
from candvg.runtime import RuntimeContext, cleanup_runtime, resolve_runtime
from candvg.tasks import build_grounding_task_dataloader, build_task
from candvg.utils import (
    build_sentence_predictions,
    normalize_pred_bbox,
    write_prediction_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument("--dataset", type=str, default="candvg")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument(
        "--runtime-mode",
        type=str,
        default="auto",
        choices=["auto", "single", "distributed"],
        help="auto: WORLD_SIZE>1 uses distributed, otherwise single.",
    )

    parser.add_argument(
        "--target-mode",
        type=str,
        default="single",
        choices=["single", "candidates"],
        help="Prediction mode for context-aware visual grounding. single: predict single bbox; candidates: predict all candidate bboxes.",
    )

    parser.add_argument("--export-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--min-new-tokens", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device-map-auto",
        action="store_true",
        help="Use Hugging Face device_map='auto' for model weight placement.",
    )
    parser.add_argument(
        "--selection-mode",
        action="store_true",
        help="Append oracle candidate bboxes from task metadata and interpret outputs as relative 0-1000.",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    """Seed common RNGs for practical inference reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(args: argparse.Namespace, runtime: RuntimeContext) -> None:
    if runtime.rank == 0:
        active_mode = "distributed" if runtime.distributed else "single"
        print(
            f"Runtime mode: {active_mode} "
            f"(world_size={runtime.world_size}, device={runtime.device})"
        )
        print(f"Seed: {args.seed}")

    model_backend = build_model_backend(
        checkpoint=args.model,
        device=runtime.device,
        device_map="auto" if args.device_map_auto else None,
        selection_mode=args.selection_mode,
    )
    print(
        f"[rank {runtime.rank}] Model backend: {model_backend.__class__.__name__}, Checkpoint: {args.model}"
    )

    task_dataset = build_task(
        dataset_name=args.dataset,
        max_samples=args.max_samples,
        target_mode=args.target_mode,
    )
    if runtime.rank == 0:
        print(f"Dataset: {args.dataset} (num_samples={len(task_dataset)})")

    dataloader = build_grounding_task_dataloader(
        dataset=task_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        distributed=runtime.distributed,
        world_size=runtime.world_size,
        rank=runtime.rank,
        pin_memory=runtime.device.type == "cuda",
    )

    outputs: list[dict[str, object]] = []
    progress = tqdm(dataloader, disable=runtime.rank != 0)
    for batch in progress:
        model_outputs = model_backend.generate_batch(
            image_paths=batch.image_paths,
            input_texts=batch.input_texts,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens,
            metadata=batch.metadata,
        )

        for (
            sample_id,
            image_path,
            sentence,
            positive_token_span,
            model_output,
        ) in zip(
            batch.sample_ids,
            batch.image_paths,
            batch.sentences,
            batch.positive_token_spans,
            model_outputs,
        ):
            answer, height, width = model_output

            pred_bboxes = normalize_pred_bbox(
                pred_bboxes=parse_bbox_2d_json(answer),
                hw=(int(height), int(width)),
                coord_mode=model_backend.coord_mode,
                coord_scale=model_backend.coord_scale,
            )
            outputs.append(
                {
                    "sample_id": sample_id,
                    "image_path": image_path,
                    "sentence": sentence,
                    "positive_token_span": positive_token_span,
                    "answer": answer,
                    "pred_bboxes": pred_bboxes,
                }
            )

    if runtime.distributed:
        import torch

        gathered_outputs: list[list[dict[str, object]] | None] = [
            None for _ in range(runtime.world_size)
        ]
        torch.distributed.all_gather_object(gathered_outputs, outputs)
        merged_outputs = list(
            itertools.chain.from_iterable(group for group in gathered_outputs if group)
        )
    else:
        merged_outputs = outputs

    if runtime.rank == 0:
        sentence_predictions = build_sentence_predictions(merged_outputs)
        exp_name = f"{args.dataset}_{args.target_mode}"
        if args.selection_mode:
            exp_name = f"{exp_name}_selection"
        run_dir = write_prediction_outputs(
            exp_name=exp_name,
            model=str(args.model),
            runtime_mode="distributed" if runtime.distributed else "single",
            world_size=runtime.world_size,
            export_dir=args.export_dir,
            predictions=sentence_predictions,
            inference_outputs=merged_outputs,
        )
        print(f"Outputs: {run_dir}")


if __name__ == "__main__":
    args = parse_args()
    runtime = resolve_runtime(args.runtime_mode)
    seed_everything(args.seed + runtime.rank)
    completed = False
    try:
        main(args, runtime)
        completed = True
    finally:
        cleanup_runtime(runtime, barrier=completed)
