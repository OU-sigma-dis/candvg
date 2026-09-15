"""Shared constants and util fuctions for tasks."""

from collections.abc import Iterator
from pathlib import Path

from torch.utils.data import DataLoader, Dataset, Sampler

from candvg.utils import GroundingBatch, GroundingSample

DATASET_COLLECTIONS: dict[str, str] = {
    "refcoco_val": "data/coco/refcoco/refcoco_val.jsonl",
    "refcoco_testA": "data/coco/refcoco/refcoco_testA.jsonl",
    "refcoco_testB": "data/coco/refcoco/refcoco_testB.jsonl",
    "refcoco+_val": "data/coco/refcoco+/refcoco+_val.jsonl",
    "refcoco+_testA": "data/coco/refcoco+/refcoco+_testA.jsonl",
    "refcoco+_testB": "data/coco/refcoco+/refcoco+_testB.jsonl",
    "refcocog_val": "data/coco/refcocog/refcocog_val.jsonl",
    "refcocog_test": "data/coco/refcocog/refcocog_test.jsonl",
    "flickr30k_entities_val": "data/flickr30k_entities/final_flickr_separateGT_val.json",
    "flickr30k_entities_test": "data/flickr30k_entities/final_flickr_separateGT_test.json",
    "flickr30k_entities_ja_val": "data/flickr30k_entities_ja/final_flickr_separateGT_val.json",
    "flickr30k_entities_ja_test": "data/flickr30k_entities_ja/final_flickr_separateGT_test.json",
    "candvg": "data/candvg/candvg.jsonl",
    "grefcoco": "data/gRefCOCO/grefs(unc).json",
}

VISUAL_GROUNDING_DATASET_NAMES: tuple[str, ...] = (
    "refcoco_val",
    "refcoco_testA",
    "refcoco_testB",
    "refcoco+_val",
    "refcoco+_testA",
    "refcoco+_testB",
    "refcocog_val",
    "refcocog_test",
)
VISUAL_GROUNDING_MULTI_DATASET_NAMES: dict[str, tuple[str, ...]] = {
    "grefcoco": ("val", "testA", "testB"),
}

PHRASE_GROUNDING_DATASET_NAMES: tuple[str, ...] = (
    "flickr30k_entities_val",
    "flickr30k_entities_test",
    "flickr30k_entities_ja_val",
    "flickr30k_entities_ja_test",
)

REF_START = "<ref>"
REF_END = "</ref>"


def build_grounding_text(sentence: str, positive_token_span: tuple[int, int]) -> str:
    """Build one grounding input text from context and one positive token span."""
    start, end = positive_token_span
    if start < 0 or end <= start or end > len(sentence):
        raise ValueError(
            f"Invalid positive_token_span={positive_token_span} for sentence length={len(sentence)}"
        )
    target = sentence[start:end]
    marked_sentence = f"{sentence[:start]}{REF_START}{target}{REF_END}{sentence[end:]}"
    return marked_sentence


def resolve_dataset_path(dataset_name: str) -> Path:
    """Resolve dataset key into the configured local data path."""
    if dataset_name not in DATASET_COLLECTIONS:
        valid = ", ".join(DATASET_COLLECTIONS.keys())
        raise KeyError(f"Unknown dataset '{dataset_name}'. Available datasets: {valid}")
    return Path(DATASET_COLLECTIONS[dataset_name])


def _collate_grounding_batch(samples: list[GroundingSample]) -> GroundingBatch:
    """Batch grounding samples and build model input texts."""
    return GroundingBatch(
        sample_ids=[item.sample_id for item in samples],
        image_paths=[item.image_path for item in samples],
        sentences=[item.sentence for item in samples],
        positive_token_spans=[item.positive_token_span for item in samples],
        input_texts=[
            build_grounding_text(item.sentence, item.positive_token_span)
            for item in samples
        ],
        gt_bboxes=[item.gt_bboxes for item in samples],
        metadata=[item.metadata for item in samples],
    )


class DistributedEvalSampler(Sampler[int]):
    """Shard eval indices across ranks without padding or dropping samples."""

    def __init__(
        self, dataset: Dataset[GroundingSample], world_size: int, rank: int
    ) -> None:
        self.dataset = dataset
        self.world_size = world_size
        self.rank = rank

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self) -> int:
        if self.rank >= len(self.dataset):
            return 0
        return ((len(self.dataset) - 1 - self.rank) // self.world_size) + 1


def build_grounding_task_dataloader(
    dataset: Dataset[GroundingSample],
    batch_size: int,
    num_workers: int,
    distributed: bool,
    world_size: int,
    rank: int,
    pin_memory: bool,
) -> DataLoader[GroundingBatch]:
    """Build grounding task DataLoader with optional distributed sampler."""
    sampler = None
    if distributed:
        sampler = DistributedEvalSampler(
            dataset=dataset, world_size=world_size, rank=rank
        )

    return DataLoader(
        dataset=dataset,
        sampler=sampler,
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=_collate_grounding_batch,
    )
