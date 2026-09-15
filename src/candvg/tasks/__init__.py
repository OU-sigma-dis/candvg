from torch.utils.data import Dataset

from candvg.utils import GroundingSample

from .flickr30k_entities import PhraseGroundingDataset
from .utils import (
    DATASET_COLLECTIONS,
    PHRASE_GROUNDING_DATASET_NAMES,
    VISUAL_GROUNDING_DATASET_NAMES,
    VISUAL_GROUNDING_MULTI_DATASET_NAMES,
    build_grounding_task_dataloader,
)
from .refcocog import VisualGroundingDataset
from .candvg import (
    ContextVisualGroundingDataset,
)
from .grefcoco import VisualGroundingMultiDataset


def build_task(
    dataset_name: str, max_samples: int | None = None, target_mode: str = "single"
) -> Dataset[GroundingSample]:
    """Build task dataset object from dataset key."""
    if dataset_name in VISUAL_GROUNDING_MULTI_DATASET_NAMES:
        if target_mode != "candidates":
            raise ValueError(
                f"{dataset_name} is a multi-target dataset and requires target_mode='candidates'"
            )
        return VisualGroundingMultiDataset(
            dataset_name=dataset_name, max_samples=max_samples
        )
    if dataset_name.startswith("candvg"):
        return ContextVisualGroundingDataset(
            dataset_name=dataset_name, max_samples=max_samples, target_mode=target_mode
        )
    if dataset_name in VISUAL_GROUNDING_DATASET_NAMES:
        return VisualGroundingDataset(
            dataset_name=dataset_name, max_samples=max_samples
        )
    if dataset_name in PHRASE_GROUNDING_DATASET_NAMES:
        return PhraseGroundingDataset(
            dataset_name=dataset_name,
            max_samples=max_samples,
            target_mode=target_mode,
        )
    valid = ", ".join(DATASET_COLLECTIONS.keys())
    raise ValueError(f"Unknown dataset '{dataset_name}'. Available datasets: {valid}")


__all__ = [
    "VisualGroundingDataset",
    "ContextVisualGroundingDataset",
    "VisualGroundingMultiDataset",
    "PhraseGroundingDataset",
    "build_grounding_task_dataloader",
    "build_task",
]
