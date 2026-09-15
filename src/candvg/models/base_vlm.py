from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from .utils import SelectionCandidateFormat, format_selection_candidate_section


class BaseVLM(ABC):
    """Abstract base class for grounding-capable VLM backends."""

    def __init__(
        self,
        checkpoint: str,
        system_prompt: str,
        device: str | None = None,
        device_map: str | None = None,
    ) -> None:
        self.checkpoint = checkpoint
        self.system_prompt = system_prompt
        self.device = device
        self.device_map = device_map
        self.model, self.processor = self._load(
            checkpoint=checkpoint,
            device=device,
            device_map=device_map,
        )
        self.coord_mode = "pixel"
        self.coord_scale = None
        self.selection_mode = False
        self.selection_candidate_format: SelectionCandidateFormat = "qwen"

    @abstractmethod
    def _load(
        self,
        checkpoint: str,
        device: Any,
        device_map: str | None,
    ) -> tuple[Any, Any]:
        """Load model assets and return (model, processor/tokenizer)."""
        raise NotImplementedError

    def _generation_kwargs(
        self,
        max_new_tokens: int,
        min_new_tokens: int,
    ) -> dict[str, Any]:
        """Build shared deterministic generation kwargs."""
        return {
            "do_sample": False,
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": min_new_tokens,
            "use_cache": True,
        }

    def prepare_input_texts(
        self,
        input_texts: list[str],
        metadata: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        """Prepare backend-specific user input texts before generation."""
        if not self.selection_mode:
            return input_texts
        if metadata is None:
            raise ValueError("selection_mode requires batch metadata")

        prepared_texts: list[str] = []
        for input_text, sample_metadata in zip(input_texts, metadata):
            candidate_bboxes = sample_metadata.get("candidate_bboxes")
            if not isinstance(candidate_bboxes, list):
                raise ValueError("selection_mode requires metadata['candidate_bboxes']")
            candidate_section = format_selection_candidate_section(
                candidate_bboxes=candidate_bboxes,
                candidate_format=self.selection_candidate_format,
            )
            prepared_texts.append(f"{input_text}\n\n{candidate_section}")
        return prepared_texts

    @abstractmethod
    def generate_batch(
        self,
        image_paths: list[Path],
        input_texts: list[str],
        max_new_tokens: int,
        min_new_tokens: int,
        metadata: list[dict[str, Any]] | None = None,
    ) -> list[tuple[str, int, int]]:
        """Generate answers and original image sizes for a batch."""
        raise NotImplementedError
