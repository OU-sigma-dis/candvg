"""Model registry and factory for VLM backends."""

from pathlib import Path
from typing import Any

from candvg.tasks.utils import REF_END, REF_START

from .base_vlm import BaseVLM
from .internvl import InternVLHFModel
from .locate_anything import LocateAnythingModel
from .openai import OpenAIGPTModel
from .qwen_vl import Qwen3VLModel, Qwen35VLModel
from .utils import SelectionCandidateFormat

# multiple boxes version
BASE_PROMPT = (
    f"Provide the bounding box coordinates of the text described between {REF_START} and {REF_END}.\n"
    "Return as box(es) as necessary, but include all plausible referents, up to a maximum of 15.\n"
)

BASE_PROMPT_WITH_TASK_AWARE_1 = (
    f"Provide the bounding box coordinates of the text described between {REF_START} and {REF_END}.\n"
    "Before answering, follow this plan:"
    "1. Identify what kind of object is described by the text."
    "2. Enumerate all plausible objects in the image."
    "3. If the dialogue context following the text is provided, remove objects that are contradicted by the context.\n"
    "Return as box(es) as necessary, but include all plausible referents, up to a maximum of 15.\n"
)

BASE_PROMPT_WITH_TASK_AWARE_2 = (
    f"Provide the bounding box coordinates of the text described between {REF_START} and {REF_END}.\n"
    "When the reference is ambiguous, retain every visually plausible referent unless it is clearly ruled out by the marked text or the dialogue context (if provided).\n"
    "Return as box(es) as necessary, but include all plausible referents, up to a maximum of 15.\n"
)

QWEN_VL_PROMPT = BASE_PROMPT + (
    "Return bbox coordinates in the following format:\n"
    "["
    '{"bbox_2d": [x1, y1, x2, y2]}, '
    " ... "
    "]\n"
)  # same as cosmos reason

RYNN_BRAIN_PROMPT = BASE_PROMPT + (
    "Return bbox coordinates in the following format:\n"
    "<object> (x1, y1), (x2, y2) </object>\n"
)

LOCATE_ANYTHING_PROMPT = BASE_PROMPT + (
    "Return bbox coordinates in the following format:\n<box><x1><y1><x2><y2></box>"
)

INTERN_VL_PROMPT = BASE_PROMPT + (
    "Return bbox coordinates in the following format:\n"
    "<box>[[x1, y1, x2, y2] ...]</box>"
)

GPT_5_PROMPT = BASE_PROMPT + (
    "Return bbox coordinates in the following format:\n"
    "["
    '{"bbox_2d": [x1, y1, x2, y2]}, '
    " ... "
    "]\n"
)

MODEL_BACKENDS: dict[str, type[BaseVLM]] = {
    "qwen3_vl": Qwen3VLModel,
    "qwen3_5": Qwen35VLModel,
    "internvl3_5": InternVLHFModel,
    "locate_anything": LocateAnythingModel,
    "cosmos_reason_2": Qwen3VLModel,
    "ryanbrain": Qwen3VLModel,
    "openai_api": OpenAIGPTModel,
}


def list_model_backends() -> list[str]:
    """List available backend names."""
    return sorted(MODEL_BACKENDS.keys())


def build_model_backend(
    checkpoint: str | Path,
    device: Any,
    device_map: str | None = None,
    selection_mode: bool = False,
) -> BaseVLM:
    """Build backend instance from registry."""
    backend_name: str
    prompt: str = BASE_PROMPT
    selection_candidate_format: SelectionCandidateFormat = "qwen"
    if "Qwen3-VL" in checkpoint:
        backend_name = "qwen3_vl"
        prompt = QWEN_VL_PROMPT
        selection_candidate_format = "qwen"
    elif "Qwen3.5" in checkpoint:
        backend_name = "qwen3_5"
        prompt = QWEN_VL_PROMPT
        selection_candidate_format = "qwen"
    elif "InternVL3_5" in checkpoint and "HF" in checkpoint:
        backend_name = "internvl3_5"
        prompt = INTERN_VL_PROMPT
        selection_candidate_format = "internvl"
    elif "Cosmos-Reason2" in checkpoint:
        backend_name = "cosmos_reason_2"
        prompt = QWEN_VL_PROMPT
        selection_candidate_format = "qwen"
    elif "RynnBrain" in checkpoint:
        backend_name = "ryanbrain"
        prompt = RYNN_BRAIN_PROMPT
        selection_candidate_format = "rynnbrain"
    elif "LocateAnything" in checkpoint:
        backend_name = "locate_anything"
        prompt = LOCATE_ANYTHING_PROMPT
        selection_candidate_format = "locate_anything"
    elif "gpt-5" in checkpoint:
        backend_name = "openai_api"
        prompt = GPT_5_PROMPT
        selection_candidate_format = "qwen"
    else:
        raise KeyError(f"Unknown model backend: {checkpoint}")

    backend_cls = MODEL_BACKENDS[backend_name]
    backend = backend_cls(
        checkpoint=checkpoint,
        device=device,
        device_map=device_map,
        system_prompt=prompt,
    )
    if selection_mode:
        backend.selection_mode = True
        backend.selection_candidate_format = selection_candidate_format
        backend.coord_mode = "relative_1000"
        backend.coord_scale = 1000
    return backend


__all__ = [
    "build_model_backend",
    "list_model_backends",
]
