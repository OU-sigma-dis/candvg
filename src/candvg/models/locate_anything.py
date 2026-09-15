"""LocateAnything backend."""

import re
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer

from candvg.tasks.utils import REF_END, REF_START

from .base_vlm import BaseVLM

REF_SPAN_PATTERN = re.compile(
    re.escape(REF_START) + r"(.*?)" + re.escape(REF_END),
    re.DOTALL,
)


def _extract_target_text(input_text: str) -> str:
    """Extract the marked target phrase from a project grounding prompt."""
    match = REF_SPAN_PATTERN.search(input_text)
    if match is None:
        return input_text.strip()
    return match.group(1).strip()


class LocateAnythingModel(BaseVLM):
    """LocateAnything backend for normalized 0-1000 bbox grounding."""

    def __init__(
        self,
        checkpoint: str,
        system_prompt: str,
        device: Any,
        device_map: str | None = None,
    ) -> None:
        super().__init__(
            checkpoint=checkpoint,
            system_prompt=system_prompt,
            device=device,
            device_map=device_map,
        )
        self.dtype = torch.bfloat16
        self.coord_mode = "relative_1000"
        self.coord_scale = 1000

    def _load(
        self,
        checkpoint: str,
        device: Any,
        device_map: str | None,
    ) -> tuple[Any, Any]:
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
        }
        if device_map is not None:
            model_kwargs["device_map"] = device_map

        model = AutoModel.from_pretrained(
            checkpoint,
            **model_kwargs,
        ).eval()
        if device_map is None:
            model.to(device)

        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint,
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(
            checkpoint,
            trust_remote_code=True,
        )
        return model, (tokenizer, processor)

    @property
    def image_processor(self) -> Any:
        """Return the LocateAnything processor."""
        return self.processor[1]

    @property
    def tokenizer(self) -> Any:
        """Return the LocateAnything tokenizer."""
        return self.processor[0]

    @staticmethod
    def _build_prompt(input_text: str) -> str:
        """Build the LocateAnything prompt from the shared project input text."""
        target_text = _extract_target_text(input_text)
        return f"Context: {input_text}\nLocate all boxes that match the following text using the context: {REF_START}{target_text}{REF_END}"

    @staticmethod
    def _open_image(image_path: Path) -> Image.Image:
        """Open one image as RGB."""
        with Image.open(image_path) as image:
            return image.convert("RGB")

    def _input_device(self) -> Any:
        """Return the device where processor outputs should be placed."""
        return getattr(self.model, "device", self.device)

    def generate_batch(
        self,
        image_paths: list[Path],
        input_texts: list[str],
        max_new_tokens: int,
        min_new_tokens: int,
        metadata: list[dict[str, Any]] | None = None,
    ) -> list[tuple[str, int, int]]:
        """Run LocateAnything generation and return (answer, height, width)."""
        input_texts = self.prepare_input_texts(
            input_texts=input_texts, metadata=metadata
        )
        outputs: list[tuple[str, int, int]] = []
        for image_path, input_text in zip(image_paths, input_texts):
            image = self._open_image(image_path)
            width, height = image.size
            prompt = self._build_prompt(input_text)
            messages = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": self.system_prompt},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                },
            ]
            # breakpoint()

            text = self.image_processor.py_apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            image_inputs, video_inputs = self.image_processor.process_vision_info(
                messages
            )
            model_inputs = self.image_processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self._input_device())

            response = self.model.generate(
                pixel_values=model_inputs["pixel_values"].to(self.dtype),
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                image_grid_hws=model_inputs.get("image_grid_hws", None),
                tokenizer=self.tokenizer,
                generation_mode="hybrid",
                verbose=False,
                **self._generation_kwargs(
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=min_new_tokens,
                ),
            )
            answer = response[0] if isinstance(response, tuple) else response
            outputs.append((answer, int(height), int(width)))
        return outputs
