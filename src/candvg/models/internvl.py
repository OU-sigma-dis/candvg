"""Shared backends for InternVL Hugging Face-format."""

from pathlib import Path
from typing import Any

from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from .base_vlm import BaseVLM


class InternVLHFModel(BaseVLM):
    """InternVL backend for Hugging Face-format checkpoints."""

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
        self.coord_mode = "relative_1000"
        self.coord_scale = 1000

    def _load(
        self,
        checkpoint: str,
        device: Any,
        device_map: str | None,
    ) -> tuple[Any, Any]:
        model_kwargs: dict[str, Any] = {
            "dtype": "auto",
            "trust_remote_code": True,
        }
        if device_map is not None:
            model_kwargs["device_map"] = device_map

        model = AutoModelForImageTextToText.from_pretrained(
            checkpoint,
            **model_kwargs,
        ).eval()
        if device_map is None:
            model.to(device)
        if model.generation_config.pad_token_id is None:
            model.generation_config.pad_token_id = model.generation_config.eos_token_id

        processor = AutoProcessor.from_pretrained(
            checkpoint,
            trust_remote_code=True,
        )
        return model, processor

    def _build_message(self, image_path: Path, text: str) -> list[dict[str, Any]]:
        """Build one HF image-text-to-text chat message."""
        return [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": self.system_prompt},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": str(image_path.expanduser().resolve())},
                    {"type": "text", "text": text},
                ],
            },
        ]

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
        """Generate outputs and return (answer, height, width)."""
        input_texts = self.prepare_input_texts(
            input_texts=input_texts, metadata=metadata
        )
        outputs: list[tuple[str, int, int]] = []
        for image_path, input_text in zip(image_paths, input_texts):
            message = self._build_message(
                image_path=image_path,
                text=input_text,
            )
            model_inputs = self.processor.apply_chat_template(
                message,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self._input_device())

            generated_ids = self.model.generate(
                **model_inputs,
                **self._generation_kwargs(
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=min_new_tokens,
                ),
            )
            answer = self.processor.decode(
                generated_ids[0][model_inputs["input_ids"].shape[-1] :],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )

            with Image.open(image_path) as image:
                width, height = image.size
            outputs.append((answer, int(height), int(width)))
        return outputs
