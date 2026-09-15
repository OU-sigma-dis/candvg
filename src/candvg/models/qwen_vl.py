"""Shared backends for Qwen 2-VL / 2.5-VL / 3-VL / 3.5 / and FT VLMs from Qwen VL3  inference."""

from pathlib import Path
from typing import Any

from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor

from .base_vlm import BaseVLM


class Qwen3VLModel(BaseVLM):
    """Qwen VL family backend."""

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
        model_kwargs: dict[str, Any] = {"torch_dtype": "auto"}
        if device_map is not None:
            model_kwargs["device_map"] = device_map

        model = AutoModelForImageTextToText.from_pretrained(
            checkpoint,
            **model_kwargs,
        ).eval()
        if device_map is None:
            model.to(device)
        processor = AutoProcessor.from_pretrained(checkpoint)
        return model, processor

    def _apply_chat_template(self, message: list[dict[str, Any]]) -> str:
        """Apply the backend chat template to one message."""
        return self.processor.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _build_message(self, image_path: str, text: str) -> list[dict[str, Any]]:
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
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": text},
                ],
            },
        ]

    def generate_batch(
        self,
        image_paths: list[Path],
        input_texts: list[str],
        max_new_tokens: int,
        min_new_tokens: int,
        metadata: list[dict[str, Any]] | None = None,
    ) -> list[tuple[str, int, int]]:
        """Run Qwen VL generation and return (answer, height, width)."""
        input_texts = self.prepare_input_texts(
            input_texts=input_texts, metadata=metadata
        )
        messages = [
            self._build_message(str(image_path), input_text)
            for image_path, input_text in zip(image_paths, input_texts)
        ]

        chat_texts = [self._apply_chat_template(message) for message in messages]
        image_inputs, video_inputs = process_vision_info(messages)

        model_inputs = self.processor(
            text=chat_texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        generated_ids = self.model.generate(
            **model_inputs,
            **self._generation_kwargs(
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
            ),
        )

        trimmed_ids = [
            output_ids[input_ids.shape[0] :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        answers = self.processor.batch_decode(
            trimmed_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        outputs: list[tuple[str, int, int]] = []
        for answer, image_path in zip(answers, image_paths):
            with Image.open(image_path) as image:
                width, height = image.size
            outputs.append((answer, int(height), int(width)))
        return outputs


class Qwen35VLModel(Qwen3VLModel):
    """Qwen3.5 VL backend with thinking mode disabled."""

    def _load(
        self,
        checkpoint: str,
        device: Any,
        device_map: str | None,
    ) -> tuple[Any, Any]:
        model, processor = super()._load(
            checkpoint=checkpoint,
            device=device,
            device_map=device_map,
        )
        if model.generation_config.pad_token_id is None:
            model.generation_config.pad_token_id = model.generation_config.eos_token_id
        return model, processor

    def _apply_chat_template(self, message: list[dict[str, Any]]) -> str:
        """Apply the Qwen3.5 chat template with thinking disabled."""
        return self.processor.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,  # HACK: disable thinking mode for Qwen3.5 to avoid generating <|THINKING|> in the output
        )
