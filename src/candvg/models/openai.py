"""OpenAI GPT backend via Responses API."""

import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import Any

from openai import OpenAI
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from .base_vlm import BaseVLM


class QwenBBox(BaseModel):
    model_config = ConfigDict(extra="forbid")
    bbox_2d: list[float] = Field(min_length=4, max_length=4)


class QwenBBoxList(BaseModel):
    model_config = ConfigDict(extra="forbid")
    bboxes: list[QwenBBox]


class OpenAIGPTModel(BaseVLM):
    """OpenAI GPT backend using Responses API."""

    def __init__(
        self,
        checkpoint: str,
        system_prompt: str,
        device: str | None = None,
        device_map: str | None = None,
    ) -> None:
        if device_map is not None:
            raise ValueError("OpenAIGPTModel does not support device_map.")
        if os.getenv("OPENAI_API_KEY") is None:
            raise ValueError("OPENAI_API_KEY environment variable is not set.")
        super().__init__(
            checkpoint=checkpoint,
            system_prompt=system_prompt,
            device=device,
            device_map=device_map,
        )
        self.coord_mode = "pixel"
        self.coord_scale = None

    def _load(
        self,
        checkpoint: str,
        device: Any,
        device_map: str | None,
    ) -> tuple[Any, Any]:
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )
        return None, client

    @staticmethod
    def _image_to_data_url(image_path: Path) -> str:
        mime_type, _ = mimetypes.guess_type(str(image_path))
        if mime_type is None:
            mime_type = "image/jpeg"
        encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"

    @classmethod
    def _build_input_items(cls, image_path: Path, text: str) -> list[dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": text},
                    {
                        "type": "input_image",
                        "image_url": cls._image_to_data_url(image_path),
                        "detail": "original",
                    },
                ],
            },
        ]

    @staticmethod
    def _build_text_format() -> dict[str, Any]:
        return {
            "format": {
                "type": "json_schema",
                "name": "qwen_bbox_list",
                "strict": True,
                "schema": QwenBBoxList.model_json_schema(),
            }
        }

    def build_request_body(
        self,
        image_path: Path,
        input_text: str,
        max_output_tokens: int,
        reasoning_effort: str = "low",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build a Responses API request body without sending it."""
        prepared_text = self.prepare_input_texts(
            input_texts=[input_text],
            metadata=[metadata] if metadata is not None else None,
        )[0]
        return {
            "model": self.checkpoint,
            "instructions": self.system_prompt,
            "input": self._build_input_items(
                image_path=image_path,
                text=prepared_text,
            ),
            "text": self._build_text_format(),
            "max_output_tokens": max_output_tokens,
            "reasoning": {"effort": reasoning_effort},
        }

    # NOTE: wo structured output
    # @staticmethod
    # def _extract_response_text(response: Any) -> str:
    #     output_text = getattr(response, "output_text", None)
    #     if isinstance(output_text, str) and output_text.strip():
    #         return output_text

    #     parts: list[str] = []
    #     for item in getattr(response, "output", []) or []:
    #         if getattr(item, "type", None) != "message":
    #             continue
    #         for content in getattr(item, "content", []) or []:
    #             text = getattr(content, "text", None)
    #             if isinstance(text, str) and text:
    #                 parts.append(text)
    #     return "\n".join(parts).strip()

    @staticmethod
    def parse_output_text(output_text: str | None) -> str:
        """Convert one structured Responses API text output to bbox JSON."""
        if not isinstance(output_text, str) or not output_text.strip():
            return "[]"

        try:
            parsed = QwenBBoxList.model_validate_json(output_text)
        except Exception:
            return "[]"
        return json.dumps(
            [bbox.model_dump() for bbox in parsed.bboxes], ensure_ascii=False
        )

    @classmethod
    def parse_response_body(cls, response_body: dict[str, Any]) -> str:
        """Extract bbox JSON from a serialized Responses API response body."""
        parts: list[str] = []
        for output_item in response_body.get("output", []):
            if output_item.get("type") != "message":
                continue
            for content_item in output_item.get("content", []):
                if content_item.get("type") != "output_text":
                    continue
                text = content_item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return cls.parse_output_text("\n".join(parts))

    @classmethod
    def _extract_response_text(cls, response: Any) -> str:
        output_text = getattr(response, "output_text", None)
        return cls.parse_output_text(output_text)

    def generate_batch(
        self,
        image_paths: list[Path],
        input_texts: list[str],
        max_new_tokens: int,
        min_new_tokens: int,
        metadata: list[dict[str, Any]] | None = None,
    ) -> list[tuple[str, int, int]]:
        """Generate outputs and return (answer, height, width)."""
        outputs: list[tuple[str, int, int]] = []

        sample_metadata: list[dict[str, Any] | None]
        if metadata is None:
            sample_metadata = [None] * len(input_texts)
        else:
            sample_metadata = metadata

        for image_path, input_text, item_metadata in zip(
            image_paths, input_texts, sample_metadata
        ):
            request_body = self.build_request_body(
                image_path=image_path,
                input_text=input_text,
                max_output_tokens=max_new_tokens,
                reasoning_effort="low",
                metadata=item_metadata,
            )
            response = self.processor.responses.create(**request_body)

            # usage = response.usage
            # print(f"Input tokens: {usage.input_tokens}")
            # print(f"Output tokens: {usage.output_tokens}")
            # print(f"Total tokens: {usage.total_tokens}")
            # print(
            #     f"Reasoning tokens: "
            #     f"{usage.output_tokens_details.reasoning_tokens}"
            # )

            answer = self._extract_response_text(response)

            with Image.open(image_path) as image:
                width, height = image.size
            outputs.append((answer, int(height), int(width)))
        return outputs
