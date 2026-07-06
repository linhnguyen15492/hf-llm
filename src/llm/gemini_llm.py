import os
from typing import Any, Dict, Optional
from google import genai

from llm.base_llm import BaseLLM


class GeminiLLM(BaseLLM):
    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set")
        if not model:
            raise ValueError("GEMINI_MODEL is not set")

        super().__init__(api_key=api_key)
        self.model = model
        self.client = genai.Client(api_key=self.api_key)

    # def generate_response(self, prompt: str, max_tokens: int = 150) -> str:
    #     payload: Dict[str, Any] = {
    #         "contents": [{"parts": [{"text": prompt}]}],
    #         "generationConfig": {
    #             "maxOutputTokens": max_tokens,
    #         },
    #     }
    #
    #     response = requests.post(self._build_url(), json=payload, timeout=60)
    #     response.raise_for_status()
    #     data = response.json()
    #
    #     try:
    #         return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    #     except (KeyError, IndexError, TypeError) as exc:
    #         raise ValueError(f"Unexpected Gemini response format: {data}") from exc

    def generate_response(self, instructions: str, prompt: str) -> str | None:
        interaction = self.client.interactions.create(
            model=self.model, system_instruction=instructions, input=prompt
        )

        return interaction.output_text
