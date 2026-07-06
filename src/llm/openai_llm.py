from typing import Optional
import openai
from llm.base_llm import BaseLLM


class OpenAILLM(BaseLLM):
    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set")
        if not model:
            raise ValueError("OPENAI_LLM_MODEL is not set")
        super().__init__(api_key=api_key)
        self.model = model
        self.client = openai.OpenAI(api_key=self.api_key)

    def generate_response(self, instructions: str, prompt: str) -> str:
        response = self.client.responses.create(
            model=self.model, instructions=instructions, input=prompt
        )

        return response.output_text.strip() if response.output_text else ""
