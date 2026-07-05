import os
import openai

from llm.base_llm import BaseLLM


class OpenAILLM(BaseLLM):
    def __init__(
            self,
            api_key: str,
            base_url: str,
            model_name: str,
    ):
        super().__init__(api_key=api_key)
        self.model_name = model_name or os.getenv("OPENAI_LLM_MODEL") or "gpt-4o-mini"
        self.base_url = base_url or os.environ.get("OPENAI_API_BASE_URL")
        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)

    def generate_response(self, prompt, max_tokens=150):
        response = self.client.responses.create(
            input=prompt,
            model=self.model_name,
        )

        return response.output_text
