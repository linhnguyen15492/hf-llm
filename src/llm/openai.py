import os


class OpenAILLM:
    def __init__(self, model_name=None, api_key=None):
        self.model_name = model_name or os.getenv("OPENAI_LLM_MODEL")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")

    def generate_response(self, prompt, max_tokens=150):
        import openai

        openai.api_key = self.api_key
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        return response.choices[0].message["content"].strip()
