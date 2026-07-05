import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm.gemini_llm import GeminiLLM


class GeminiLLMTest(unittest.TestCase):
    def test_generate_response_uses_gemini_api(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "dummy-key"}, clear=False):
            with patch("llm.gemini.requests.post") as mock_post:
                mock_post.return_value.ok = True
                mock_post.return_value.json.return_value = {
                    "candidates": [
                        {
                            "content": {
                                "parts": [{"text": "Xin chào từ Gemini"}]
                            }
                        }
                    ]
                }

                llm = GeminiLLM(model_name="gemini-2.0-flash")
                result = llm.generate_response("Chào bạn")

                self.assertEqual(result, "Xin chào từ Gemini")
                mock_post.assert_called_once()


if __name__ == "__main__":
    unittest.main()
