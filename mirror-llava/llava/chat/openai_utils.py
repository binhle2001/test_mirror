import os
from typing import Any, Dict, List, Optional

try:
    import openai
except Exception:
    openai = None


class OpenaiGpt:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def _should_use_gemini(self) -> bool:
        return self.model_name.lower().startswith("gemini") or bool(os.environ.get("GEMINI_API_KEY"))

    def _messages_to_text(self, messages: List[Dict[str, str]]) -> str:
        return "\n".join([f"{m.get('role','user')}: {m.get('content','')}" for m in messages]).strip()

    def _request_gemini(self, *, messages: List[Dict[str, str]], temperature: float, max_tokens: int, top_p: float, stop: Optional[List[str]]):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set")

        try:
            from google import genai
            from google.genai import types
        except Exception as e:
            raise ImportError("google-genai package is required for Gemini models") from e

        client = genai.Client(api_key=api_key)
        prompt = self._messages_to_text(messages)
        config = types.GenerateContentConfig(
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_tokens,
            stop_sequences=stop or [],
        )
        resp = client.models.generate_content(model=self.model_name, contents=prompt, config=config)
        text = getattr(resp, "text", None) or ""
        usage_meta = getattr(resp, "usage_metadata", None)

        prompt_tokens = int(getattr(usage_meta, "prompt_token_count", 0) or 0)
        completion_tokens = int(getattr(usage_meta, "candidates_token_count", 0) or 0)
        total_tokens = int(getattr(usage_meta, "total_token_count", prompt_tokens + completion_tokens) or (prompt_tokens + completion_tokens))

        return {
            "content": text,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
            "model": self.model_name,
        }

    def request(self, **kwargs: Any) -> Dict[str, Any]:
        messages = kwargs["messages"]
        temperature = kwargs["temperature"]
        max_tokens = kwargs["max_tokens"]
        top_p = kwargs["top_p"]
        stop = kwargs.get("stop")

        if self._should_use_gemini():
            return self._request_gemini(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stop=stop,
            )

        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is not set")
        if openai is None:
            raise ImportError("openai package is required for OpenAI models")

        openai.api_key = os.environ.get("OPENAI_API_KEY")
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=kwargs["frequency_penalty"],
            presence_penalty=kwargs["presence_penalty"],
            stop=stop,
        )
        return {
            "content": response["choices"][0]["message"]["content"],
            "usage": response["usage"],
            "model": response["model"],
        }
