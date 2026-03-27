from __future__ import annotations

import os
from typing import Protocol

from openai import OpenAI


class LLMClient(Protocol):
    def generate(self, prompt: str) -> str: ...


class EnvLLMClient:
    def __init__(self, env_var_name: str = "GO2_AUTO_TRAINER_LLM_RESPONSE") -> None:
        self.env_var_name = env_var_name

    def generate(self, prompt: str) -> str:
        del prompt

        response = os.environ.get(self.env_var_name)
        if not response:
            raise RuntimeError(
                "No LLM backend configured. Set GO2_AUTO_TRAINER_LLM_RESPONSE to a JSON suggestion "
                "or replace EnvLLMClient with a real provider implementation."
            )
        return response


class DeepSeekLLMClient:
    def __init__(
        self,
        api_key_env: str = "DEEPSEEK_API_KEY",
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com/v1",
    ) -> None:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing {api_key_env}")

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def generate(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
        except Exception as exc:
            raise RuntimeError(f"DeepSeek request failed: {exc}") from exc

        content = response.choices[0].message.content if response.choices else None
        if not content or not content.strip():
            raise RuntimeError("DeepSeek returned an empty response")
        return content


def build_default_llm_client() -> LLMClient:
    if os.environ.get("DEEPSEEK_API_KEY"):
        return DeepSeekLLMClient()
    return EnvLLMClient()
