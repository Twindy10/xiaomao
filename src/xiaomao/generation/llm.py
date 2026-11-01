"""生成模型客户端：一次性回答和流式输出。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[3]  # src/xiaomao/generation/llm.py -> 项目根

Message = dict[str, str]


@dataclass
class GenerationResult:
    """一次生成的结果。"""

    text: str
    prompt_tokens: int
    completion_tokens: int


class Generator:
    """OpenAI 兼容的生成模型封装。"""

    def __init__(self, model: str | None = None, timeout: float = 120.0, client=None) -> None:
        load_dotenv(ROOT_DIR / ".env", override=True)
        if client is None:
            from openai import OpenAI

            client = OpenAI(
                base_url=os.environ["LLM_BASE_URL"],
                api_key=os.environ["LLM_API_KEY"],
                timeout=timeout,
                max_retries=1,
            )
        self.client = client
        self.model = model or os.environ.get("LLM_MODEL", "qwen-plus")
        self.enable_thinking = os.environ.get("LLM_ENABLE_THINKING", "false").lower() == "true"
        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0}

    def complete(
        self,
        messages: list[Message],
        max_tokens: int = 1200,
        temperature: float = 0.3,
    ) -> GenerationResult:
        """一次性拿到完整回答。"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            extra_body={"enable_thinking": self.enable_thinking},
        )
        usage = response.usage
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        self.last_usage = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}
        return GenerationResult(
            text=(response.choices[0].message.content or "").strip(),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def stream(
        self,
        messages: list[Message],
        max_tokens: int = 1200,
        temperature: float = 0.3,
    ) -> Iterator[str]:
        """逐段产出文本；结束时把 tokens 写进 last_usage（M4b 用）。"""
        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0}
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            stream_options={"include_usage": True},
            extra_body={"enable_thinking": self.enable_thinking},
        )
        for chunk in stream:
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                self.last_usage = {
                    "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                    "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                }
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            content = getattr(delta, "content", None) if delta is not None else None
            if content:
                yield content
