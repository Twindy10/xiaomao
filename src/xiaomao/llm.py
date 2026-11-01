"""调用生成模型的公共封装：JSON 输出、重试、并发、用量统计。

任务书第 14 条要求：openai SDK、配置来自 .env、关思考模式、JSON 输出、
并发不超 4、失败重试 3 次指数退避、结束时打印 token 合计。
"""

from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]  # src/xiaomao/llm.py -> 项目根

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class Usage:
    """token 用量合计。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    calls: int = 0

    def add(self, prompt: int, completion: int, total: int) -> None:
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += total
        self.calls += 1

    def summary(self) -> str:
        """一行汇总，脚本结束时打印。"""
        return (
            f"调用 {self.calls} 次，prompt {self.prompt_tokens} token，"
            f"completion {self.completion_tokens} token，合计 {self.total_tokens} token"
        )


class LLMClient:
    """生成模型客户端；线程安全，可在并发任务里共用。"""

    def __init__(self, max_workers: int = 4, retries: int = 3, timeout: float = 120.0) -> None:
        load_dotenv(ROOT_DIR / ".env", override=True)
        from openai import OpenAI

        base_url = os.getenv("LLM_BASE_URL", "")
        api_key = os.getenv("LLM_API_KEY", "")
        if not base_url or not api_key:
            raise RuntimeError("LLM_BASE_URL / LLM_API_KEY 未配置，先填 .env")
        self.model = os.getenv("LLM_MODEL", "qwen-plus")
        self.enable_thinking = os.getenv("LLM_ENABLE_THINKING", "false").lower() == "true"
        self.max_workers = max_workers
        self.retries = retries
        self.usage = Usage()
        self._lock = threading.Lock()
        # 本地网关与云端 API 都不走系统代理（Clash 会把 127.0.0.1 的请求也劫持）
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=0)

    def chat_json(
        self, prompt: str, system: str, max_tokens: int = 3000, temperature: float = 0.7
    ) -> dict[str, Any]:
        """发一条请求，要求模型返回 JSON 对象；失败按指数退避重试。审核类任务传 temperature=0。"""
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    response_format={"type": "json_object"},
                    extra_body={"enable_thinking": self.enable_thinking},
                )
                usage = response.usage
                if usage is not None:
                    with self._lock:
                        self.usage.add(
                            usage.prompt_tokens or 0,
                            usage.completion_tokens or 0,
                            usage.total_tokens or 0,
                        )
                content = response.choices[0].message.content or "{}"
                return json.loads(content)
            except Exception as exc:  # 网络、限流、JSON 解析失败都在这里重试
                last_error = exc
                if attempt < self.retries - 1:
                    time.sleep(2**attempt)
        raise RuntimeError(f"模型调用失败：{type(last_error).__name__}: {last_error}")

    def map(self, func: Callable[[T], R], items: list[T]) -> list[R]:
        """并发执行，结果顺序与输入一致。"""
        if not items:
            return []
        results: list[R | None] = [None] * len(items)
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(func, item): index for index, item in enumerate(items)}
            for future in as_completed(futures):
                results[futures[future]] = future.result()
        return [item for item in results if item is not None]
