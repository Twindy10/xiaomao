"""决策层对外入口。"""

from __future__ import annotations

import os

from xiaomao.decision.base import Answer, Decider, DeciderError, Question
from xiaomao.decision.fallback import FallbackDecider
from xiaomao.decision.jev_style import JevStyleDecider
from xiaomao.decision.llm_json import LLMJsonDecider
from xiaomao.decision.system_one import SystemOneDecider
from xiaomao.decision.questions import (
    EVIDENCE_QUESTION,
    INTENT_QUESTION,
    PLATFORM_QUESTION,
    ROUTING_QUESTIONS,
    SITE_QUESTION,
    STRATEGY_QUESTION,
)


def get_decider() -> Decider:
    """按 DECIDER_BACKEND 返回决策后端。"""
    backend = os.environ.get("DECIDER_BACKEND", "jev_style").lower()
    if backend == "llm_json":
        return LLMJsonDecider()
    if backend == "jev_style":
        return FallbackDecider(JevStyleDecider(), LLMJsonDecider())
    if backend in ("kev", "laya"):
        return FallbackDecider(
            SystemOneDecider(
                os.environ["DECIDER_BASE_URL"],
                backend=backend,
                model=os.environ.get("DECIDER_MODEL") or None,
                api_key=os.environ.get("DECIDER_API_KEY") or None,
            ),
            LLMJsonDecider(),
        )
    raise ValueError(f"未知的 DECIDER_BACKEND：{backend}")


__all__ = [
    "Answer", "Decider", "DeciderError", "Question",
    "FallbackDecider", "JevStyleDecider", "LLMJsonDecider", "SystemOneDecider", "get_decider",
    "EVIDENCE_QUESTION", "INTENT_QUESTION", "PLATFORM_QUESTION", "SITE_QUESTION",
    "STRATEGY_QUESTION", "ROUTING_QUESTIONS",
]
