"""决策层的公共类型和接口。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


class DeciderError(RuntimeError):
    """决策模型不可用或返回结构不对。"""


@dataclass
class Question:
    """一个要问决策模型的问题。"""

    name: str
    type: str                     # "noul" | "choice" | "score"
    instructions: str
    criteria: dict[str, str] | list[str] | None = None


@dataclass
class Answer:
    """一个问题的回答。"""

    value: str | float
    probabilities: dict[str, float]
    confidence: float
    backend: str
    fell_back: bool = False


class Decider(Protocol):
    """决策后端统一接口。"""

    def decide(self, state: str, questions: list[Question]) -> dict[str, Answer]:
        """一次回答多个问题。"""
        ...
