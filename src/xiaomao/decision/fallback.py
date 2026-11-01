"""置信度兜底：低置信度的问题合并成一次请求交给兜底后端。"""

from __future__ import annotations

from xiaomao.decision.base import Answer, Decider, DeciderError, Question


class FallbackDecider:
    """先问主后端，主后端不可用或置信度低时问兜底后端。"""

    def __init__(
        self,
        primary: Decider,
        fallback: Decider,
        threshold: float = 0.5,
        thresholds: dict[str, float] | None = None,
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.threshold = threshold
        self.thresholds = thresholds or {"strategy": 0.0}
        self.last_tokens = 0

    def decide(self, state: str, questions: list[Question]) -> dict[str, Answer]:
        """返回每个问题的答案，标出哪些是兜底来的。"""
        self.last_tokens = 0
        try:
            answers = self.primary.decide(state, questions)
        except DeciderError:
            return self._fallback_all(state, questions)

        low = [
            question for question in questions
            if answers[question.name].confidence < self.thresholds.get(question.name, self.threshold)
        ]
        if not low:
            return answers

        try:
            replacement = self.fallback.decide(state, low)
        except Exception as exc:
            raise DeciderError(f"兜底也失败：{type(exc).__name__}: {exc}") from exc
        self.last_tokens = getattr(self.fallback, "last_tokens", 0)
        for question in low:
            answer = replacement[question.name]
            answers[question.name] = Answer(
                value=answer.value,
                probabilities=answer.probabilities,
                confidence=answer.confidence,
                backend=answer.backend,
                fell_back=True,
            )
        return answers

    def _fallback_all(self, state: str, questions: list[Question]) -> dict[str, Answer]:
        """主后端整体失败时全部交给兜底。"""
        try:
            replacement = self.fallback.decide(state, questions)
        except Exception as exc:
            raise DeciderError(f"主后端和兜底都失败：{type(exc).__name__}: {exc}") from exc
        self.last_tokens = getattr(self.fallback, "last_tokens", 0)
        return {
            question.name: Answer(
                value=replacement[question.name].value,
                probabilities=replacement[question.name].probabilities,
                confidence=replacement[question.name].confidence,
                backend=replacement[question.name].backend,
                fell_back=True,
            )
            for question in questions
        }
