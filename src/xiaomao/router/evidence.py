"""证据充分性判断：先用 rerank 分数卡阈值，中间段再问决策模型。"""

from __future__ import annotations

from dataclasses import dataclass

from xiaomao.decision.base import Decider, DeciderError, Question
from xiaomao.router.retrieve import RetrievalOutcome

EVIDENCE_QUESTION = Question(
    name="evidence",
    type="noul",
    instructions="检索到的资料里包含能直接回答用户问题的具体信息（不是只沾边）",
)

CONTEXT_LIMIT = 1200
THRESHOLD = 0.5


@dataclass
class EvidenceVerdict:
    """证据是否充分以及依据。"""

    sufficient: bool
    by: str            # rerank_high / rerank_low / decider / decider_fallback
    score: float


def _top_score(outcome: RetrievalOutcome) -> float:
    """第 1 名的分数。"""
    return float(outcome.candidates[0].score) if outcome.candidates else 0.0


def _state(query: str, outcome: RetrievalOutcome) -> str:
    """给决策模型的文本：问题 + 编号资料。"""
    lines = [f"用户问题：{query}", "", "检索到的资料："]
    for index, candidate in enumerate(outcome.candidates, start=1):
        text = str(candidate.fields.get("text") or "")[:CONTEXT_LIMIT]
        lines.append(f"[{index}] {text}")
        lines.append("")
    return "\n".join(lines)


def judge_evidence(
    query: str,
    outcome: RetrievalOutcome,
    decider: Decider,
    high: float = 0.7,
    low: float = 0.3,
) -> EvidenceVerdict:
    """三段判断：高分直接充分、低分直接不充分、中间问决策模型。"""
    if not outcome.candidates:
        return EvidenceVerdict(sufficient=False, by="rerank_low", score=0.0)

    score = _top_score(outcome)
    if not outcome.rerank_skipped:
        if score >= high:
            return EvidenceVerdict(sufficient=True, by="rerank_high", score=score)
        if score < low:
            return EvidenceVerdict(sufficient=False, by="rerank_low", score=score)

    try:
        answers = decider.decide(_state(query, outcome), [EVIDENCE_QUESTION])
    except (DeciderError, Exception):
        # 决策模型不可用时按"不充分"处理，交给 Backtrack 或拒答
        return EvidenceVerdict(
            sufficient=False,
            by="decider_fallback",
            score=score,
        )

    answer = answers[EVIDENCE_QUESTION.name]
    probability = float(answer.value)
    return EvidenceVerdict(
        sufficient=probability >= THRESHOLD,
        by="decider",
        score=probability,
    )
