"""该不该把这次回答写进缓存。"""

from __future__ import annotations

PARTIAL_MARKERS = ("资料里没有提到", "资料中没有提到", "没有相关的资料")


def should_cache(result) -> tuple[bool, str]:
    """返回（是否写缓存, 不写的原因）。"""
    trace = result.trace or {}

    if result.strategy in ("none", None):
        return False, "chitchat_or_out_of_scope"
    if result.refused:
        return False, "refused"
    if trace.get("cache_hit"):
        return False, "from_cache"
    if trace.get("decision_error"):
        return False, "decision_degraded"
    if trace.get("rerank_skipped"):
        return False, "rerank_skipped"
    if not result.citations:
        return False, "no_citations"
    if any(marker in result.answer for marker in PARTIAL_MARKERS):
        return False, "partial_answer"
    return True, ""
