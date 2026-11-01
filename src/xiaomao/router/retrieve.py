"""三种检索策略（Direct / SubQuery / HyDE）和 Backtrack 的放宽步骤。"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from xiaomao.retrieval.search import Candidate

CANDIDATES = 20  # 先取 20 条交给 reranker

# 只写"拆成 2–3 个子问题"时，模型会补上原问题没有的法规术语、丢掉原问题的另一个方面（M4 评估）
SUBQUERY_SYSTEM = (
    "你是检索助手。用户的问题里如果包含几个不同的方面，把它按原问题里的方面拆开，"
    "每个子问题只对应原问题里的一个方面，用原问题里的词，不要加入原问题没有提到的法规、术语或方面。"
    "原问题里的平台、站点、产品要保留在每个子问题里。通常拆成 2 个，最多 3 个。"
    "只输出 JSON：{\"subqueries\": [\"...\", \"...\"]}。不要解释。"
)
HYDE_SYSTEM = (
    "你是跨境电商专家。针对用户问题写一段 150 字以内的假设答案，用来帮助检索。"
    "不需要完全准确，但要包含可能的关键词和结论。只输出这段文字。"
)


@dataclass
class RetrievalOutcome:
    """一次检索的结果。"""

    candidates: list[Candidate]
    rerank_skipped: bool
    subqueries: list[str] = field(default_factory=list)
    hypothetical: str | None = None
    filters: dict = field(default_factory=dict)
    tokens: int = 0
    rerank_tokens: int = 0


def _before_tokens(llm) -> int:
    """当前累计 tokens。"""
    usage = getattr(llm, "usage", None)
    return int(getattr(usage, "total_tokens", 0) or 0) if usage is not None else 0


def _used_tokens(llm, before: int) -> int:
    """LLM 调用的 token 增量。"""
    usage = getattr(llm, "usage", None)
    if usage is None:
        return 0
    return int(getattr(usage, "total_tokens", 0) or 0) - before


def _search(retriever, query: str, *, platform, site, dense_text: str | None = None,
            vectors=None) -> list[Candidate]:
    """检索 Top-20；vectors 传了就直接用，省一次编码。"""
    return retriever.search(
        query, platform=platform, site=site, top_k=CANDIDATES, dense_text=dense_text, vectors=vectors
    )


def _rerank(reranker, query: str, candidates: list[Candidate], top_n: int, use_rerank: bool):
    """重排并截断；不重排时直接取融合顺序。返回（候选, 是否跳过, rerank tokens）。"""
    if not use_rerank:
        return candidates[:top_n], True, 0
    result = reranker.rerank(query, candidates, top_n=top_n)
    return result.candidates, result.skipped, int(getattr(result, "tokens", 0) or 0)


def backtrack_steps(platform: str | None, site: str | None) -> list[dict]:
    """依次放宽：去掉 site → 去掉 platform。"""
    steps: list[dict] = []
    if site is not None:
        steps.append({"platform": platform, "site": None})
    if platform is not None:
        steps.append({"platform": None, "site": None})
    return steps


def _hyde_text(llm, query: str) -> str:
    """让 LLM 写一段假设答案。"""
    try:
        payload = llm.chat_json(
            f"用户问题：{query}\n请写一段 150 字以内的假设答案，只输出 JSON：{{\"answer\": \"...\"}}",
            HYDE_SYSTEM,
            max_tokens=300,
            temperature=0.3,
        )
        return str(payload.get("answer") or "").strip()
    except Exception:
        return ""


def _subqueries(llm, query: str) -> list[str]:
    """让 LLM 拆子问题。"""
    try:
        payload = llm.chat_json(
            f"用户问题：{query}", SUBQUERY_SYSTEM, max_tokens=300, temperature=0.0
        )
    except Exception:
        return []
    return [str(item).strip() for item in (payload.get("subqueries") or []) if str(item).strip()]


def _round_robin(ranked_lists: list[list[Candidate]], top_n: int) -> list[Candidate]:
    """轮流取：先取每个子问题的第 1 名，再取第 2 名……去重后取满。"""
    merged: list[Candidate] = []
    seen: set[str] = set()
    depth = 0
    while len(merged) < top_n and any(depth < len(rows) for rows in ranked_lists):
        for rows in ranked_lists:
            if depth >= len(rows):
                continue
            candidate = rows[depth]
            if candidate.chunk_id in seen:
                continue
            seen.add(candidate.chunk_id)
            merged.append(replace(candidate, rank=len(merged) + 1))
            if len(merged) >= top_n:
                break
        depth += 1
    return merged


def _with_extra(retriever, candidates: list[Candidate], extra_queries: list[str], *,
                platform, site) -> list[Candidate]:
    """把附加改写（英文译文）召回的候选并进原问题的候选池，去重；之后统一用原问题 rerank。"""
    merged = list(candidates)
    seen = {candidate.chunk_id for candidate in merged}
    for extra in extra_queries:
        for candidate in _search(retriever, extra, platform=platform, site=site):
            if candidate.chunk_id not in seen:
                seen.add(candidate.chunk_id)
                merged.append(candidate)
    return merged


def retrieve(
    query: str,
    strategy: str,
    *,
    retriever,
    reranker,
    llm,
    platform: str | None = None,
    site: str | None = None,
    use_rerank: bool = True,
    top_n: int = 5,
    extra_queries: list[str] | None = None,
    vectors=None,
) -> RetrievalOutcome:
    """按策略检索并重排，返回 Top-n。

    extra_queries（英文译文）只在重排时生效：多出来的候选要靠 rerank 和原问题的候选放在一起比，
    不重排时没法合并，直接忽略。
    """
    filters = {"platform": platform, "site": site}
    before = _before_tokens(llm)
    extra = [text for text in (extra_queries or []) if text and text != query] if use_rerank else []

    def original_pool(dense_text: str | None = None) -> list[Candidate]:
        candidates = _search(retriever, query, platform=platform, site=site, dense_text=dense_text,
                             vectors=vectors if dense_text is None else None)
        return _with_extra(retriever, candidates, extra, platform=platform, site=site) if extra else candidates

    if strategy == "subquery":
        subqueries = _subqueries(llm, query)
        if len(subqueries) >= 2 and use_rerank:
            # 原问题、英文译文、子问题各自召回，合并后统一用原问题 rerank 一次。
            # 以前是每个子问题各自 rerank 再轮流取：子问题没翻译，偏中文的结果会把译文召回的英文证据挤掉
            # （M4 v3 里 3 道多证据题因此丢了证据）。原问题也在池子里，拆错了不至于丢掉它本来能找到的片段。
            pool = _with_extra(retriever, original_pool(), subqueries, platform=platform, site=site)
            ranked, skipped, rerank_tokens = _rerank(reranker, query, pool, top_n, use_rerank)
            return RetrievalOutcome(
                candidates=ranked,
                rerank_skipped=skipped,
                subqueries=subqueries,
                filters=filters,
                tokens=_used_tokens(llm, before),
                rerank_tokens=rerank_tokens,
            )
        if len(subqueries) >= 2:
            # 不重排时分数没法放在一起比，退回原问题和子问题轮流取
            ranked_lists = [
                _rerank(reranker, text, _search(retriever, text, platform=platform, site=site), top_n, False)[0]
                for text in [query, *subqueries]
            ]
            return RetrievalOutcome(
                candidates=_round_robin(ranked_lists, top_n),
                rerank_skipped=True,
                subqueries=subqueries,
                filters=filters,
                tokens=_used_tokens(llm, before),
            )

    if strategy == "hyde":
        hypothetical = _hyde_text(llm, query)
        if hypothetical:
            candidates = original_pool(dense_text=f"{query}\n{hypothetical}")
            ranked, skipped, rerank_tokens = _rerank(reranker, query, candidates, top_n, use_rerank)
            return RetrievalOutcome(
                candidates=ranked,
                rerank_skipped=skipped,
                hypothetical=hypothetical,
                filters=filters,
                tokens=_used_tokens(llm, before),
                rerank_tokens=rerank_tokens,
            )

    ranked, skipped, rerank_tokens = _rerank(reranker, query, original_pool(), top_n, use_rerank)
    return RetrievalOutcome(
        candidates=ranked, rerank_skipped=skipped, filters=filters, rerank_tokens=rerank_tokens,
        tokens=_used_tokens(llm, before),
    )
