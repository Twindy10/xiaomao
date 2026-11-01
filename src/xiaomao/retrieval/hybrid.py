"""融合与对外入口：Python 里做 RRF，便于跨 collection 合并和消融复用。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from xiaomao.retrieval.filters import build_filter
from xiaomao.retrieval.search import (
    DEFAULT_COLLECTIONS,
    ROUTES,
    Candidate,
    QueryVectors,
    encode_queries,
    search_route,
)


FUSION_METHODS = {"rrf": "rrf", "z": "zscore"}


@dataclass
class FusionConfig:
    """融合配置：参与的路、权重和融合方式（rrf 按名次，zscore 按每路 z-score 后的分数加权）。"""

    name: str
    weights: dict[str, float]
    k: int = 60
    method: str = "rrf"


def parse_config(text: str) -> FusionConfig:
    """把 "dense+bm25"、"dense:2+bm25" 或 "dense+bm25:0.2@z" 解析成配置；@z 表示 z-score 加权融合。"""
    body, _, suffix = text.partition("@")
    if (suffix or "rrf") not in FUSION_METHODS:
        raise ValueError(f"未知的融合方式：{suffix!r}，可选 {sorted(FUSION_METHODS)}")
    weights: dict[str, float] = {}
    for part in body.split("+"):
        piece = part.strip()
        if not piece:
            continue
        if ":" in piece:
            name, value = piece.split(":", 1)
            weight = float(value)
        else:
            name, weight = piece, 1.0
        name = name.strip()
        if name not in ROUTES:
            raise ValueError(f"未知的检索路：{name!r}，可选 {ROUTES}")
        weights[name] = weight
    if not weights:
        raise ValueError("融合配置不能为空")
    return FusionConfig(name=text, weights=weights, method=FUSION_METHODS[suffix or "rrf"])


def merge_collections(per_collection: dict[str, list[Candidate]]) -> list[Candidate]:
    """同一路在多个 collection 的结果按原始分数合并，重新编 rank。"""
    merged = [candidate for rows in per_collection.values() for candidate in rows]
    merged.sort(key=lambda item: -item.score)
    return [replace(candidate, rank=index) for index, candidate in enumerate(merged, start=1)]


def rrf_fuse(route_lists: dict[str, list[Candidate]], config: FusionConfig, limit: int = 20) -> list[Candidate]:
    """RRF：score = Σ weight / (k + rank)；同分按最好名次、再按 chunk_id 排。"""
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    first_seen: dict[str, Candidate] = {}

    for route, weight in config.weights.items():
        if weight == 0:
            continue
        for candidate in route_lists.get(route, []):
            chunk_id = candidate.chunk_id
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (config.k + candidate.rank)
            best_rank[chunk_id] = min(best_rank.get(chunk_id, candidate.rank), candidate.rank)
            first_seen.setdefault(chunk_id, candidate)

    ordered = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], best_rank[chunk_id], chunk_id))
    fused = [
        replace(first_seen[chunk_id], score=scores[chunk_id], rank=index)
        for index, chunk_id in enumerate(ordered[:limit], start=1)
    ]
    return fused


def zscore_fuse(route_lists: dict[str, list[Candidate]], config: FusionConfig, limit: int = 20) -> list[Candidate]:
    """每一路的原始分数先在本题的候选里做 z-score，再按权重相加；某路没召回到的候选记这一路的最低分。

    RRF 只看名次：中文问题配英文文档时，稀疏路分数整体很低，但第 1 名照样拿满分加成，会挤掉 dense 的正确答案。
    z-score 消除了 BM25（几十）和余弦（0–1）的量纲差，又保留了每路分数的相对强弱（M3 实测，见 eval/reports/M3_retrieval.md）。
    """
    normalized: dict[str, dict[str, float]] = {}
    first_seen: dict[str, Candidate] = {}
    for route, weight in config.weights.items():
        rows = route_lists.get(route, [])
        if weight == 0 or not rows:
            continue
        values = [candidate.score for candidate in rows]
        mean = sum(values) / len(values)
        std = (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5 or 1.0
        normalized[route] = {candidate.chunk_id: (candidate.score - mean) / std for candidate in rows}
        for candidate in rows:
            first_seen.setdefault(candidate.chunk_id, candidate)

    floors = {route: min(scores.values()) for route, scores in normalized.items()}
    totals = {
        chunk_id: sum(
            config.weights[route] * scores.get(chunk_id, floors[route])
            for route, scores in normalized.items()
        )
        for chunk_id in first_seen
    }
    ordered = sorted(totals, key=lambda chunk_id: (-totals[chunk_id], chunk_id))
    return [
        replace(first_seen[chunk_id], score=totals[chunk_id], rank=index)
        for index, chunk_id in enumerate(ordered[:limit], start=1)
    ]


def fuse(route_lists: dict[str, list[Candidate]], config: FusionConfig, limit: int = 20) -> list[Candidate]:
    """按 config.method 选融合方式。"""
    if config.method == "zscore":
        return zscore_fuse(route_lists, config, limit)
    return rrf_fuse(route_lists, config, limit)


class Retriever:
    """给 M4 用的检索入口：一次查询 → 多路召回 → 融合。"""

    def __init__(
        self,
        client: Any,
        embedder,
        collections=DEFAULT_COLLECTIONS,
        config: FusionConfig | str = "dense+bm25:0.2@z",  # M3 实测，见 eval/reports/M3_retrieval.md 第 5 节
        candidates: int = 20,
    ) -> None:
        self.client = client
        self.embedder = embedder
        self.collections = tuple(collections)
        self.config = parse_config(config) if isinstance(config, str) else config
        self.candidates = candidates

    def route_lists(self, query: str, vectors: QueryVectors, *, filter_expr: str) -> dict[str, list[Candidate]]:
        """config 里用到的每一路，在每个 collection 各查 candidates 条再合并。"""
        lists: dict[str, list[Candidate]] = {}
        for route in self.config.weights:
            per_collection = {
                collection: search_route(
                    self.client, collection, route, query, vectors,
                    filter_expr=filter_expr, limit=self.candidates,
                )
                for collection in self.collections
            }
            lists[route] = merge_collections(per_collection)
        return lists

    def search(
        self,
        query: str,
        *,
        platform: str | None = None,
        site: str | None = None,
        top_k: int = 20,
        dense_text: str | None = None,
        vectors: QueryVectors | None = None,
    ) -> list[Candidate]:
        """检索并融合，返回前 top_k 条。

        dense_text 只影响 dense 那一路（HyDE 用）；vectors 传了就直接用，
        避免同一条问题在查缓存和检索时各编码一次。
        """
        if vectors is None or dense_text is not None:
            vectors = encode_queries(self.embedder, [dense_text or query])[0]
        filter_expr = build_filter(platform, site)
        lists = self.route_lists(query, vectors, filter_expr=filter_expr)
        return fuse(lists, self.config, limit=top_k)
