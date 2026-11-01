"""多路召回：dense、BM25、BGE-M3 稀疏，各自向 Milvus 查一次。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from xiaomao.ingest.embed import Embedder

ROUTES = ("dense", "bm25", "sparse_m3")
DEFAULT_COLLECTIONS = ("xm_platform_policy", "xm_compliance")

OUTPUT_FIELDS = (
    "doc_id", "text", "title_path", "source_url",
    "platform", "site", "doc_type", "effective_date", "expire_date",
)


@dataclass
class Candidate:
    """一条召回结果。"""

    chunk_id: str
    collection: str
    score: float
    rank: int
    fields: dict


@dataclass
class QueryVectors:
    """一条问题的向量。"""

    dense: list[float]
    sparse: dict[int, float]


def encode_queries(embedder: Embedder, queries: list[str]) -> list[QueryVectors]:
    """一次编码多条问题。"""
    embeddings = embedder.encode(queries)
    return [QueryVectors(dense=item.dense, sparse=item.sparse) for item in embeddings]


def search_route(
    client: Any,
    collection: str,
    route: str,
    query: str,
    vectors: QueryVectors,
    *,
    filter_expr: str,
    limit: int = 20,
) -> list[Candidate]:
    """在某一路、某个 collection 上检索 Top-limit。"""
    if route == "dense":
        data = vectors.dense
        anns_field = "dense"
        params = {"metric_type": "COSINE", "params": {"ef": 100}}
    elif route == "bm25":
        data = query  # Milvus 用 text 字段自动分词
        anns_field = "sparse_bm25"
        params = {"metric_type": "BM25"}
    elif route == "sparse_m3":
        data = vectors.sparse
        anns_field = "sparse_m3"
        params = {"metric_type": "IP"}
    else:
        raise ValueError(f"未知的检索路：{route}")

    hits = client.search(
        collection_name=collection,
        data=[data],
        anns_field=anns_field,
        search_params=params,
        limit=limit,
        filter=filter_expr,
        output_fields=list(OUTPUT_FIELDS),
    )[0]

    candidates: list[Candidate] = []
    for rank, hit in enumerate(hits, start=1):
        entity = hit.get("entity") or {}
        fields = {name: entity.get(name, hit.get(name)) for name in OUTPUT_FIELDS}
        candidates.append(
            Candidate(
                chunk_id=str(hit["chunk_id"]),
                collection=collection,
                score=float(hit.get("distance", hit.get("score", 0.0))),
                rank=rank,
                fields=fields,
            )
        )
    return candidates
