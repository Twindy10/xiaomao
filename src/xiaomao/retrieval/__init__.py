"""检索层：过滤、多路召回、RRF 融合、rerank。"""

from xiaomao.retrieval.filters import EU_MEMBERS, build_filter
from xiaomao.retrieval.hybrid import FusionConfig, Retriever, fuse, merge_collections, parse_config, rrf_fuse, zscore_fuse
from xiaomao.retrieval.rerank import (
    ApiReranker,
    LocalReranker,
    NoopReranker,
    RerankResult,
    Reranker,
    get_reranker,
    rerank_text,
)
from xiaomao.retrieval.search import (
    DEFAULT_COLLECTIONS,
    ROUTES,
    Candidate,
    QueryVectors,
    encode_queries,
    search_route,
)

__all__ = [
    "EU_MEMBERS", "build_filter",
    "ROUTES", "DEFAULT_COLLECTIONS", "Candidate", "QueryVectors", "encode_queries", "search_route",
    "FusionConfig", "parse_config", "merge_collections", "rrf_fuse", "zscore_fuse", "fuse", "Retriever",
    "RerankResult", "Reranker", "ApiReranker", "LocalReranker", "NoopReranker",
    "get_reranker", "rerank_text",
]
