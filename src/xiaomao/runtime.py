"""运行时组件工厂：检索、rerank、缓存、链路池共用一套模型。"""

from __future__ import annotations

import os
import threading
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]


class LockedRetriever:
    """多线程共用一份 BGE-M3：编码和检索串行，LLM / rerank 调用并发。"""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.lock = threading.Lock()

    @property
    def embedder(self):
        """底层的 Embedder。"""
        return getattr(self.inner, "embedder", None)

    def encode(self, text: str):
        """编码一条问题，返回 QueryVectors。"""
        from xiaomao.retrieval.search import encode_queries

        with self.lock:
            return encode_queries(self.inner.embedder, [text])[0]

    def search(self, *args, **kwargs):
        """串行检索。"""
        with self.lock:
            return self.inner.search(*args, **kwargs)


def redis_from_env():
    """按 REDIS_URL 建 Redis 客户端。"""
    import redis

    load_dotenv(ROOT_DIR / ".env", override=True)
    return redis.Redis.from_url(os.environ["REDIS_URL"], socket_timeout=10)


def build_pool():
    """MySQL 连接池。"""
    from xiaomao.storage.db import MySQLPool

    load_dotenv(ROOT_DIR / ".env", override=True)
    return MySQLPool(size=int(os.environ.get("MYSQL_POOL_SIZE", "8")))


def build_cache(redis_client=None):
    """语义缓存，把关用 LLM（实测 0.8B 分不开）。"""
    from xiaomao.cache.semantic import SemanticCache
    from xiaomao.decision.llm_json import LLMJsonDecider

    return SemanticCache(redis_client or redis_from_env(), gate=LLMJsonDecider())


def build_pipelines(workers: int = 1, *, with_cache: bool = False, with_rewriter: bool = False) -> list:
    """构建 N 条链路：共用检索 / rerank / Redis，各自持有决策器、LLM 客户端和缓存把关。"""
    from pymilvus import MilvusClient

    from xiaomao.decision import get_decider
    from xiaomao.generation.llm import Generator
    from xiaomao.ingest.embed import Embedder
    from xiaomao.llm import LLMClient
    from xiaomao.qa import QAPipeline
    from xiaomao.retrieval.hybrid import Retriever
    from xiaomao.retrieval.rerank import get_reranker
    from xiaomao.router.rewrite import Rewriter
    from xiaomao.router.translate import Translator

    load_dotenv(ROOT_DIR / ".env", override=True)
    client = MilvusClient(uri=os.environ["MILVUS_URI"], token=os.environ.get("MILVUS_TOKEN", ""))
    embedder = Embedder(model_name=os.environ.get("EMBED_MODEL"), max_length=512)
    retriever = LockedRetriever(Retriever(client, embedder))
    reranker = get_reranker()
    # 每条链路一个 SemanticCache（各自的把关 LLM 客户端，tokens 不串），共用同一个 Redis 客户端
    redis_client = redis_from_env() if with_cache else None

    return [
        QAPipeline(
            decider=get_decider(),
            retriever=retriever,
            reranker=reranker,
            llm_json=LLMClient(),
            generator=Generator(),
            translator=Translator(),
            cache=build_cache(redis_client) if with_cache else None,
            rewriter=Rewriter() if with_rewriter else None,
        )
        for _ in range(workers)
    ]
