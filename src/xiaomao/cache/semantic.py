"""语义缓存：Redis 8 向量索引 + LLM 把关。"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass

import numpy as np
import redis
from redis.commands.search.field import TagField, TextField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from xiaomao.decision.base import Question

NO_VALUE = "none"
TAG_SPECIAL = re.compile(r"([,\\.<>{}[\]\"':;!@#$%^&*()\-+=~ ])")


@dataclass
class CacheHit:
    """一次缓存命中的结果。"""

    cache_id: str
    question: str
    answer: str
    citations: list[dict]
    similarity: float
    gate_p: float
    gate_tokens: int


REUSE_QUESTION = Question(
    name="reuse",
    type="noul",
    instructions=(
        "缓存的问题和当前问题问的是同一件事（同一个平台、站点、同一个要点），"
        "缓存的回答能直接完整地回答当前问题"
    ),
)


def _escape_tag(value: str) -> str:
    """TAG 值里的特殊字符要转义。"""
    return TAG_SPECIAL.sub(r"\\\1", value)


def build_lookup_query(platform: str | None, site: str | None) -> str:
    """TAG 过滤 + KNN 的查询字符串；只在与平台、站点一致的缓存里找。"""
    platform_tag = _escape_tag(platform or NO_VALUE)
    site_tag = _escape_tag(site or NO_VALUE)
    return (
        f"(@platform:{{{platform_tag}}} @site:{{{site_tag}}})"
        "=>[KNN 1 @vec $v AS dist]"
    )


def _vector_bytes(vector: list[float]) -> bytes:
    """FLOAT16 存进 HASH，体积减半。"""
    return np.asarray(vector, dtype=np.float16).tobytes()


class SemanticCache:
    """跨用户共享的语义缓存；只存通用内容。"""

    def __init__(
        self,
        redis_client,
        *,
        gate=None,
        tau: float = 0.80,
        gate_threshold: float = 0.5,
        ttl_seconds: int = 7 * 24 * 3600,
        prefix: str = "xm:cache:",
        index_name: str = "xm:idx:cache",
    ) -> None:
        self.redis = redis_client
        self.gate = gate
        self.tau = tau
        self.gate_threshold = gate_threshold
        self.ttl = ttl_seconds
        self.prefix = prefix
        self.index_name = index_name
        self._index_ready = False

    # ------------------------------------------------------------------
    # 索引
    # ------------------------------------------------------------------

    def ensure_index(self) -> None:
        """不存在就建索引；已存在跳过。"""
        if self._index_ready:
            return
        try:
            self.redis.ft(self.index_name).create_index(
                [
                    TextField("question"),
                    TagField("platform"),
                    TagField("site"),
                    TagField("intent"),
                    VectorField(
                        "vec", "HNSW",
                        {"TYPE": "FLOAT16", "DIM": 1024, "DISTANCE_METRIC": "COSINE"},
                    ),
                ],
                definition=IndexDefinition(prefix=[self.prefix], index_type=IndexType.HASH),
            )
        except redis.ResponseError as exc:
            if "already exists" not in str(exc).lower():
                raise
        self._index_ready = True

    def reset(self) -> None:
        """清空命名空间并重建索引（评估脚本用）。"""
        try:
            self.redis.ft(self.index_name).dropindex(delete_documents=True)
        except redis.ResponseError:
            pass
        self._index_ready = False
        self.ensure_index()

    def wait_until_indexed(self, expected: int, timeout: float = 15.0) -> int:
        """等后台索引追上写入条数，返回最终条数（评估脚本用）。"""
        deadline = time.time() + timeout
        count = 0
        while time.time() < deadline:
            try:
                info = self.redis.ft(self.index_name).info()
                count = int(info.get("num_docs", 0) or 0)
            except Exception:
                count = 0
            if count >= expected:
                return count
            time.sleep(0.1)
        return count

    # ------------------------------------------------------------------
    # 查
    # ------------------------------------------------------------------

    def lookup(
        self,
        question: str,
        vector: list[float],
        *,
        platform: str | None,
        site: str | None,
    ) -> tuple[CacheHit | None, dict]:
        """按平台、站点一致的缓存里找最相似的一条，再过 LLM 把关。"""
        self.ensure_index()
        query_text = build_lookup_query(platform, site)
        detail: dict = {
            "similarity": None, "candidate_id": None, "gate_p": None,
            "gate_tokens": 0, "lookup_ms": 0.0, "gate_ms": 0.0,
        }

        query = (
            Query(query_text)
            .sort_by("dist")
            .return_fields("question", "answer", "citations", "platform", "site", "hit_count", "dist")
            .dialect(2)
        )

        started = time.perf_counter()
        try:
            result = self.redis.ft(self.index_name).search(
                query, query_params={"v": _vector_bytes(vector)}
            )
        except redis.ResponseError:
            detail["lookup_ms"] = (time.perf_counter() - started) * 1000
            return None, detail
        detail["lookup_ms"] = (time.perf_counter() - started) * 1000

        if not result.docs:
            return None, detail
        document = result.docs[0]
        similarity = 1.0 - float(getattr(document, "dist", 1.0))
        detail["similarity"] = similarity
        detail["candidate_id"] = str(getattr(document, "id", "")).split(self.prefix)[-1] or None

        if similarity < self.tau:
            return None, detail

        cached_question = str(getattr(document, "question", ""))
        answer = str(getattr(document, "answer", ""))
        if self.gate is not None:
            gate_started = time.perf_counter()
            state = (
                f"当前问题：{question}\n\n缓存的问题：{cached_question}\n\n"
                f"缓存的回答：{answer[:1500]}"
            )
            try:
                answers = self.gate.decide(state, [REUSE_QUESTION])
                probability = float(answers["reuse"].value)
            except Exception:
                detail["gate_ms"] = (time.perf_counter() - gate_started) * 1000
                return None, detail
            detail["gate_ms"] = (time.perf_counter() - gate_started) * 1000
            detail["gate_p"] = probability
            detail["gate_tokens"] = int(getattr(self.gate, "last_tokens", 0) or 0)
            if probability < self.gate_threshold:
                return None, detail

        cache_id = detail["candidate_id"] or ""
        try:
            raw = self.redis.hget(self.prefix + cache_id, "citations")
            citations = json.loads(raw) if raw else []
        except Exception:
            citations = []
        try:
            self.redis.hincrby(self.prefix + cache_id, "hit_count", 1)
        except Exception:
            pass

        hit = CacheHit(
            cache_id=cache_id,
            question=cached_question,
            answer=answer,
            citations=citations,
            similarity=similarity,
            gate_p=float(detail["gate_p"]) if detail["gate_p"] is not None else 1.0,
            gate_tokens=int(detail["gate_tokens"]),
        )
        return hit, detail

    # ------------------------------------------------------------------
    # 写
    # ------------------------------------------------------------------

    def write(
        self,
        *,
        question: str,
        vector: list[float],
        platform: str | None,
        site: str | None,
        intent: str,
        answer: str,
        citations: list[dict],
        doc_ids: list[int],
    ) -> str:
        """写入一条缓存，并登记到 bydoc 集合。"""
        self.ensure_index()
        cache_id = uuid.uuid4().hex
        key = self.prefix + cache_id
        mapping = {
            "question": question,
            "vec": _vector_bytes(vector),
            "answer": answer,
            "citations": json.dumps(citations, ensure_ascii=False),
            "platform": platform or NO_VALUE,
            "site": site or NO_VALUE,
            "intent": intent,
            "hit_count": 0,
            "created_at": int(time.time()),
        }
        pipe = self.redis.pipeline()
        pipe.hset(key, mapping=mapping)
        pipe.expire(key, self.ttl)
        for doc_id in {int(item) for item in doc_ids}:
            index_key = f"xm:cache:bydoc:{doc_id}"
            pipe.sadd(index_key, cache_id)
            pipe.expire(index_key, self.ttl)
        pipe.execute()
        return cache_id

    def delete(self, cache_id: str) -> None:
        """删除一条缓存。"""
        self.redis.delete(self.prefix + cache_id)
