"""语义缓存测试：假 Redis，覆盖阈值、TAG、把关、TTL 和写缓存规则。"""

from __future__ import annotations

import json

import pytest

from xiaomao.cache.policy import should_cache
from xiaomao.cache.semantic import SemanticCache
from xiaomao.decision.base import Answer
from xiaomao.qa import QAResult


class FakeDoc:
    """模拟 redis 返回的文档。"""

    def __init__(self, doc_id: str, **fields) -> None:
        self.id = doc_id
        for key, value in fields.items():
            setattr(self, key, value)


class FakeResult:
    """模拟 SearchResult。"""

    def __init__(self, docs: list[FakeDoc]) -> None:
        self.docs = docs


class FakeFT:
    """模拟 ft(index)。"""

    def __init__(self, redis_client: "FakeRedis") -> None:
        self.redis = redis_client

    def create_index(self, fields, definition=None):
        """第一次成功，第二次报已存在。"""
        if self.redis.index_exists:
            import redis

            raise redis.ResponseError("Index already exists")
        self.redis.index_exists = True

    def dropindex(self, delete_documents: bool = False):
        """删除索引。"""
        if not self.redis.index_exists:
            import redis

            raise redis.ResponseError("Unknown Index name")
        self.redis.index_exists = False
        if delete_documents:
            self.redis.hashes.clear()

    def search(self, query, query_params=None):
        """返回测试预设的文档。"""
        if self.redis.search_error:
            import redis

            raise redis.ResponseError("boom")
        self.redis.last_query = query
        return FakeResult(self.redis.search_docs)


class FakePipeline:
    """立即执行命令的假 pipeline。"""

    def __init__(self, redis_client: "FakeRedis") -> None:
        self.redis = redis_client
        self.commands: list[tuple] = []

    def hset(self, key, mapping):
        """记录并写入。"""
        self.commands.append(("hset", key))
        self.redis.hashes[key] = dict(mapping)
        return self

    def expire(self, key, ttl):
        """记录 TTL。"""
        self.commands.append(("expire", key, ttl))
        self.redis.expires[key] = ttl
        return self

    def sadd(self, key, member):
        """记录集合。"""
        self.commands.append(("sadd", key, member))
        self.redis.sets.setdefault(key, set()).add(member)
        return self

    def execute(self):
        """执行完成。"""
        return [1 for _ in self.commands]


class FakeRedis:
    """只实现 SemanticCache 用到的方法。"""

    def __init__(self) -> None:
        self.hashes: dict[str, dict] = {}
        self.sets: dict[str, set] = {}
        self.expires: dict[str, int] = {}
        self.index_exists = False
        self.search_docs: list[FakeDoc] = []
        self.search_error = False
        self.last_query = None

    def ft(self, name):
        """返回假 FT 对象。"""
        return FakeFT(self)

    def pipeline(self):
        """返回假 pipeline。"""
        return FakePipeline(self)

    def hget(self, key, field):
        """读字段。"""
        return self.hashes.get(key, {}).get(field)

    def hincrby(self, key, field, amount):
        """自增。"""
        self.hashes.setdefault(key, {})
        self.hashes[key][field] = int(self.hashes[key].get(field, 0)) + amount
        return self.hashes[key][field]

    def delete(self, key):
        """删除 key。"""
        self.hashes.pop(key, None)
        return 1


class FakeGate:
    """假把关决策器。"""

    def __init__(self, value: float = 0.9, error: Exception | None = None) -> None:
        self.value = value
        self.error = error
        self.calls: list[str] = []
        self.last_tokens = 42

    def decide(self, state, questions):
        """返回预设概率。"""
        self.calls.append(state)
        if self.error:
            raise self.error
        return {"reuse": Answer(self.value, {"yes": self.value, "no": 1 - self.value},
                                abs(2 * self.value - 1), "fake")}


def cached_doc(similarity: float = 0.9) -> FakeDoc:
    """构造一条命中候选（dist = 1 - 相似度）。"""
    return FakeDoc(
        "xm:cache:abc123",
        question="德国站耳机要什么认证",
        answer="需要 CE-RED 认证。",
        dist=1 - similarity,
    )


def build_cache(gate=None, tau: float = 0.80) -> tuple[SemanticCache, FakeRedis]:
    """构造缓存和假 Redis。"""
    redis_client = FakeRedis()
    cache = SemanticCache(redis_client, gate=gate, tau=tau)
    cache.ensure_index()
    return cache, redis_client


def test_low_similarity_misses() -> None:
    """相似度低于 τ 不命中。"""
    cache, redis_client = build_cache(tau=0.80)
    redis_client.search_docs = [cached_doc(0.79)]

    hit, detail = cache.lookup("问题", [0.1] * 1024, platform="amazon", site="DE")
    assert hit is None
    assert detail["similarity"] == pytest.approx(0.79)


def test_gate_rejects() -> None:
    """把关概率低于阈值不命中。"""
    gate = FakeGate(value=0.2)
    cache, redis_client = build_cache(gate=gate)
    redis_client.search_docs = [cached_doc(0.9)]
    redis_client.hashes["xm:cache:abc123"] = {"citations": "[]"}

    hit, detail = cache.lookup("问题", [0.1] * 1024, platform="amazon", site="DE")
    assert hit is None
    assert detail["gate_p"] == pytest.approx(0.2)
    assert gate.calls


def test_gate_error_misses() -> None:
    """把关出错按未命中处理。"""
    cache, redis_client = build_cache(gate=FakeGate(error=RuntimeError("boom")))
    redis_client.search_docs = [cached_doc(0.9)]

    hit, _ = cache.lookup("问题", [0.1] * 1024, platform=None, site=None)
    assert hit is None


def test_hit_returns_answer_and_increments() -> None:
    """命中返回答案和引用，hit_count 加 1。"""
    cache, redis_client = build_cache(gate=FakeGate(value=0.9))
    redis_client.search_docs = [cached_doc(0.95)]
    redis_client.hashes["xm:cache:abc123"] = {
        "citations": json.dumps([{"n": 1, "chunk_id": "c1"}], ensure_ascii=False)
    }

    hit, detail = cache.lookup("问题", [0.1] * 1024, platform="amazon", site="DE")

    assert hit is not None
    assert hit.answer == "需要 CE-RED 认证。"
    assert hit.citations == [{"n": 1, "chunk_id": "c1"}]
    assert hit.gate_tokens == 42
    assert redis_client.hashes["xm:cache:abc123"]["hit_count"] == 1
    assert detail["candidate_id"] == "abc123"


def test_tag_filter_in_query() -> None:
    """查询里带平台和站点的 TAG 条件，None 写成 none。"""
    from xiaomao.cache.semantic import build_lookup_query

    assert build_lookup_query("amazon", "DE") == "(@platform:{amazon} @site:{DE})=>[KNN 1 @vec $v AS dist]"
    assert build_lookup_query(None, None) == "(@platform:{none} @site:{none})=>[KNN 1 @vec $v AS dist]"


def test_write_registers_bydoc_and_ttl() -> None:
    """写缓存登记 bydoc 集合并给所有 key 设 TTL。"""
    cache, redis_client = build_cache()

    cache_id = cache.write(
        question="问题", vector=[0.1] * 1024, platform="amazon", site="DE",
        intent="compliance", answer="答案", citations=[], doc_ids=[7, 8],
    )

    key = f"xm:cache:{cache_id}"
    assert key in redis_client.hashes
    assert redis_client.hashes[key]["platform"] == "amazon"
    assert redis_client.expires[key] == 7 * 24 * 3600
    assert redis_client.sets["xm:cache:bydoc:7"] == {cache_id}
    assert redis_client.expires["xm:cache:bydoc:8"] == 7 * 24 * 3600


def test_ensure_index_is_idempotent() -> None:
    """重复建索引不报错。"""
    cache, redis_client = build_cache()
    cache.ensure_index()
    assert redis_client.index_exists is True


def make_result(**overrides) -> QAResult:
    """构造 QAResult。"""
    result = QAResult(
        answer="回答[1]", citations=[{"n": 1, "chunk_id": "c1"}], refused=False,
        intent="compliance", strategy="direct", evidence=[], trace={"cache_hit": False},
    )
    for key, value in overrides.items():
        setattr(result, key, value)
    return result


def test_should_cache_rules() -> None:
    """不写缓存的六种情况，以及正常写。"""
    assert should_cache(make_result())[0] is True

    assert should_cache(make_result(strategy="none"))[0] is False
    assert should_cache(make_result(refused=True))[0] is False
    assert should_cache(make_result(trace={"cache_hit": True}))[0] is False
    assert should_cache(make_result(trace={"decision_error": True}))[0] is False
    assert should_cache(make_result(trace={"rerank_skipped": True}))[0] is False
    assert should_cache(make_result(citations=[]))[0] is False
    assert should_cache(make_result(answer="资料里没有提到这一点"))[0] is False
