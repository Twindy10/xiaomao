"""存储层测试：embed / index / kb_store 的纯函数和工具，全部用假对象。"""

from __future__ import annotations

import json
from datetime import date

import pytest

from xiaomao.ingest.embed import Embedder, Embedding, embed_input
from xiaomao.ingest.index import (
    ANALYZER_PARAMS,
    NO_EXPIRE,
    build_schema,
    chunk_to_row,
)
from xiaomao.ingest.kb_store import IngestLock, invalidate_doc_cache, meta_changed


class FakeModel:
    """假的 BGE-M3：记录调用、可按批大小抛 OOM。"""

    def __init__(self, fail_batch: int | None = None, sparse: dict | None = None) -> None:
        self.calls: list[list[str]] = []
        self.fail_batch = fail_batch
        self.sparse = sparse if sparse is not None else {"7": 0.5, "9": 0.0, "11": 0.2}

    def encode(self, batch, batch_size, max_length, return_dense, return_sparse, return_colbert_vecs):
        """返回固定的 dense 和稀疏向量。"""
        self.calls.append(list(batch))
        if self.fail_batch is not None and len(batch) == self.fail_batch:
            import torch

            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return {
            "dense_vecs": [[0.1, 0.2, 0.3] for _ in batch],
            "lexical_weights": [dict(self.sparse) for _ in batch],
        }


class FakeRedis:
    """只实现用到的方法。"""

    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.sets: dict[str, set[str]] = {}

    def set(self, key, value, nx=False, ex=None):
        """SET NX。"""
        if nx and key in self.values:
            return None
        self.values[key] = value
        return True

    def eval(self, script, numkeys, key, token):
        """比较后删除。"""
        if self.values.get(key) == token:
            del self.values[key]
            return 1
        return 0

    def smembers(self, key):
        """集合成员。"""
        return set(self.sets.get(key, set()))

    def delete(self, *keys):
        """删除若干 key。"""
        removed = 0
        for key in keys:
            if key in self.values:
                del self.values[key]
                removed += 1
            if key in self.sets:
                del self.sets[key]
                removed += 1
        return removed


def test_embed_input() -> None:
    """FAQ 只算问题，其他算标题路径 + 正文。"""
    faq = {"collection": "xm_faq", "question": "德国站耳机要认证吗", "text": "问题\n答案"}
    assert embed_input(faq) == "德国站耳机要认证吗"

    policy = {"collection": "xm_platform_policy", "title_path": "指南 > 一、认证", "text": "正文"}
    assert embed_input(policy) == "指南 > 一、认证\n正文"


def test_embedder_keeps_order_and_converts_sparse() -> None:
    """按长度排序编码，但返回顺序与输入一致；sparse key 转 int 并去掉 ≤0。"""
    model = FakeModel()
    embedder = Embedder(batch_size=4, model=model)
    texts = ["长" * 30, "短", "中等长度" * 5, "x"]

    result = embedder.encode(texts)

    assert [len(item.dense) for item in result] == [3, 3, 3, 3]
    assert result[1].sparse == {7: 0.5, 11: 0.2}
    assert all(isinstance(key, int) for key in result[0].sparse)
    # 编码时按长度从短到长
    assert [len(text) for text in model.calls[0]] == sorted(len(text) for text in texts)
    assert embedder.encode_seconds >= 0


def test_embedder_oom_downgrades_batch() -> None:
    """batch_size=16 时 OOM，自动降到 8 并成功。"""
    model = FakeModel(fail_batch=16)
    embedder = Embedder(batch_size=16, model=model)

    result = embedder.encode(["文本" * index for index in range(20)])

    assert len(result) == 20
    assert embedder._batch_size == 8
    assert any(len(batch) == 16 for batch in model.calls)
    assert any(len(batch) == 8 for batch in model.calls)


def test_embedder_rejects_empty_sparse() -> None:
    """稀疏向量为空要报错。"""
    embedder = Embedder(batch_size=4, model=FakeModel(sparse={"1": 0.0}))
    with pytest.raises(ValueError):
        embedder.encode(["正文"])


def test_embedder_empty_input() -> None:
    """空输入不调用模型。"""
    model = FakeModel()
    assert Embedder(model=model).encode([]) == []
    assert model.calls == []


def make_chunk(**overrides) -> dict:
    """构造一条片段。"""
    chunk = {
        "chunk_id": "a" * 16 + "-0000",
        "doc_key": "a" * 16,
        "chunk_index": 0,
        "collection": "xm_compliance",
        "platform": None,
        "site": "DE",
        "category": None,
        "doc_type": "regulation",
        "lang": "zh",
        "effective_date": "2026-01-01",
        "expire_date": None,
        "source_url": "https://example.com/a",
        "title_path": "法规 > 第一条",
        "text": "正文",
        "question": None,
        "answer": None,
        "token_count": 10,
    }
    chunk.update(overrides)
    return chunk


def test_chunk_to_row_conversions() -> None:
    """None 转空串、日期转 int、expire 默认 99991231、不含 sparse_bm25。"""
    embedding = Embedding(dense=[0.1, 0.2], sparse={7: 0.5})
    row = chunk_to_row(make_chunk(), 42, embedding)

    assert row["doc_id"] == 42
    assert row["text"] == "法规 > 第一条\n正文"  # 政策法规的 BM25 文本带标题路径
    faq = chunk_to_row(make_chunk(collection="xm_faq", title_path="问题", text="问题\n答案"), 1, embedding)
    assert faq["text"] == "问题\n答案"
    assert row["platform"] == "" and row["site"] == "DE"
    assert row["effective_date"] == 20260101
    assert row["expire_date"] == NO_EXPIRE
    assert row["dense"] == [0.1, 0.2]
    assert row["sparse_m3"] == {7: 0.5}
    assert "sparse_bm25" not in row


def test_chunk_to_row_rejects_overflow() -> None:
    """超长字段按 UTF-8 字节数报错。"""
    embedding = Embedding(dense=[0.1], sparse={1: 0.5})
    with pytest.raises(ValueError) as error:
        chunk_to_row(make_chunk(text="中" * 2731), 1, embedding)
    assert "text" in str(error.value)
    assert "8192" in str(error.value)


def test_chunk_to_row_rejects_bad_date() -> None:
    """日期格式不对要报错。"""
    with pytest.raises(ValueError):
        chunk_to_row(make_chunk(effective_date="2026/01/01"), 1, Embedding(dense=[0.1], sparse={1: 1.0}))


def test_build_schema_fields_and_function() -> None:
    """字段齐全，text 用 jieba + 业务词典，带 BM25 Function。"""
    schema = build_schema()
    names = {field.name for field in schema.fields}
    assert names == {
        "chunk_id", "doc_id", "text", "dense", "sparse_m3", "sparse_bm25",
        "platform", "site", "doc_type", "lang", "effective_date", "expire_date",
        "title_path", "source_url",
    }

    text_field = next(field for field in schema.fields if field.name == "text")
    analyzer = json.loads(text_field.params["analyzer_params"])
    assert analyzer["tokenizer"]["type"] == "jieba"
    assert "半托管" in analyzer["tokenizer"]["dict"]
    assert analyzer["filter"] == ["lowercase", "cnalphanumonly"]
    assert analyzer["tokenizer"]["dict"][0] == "_default_"

    function = schema.functions[0]
    assert function.type.name == "BM25"
    assert function.input_field_names == ["text"]
    assert function.output_field_names == ["sparse_bm25"]
    assert ANALYZER_PARAMS["tokenizer"]["mode"] == "exact"


def test_meta_changed_normalizes_date_and_null() -> None:
    """DATE 与字符串等价，NULL 与 None 等价。"""
    doc = {"title": "a", "platform": None, "site": "DE", "doc_type": "regulation",
           "lang": "zh", "effective_date": "2026-10-28", "expire_date": None}

    row_same = {**doc, "effective_date": date(2026, 10, 28)}
    assert meta_changed(row_same, doc) is False

    row_diff = {**doc, "expire_date": "2027-01-01"}
    assert meta_changed(row_diff, doc) is True


def test_invalidate_doc_cache() -> None:
    """按文档删除缓存并删掉索引集合。"""
    redis = FakeRedis()
    redis.sets["xm:cache:bydoc:7"] = {"c1", "c2"}
    redis.values["xm:cache:c1"] = "1"
    redis.values["xm:cache:c2"] = "2"

    assert invalidate_doc_cache(redis, 7) == 2
    assert "xm:cache:c1" not in redis.values
    assert "xm:cache:bydoc:7" not in redis.sets


def test_invalidate_doc_cache_empty() -> None:
    """没有登记过缓存时返回 0。"""
    assert invalidate_doc_cache(FakeRedis(), 9) == 0


def test_ingest_lock_exclusive_and_safe_release() -> None:
    """锁互斥；释放时不删别人的锁。"""
    redis = FakeRedis()
    with IngestLock(redis, "xm_faq"):
        assert "xm:lock:ingest:xm_faq" in redis.values
        with pytest.raises(RuntimeError):
            with IngestLock(redis, "xm_faq"):
                pass
    assert "xm:lock:ingest:xm_faq" not in redis.values

    redis.values["xm:lock:ingest:xm_faq"] = "someone-else"
    lock = IngestLock(redis, "xm_faq")
    lock.__exit__(None, None, None)
    assert redis.values["xm:lock:ingest:xm_faq"] == "someone-else"
