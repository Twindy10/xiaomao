"""增量入库测试：全部用假 Milvus / MySQL / Redis / 向量模型。"""

from __future__ import annotations

import re

import pytest

from xiaomao.ingest.embed import Embedding
from xiaomao.ingest.incremental import Action, apply_plan, plan_collection, reconcile


class FakeEmbedder:
    """按文本长度生成固定向量，记录被编码过的文本。"""

    def __init__(self) -> None:
        self.encode_seconds = 0.0
        self.texts: list[str] = []

    def encode(self, texts: list[str]) -> list[Embedding]:
        """返回固定向量。"""
        self.texts.extend(texts)
        return [Embedding(dense=[float(len(text)), 0.0], sparse={1: 0.5}) for text in texts]

    def peak_vram_mb(self) -> float | None:
        """测试里不涉及显存。"""
        return None


class FakeMilvus:
    """用 dict 模拟 collection。"""

    def __init__(self, fail_upsert: bool = False) -> None:
        self.collections: dict[str, dict[str, dict]] = {}
        self.fail_upsert = fail_upsert
        self.finished: list[str] = []

    def upsert(self, collection_name: str, data: list[dict]) -> None:
        """写入或覆盖。"""
        if self.fail_upsert:
            raise RuntimeError("milvus upsert 失败")
        rows = self.collections.setdefault(collection_name, {})
        for row in data:
            rows[row["chunk_id"]] = row

    def delete(self, collection_name: str, filter: str) -> None:
        """解析 doc_id == N 并删除。"""
        match = re.match(r"doc_id == (\d+)", filter)
        assert match, filter
        doc_id = int(match.group(1))
        rows = self.collections.setdefault(collection_name, {})
        for chunk_id in [key for key, row in rows.items() if row["doc_id"] == doc_id]:
            del rows[chunk_id]

    def query(self, collection_name: str, filter: str, output_fields, consistency_level):
        """只支持 count(*)。"""
        return [{"count(*)": len(self.collections.get(collection_name, {}))}]

    def flush(self, name: str) -> None:
        """记录调用。"""
        self.finished.append(name)

    def load_collection(self, name: str) -> None:
        """无需实现。"""

    def rows(self, name: str) -> list[dict]:
        """取集合里的行。"""
        return list(self.collections.get(name, {}).values())


class FakeStore:
    """内存版 kb_document。"""

    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.next_id = 1
        self._snapshot: list[dict] = []

    def _copy(self) -> list[dict]:
        return [dict(row) for row in self.rows]

    def find_active(self, collection: str, source_url: str) -> dict | None:
        """按 collection + url 找 active。"""
        for row in self.rows:
            if row["collection"] == collection and row["source_url"] == source_url and row["status"] == "active":
                return dict(row)
        return None

    def find_by_hash(self, collection: str, content_hash: str) -> dict | None:
        """按 collection + hash 找任意状态。"""
        for row in self.rows:
            if row["collection"] == collection and row["content_hash"] == content_hash:
                return dict(row)
        return None

    def list_active(self, collection: str) -> list[dict]:
        """列出 active。"""
        return [dict(row) for row in self.rows if row["collection"] == collection and row["status"] == "active"]

    def insert(self, doc: dict, version: int) -> int:
        """插入一行。"""
        row = {
            "id": self.next_id,
            "collection": doc["collection"],
            "source_url": doc["source_url"],
            "content_hash": doc["content_hash"],
            "version": version,
            "status": "active",
            "chunk_count": 0,
            **{field: doc.get(field) for field in ("title", "platform", "site", "doc_type", "lang",
                                                   "effective_date", "expire_date")},
        }
        self.rows.append(row)
        self.next_id += 1
        return row["id"]

    def reactivate(self, row_id: int, doc: dict, version: int) -> None:
        """重新激活。"""
        row = self._get(row_id)
        row.update(
            source_url=doc["source_url"], content_hash=doc["content_hash"], version=version,
            status="active",
            **{field: doc.get(field) for field in ("title", "platform", "site", "doc_type", "lang",
                                                   "effective_date", "expire_date")},
        )

    def update_meta(self, row_id: int, doc: dict) -> None:
        """只更新元数据。"""
        row = self._get(row_id)
        row.update({field: doc.get(field) for field in ("title", "platform", "site", "doc_type", "lang",
                                                        "effective_date", "expire_date")})

    def set_status(self, row_id: int, status: str, expire_date: str | None = None) -> None:
        """改状态。"""
        row = self._get(row_id)
        row["status"] = status
        if expire_date is not None:
            row["expire_date"] = expire_date

    def mark_ingested(self, row_id: int, chunk_count: int) -> None:
        """写片段数。"""
        self._get(row_id)["chunk_count"] = chunk_count

    def active_chunk_sum(self, collection: str) -> int:
        """active 行的 chunk_count 之和。"""
        return sum(row["chunk_count"] for row in self.rows
                   if row["collection"] == collection and row["status"] == "active")

    def commit(self) -> None:
        """记录快照。"""
        self._snapshot = self._copy()

    def rollback(self) -> None:
        """回到上次提交。"""
        self.rows = [dict(row) for row in self._snapshot]

    def _get(self, row_id: int) -> dict:
        for row in self.rows:
            if row["id"] == row_id:
                return row
        raise KeyError(row_id)


class FakeRedis:
    """只实现缓存失效需要的方法。"""

    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.sets: dict[str, set[str]] = {}

    def smembers(self, key):
        """集合成员。"""
        return set(self.sets.get(key, set()))

    def delete(self, *keys):
        """删除 key。"""
        removed = 0
        for key in keys:
            if key in self.values:
                del self.values[key]
                removed += 1
            if key in self.sets:
                del self.sets[key]
                removed += 1
        return removed


def make_doc(doc_key: str, url: str, content_hash: str, **overrides) -> dict:
    """构造 documents 的一行。"""
    doc = {
        "doc_key": doc_key,
        "collection": "xm_compliance",
        "source_url": url,
        "title": f"文档{doc_key}",
        "platform": None,
        "site": "DE",
        "category": None,
        "doc_type": "regulation",
        "lang": "zh",
        "effective_date": "2026-01-01",
        "expire_date": None,
        "content_hash": content_hash,
        "fetched_at": "2026-10-31T10:00:00+08:00",
        "text": "正文",
        "question": None,
        "answer": None,
    }
    doc.update(overrides)
    return doc


def make_chunks(doc_key: str, count: int = 2) -> list[dict]:
    """构造该文档的片段。"""
    return [
        {
            "chunk_id": f"{doc_key}-{index:04d}",
            "doc_key": doc_key,
            "chunk_index": index,
            "collection": "xm_compliance",
            "platform": None,
            "site": "DE",
            "category": None,
            "doc_type": "regulation",
            "lang": "zh",
            "effective_date": "2026-01-01",
            "expire_date": None,
            "source_url": f"https://example.com/{doc_key}",
            "title_path": "法规 > 第一条",
            "text": f"正文{index}",
            "question": None,
            "answer": None,
            "token_count": 10,
        }
        for index in range(count)
    ]


def factory_with_counter():
    """返回 (工厂函数, 创建出来的 embedder 列表)。"""
    created: list[FakeEmbedder] = []

    def factory() -> FakeEmbedder:
        embedder = FakeEmbedder()
        created.append(embedder)
        return embedder

    return factory, created


def run(actions, chunks_by_doc, *, store, milvus, redis, expire_missing=False, factory=None):
    """跑一次 apply_plan。"""
    factory = factory or factory_with_counter()[0]
    return apply_plan(
        "xm_compliance", actions, chunks_by_doc,
        embedder_factory=factory, milvus=milvus, store=store, redis_client=redis,
        expire_missing=expire_missing,
    )


def test_first_ingest() -> None:
    """首次入库：全是 new，对账一致。"""
    store, milvus, redis = FakeStore(), FakeMilvus(), FakeRedis()
    doc = make_doc("d1", "https://example.com/1", "h1")
    chunks = {"d1": make_chunks("d1")}

    actions = plan_collection("xm_compliance", [doc], store)
    assert [action.kind for action in actions] == ["new"]

    stats = run(actions, chunks, store=store, milvus=milvus, redis=redis)

    assert stats.actions["new"] == 1
    assert stats.chunks_written == 2
    assert len(store.rows) == 1 and store.rows[0]["chunk_count"] == 2
    assert len(milvus.rows("xm_compliance")) == 2
    assert reconcile("xm_compliance", [doc], chunks, actions, milvus=milvus, store=store) == []


def test_second_run_skips_without_loading_model() -> None:
    """再跑一次全是 skip，且不加载模型。"""
    store, milvus, redis = FakeStore(), FakeMilvus(), FakeRedis()
    doc = make_doc("d1", "https://example.com/1", "h1")
    chunks = {"d1": make_chunks("d1")}
    run(plan_collection("xm_compliance", [doc], store), chunks, store=store, milvus=milvus, redis=redis)

    actions = plan_collection("xm_compliance", [doc], store)
    assert [action.kind for action in actions] == ["skip"]

    factory, created = factory_with_counter()
    stats = run(actions, chunks, store=store, milvus=milvus, redis=redis, factory=factory)
    assert stats.actions["skip"] == 1
    assert created == []


def test_changed_document_creates_new_version() -> None:
    """改正文：旧行 superseded、新行 v2、Milvus 里没有旧片段、旧缓存被失效。"""
    store, milvus, redis = FakeStore(), FakeMilvus(), FakeRedis()
    old_doc = make_doc("d1", "https://example.com/1", "h1")
    old_chunks = {"d1": make_chunks("d1")}
    run(plan_collection("xm_compliance", [old_doc], store), old_chunks, store=store, milvus=milvus, redis=redis)
    old_id = store.rows[0]["id"]
    redis.sets[f"xm:cache:bydoc:{old_id}"] = {"c1"}
    redis.values["xm:cache:c1"] = "1"

    new_doc = make_doc("d1", "https://example.com/1", "h2", text="新正文")
    actions = plan_collection("xm_compliance", [new_doc], store)
    assert [action.kind for action in actions] == ["changed"]

    factory, created = factory_with_counter()
    stats = run(actions, {"d1": make_chunks("d1")}, store=store, milvus=milvus, redis=redis, factory=factory)

    assert stats.cache_invalidated == 1
    assert len(created[0].texts) == 2  # 只编码这一篇的片段
    assert len(store.rows) == 2
    assert store.rows[0]["status"] == "superseded"
    assert store.rows[1]["status"] == "active" and store.rows[1]["version"] == 2
    assert all(row["doc_id"] == store.rows[1]["id"] for row in milvus.rows("xm_compliance"))
    assert reconcile("xm_compliance", [new_doc], {"d1": make_chunks("d1")}, actions,
                     milvus=milvus, store=store) == []


def test_change_back_reactivates_old_row() -> None:
    """改回旧内容：重新激活旧行，不新插行，version 继续加。"""
    store, milvus, redis = FakeStore(), FakeMilvus(), FakeRedis()
    doc_v1 = make_doc("d1", "https://example.com/1", "h1")
    chunks = {"d1": make_chunks("d1")}
    run(plan_collection("xm_compliance", [doc_v1], store), chunks, store=store, milvus=milvus, redis=redis)

    doc_v2 = make_doc("d1", "https://example.com/1", "h2")
    run(plan_collection("xm_compliance", [doc_v2], store), chunks, store=store, milvus=milvus, redis=redis)
    assert len(store.rows) == 2

    actions = plan_collection("xm_compliance", [doc_v1], store)
    assert actions[0].kind == "changed" and actions[0].reuse is not None

    run(actions, chunks, store=store, milvus=milvus, redis=redis)
    assert len(store.rows) == 2  # 没有新插行
    reactivated = store.rows[0]
    assert reactivated["status"] == "active" and reactivated["version"] == 3
    assert store.rows[1]["status"] == "superseded"
    assert reconcile("xm_compliance", [doc_v1], chunks, actions, milvus=milvus, store=store) == []


def test_meta_change_keeps_doc_id() -> None:
    """只改 expire_date：doc_id 不变，Milvus 里的日期更新。"""
    store, milvus, redis = FakeStore(), FakeMilvus(), FakeRedis()
    doc = make_doc("d1", "https://example.com/1", "h1")
    chunks = {"d1": make_chunks("d1")}
    run(plan_collection("xm_compliance", [doc], store), chunks, store=store, milvus=milvus, redis=redis)
    doc_id = store.rows[0]["id"]

    changed = make_doc("d1", "https://example.com/1", "h1", expire_date="2027-01-01")
    actions = plan_collection("xm_compliance", [changed], store)
    assert [action.kind for action in actions] == ["meta"]

    run(actions, chunks, store=store, milvus=milvus, redis=redis)

    assert store.rows[0]["id"] == doc_id
    assert all(row["doc_id"] == doc_id for row in milvus.rows("xm_compliance"))
    assert all(row["expire_date"] == 20270101 for row in milvus.rows("xm_compliance"))


def test_missing_document_warn_then_expire() -> None:
    """jsonl 里删掉一篇：默认只警告，加 expire_missing 才下线。"""
    store, milvus, redis = FakeStore(), FakeMilvus(), FakeRedis()
    doc = make_doc("d1", "https://example.com/1", "h1")
    chunks = {"d1": make_chunks("d1")}
    run(plan_collection("xm_compliance", [doc], store), chunks, store=store, milvus=milvus, redis=redis)

    actions = plan_collection("xm_compliance", [], store)
    assert [action.kind for action in actions] == ["missing"]

    run(actions, {}, store=store, milvus=milvus, redis=redis)
    assert store.rows[0]["status"] == "active"
    assert len(milvus.rows("xm_compliance")) == 2  # 片段还在

    run(actions, {}, store=store, milvus=milvus, redis=redis, expire_missing=True)
    assert store.rows[0]["status"] == "expired"
    assert milvus.rows("xm_compliance") == []
    assert reconcile("xm_compliance", [], {}, actions, milvus=milvus, store=store) == []


def test_duplicate_content_is_not_written() -> None:
    """两个网址正文相同：第二篇是 duplicate，不写入。"""
    store, milvus, redis = FakeStore(), FakeMilvus(), FakeRedis()
    doc_a = make_doc("d1", "https://example.com/1", "h1")
    chunks = {"d1": make_chunks("d1")}
    run(plan_collection("xm_compliance", [doc_a], store), chunks, store=store, milvus=milvus, redis=redis)

    doc_b = make_doc("d2", "https://example.com/2", "h1")
    actions = plan_collection("xm_compliance", [doc_a, doc_b], store)
    kinds = {action.doc["doc_key"]: action.kind for action in actions}
    assert kinds["d2"] == "duplicate"

    run(actions, chunks, store=store, milvus=milvus, redis=redis)
    assert len(store.rows) == 1
    assert len(milvus.rows("xm_compliance")) == 2


def test_rebuild_rewrites_with_same_doc_id() -> None:
    """rebuild=True：全部 rewrite，doc_id 不变。"""
    store, milvus, redis = FakeStore(), FakeMilvus(), FakeRedis()
    doc = make_doc("d1", "https://example.com/1", "h1")
    chunks = {"d1": make_chunks("d1")}
    run(plan_collection("xm_compliance", [doc], store), chunks, store=store, milvus=milvus, redis=redis)
    doc_id = store.rows[0]["id"]

    actions = plan_collection("xm_compliance", [doc], store, rebuild=True)
    assert [action.kind for action in actions] == ["rewrite"]

    run(actions, chunks, store=store, milvus=milvus, redis=redis)
    assert len(store.rows) == 1 and store.rows[0]["id"] == doc_id
    assert all(row["doc_id"] == doc_id for row in milvus.rows("xm_compliance"))


def test_milvus_failure_rolls_back_mysql() -> None:
    """Milvus 写入失败：MySQL 回滚，不留新行。"""
    store, redis = FakeStore(), FakeRedis()
    milvus = FakeMilvus(fail_upsert=True)
    doc = make_doc("d1", "https://example.com/1", "h1")
    actions = plan_collection("xm_compliance", [doc], store)

    with pytest.raises(RuntimeError):
        run(actions, {"d1": make_chunks("d1")}, store=store, milvus=milvus, redis=redis)

    assert store.rows == []


def test_plan_duplicate_when_hash_hits_active_row() -> None:
    """有 active 行但正文改成另一篇 active 文档的内容：duplicate，旧行保持。"""
    store, milvus, redis = FakeStore(), FakeMilvus(), FakeRedis()
    doc_a = make_doc("d1", "https://example.com/1", "h1")
    doc_b = make_doc("d2", "https://example.com/2", "h2")
    chunks = {"d1": make_chunks("d1"), "d2": make_chunks("d2")}
    run(plan_collection("xm_compliance", [doc_a], store), chunks, store=store, milvus=milvus, redis=redis)
    run(plan_collection("xm_compliance", [doc_a, doc_b], store), chunks, store=store, milvus=milvus, redis=redis)

    changed_a = make_doc("d1", "https://example.com/1", "h2")
    actions = plan_collection("xm_compliance", [changed_a, doc_b], store)
    kinds = {action.doc["doc_key"]: action.kind for action in actions}
    assert kinds["d1"] == "duplicate"


def test_rebuild_still_versions_changed_text() -> None:
    """rebuild 时正文变了的文档仍走 changed：新行 v2、旧行 superseded，而不是在旧行上重写。"""
    store, milvus, redis = FakeStore(), FakeMilvus(), FakeRedis()
    doc = make_doc("d1", "https://example.com/1", "h1")
    run(plan_collection("xm_compliance", [doc], store), {"d1": make_chunks("d1")},
        store=store, milvus=milvus, redis=redis)

    changed = make_doc("d1", "https://example.com/1", "h2")
    actions = plan_collection("xm_compliance", [changed], store, rebuild=True)
    assert [action.kind for action in actions] == ["changed"]
    run(actions, {"d1": make_chunks("d1")}, store=store, milvus=milvus, redis=redis)
    active = [row for row in store.rows if row["status"] == "active"]
    assert len(active) == 1 and active[0]["content_hash"] == "h2" and active[0]["version"] == 2


def test_same_text_twice_in_one_batch() -> None:
    """同一批里两个网址正文相同：第二篇 duplicate，不会撞唯一索引。"""
    store, milvus, redis = FakeStore(), FakeMilvus(), FakeRedis()
    doc_a = make_doc("d1", "https://example.com/1", "h1")
    doc_b = make_doc("d2", "https://example.com/2", "h1")
    chunks = {"d1": make_chunks("d1"), "d2": make_chunks("d2")}
    actions = plan_collection("xm_compliance", [doc_a, doc_b], store)
    assert [action.kind for action in actions] == ["new", "duplicate"]
    run(actions, chunks, store=store, milvus=milvus, redis=redis)
    assert len(store.rows) == 1
    assert reconcile("xm_compliance", [doc_a, doc_b], chunks, actions, milvus=milvus, store=store) == []
