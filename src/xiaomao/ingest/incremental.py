"""增量入库：先规划每篇文档做什么，再逐篇执行并核对。"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Callable

from xiaomao.ingest.embed import Embedder, Embedding, embed_input
from xiaomao.ingest.index import (
    chunk_to_row,
    count_rows,
    delete_doc,
    finish,
    upsert_rows,
)
from xiaomao.ingest.kb_store import KbDocumentStore, invalidate_doc_cache, meta_changed

logger = logging.getLogger(__name__)

WRITE_KINDS = ("new", "reactivate", "changed", "meta", "rewrite")


@dataclass
class Action:
    """一篇文档要做的动作。"""

    kind: str
    doc: dict | None
    old: dict | None = None
    reuse: dict | None = None


@dataclass
class RunStats:
    """一次入库的统计。"""

    actions: dict[str, int] = field(default_factory=dict)
    chunks_written: int = 0
    embedded: int = 0
    encode_seconds: float = 0.0
    write_seconds: float = 0.0
    cache_invalidated: int = 0


def plan_collection(
    collection: str,
    documents: list[dict],
    store: KbDocumentStore,
    rebuild: bool = False,
) -> list[Action]:
    """对比 jsonl 与 kb_document，算出每篇文档的动作。"""
    actions: list[Action] = []
    seen_urls: set[str] = set()
    claimed: dict[str, str] = {}  # 本次已经分给某篇文档的 content_hash -> source_url

    for doc in documents:
        seen_urls.add(doc["source_url"])
        active = store.find_active(collection, doc["source_url"])

        # 同一批里两个网址正文相同：库里都还没有，不拦下来第二篇会撞唯一索引
        if doc["content_hash"] in claimed:
            actions.append(Action("duplicate", doc, old=active,
                                  reuse={"source_url": claimed[doc["content_hash"]]}))
            continue
        claimed[doc["content_hash"]] = doc["source_url"]

        if active is None:
            by_hash = store.find_by_hash(collection, doc["content_hash"])
            if by_hash is not None and by_hash.get("status") == "active":
                # 同一集合里另一篇网址的正文完全相同，撞唯一索引
                actions.append(Action("duplicate", doc, reuse=by_hash))
            elif by_hash is not None:
                actions.append(Action("reactivate", doc, reuse=by_hash))
            else:
                actions.append(Action("new", doc))
            continue

        if active["content_hash"] == doc["content_hash"]:
            if meta_changed(active, doc):
                actions.append(Action("meta", doc, old=active))
            else:
                # 重建时 Milvus 已清空，没变的也要重写；正文变了的仍走 changed，版本号照常递增
                actions.append(Action("rewrite" if rebuild else "skip", doc, old=active))
            continue

        by_hash = store.find_by_hash(collection, doc["content_hash"])
        if by_hash is not None and by_hash["id"] != active["id"] and by_hash.get("status") == "active":
            actions.append(Action("duplicate", doc, old=active, reuse=by_hash))
        elif by_hash is not None and by_hash["id"] != active["id"]:
            actions.append(Action("changed", doc, old=active, reuse=by_hash))
        else:
            actions.append(Action("changed", doc, old=active))

    for row in store.list_active(collection):
        if row["source_url"] not in seen_urls:
            actions.append(Action("missing", None, old=row))
    return actions


def _count(stats: RunStats, kind: str) -> None:
    stats.actions[kind] = stats.actions.get(kind, 0) + 1


def apply_plan(
    collection: str,
    actions: list[Action],
    chunks_by_doc: dict[str, list[dict]],
    *,
    embedder_factory: Callable[[], Embedder],
    milvus,
    store: KbDocumentStore,
    redis_client,
    expire_missing: bool = False,
) -> RunStats:
    """执行规划：先统一编码，再逐篇写 MySQL 和 Milvus。"""
    stats = RunStats()
    todo = [action for action in actions if action.kind in WRITE_KINDS]
    for action in todo:
        if not chunks_by_doc.get(action.doc["doc_key"]):
            raise ValueError(f"{action.doc['doc_key']} 没有片段，不能登记 active 行")

    embeddings: dict[str, list[Embedding]] = {}
    if todo:
        embedder = embedder_factory()
        texts: list[str] = []
        owners: list[str] = []
        for action in todo:
            for chunk in chunks_by_doc[action.doc["doc_key"]]:
                texts.append(embed_input(chunk))
                owners.append(action.doc["doc_key"])
        before = embedder.encode_seconds  # 同一个 embedder 跨 collection 共用，记的是累计值
        vectors = embedder.encode(texts)
        for doc_key, vector in zip(owners, vectors):
            embeddings.setdefault(doc_key, []).append(vector)
        stats.embedded = len(texts)
        stats.encode_seconds = embedder.encode_seconds - before

    started = time.perf_counter()
    for action in actions:
        if action.kind == "skip":
            _count(stats, action.kind)
            continue
        if action.kind == "duplicate":
            target = (action.reuse or {}).get("source_url", "")
            print(f"[WARN] 正文与已有文档重复，跳过：{action.doc['source_url']}（重复于 {target}）")
            _count(stats, action.kind)
            continue
        if action.kind == "missing":
            _handle_missing(action, collection, stats, milvus, store, redis_client, expire_missing)
            continue

        _write_one(action, collection, chunks_by_doc, embeddings, stats, milvus, store, redis_client)

    finish(milvus, collection)
    stats.write_seconds = time.perf_counter() - started
    return stats


def _handle_missing(
    action: Action,
    collection: str,
    stats: RunStats,
    milvus,
    store: KbDocumentStore,
    redis_client,
    expire_missing: bool,
) -> None:
    """jsonl 里已经没有的文档：默认只警告，加 --expire-missing 才下线。"""
    row = action.old or {}
    if not expire_missing:
        print(f"[WARN] 文档已从 jsonl 消失，仍在库里：{row.get('title')}（{row.get('source_url')}）")
        _count(stats, "missing")
        return

    delete_doc(milvus, collection, row["id"])
    store.set_status(row["id"], "expired", expire_date=date.today().strftime("%Y-%m-%d"))
    store.mark_ingested(row["id"], 0)
    store.commit()
    stats.cache_invalidated += invalidate_doc_cache(redis_client, row["id"])
    _count(stats, "missing_expired")


def _with_doc_meta(chunk: dict, doc: dict) -> dict:
    """片段里的文档级字段以 documents.jsonl 为准（只改元数据时 chunks 可能还没重跑）。"""
    merged = dict(chunk)
    for field in ("platform", "site", "category", "doc_type", "lang",
                  "effective_date", "expire_date", "source_url"):
        if field in doc:
            merged[field] = doc[field]
    return merged


def _write_one(
    action: Action,
    collection: str,
    chunks_by_doc: dict[str, list[dict]],
    embeddings: dict[str, list[Embedding]],
    stats: RunStats,
    milvus,
    store: KbDocumentStore,
    redis_client,
) -> None:
    """写一篇文档：MySQL 登记 → Milvus 删除旧片段并写入 → 提交 → 失效缓存。"""
    doc = action.doc
    chunks = chunks_by_doc[doc["doc_key"]]
    vectors = embeddings[doc["doc_key"]]
    old = action.old
    doc_id: int

    try:
        # 1) MySQL 登记
        if action.kind == "new":
            doc_id = store.insert(doc, 1)
        elif action.kind == "reactivate":
            reuse = action.reuse or {}
            doc_id = int(reuse["id"])
            store.reactivate(doc_id, doc, int(reuse["version"]) + 1)
        elif action.kind in ("meta", "rewrite"):
            doc_id = int(old["id"])
            store.update_meta(doc_id, doc)
        else:  # changed
            reuse = action.reuse
            if reuse is not None:
                doc_id = int(reuse["id"])
                store.reactivate(doc_id, doc, int(old["version"]) + 1)
            else:
                doc_id = store.insert(doc, int(old["version"]) + 1)

        # 2) Milvus：先删旧片段再写入
        if action.kind == "changed" and old is not None:
            delete_doc(milvus, collection, int(old["id"]))
        elif action.kind in ("meta", "rewrite", "reactivate"):
            delete_doc(milvus, collection, doc_id)

        rows = [
            chunk_to_row(_with_doc_meta(chunk, doc), doc_id, vector)
            for chunk, vector in zip(chunks, vectors)
        ]
        upsert_rows(milvus, collection, rows)
        stats.chunks_written += len(rows)

        # 3) 收尾：片段数、旧版本标记、提交
        store.mark_ingested(doc_id, len(rows))
        if action.kind == "changed" and old is not None:
            store.set_status(int(old["id"]), "superseded")
        store.commit()
    except Exception:
        store.rollback()
        raise

    # 4) 缓存失效
    if action.kind in ("changed", "meta") and old is not None:
        stats.cache_invalidated += invalidate_doc_cache(redis_client, int(old["id"]))
    _count(stats, action.kind)


def reconcile(
    collection: str,
    documents: list[dict],
    chunks_by_doc: dict[str, list[dict]],
    actions: list[Action],
    *,
    milvus,
    store: KbDocumentStore,
) -> list[str]:
    """核对三个数，返回不一致的说明（空列表表示一致）。"""
    problems: list[str] = []
    milvus_count = count_rows(milvus, collection)
    db_sum = store.active_chunk_sum(collection)
    if milvus_count != db_sum:
        problems.append(f"Milvus 条数 {milvus_count} != kb_document active 之和 {db_sum}")

    skipped = {action.doc["doc_key"] for action in actions if action.kind == "duplicate"}
    jsonl_count = sum(
        len(chunks_by_doc.get(doc["doc_key"], []))
        for doc in documents
        if doc["doc_key"] not in skipped
    )
    # 只在 jsonl 里还存在 active 的 missing 文档才要扣掉
    doc_urls = {doc["source_url"] for doc in documents}
    missing_sum = sum(
        int(row.get("chunk_count") or 0)
        for row in store.list_active(collection)
        if row["source_url"] not in doc_urls
    )
    if jsonl_count != db_sum - missing_sum:
        problems.append(
            f"jsonl 片段数 {jsonl_count} != active 之和 {db_sum} 减去 missing 的 {missing_sum}"
        )
    return problems
