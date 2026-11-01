"""MySQL `kb_document` 存储层，以及 Redis 的缓存失效和入库锁。"""

from __future__ import annotations

import os
import secrets
from datetime import date
from pathlib import Path
from typing import Any

import pymysql
from dotenv import load_dotenv
from pymysql.cursors import DictCursor

ROOT_DIR = Path(__file__).resolve().parents[3]  # src/xiaomao/ingest/kb_store.py -> 项目根
DOC_FIELDS = ("title", "platform", "site", "doc_type", "lang", "effective_date", "expire_date")
VALID_STATUS = {"active", "expired", "superseded"}

# 只有 token 还是自己的才删，避免误删别人的锁
RELEASE_LOCK_SCRIPT = """
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("del", KEYS[1])
end
return 0
"""


def _as_text(value: Any) -> str | None:
    """把 MySQL 的 DATE / None / 字符串统一成可比较的形式。"""
    if value is None:
        return None
    if isinstance(value, date):
        return value.strftime("%Y-%m-%d")
    text = str(value)
    return text or None


def meta_changed(row: dict, doc: dict) -> bool:
    """比较文档级元数据是否变化；DATE 与字符串、NULL 与 None 视为相同。"""
    for field in DOC_FIELDS:
        if _as_text(row.get(field)) != _as_text(doc.get(field)):
            return True
    return False


class KbDocumentStore:
    """`kb_document` 的读写；写方法不自动 commit。"""

    def __init__(self, conn: pymysql.connections.Connection) -> None:
        self.conn = conn
        self.conn.autocommit(False)

    def _cursor(self) -> DictCursor:
        return self.conn.cursor(DictCursor)

    def find_active(self, collection: str, source_url: str) -> dict | None:
        """按 collection + source_url 找 active 的那行。"""
        with self._cursor() as cursor:
            cursor.execute(
                "SELECT * FROM kb_document WHERE collection=%s AND source_url=%s AND status='active' "
                "ORDER BY id DESC LIMIT 1",
                (collection, source_url),
            )
            return cursor.fetchone()

    def find_by_hash(self, collection: str, content_hash: str) -> dict | None:
        """按 collection + content_hash 找任意状态的一行。"""
        with self._cursor() as cursor:
            cursor.execute(
                "SELECT * FROM kb_document WHERE collection=%s AND content_hash=%s LIMIT 1",
                (collection, content_hash),
            )
            return cursor.fetchone()

    def list_active(self, collection: str) -> list[dict]:
        """列出某集合所有 active 行。"""
        with self._cursor() as cursor:
            cursor.execute(
                "SELECT * FROM kb_document WHERE collection=%s AND status='active' ORDER BY id",
                (collection,),
            )
            return list(cursor.fetchall())

    def insert(self, doc: dict, version: int) -> int:
        """插入一行 active 文档，返回自增 id。"""
        with self._cursor() as cursor:
            cursor.execute(
                "INSERT INTO kb_document (collection, source_url, title, platform, site, doc_type, lang, "
                "effective_date, expire_date, content_hash, version, status, chunk_count) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'active', 0)",
                (
                    doc["collection"], doc["source_url"], doc.get("title"), doc.get("platform"),
                    doc.get("site"), doc.get("doc_type"), doc.get("lang"), doc.get("effective_date"),
                    doc.get("expire_date"), doc["content_hash"], version,
                ),
            )
            return int(cursor.lastrowid)

    def reactivate(self, row_id: int, doc: dict, version: int) -> None:
        """把一行重新激活，并用最新文档更新元数据。"""
        with self._cursor() as cursor:
            cursor.execute(
                "UPDATE kb_document SET source_url=%s, title=%s, platform=%s, site=%s, doc_type=%s, "
                "lang=%s, effective_date=%s, expire_date=%s, version=%s, status='active' WHERE id=%s",
                (
                    doc["source_url"], doc.get("title"), doc.get("platform"), doc.get("site"),
                    doc.get("doc_type"), doc.get("lang"), doc.get("effective_date"),
                    doc.get("expire_date"), version, row_id,
                ),
            )

    def update_meta(self, row_id: int, doc: dict) -> None:
        """只更新文档级元数据。"""
        with self._cursor() as cursor:
            cursor.execute(
                "UPDATE kb_document SET title=%s, platform=%s, site=%s, doc_type=%s, lang=%s, "
                "effective_date=%s, expire_date=%s WHERE id=%s",
                (
                    doc.get("title"), doc.get("platform"), doc.get("site"), doc.get("doc_type"),
                    doc.get("lang"), doc.get("effective_date"), doc.get("expire_date"), row_id,
                ),
            )

    def set_status(self, row_id: int, status: str, expire_date: str | None = None) -> None:
        """改文档状态；给 expire_date 时一起更新。"""
        if status not in VALID_STATUS:
            raise ValueError(f"status 只能是 {VALID_STATUS}，收到 {status!r}")
        with self._cursor() as cursor:
            if expire_date is None:
                cursor.execute("UPDATE kb_document SET status=%s WHERE id=%s", (status, row_id))
            else:
                cursor.execute(
                    "UPDATE kb_document SET status=%s, expire_date=%s WHERE id=%s",
                    (status, expire_date, row_id),
                )

    def mark_ingested(self, row_id: int, chunk_count: int) -> None:
        """写片段数和入库时间。"""
        with self._cursor() as cursor:
            cursor.execute(
                "UPDATE kb_document SET chunk_count=%s, ingested_at=NOW(3) WHERE id=%s",
                (chunk_count, row_id),
            )

    def active_chunk_sum(self, collection: str) -> int:
        """某集合所有 active 行的 chunk_count 之和。"""
        with self._cursor() as cursor:
            cursor.execute(
                "SELECT COALESCE(SUM(chunk_count), 0) AS total FROM kb_document "
                "WHERE collection=%s AND status='active'",
                (collection,),
            )
            row = cursor.fetchone()
            return int(row["total"]) if row else 0

    def commit(self) -> None:
        """提交事务。"""
        self.conn.commit()

    def rollback(self) -> None:
        """回滚事务。"""
        self.conn.rollback()


def connect_mysql() -> pymysql.connections.Connection:
    """按 .env 连接 MySQL。"""
    load_dotenv(os.path.join(ROOT_DIR, ".env"), override=True)
    return pymysql.connect(
        host=os.environ["MYSQL_HOST"],
        port=int(os.environ["MYSQL_PORT"]),
        user=os.environ["MYSQL_USER"],
        password=os.environ["MYSQL_PASSWORD"],
        database=os.environ["MYSQL_DATABASE"],
        charset="utf8mb4",
        autocommit=False,
    )


def _decode(value: Any) -> str:
    """Redis 返回 bytes 时解码。"""
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def invalidate_doc_cache(redis_client, doc_id: int) -> int:
    """按文档失效语义缓存：删掉 bydoc 集合里登记的每个缓存，再删集合本身。"""
    index_key = f"xm:cache:bydoc:{doc_id}"
    members = redis_client.smembers(index_key) or set()
    removed = 0
    if members:
        keys = [f"xm:cache:{_decode(member)}" for member in members]
        removed = int(redis_client.delete(*keys) or 0)
    redis_client.delete(index_key)
    return removed


class IngestLock:
    """入库任务锁：同一个 collection 同时只允许一个任务。"""

    def __init__(self, redis_client, collection: str, ttl: int = 3600) -> None:
        self.redis = redis_client
        self.key = f"xm:lock:ingest:{collection}"
        self.ttl = ttl
        self.token = secrets.token_hex(8)

    def __enter__(self) -> "IngestLock":
        acquired = self.redis.set(self.key, self.token, nx=True, ex=self.ttl)
        if not acquired:
            raise RuntimeError(f"{self.key} 已被占用，有别的入库任务在跑")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            self.redis.eval(RELEASE_LOCK_SCRIPT, 1, self.key, self.token)
        except Exception:  # 释放失败不影响主流程，锁会自己过期
            pass
