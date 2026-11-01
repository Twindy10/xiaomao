"""对话与链路日志的 MySQL 存储。所有查询都带 user_id。"""

from __future__ import annotations

import json
from typing import Any

from pymysql.cursors import DictCursor

from xiaomao.ingest.kb_store import connect_mysql
from xiaomao.storage.db import MySQLPool

EXTRA_KEYS = (
    "query_en", "subqueries", "hypothetical", "filters",
    "evidence_by", "llm_tokens", "cache", "intent_overridden",
    "cache_write", "cache_write_error", "cache_write_id", "rerank_tokens",
    "rerank_skipped", "decision_error",
)


class ChatStore:
    """连接从连接池借，一次方法调用一个连接。"""

    def __init__(self, pool: MySQLPool | None = None, connect=None) -> None:
        self.pool = pool or MySQLPool(connect=connect or connect_mysql)

    def _run(self, sql: str, params: tuple, *, mode: str = "id") -> Any:
        """执行一条 SQL；mode 决定返回自增 id、影响行数、一行还是多行。"""
        with self.pool.connection() as conn, conn.cursor(DictCursor) as cursor:
            cursor.execute(sql, params)
            if mode == "one":
                return cursor.fetchone()
            if mode == "all":
                return cursor.fetchall()
            if mode == "rowcount":
                return cursor.rowcount
            return cursor.lastrowid

    # ------------------------------------------------------------------
    # 会话
    # ------------------------------------------------------------------

    def create_conversation(self, user_id: int, title: str) -> int:
        """新建会话，标题取首问前 30 字。"""
        return int(self._run(
            "INSERT INTO conversation (user_id, title) VALUES (%s, %s)",
            (user_id, title[:30]),
        ))

    def get_conversation(self, user_id: int, conversation_id: int) -> dict | None:
        """取会话；不是这个用户的或已删除返回 None。"""
        return self._run(
            "SELECT * FROM conversation WHERE id=%s AND user_id=%s AND is_deleted=0",
            (conversation_id, user_id), mode="one",
        )

    def list_conversations(
        self,
        user_id: int,
        *,
        limit: int = 20,
        before: str | None = None,
        max_history_days: int | None = None,
    ) -> list[dict]:
        """会话列表：按 updated_at 倒序，支持保留天数和游标分页。"""
        sql = (
            "SELECT id, title, updated_at FROM conversation "
            "WHERE user_id=%s AND is_deleted=0"
        )
        params: list[Any] = [user_id]
        if max_history_days is not None:
            sql += " AND updated_at >= NOW(3) - INTERVAL %s DAY"
            params.append(int(max_history_days))
        if before:
            sql += " AND updated_at < %s"
            params.append(before)
        sql += " ORDER BY updated_at DESC, id DESC LIMIT %s"
        params.append(int(limit))
        return list(self._run(sql, tuple(params), mode="all") or [])

    def delete_conversation(self, user_id: int, conversation_id: int) -> bool:
        """软删除；不是这个用户的返回 False。"""
        affected = self._run(
            "UPDATE conversation SET is_deleted=1 WHERE id=%s AND user_id=%s AND is_deleted=0",
            (conversation_id, user_id), mode="rowcount",
        )
        return bool(affected)

    # ------------------------------------------------------------------
    # 消息
    # ------------------------------------------------------------------

    def add_message(
        self,
        user_id: int,
        conversation_id: int,
        role: str,
        content: str,
        citations: list[dict] | None = None,
    ) -> int:
        """写一条消息，并在同一个事务里更新会话的 updated_at。"""
        with self.pool.connection() as conn, conn.cursor(DictCursor) as cursor:
            cursor.execute(
                "INSERT INTO message (conversation_id, user_id, role, content, citations) "
                "VALUES (%s, %s, %s, %s, %s)",
                (
                    conversation_id, user_id, role, content,
                    json.dumps(citations, ensure_ascii=False) if citations is not None else None,
                ),
            )
            message_id = int(cursor.lastrowid)
            cursor.execute(
                "UPDATE conversation SET updated_at=NOW(3) WHERE id=%s AND user_id=%s",
                (conversation_id, user_id),
            )
        return message_id

    def recent_messages(self, user_id: int, conversation_id: int, limit: int = 6) -> list[dict]:
        """最近几条消息，按时间正序返回。"""
        rows = self._run(
            "SELECT id, role, content, citations, created_at FROM message "
            "WHERE conversation_id=%s AND user_id=%s ORDER BY id DESC LIMIT %s",
            (conversation_id, user_id, int(limit)), mode="all",
        )
        return list(reversed(rows or []))

    def list_messages(self, user_id: int, conversation_id: int) -> list[dict]:
        """会话里的全部消息，带当前用户的反馈 rating。"""
        rows = self._run(
            "SELECT m.id, m.role, m.content, m.citations, m.created_at, f.rating "
            "FROM message m LEFT JOIN feedback f "
            "ON f.message_id = m.id AND f.user_id = %s "
            "WHERE m.conversation_id=%s AND m.user_id=%s ORDER BY m.id",
            (user_id, conversation_id, user_id), mode="all",
        )
        return list(rows or [])

    def get_message(self, user_id: int, message_id: int) -> dict | None:
        """按用户取一条消息。"""
        return self._run(
            "SELECT * FROM message WHERE id=%s AND user_id=%s",
            (message_id, user_id), mode="one",
        )

    # ------------------------------------------------------------------
    # 链路日志
    # ------------------------------------------------------------------

    def get_trace(self, user_id: int, message_id: int) -> dict | None:
        """按用户取一条 qa_trace。"""
        return self._run(
            "SELECT * FROM qa_trace WHERE message_id=%s AND user_id=%s",
            (message_id, user_id), mode="one",
        )

    def save_trace(self, user_id: int, message_id: int, query_raw: str, trace: dict) -> int:
        """把 trace 映射到 qa_trace 的列；其余字段并进 decisions._extra。"""
        decisions = dict(trace.get("decisions") or {})
        extra = {key: trace[key] for key in EXTRA_KEYS if key in trace}
        if extra:
            decisions["_extra"] = extra

        evidence_sufficient = trace.get("evidence_sufficient")
        if evidence_sufficient is None:
            evidence_value = None
        else:
            evidence_value = 1 if evidence_sufficient else 0

        latency = trace.get("latency_ms") or {}
        total_latency = int(round(float(latency.get("total") or 0)))
        strategy = trace.get("strategy")
        if strategy in ("none", None):
            strategy = None

        return int(self._run(
            "INSERT INTO qa_trace (message_id, user_id, query_raw, query_rewritten, intent, strategy, "
            "decisions, cache_hit, chunk_ids, rerank_scores, evidence_sufficient, backtracked, "
            "llm_model, prompt_tokens, completion_tokens, latency_ms, total_latency_ms) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (
                message_id,
                user_id,
                query_raw,
                trace.get("query_rewritten") or query_raw,
                trace.get("intent"),
                strategy,
                json.dumps(decisions, ensure_ascii=False) if decisions else None,
                1 if trace.get("cache_hit") else 0,
                json.dumps(trace.get("chunk_ids") or [], ensure_ascii=False),
                json.dumps(trace.get("rerank_scores") or [], ensure_ascii=False),
                evidence_value,
                1 if int(trace.get("backtracked") or 0) > 0 else 0,
                trace.get("llm_model"),
                int(trace.get("prompt_tokens") or 0),
                int(trace.get("completion_tokens") or 0),
                json.dumps(latency, ensure_ascii=False),
                total_latency,
            ),
        ))
