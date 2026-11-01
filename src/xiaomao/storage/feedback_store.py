"""反馈存储。"""

from __future__ import annotations

from pymysql.cursors import DictCursor

from xiaomao.storage.db import MySQLPool


class FeedbackStore:
    """`feedback` 表读写。"""

    def __init__(self, pool: MySQLPool) -> None:
        self.pool = pool

    def upsert(
        self,
        user_id: int,
        message_id: int,
        rating: int,
        reason: str | None,
        comment: str | None,
    ) -> int:
        """(message_id, user_id) 唯一；改主意时覆盖并把 status 重置为 pending。"""
        with self.pool.connection() as conn, conn.cursor(DictCursor) as cursor:
            cursor.execute(
                "INSERT INTO feedback (message_id, user_id, rating, reason, comment, status) "
                "VALUES (%s, %s, %s, %s, %s, 'pending') "
                "ON DUPLICATE KEY UPDATE rating=VALUES(rating), reason=VALUES(reason), "
                "comment=VALUES(comment), status='pending'",
                (message_id, user_id, rating, reason, comment),
            )
            feedback_id = int(cursor.lastrowid)
            if feedback_id == 0:  # ON DUPLICATE 时 lastrowid 可能是 0
                cursor.execute(
                    "SELECT id FROM feedback WHERE message_id=%s AND user_id=%s",
                    (message_id, user_id),
                )
                feedback_id = int(cursor.fetchone()["id"])
        return feedback_id

    def list_pending(self, limit: int = 50) -> list[dict]:
        """运营复核：待处理的点踩，按时间倒序。"""
        with self.pool.connection() as conn, conn.cursor(DictCursor) as cursor:
            cursor.execute(
                "SELECT f.id, f.message_id, f.user_id, f.rating, f.reason, f.comment, f.created_at, "
                "m.content AS answer, m.citations, t.query_raw, t.query_rewritten, t.intent, t.strategy, "
                "t.cache_hit, t.evidence_sufficient, t.rerank_scores, t.decisions, t.chunk_ids "
                "FROM feedback f "
                "JOIN message m ON m.id = f.message_id "
                "LEFT JOIN qa_trace t ON t.message_id = f.message_id "
                "WHERE f.status='pending' AND f.rating=-1 "
                "ORDER BY f.created_at DESC LIMIT %s",
                (int(limit),),
            )
            return list(cursor.fetchall() or [])
