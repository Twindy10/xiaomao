"""把 Redis 里的当日用量写回 MySQL。写绝对值，重复同步幂等。"""

from __future__ import annotations

import logging
import threading
import time

from xiaomao.storage.db import MySQLPool
from xiaomao.user.limits import DIRTY_KEY, day_to_sql, quota_key

logger = logging.getLogger("xiaomao.usage")
SYNC_INTERVAL = 300


def _decode(value) -> str:
    """Redis 可能返回 bytes。"""
    return value.decode() if isinstance(value, bytes) else str(value)


def sync_usage(redis_client, pool: MySQLPool) -> int:
    """同步所有待同步成员，返回成功条数。"""
    synced = 0
    members = redis_client.smembers(DIRTY_KEY) or set()
    for member in members:
        text = _decode(member)
        try:
            user_id_text, day = text.split(":", 1)
            user_id = int(user_id_text)
        except ValueError:
            logger.warning("待同步成员格式不对：%s", text)
            redis_client.srem(DIRTY_KEY, member)
            continue

        # 先移出待同步集合再读计数：读完之后又来的请求会把它重新加进来，下一轮再同步。
        # 反过来（读完再移除）的话，读和移除之间的那次请求就永远不会写回 MySQL
        redis_client.srem(DIRTY_KEY, member)
        key = quota_key(user_id, day)
        if not redis_client.exists(key):
            continue  # Hash 已过期，没有可同步的数据

        values = redis_client.hgetall(key) or {}
        used = int(values.get("q", values.get(b"q", 0)) or 0)
        tokens = int(values.get("tokens", values.get(b"tokens", 0)) or 0)
        try:
            with pool.connection() as conn, conn.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO usage_daily (user_id, day, query_count, tokens) "
                    "VALUES (%s, %s, %s, %s) "
                    "ON DUPLICATE KEY UPDATE query_count=VALUES(query_count), tokens=VALUES(tokens)",
                    (user_id, day_to_sql(day), used, tokens),
                )
        except Exception:
            logger.warning("同步用量失败：%s", text, exc_info=True)
            redis_client.sadd(DIRTY_KEY, member)  # 放回去，下一轮重试
            continue
        synced += 1
    return synced


def start_sync_thread(redis_client, pool: MySQLPool, stop: threading.Event) -> threading.Thread:
    """后台线程：每 5 分钟同步一次，异常只记日志。"""

    def run() -> None:
        while not stop.wait(SYNC_INTERVAL):
            try:
                count = sync_usage(redis_client, pool)
                if count:
                    logger.info("用量同步完成：%s 条", count)
            except Exception:
                logger.warning("用量同步异常", exc_info=True)
        try:
            sync_usage(redis_client, pool)
        except Exception:
            logger.warning("退出前最后一次同步失败", exc_info=True)

    thread = threading.Thread(target=run, name="usage-sync", daemon=True)
    thread.start()
    return thread
