"""限流和每日配额。检查和计数在同一个 Lua 脚本里。"""

from __future__ import annotations

import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from pymysql.cursors import DictCursor

from xiaomao.storage.db import MySQLPool

logger = logging.getLogger("xiaomao.limits")

DIRTY_KEY = "xm:usage:dirty"
QUOTA_TTL = 2 * 24 * 3600
RATE_TTL = 120
TZ = ZoneInfo("Asia/Shanghai")  # 按北京时间记"今天"

CONSUME_SCRIPT = """
local used = tonumber(redis.call('HGET', KEYS[1], 'q') or '0')
if used >= tonumber(ARGV[1]) then return {0, used} end
used = redis.call('HINCRBY', KEYS[1], 'q', 1)
redis.call('EXPIRE', KEYS[1], ARGV[2])
redis.call('SADD', KEYS[2], ARGV[3])
return {1, used}
"""


def utc_day() -> str:
    """UTC 日期 YYYYMMDD。"""
    return datetime.now(TZ).strftime("%Y%m%d")


def day_to_sql(day: str) -> str:
    """yyyyMMdd → YYYY-MM-DD。"""
    return f"{day[:4]}-{day[4:6]}-{day[6:8]}"


def quota_key(user_id: int, day: str) -> str:
    """配额 Hash 的 key。"""
    return f"xm:quota:{user_id}:{day}"


def dirty_member(user_id: int, day: str) -> str:
    """待同步集合的成员。"""
    return f"{user_id}:{day}"


class RateLimiter:
    """按分钟计数。"""

    def __init__(self, redis_client, per_minute: int = 10) -> None:
        self.redis = redis_client
        self.per_minute = int(per_minute)

    def hit(self, user_id: int) -> tuple[bool, int]:
        """返回（是否放行, 本分钟已用次数）。"""
        minute = datetime.now(TZ).strftime("%Y%m%d%H%M")
        key = f"xm:rl:{user_id}:{minute}"
        used = int(self.redis.incr(key))
        if used == 1:
            self.redis.expire(key, RATE_TTL)
        return used <= self.per_minute, used


class QuotaService:
    """每日配额：Redis 实时计数，定时写回 MySQL。"""

    def __init__(self, redis_client, pool: MySQLPool) -> None:
        self.redis = redis_client
        self.pool = pool
        self._consume = redis_client.register_script(CONSUME_SCRIPT)

    def consume(self, user_id: int, daily_quota: int) -> tuple[bool, int]:
        """检查并加一；Redis 里没有时先从 usage_daily 补回。"""
        day = utc_day()
        key = quota_key(user_id, day)
        if not self.redis.exists(key):
            self._restore(user_id, day, key)
        allowed, used = self._consume(
            keys=[key, DIRTY_KEY],
            args=[int(daily_quota), QUOTA_TTL, dirty_member(user_id, day)],
        )
        return bool(int(allowed)), int(used)

    def refund(self, user_id: int) -> None:
        """链路出错时退回这次，不低于 0。"""
        day = utc_day()
        key = quota_key(user_id, day)
        if not self.redis.exists(key):
            return
        remaining = int(self.redis.hincrby(key, "q", -1))
        if remaining < 0:
            self.redis.hset(key, "q", 0)

    def add_tokens(self, user_id: int, tokens: int) -> None:
        """记 token，并把用户放进待同步集合。"""
        if int(tokens) <= 0:
            return
        day = utc_day()
        key = quota_key(user_id, day)
        self.redis.hincrby(key, "tokens", int(tokens))
        self.redis.expire(key, QUOTA_TTL)
        self.redis.sadd(DIRTY_KEY, dirty_member(user_id, day))

    def today(self, user_id: int) -> dict:
        """当天用量。"""
        day = utc_day()
        key = quota_key(user_id, day)
        values = self.redis.hgetall(key) or {}
        used = int(values.get("q", values.get(b"q", 0)) or 0)
        tokens = int(values.get("tokens", values.get(b"tokens", 0)) or 0)
        return {"day": day_to_sql(day), "used": used, "tokens": tokens}

    def _restore(self, user_id: int, day: str, key: str) -> None:
        """Redis 丢了就从 usage_daily 补回，避免重启后额度翻倍。"""
        used, tokens = 0, 0
        try:
            with self.pool.connection() as conn, conn.cursor(DictCursor) as cursor:
                cursor.execute(
                    "SELECT query_count, tokens FROM usage_daily WHERE user_id=%s AND day=%s",
                    (user_id, day_to_sql(day)),
                )
                row = cursor.fetchone()
            if row:
                used, tokens = int(row["query_count"]), int(row["tokens"])
        except Exception:
            logger.warning("读 usage_daily 失败，配额按 0 起步", exc_info=True)
        pipe = self.redis.pipeline()
        pipe.hsetnx(key, "q", used)
        pipe.hsetnx(key, "tokens", tokens)
        pipe.expire(key, QUOTA_TTL)
        pipe.execute()
