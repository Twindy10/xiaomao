"""限流和配额测试。"""

from __future__ import annotations

import threading

import pytest

from xiaomao.user.limits import DIRTY_KEY, QuotaService, RateLimiter, day_to_sql, utc_day


class FakeRedis:
    """带 Lua 模拟的 Redis。"""

    def __init__(self) -> None:
        self.data: dict[str, str] = {}
        self.hashes: dict[str, dict] = {}
        self.sets: dict[str, set] = {}
        self.expires: dict[str, int] = {}
        self.lock = threading.Lock()

    def incr(self, key):
        """自增。"""
        self.data[key] = str(int(self.data.get(key, 0)) + 1)
        return int(self.data[key])

    def expire(self, key, ttl):
        """记 TTL。"""
        self.expires[key] = ttl

    def exists(self, key):
        """是否存在。"""
        return key in self.hashes or key in self.data

    def hget(self, key, field):
        """读字段。"""
        return self.hashes.get(key, {}).get(field)

    def hset(self, key, field, value):
        """写字段。"""
        self.hashes.setdefault(key, {})[field] = str(value)
        return 1

    def hincrby(self, key, field, amount):
        """自增字段。"""
        self.hashes.setdefault(key, {})
        current = int(self.hashes[key].get(field, 0))
        self.hashes[key][field] = str(current + amount)
        return current + amount

    def hgetall(self, key):
        """读整个 Hash。"""
        return dict(self.hashes.get(key, {}))

    def sadd(self, key, member):
        """加集合成员。"""
        self.sets.setdefault(key, set()).add(member)

    def srem(self, key, member):
        """删集合成员。"""
        self.sets.get(key, set()).discard(member)

    def pipeline(self):
        """极简 pipeline。"""
        redis_client = self

        class _Pipe:
            def __init__(self):
                self.ops = []

            def hsetnx(self, key, field, value):
                if field not in redis_client.hashes.get(key, {}):
                    redis_client.hset(key, field, value)
                self.ops.append(("hsetnx", key, field))
                return self

            def expire(self, key, ttl):
                redis_client.expires[key] = ttl
                self.ops.append(("expire", key))
                return self

            def execute(self):
                return [1 for _ in self.ops]

        return _Pipe()

    def register_script(self, script):
        """用 Python 模拟同一个 Lua 脚本的原子语义。"""
        redis_client = self

        def run(keys, args):
            key, dirty = keys
            quota, ttl, member = int(args[0]), int(args[1]), args[2]
            with redis_client.lock:
                used = int(redis_client.hashes.get(key, {}).get("q", 0))
                if used >= quota:
                    return [0, used]
                used += 1
                redis_client.hset(key, "q", used)
                redis_client.expire(key, ttl)
                redis_client.sadd(dirty, member)
                return [1, used]

        return run


class FakePool:
    """配额补回时读 usage_daily。"""

    def __init__(self, row=None) -> None:
        self.row = row

    def connection(self):
        """上下文管理器。"""
        pool = self

        class _Cursor:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def execute(self, sql, params=None):
                pass

            def fetchone(self):
                return pool.row

        class _Ctx:
            def __enter__(self):
                return type("Conn", (), {"cursor": lambda self, *a, **k: _Cursor()})()

            def __exit__(self, *args):
                return False

        return _Ctx()


def test_rate_limit_counts_per_minute() -> None:
    """超过每分钟上限不放行。"""
    limiter = RateLimiter(FakeRedis(), per_minute=2)
    assert limiter.hit(1) == (True, 1)
    assert limiter.hit(1) == (True, 2)
    allowed, used = limiter.hit(1)
    assert allowed is False and used == 3
    assert limiter.hit(2) == (True, 1)  # 用户之间独立


def test_quota_limit() -> None:
    """到上限不放行，且不再计数。"""
    redis_client = FakeRedis()
    service = QuotaService(redis_client, FakePool())
    assert service.consume(1, 2) == (True, 1)
    assert service.consume(1, 2) == (True, 2)
    assert service.consume(1, 2) == (False, 2)


def test_refund_not_below_zero() -> None:
    """退回不低于 0。"""
    redis_client = FakeRedis()
    service = QuotaService(redis_client, FakePool())
    service.consume(1, 10)
    service.refund(1)
    for _ in range(3):
        service.refund(1)
    assert service.today(1)["used"] == 0


def test_restore_from_usage_daily() -> None:
    """Redis 里没有时从 MySQL 补回。"""
    redis_client = FakeRedis()
    pool = FakePool(row={"query_count": 5, "tokens": 120})
    service = QuotaService(redis_client, pool)

    allowed, used = service.consume(1, 10)
    assert allowed is True
    assert used == 6
    assert redis_client.hashes[f"xm:quota:1:{utc_day()}"]["tokens"] == "120"


def test_add_tokens_marks_dirty() -> None:
    """加 token 时进待同步集合。"""
    redis_client = FakeRedis()
    service = QuotaService(redis_client, FakePool())
    service.add_tokens(1, 150)

    key = f"xm:quota:1:{utc_day()}"
    assert redis_client.hashes[key]["tokens"] == "150"
    assert DIRTY_KEY in redis_client.sets
    assert f"1:{utc_day()}" in redis_client.sets[DIRTY_KEY]


def test_concurrent_consume_exactly_quota() -> None:
    """30 个并发请求，上限 20，正好放行 20 个。"""
    service = QuotaService(FakeRedis(), FakePool(row=None))
    results: list[bool] = []
    lock = threading.Lock()

    def worker() -> None:
        allowed, _ = service.consume(1, 20)
        with lock:
            results.append(allowed)

    threads = [threading.Thread(target=worker) for _ in range(30)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sum(results) == 20


def test_day_to_sql() -> None:
    """yyyyMMdd 转日期。"""
    assert day_to_sql("20261031") == "2026-10-31"
