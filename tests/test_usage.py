"""用量同步测试：绝对值、幂等、SREM 时机。"""

from __future__ import annotations

from xiaomao.user.limits import DIRTY_KEY, quota_key, utc_day
from xiaomao.user.usage import sync_usage


class FakeRedis:
    """最小 Redis。"""

    def __init__(self) -> None:
        self.hashes: dict[str, dict[str, str]] = {}
        self.sets: dict[str, set] = {}

    def smembers(self, key):
        """集合成员。"""
        return set(self.sets.get(key, set()))

    def exists(self, key):
        """Hash 是否存在。"""
        return key in self.hashes

    def hgetall(self, key):
        """读 Hash。"""
        return dict(self.hashes.get(key, {}))

    def srem(self, key, member):
        """删成员。"""
        self.sets.get(key, set()).discard(member)

    def sadd(self, key, member):
        """加成员。"""
        self.sets.setdefault(key, set()).add(member)


class FakeCursor:
    """记录 SQL。"""

    def __init__(self, pool: "FakePool") -> None:
        self.pool = pool

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, sql, params=None):
        """记录。"""
        if self.pool.fail:
            raise RuntimeError("db down")
        self.pool.executed.append((sql, params))


class FakeConn:
    """假连接。"""

    def __init__(self, pool: "FakePool") -> None:
        self.pool = pool

    def cursor(self, *args, **kwargs):
        """返回游标。"""
        return FakeCursor(self.pool)

    def commit(self):
        """提交。"""

    def rollback(self):
        """回滚。"""

    def close(self):
        """关闭。"""

    def ping(self):
        """体检。"""


class FakePool:
    """记录执行过的 SQL。"""

    def __init__(self, fail: bool = False) -> None:
        self.executed: list[tuple[str, tuple]] = []
        self.fail = fail

    def connection(self):
        """上下文管理器。"""
        pool = self

        class _Ctx:
            def __enter__(self):
                return FakeConn(pool)

            def __exit__(self, *args):
                return False

        return _Ctx()


def test_sync_writes_absolute_values_and_removes_member() -> None:
    """写绝对值并 SREM。"""
    redis_client = FakeRedis()
    pool = FakePool()
    key = quota_key(1, utc_day())
    redis_client.hashes[key] = {"q": "7", "tokens": "900"}
    redis_client.sets[DIRTY_KEY] = {f"1:{utc_day()}"}

    assert sync_usage(redis_client, pool) == 1
    assert redis_client.sets[DIRTY_KEY] == set()
    sql, params = pool.executed[0]
    assert "ON DUPLICATE KEY UPDATE query_count=VALUES(query_count)" in sql
    assert params[2] == 7 and params[3] == 900


def test_sync_is_idempotent() -> None:
    """重复同步结果不变。"""
    redis_client = FakeRedis()
    pool = FakePool()
    key = quota_key(1, utc_day())
    redis_client.hashes[key] = {"q": "3", "tokens": "30"}
    redis_client.sets[DIRTY_KEY] = {f"1:{utc_day()}"}

    assert sync_usage(redis_client, pool) == 1
    assert sync_usage(redis_client, pool) == 0  # 没有待同步成员了
    assert len(pool.executed) == 1


def test_expired_hash_is_removed_without_write() -> None:
    """Hash 过期的成员直接 SREM。"""
    redis_client = FakeRedis()
    pool = FakePool()
    redis_client.sets[DIRTY_KEY] = {f"1:{utc_day()}"}

    assert sync_usage(redis_client, pool) == 0
    assert redis_client.sets[DIRTY_KEY] == set()
    assert pool.executed == []


def test_failed_write_keeps_member() -> None:
    """写库失败不 SREM，下次还能重试。"""
    redis_client = FakeRedis()
    pool = FakePool(fail=True)
    key = quota_key(1, utc_day())
    redis_client.hashes[key] = {"q": "1", "tokens": "10"}
    redis_client.sets[DIRTY_KEY] = {f"1:{utc_day()}"}

    assert sync_usage(redis_client, pool) == 0
    assert redis_client.sets[DIRTY_KEY] == {f"1:{utc_day()}"}


def test_request_during_sync_stays_dirty() -> None:
    """同步写库的同时又来了一次请求：这个用户要留在待同步集合里，下一轮把新计数写回去。"""
    redis_client = FakeRedis()
    member = f"1:{utc_day()}"
    key = quota_key(1, utc_day())
    redis_client.hashes[key] = {"q": "4", "tokens": "40"}
    redis_client.sets[DIRTY_KEY] = {member}

    class ConcurrentPool(FakePool):
        def connection(self):
            # 写库这一刻，另一个请求计了数并 SADD
            redis_client.hashes[key]["q"] = "5"
            redis_client.sadd(DIRTY_KEY, member)
            return super().connection()

    pool = ConcurrentPool()
    assert sync_usage(redis_client, pool) == 1
    assert pool.executed[0][1][2] == 4
    assert redis_client.sets[DIRTY_KEY] == {member}
    assert sync_usage(redis_client, pool) == 1
    assert pool.executed[1][1][2] == 5
