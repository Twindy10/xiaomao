"""画像测试：校验、cache-aside、profile_filters。"""

from __future__ import annotations

import json

import pytest

from xiaomao.user.profile import (
    ProfileStore,
    profile_filters,
    profile_options,
    validate_profile,
)


class FakeCursor:
    """记录 SQL 的假游标。"""

    def __init__(self, conn: "FakeConn") -> None:
        self.conn = conn
        self.row = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, sql, params=None):
        """记录并回填预设行。"""
        self.conn.log.append((sql, params))
        self.row = self.conn.row

    def fetchone(self):
        """返回预设行。"""
        return self.row

    def fetchall(self):
        """返回空。"""
        return []


class FakeConn:
    """假连接。"""

    def __init__(self) -> None:
        self.log: list[tuple[str, tuple]] = []
        self.row = None
        self.commits = 0
        self.pings = 0

    def cursor(self, *args, **kwargs):
        """返回游标。"""
        return FakeCursor(self)

    def commit(self):
        """提交。"""
        self.commits += 1

    def rollback(self):
        """回滚。"""
        pass

    def close(self):
        """关闭。"""
        pass

    def ping(self):
        """体检。"""
        self.pings += 1


class FakePool:
    """一个假连接。"""

    def __init__(self) -> None:
        self.conn = FakeConn()

    def connection(self):
        """上下文管理器。"""
        pool = self

        class _Ctx:
            def __enter__(self):
                return pool.conn

            def __exit__(self, *args):
                pool.conn.commits += 1
                return False

        return _Ctx()


class FakeRedis:
    """最小 Redis。"""

    def __init__(self) -> None:
        self.data: dict[str, str] = {}
        self.deleted: list[str] = []

    def get(self, key):
        """读。"""
        return self.data.get(key)

    def set(self, key, value, ex=None):
        """写。"""
        self.data[key] = value

    def delete(self, key):
        """删。"""
        self.deleted.append(key)
        self.data.pop(key, None)


def test_validate_platform_and_site() -> None:
    """非法平台、站点报错。"""
    with pytest.raises(ValueError):
        validate_profile({"platforms": ["ebay"]})
    with pytest.raises(ValueError):
        validate_profile({"sites": ["FR2"]})


def test_validate_categories() -> None:
    """品类最多 10 个，每个 1–20 字。"""
    with pytest.raises(ValueError):
        validate_profile({"categories": ["长" * 21]})
    with pytest.raises(ValueError):
        validate_profile({"categories": [f"品类{index}" for index in range(11)]})


def test_validate_stage_and_dedupe() -> None:
    """阶段取值、去重保序。"""
    profile = validate_profile({
        "platforms": ["amazon", "amazon", "temu"],
        "sites": ["DE", "DE"],
        "categories": ["耳机", "耳机", "充电器"],
        "stage": "growing",
    })
    assert profile["platforms"] == ["amazon", "temu"]
    assert profile["categories"] == ["耳机", "充电器"]
    with pytest.raises(ValueError):
        validate_profile({"stage": "veteran"})


def test_profile_filters_single_and_multiple() -> None:
    """单一补全，多个不猜。"""
    assert profile_filters({"platforms": ["amazon"], "sites": ["DE"]}) == {"platform": "amazon", "site": "DE"}
    assert profile_filters({"platforms": ["amazon", "temu"], "sites": ["US"]}) == {
        "platform": None, "site": "US"
    }
    assert profile_filters({"platforms": ["amazon", "temu"], "sites": ["US", "DE"]}) is None
    assert profile_filters(None) is None


def test_options_include_platforms_and_sites() -> None:
    """选项里有平台和站点，去掉 none。"""
    options = profile_options()
    assert "amazon" in options["platforms"] and "none" not in options["platforms"]
    assert "DE" in options["sites"]


def test_get_fills_cache() -> None:
    """第一次读库并回填 Redis，第二次直接读缓存。"""
    pool = FakePool()
    redis_client = FakeRedis()
    pool.conn.row = {"platforms": '["amazon"]', "sites": '["DE"]', "categories": "[]", "stage": "newbie"}
    store = ProfileStore(pool, redis_client)

    first = store.get(1)
    assert first["platforms"] == ["amazon"]
    assert "xm:profile:1" in redis_client.data

    pool.conn.row = None
    second = store.get(1)
    assert second["platforms"] == ["amazon"]  # 来自缓存
    assert len(pool.conn.log) == 1


def test_get_returns_empty_when_missing() -> None:
    """没填过返回空画像。"""
    pool = FakePool()
    store = ProfileStore(pool, FakeRedis())
    assert store.get(1) == {"platforms": [], "sites": [], "categories": [], "stage": None}


def test_save_writes_then_deletes_cache() -> None:
    """先写 MySQL，再删缓存。"""
    pool = FakePool()
    redis_client = FakeRedis()
    redis_client.data["xm:profile:1"] = json.dumps({"platforms": ["amazon"]})
    store = ProfileStore(pool, redis_client)

    saved = store.save(1, {"platforms": ["amazon"], "sites": ["DE"]})

    assert saved["sites"] == ["DE"]
    assert redis_client.deleted == ["xm:profile:1"]
    sql, params = pool.conn.log[0]
    assert "ON DUPLICATE KEY UPDATE" in sql
    assert params[1] == '["amazon"]'


def test_save_rejects_invalid() -> None:
    """保存前校验。"""
    store = ProfileStore(FakePool(), FakeRedis())
    with pytest.raises(ValueError):
        store.save(1, {"platforms": ["ebay"]})
