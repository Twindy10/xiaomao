"""连接池测试：复用、上限、体检、回滚。"""

from __future__ import annotations

import threading
import time

import pytest

from xiaomao.storage.db import MySQLPool


class FakeConn:
    """假连接。"""

    def __init__(self, alive: bool = True) -> None:
        self.alive = alive
        self.pings = 0
        self.commits = 0
        self.rollbacks = 0
        self.closed = False

    def ping(self):
        """体检。"""
        self.pings += 1
        if not self.alive:
            raise RuntimeError("gone")

    def commit(self):
        """提交。"""
        self.commits += 1

    def rollback(self):
        """回滚。"""
        self.rollbacks += 1

    def close(self):
        """关闭。"""
        self.closed = True


def build(size: int = 2) -> tuple[MySQLPool, list[FakeConn]]:
    """每次 connect 生成一个新连接并记录。"""
    conns: list[FakeConn] = []

    def connect():
        conn = FakeConn()
        conns.append(conn)
        return conn

    return MySQLPool(connect=connect, size=size, borrow_timeout=0.3), conns


def test_reuses_same_connection() -> None:
    """借还两次复用同一个连接。"""
    pool, conns = build()
    with pool.connection() as conn:
        first = conn
    with pool.connection() as conn:
        second = conn

    assert first is second
    assert len(conns) == 1
    assert conns[0].commits == 2
    assert conns[0].closed is False


def test_waits_when_exhausted() -> None:
    """全部借出时等待，超时抛 TimeoutError。"""
    pool, _ = build(size=1)
    with pool.connection():
        with pytest.raises(TimeoutError):
            with pool.connection():
                pass


def test_broken_connection_is_replaced() -> None:
    """ping 失败时重建。"""
    pool, conns = build(size=1)
    with pool.connection() as conn:
        conn.alive = False
    with pool.connection() as conn:
        assert conn is not conns[0]
    assert len(conns) == 2  # 原连接坏掉后只新建一个
    assert conns[0].closed is True


def test_exception_rolls_back() -> None:
    """异常时回滚，连接可继续使用。"""
    pool, conns = build()
    with pytest.raises(RuntimeError):
        with pool.connection() as conn:
            raise RuntimeError("boom")

    assert conns[0].rollbacks == 1
    with pool.connection() as conn:
        assert conn is conns[0]


def test_threads_never_exceed_size() -> None:
    """多线程同时借，创建数不超过 size。"""
    pool, conns = build(size=3)
    peak = 0
    lock = threading.Lock()

    def worker() -> None:
        nonlocal peak
        with pool.connection():
            with lock:
                peak = max(peak, len(conns))
            time.sleep(0.05)

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(conns) <= 3
    assert peak <= 3
