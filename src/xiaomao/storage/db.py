"""MySQL 连接池：按需创建、借出前体检、异常回滚。"""

from __future__ import annotations

import queue
import threading
import time
from contextlib import contextmanager
from typing import Iterator

import pymysql

from xiaomao.ingest.kb_store import connect_mysql


class MySQLPool:
    """LIFO 空闲队列；最多 size 个连接，借不到时最多等 borrow_timeout 秒。"""

    def __init__(
        self,
        connect=connect_mysql,
        size: int = 8,
        recycle_seconds: int = 3600,
        borrow_timeout: float = 10.0,
    ) -> None:
        self._connect = connect
        self._size = size
        self._recycle = recycle_seconds
        self._timeout = borrow_timeout
        self._idle: queue.LifoQueue = queue.LifoQueue()
        self._lock = threading.Lock()
        self._created = 0
        self._born: dict[int, float] = {}
        self._closed = False

    @contextmanager
    def connection(self) -> Iterator[pymysql.connections.Connection]:
        """借一个连接：正常退出提交，异常回滚；坏连接直接丢弃。"""
        conn = self._acquire()
        try:
            yield conn
        except BaseException:
            try:
                conn.rollback()
            except Exception:
                self._discard(conn)
                raise
            self._release(conn)
            raise
        else:
            try:
                conn.commit()
            except Exception:
                self._discard(conn)
                raise
            self._release(conn)

    def close(self) -> None:
        """关掉所有空闲连接。"""
        self._closed = True
        while True:
            try:
                conn = self._idle.get_nowait()
            except queue.Empty:
                break
            self._close_quietly(conn)
        with self._lock:
            self._created = 0
            self._born.clear()

    # ------------------------------------------------------------------

    def _acquire(self):
        """拿一个可用连接。"""
        if self._closed:
            raise RuntimeError("连接池已关闭")
        try:
            conn = self._idle.get_nowait()
        except queue.Empty:
            conn = self._new_or_wait()
            if conn is None:
                try:
                    conn = self._idle.get(timeout=self._timeout)
                except queue.Empty as exc:
                    raise TimeoutError(f"等连接超时（{self._timeout}s），池大小 {self._size}") from exc
        if not self._healthy(conn):
            self._close_quietly(conn)
            with self._lock:
                self._created = max(0, self._created - 1)
            conn = self._connect()
            with self._lock:
                self._created += 1
            self._born[id(conn)] = time.monotonic()
        return conn

    def _new_or_wait(self):
        """还有配额就直接建一个，没有返回 None。"""
        with self._lock:
            if self._created < self._size:
                self._created += 1
                created = True
            else:
                created = False
        if not created:
            return None
        try:
            conn = self._connect()
        except Exception:
            with self._lock:
                self._created -= 1
            raise
        self._born[id(conn)] = time.monotonic()
        return conn

    def _healthy(self, conn) -> bool:
        """用得太久或 ping 不通就换掉。"""
        born = self._born.get(id(conn), 0.0)
        if time.monotonic() - born > self._recycle:
            return False
        try:
            conn.ping()
            return True
        except Exception:
            return False

    def _release(self, conn) -> None:
        """归还。"""
        if self._closed:
            self._close_quietly(conn)
            return
        self._idle.put(conn)

    def _discard(self, conn) -> None:
        """坏连接：关掉并从计数里移除。"""
        self._close_quietly(conn)
        with self._lock:
            self._created = max(0, self._created - 1)
        self._born.pop(id(conn), None)

    @staticmethod
    def _close_quietly(conn) -> None:
        """关连接，忽略异常。"""
        try:
            conn.close()
        except Exception:
            pass
