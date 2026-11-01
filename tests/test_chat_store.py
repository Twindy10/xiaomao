"""对话存储测试：假连接池，检查 SQL 带 user_id 和字段映射。"""

from __future__ import annotations

import json

from xiaomao.storage.chat_store import EXTRA_KEYS, ChatStore


class FakeCursor:
    """记录 SQL 的假游标。"""

    def __init__(self, conn: "FakeConnection") -> None:
        self.conn = conn
        self.row = None
        self.rowcount = 0
        self.lastrowid = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, sql, params=None):
        """记录 SQL；INSERT 时分配 id。"""
        self.conn.log.append((sql, params))
        self.row = self.conn.one
        self.lastrowid = self.conn.next_id
        if sql.strip().upper().startswith("INSERT"):
            self.conn.next_id += 1
            self.rowcount = 1
        elif sql.strip().upper().startswith("UPDATE"):
            self.rowcount = self.conn.update_count
        elif sql.strip().upper().startswith("DELETE"):
            self.rowcount = self.conn.update_count

    def fetchone(self):
        """返回预设单行。"""
        return self.row

    def fetchall(self):
        """返回预设多行。"""
        return self.conn.all


class FakeConnection:
    """假连接。"""

    def __init__(self) -> None:
        self.log: list[tuple[str, tuple]] = []
        self.one = None
        self.all: list[dict] = []
        self.next_id = 100
        self.update_count = 1
        self.commits = 0
        self.rollbacks = 0

    def cursor(self, *args, **kwargs):
        """返回假游标。"""
        return FakeCursor(self)

    def commit(self):
        """记录提交。"""
        self.commits += 1

    def rollback(self):
        """记录回滚。"""
        self.rollbacks += 1

    def close(self):
        """关闭。"""

    def ping(self):
        """体检。"""


class FakePool:
    """单连接假池。"""

    def __init__(self) -> None:
        self.conn = FakeConnection()

    def connection(self):
        """上下文管理器。"""
        pool = self

        class _Ctx:
            def __enter__(self):
                return pool.conn

            def __exit__(self, exc_type, *args):
                if exc_type is None:
                    pool.conn.commit()
                else:
                    pool.conn.rollback()
                return False

        return _Ctx()


def make_store() -> tuple[ChatStore, FakeConnection]:
    """构造 store 和连接。"""
    pool = FakePool()
    return ChatStore(pool=pool), pool.conn


def test_create_conversation_truncates_title() -> None:
    """标题取前 30 字。"""
    store, conn = make_store()
    conversation_id = store.create_conversation(1, "很长的问题" * 20)

    assert conversation_id == 100
    sql, params = conn.log[0]
    assert params[1] == ("很长的问题" * 20)[:30]
    assert conn.commits == 1


def test_get_conversation_filters_user() -> None:
    """查会话必须带 user_id。"""
    store, conn = make_store()
    conn.one = {"id": 5, "user_id": 1}
    store.get_conversation(1, 5)

    sql, params = conn.log[0]
    assert "user_id" in sql and "is_deleted=0" in sql
    assert params == (5, 1)


def test_add_message_writes_and_touches_conversation() -> None:
    """写消息后更新会话时间，同一个事务。"""
    store, conn = make_store()
    message_id = store.add_message(1, 5, "user", "问题")

    assert message_id == 100
    assert conn.log[0][1][0] == 5 and conn.log[0][1][1] == 1
    assert "UPDATE conversation" in conn.log[1][0]
    assert conn.commits == 1


def test_recent_messages_reverses_order() -> None:
    """最近消息按时间正序返回。"""
    store, conn = make_store()
    conn.all = [{"id": 3}, {"id": 2}, {"id": 1}]
    rows = store.recent_messages(1, 5, limit=3)

    assert [row["id"] for row in rows] == [1, 2, 3]
    assert conn.log[0][1] == (5, 1, 3)


def test_list_conversations_filters_and_paginates() -> None:
    """列表带 user_id、软删除、保留天数、游标。"""
    store, conn = make_store()
    conn.all = [{"id": 1, "title": "x", "updated_at": "2026-01-01"}]
    store.list_conversations(7, limit=10, before="2026-01-02", max_history_days=30)

    sql, params = conn.log[0]
    assert "user_id=%s AND is_deleted=0" in sql
    assert "INTERVAL %s DAY" in sql
    assert "updated_at < %s" in sql
    assert params == (7, 30, "2026-01-02", 10)


def test_list_conversations_without_history_days() -> None:
    """没有套餐限制时不加天数条件。"""
    store, conn = make_store()
    store.list_conversations(7)
    sql, params = conn.log[0]
    assert "INTERVAL" not in sql
    assert params == (7, 20)


def test_delete_conversation_uses_user_id() -> None:
    """删除带 user_id，缺行返回 False。"""
    store, conn = make_store()
    assert store.delete_conversation(1, 5) is True
    sql, params = conn.log[0]
    assert "is_deleted=1" in sql and params == (5, 1)

    conn.update_count = 0
    assert store.delete_conversation(1, 5) is False


def test_list_messages_joins_feedback() -> None:
    """消息列表带反馈 rating。"""
    store, conn = make_store()
    conn.all = [{"id": 1, "role": "assistant", "rating": -1}]
    rows = store.list_messages(1, 5)

    sql, params = conn.log[0]
    assert "LEFT JOIN feedback" in sql
    assert params == (1, 5, 1)
    assert rows[0]["rating"] == -1


def test_get_message_and_trace_filter_user() -> None:
    """消息和 trace 都按用户查。"""
    store, conn = make_store()
    conn.one = {"id": 9, "user_id": 1}
    store.get_message(1, 9)
    assert "message WHERE id=%s AND user_id=%s" in conn.log[0][0]

    store.get_trace(1, 9)
    assert "qa_trace" in conn.log[1][0]
    assert conn.log[1][1] == (9, 1)


def test_save_trace_mapping() -> None:
    """trace 映射到 qa_trace 列，额外字段进 _extra。"""
    store, conn = make_store()
    trace = {
        "query_rewritten": "改写后的问题",
        "intent": "compliance",
        "strategy": "none",
        "decisions": {"intent": {"value": "compliance"}},
        "cache_hit": True,
        "chunk_ids": ["c1", "c2"],
        "rerank_scores": [0.9, 0.8],
        "evidence_sufficient": True,
        "backtracked": 2,
        "llm_model": "qwen",
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "latency_ms": {"total": 1234.6},
        "query_en": "english",
        "cache": {"similarity": 0.9},
        "cache_write_id": "abc",
    }
    store.save_trace(7, 55, "原问题", trace)

    sql, params = conn.log[0]
    assert "INSERT INTO qa_trace" in sql
    assert params[0] == 55 and params[1] == 7
    assert params[2] == "原问题" and params[3] == "改写后的问题"
    assert params[5] is None  # strategy none -> NULL
    decisions = json.loads(params[6])
    assert decisions["_extra"]["query_en"] == "english"
    assert decisions["_extra"]["cache_write_id"] == "abc"
    assert params[7] == 1
    assert params[10] == 1
    assert params[11] == 1
    assert params[16] == 1235


def test_extra_keys_include_m6_fields() -> None:
    """M6 的新字段要能落库。"""
    assert "rerank_tokens" in EXTRA_KEYS
    assert "cache_write_id" in EXTRA_KEYS
