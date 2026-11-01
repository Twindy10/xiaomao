"""接口测试：账号、画像、会话、反馈、限流配额、SSE。"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from xiaomao.api.app import create_app, current_user_id
from xiaomao.qa import QAResult

USER = {
    "id": 1, "phone": "13800138000", "nickname": "小贸", "status": 1,
    "plan_id": 1, "plan_code": "free", "daily_quota": 20, "max_history_days": 30,
}


class FakeUsers:
    """内存用户表。"""

    def __init__(self, user: dict | None = None) -> None:
        self.user = dict(user or USER)
        self.others: dict[int, dict] = {}

    def get_by_id(self, user_id: int):
        """按 id 查。"""
        if user_id == self.user["id"]:
            return self.user
        return self.others.get(user_id)

    def get_by_phone(self, phone: str):
        """按手机号查。"""
        return self.user if phone == self.user["phone"] else None

    def create_user(self, phone, password_hash, nickname):
        """建号。"""
        self.user["id"] += 1
        return self.user["id"]

    def touch_login(self, user_id):
        """记录。"""


class FakeAuth:
    """固定 token 语义：任何非空 token 都算用户 1。"""

    def __init__(self) -> None:
        self.registered: list[tuple] = []
        self.logged_out: list[tuple] = []
        self.refreshed: list[str] = []

    def verify(self, token: str) -> int:
        """token 非空即通过。"""
        from xiaomao.user.auth import AuthError

        if not token:
            raise AuthError("unauthorized")
        return 1

    def register(self, phone, password, nickname=None):
        """记录并返回 id。"""
        self.registered.append((phone, password, nickname))
        return 2

    def login(self, phone, password):
        """返回一对 token。"""
        from xiaomao.user.auth import TokenPair

        if password != "abcd1234":
            from xiaomao.user.auth import AuthError

            raise AuthError("手机号或密码错误")
        return TokenPair("access-token", "refresh-token", 7200)

    def refresh(self, token):
        """轮换。"""
        from xiaomao.user.auth import TokenPair

        self.refreshed.append(token)
        return TokenPair("new-access", "new-refresh", 7200)

    def logout(self, access, refresh):
        """记录。"""
        self.logged_out.append((access, refresh))


class FakeStore:
    """内存版 ChatStore。"""

    def __init__(self) -> None:
        self.conversations: dict[int, dict] = {}
        self.messages: list[dict] = []
        self.traces: dict[int, dict] = {}
        self._next = 0

    def create_conversation(self, user_id: int, title: str) -> int:
        """新建会话。"""
        self._next += 1
        self.conversations[self._next] = {"id": self._next, "user_id": user_id,
                                          "title": title, "is_deleted": 0}
        return self._next

    def get_conversation(self, user_id, conversation_id):
        """按用户取。"""
        row = self.conversations.get(conversation_id)
        if row and row["user_id"] == user_id and not row["is_deleted"]:
            return row
        return None

    def list_conversations(self, user_id, *, limit=20, before=None, max_history_days=None):
        """列表。"""
        rows = [row for row in self.conversations.values()
                if row["user_id"] == user_id and not row["is_deleted"]]
        return [{"id": row["id"], "title": row["title"], "updated_at": "2026-10-31 10:00:00"}
                for row in rows[:limit]]

    def delete_conversation(self, user_id, conversation_id) -> bool:
        """软删除。"""
        row = self.get_conversation(user_id, conversation_id)
        if row is None:
            return False
        row["is_deleted"] = 1
        return True

    def add_message(self, user_id, conversation_id, role, content, citations=None) -> int:
        """记录消息。"""
        self._next += 1
        self.messages.append({"id": self._next, "user_id": user_id,
                              "conversation_id": conversation_id, "role": role,
                              "content": content, "citations": citations, "rating": None})
        return self._next

    def recent_messages(self, user_id, conversation_id, limit: int = 6):
        """最近消息。"""
        return [row for row in self.messages
                if row["user_id"] == user_id and row["conversation_id"] == conversation_id][-limit:]

    def list_messages(self, user_id, conversation_id):
        """全部消息。"""
        return [row for row in self.messages
                if row["user_id"] == user_id and row["conversation_id"] == conversation_id]

    def get_message(self, user_id, message_id):
        """取消息。"""
        return next((row for row in self.messages
                     if row["id"] == message_id and row["user_id"] == user_id), None)

    def get_trace(self, user_id, message_id):
        """取 trace。"""
        return self.traces.get(message_id)

    def save_trace(self, user_id, message_id, query_raw, trace) -> int:
        """记录 trace。"""
        self.traces[message_id] = {"message_id": message_id, "user_id": user_id,
                                   "query_raw": query_raw, "decisions": json.dumps(
                                       {"_extra": trace.get("_extra") or {}}), "cache_hit": 0}
        if trace.get("cache_hit"):
            self.traces[message_id]["cache_hit"] = 1
            self.traces[message_id]["decisions"] = json.dumps({
                "_extra": {"cache": {"candidate_id": "cache-1"}}
            })
        return message_id


class FakeProfiles:
    """内存画像。"""

    def __init__(self) -> None:
        self.saved: dict[int, dict] = {}

    def get(self, user_id: int) -> dict:
        """读。"""
        return self.saved.get(user_id) or {"platforms": [], "sites": [],
                                           "categories": [], "stage": None}

    def save(self, user_id: int, profile: dict) -> dict:
        """写。"""
        from xiaomao.user.profile import validate_profile

        cleaned = validate_profile(profile)
        self.saved[user_id] = cleaned
        return cleaned


class FakeFeedback:
    """内存反馈。"""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def upsert(self, user_id, message_id, rating, reason, comment) -> int:
        """覆盖式写入。"""
        self.rows = [row for row in self.rows if row["message_id"] != message_id]
        self.rows.append({"user_id": user_id, "message_id": message_id, "rating": rating,
                          "reason": reason, "comment": comment})
        return len(self.rows)


class FakeQuota:
    """可编排的配额。"""

    def __init__(self, allow: bool = True, used: int = 1) -> None:
        self.allow = allow
        self.used = used
        self.refunds = 0
        self.tokens: list[int] = []

    def consume(self, user_id, daily_quota):
        """返回预设。"""
        return self.allow, self.used

    def refund(self, user_id):
        """记退回。"""
        self.refunds += 1

    def add_tokens(self, user_id, tokens):
        """记 token。"""
        self.tokens.append(tokens)

    def today(self, user_id):
        """当天用量。"""
        return {"day": "2026-10-31", "used": self.used, "tokens": 100}


class FakeLimiter:
    """可编排的限流。"""

    def __init__(self, allow: bool = True) -> None:
        self.allow = allow

    def hit(self, user_id):
        """返回预设。"""
        return self.allow, 1


class FakeCache:
    """记录删除。"""

    def __init__(self) -> None:
        self.deleted: list[str] = []

    def delete(self, cache_id: str):
        """记录删除。"""
        self.deleted.append(cache_id)


class FakePipeline:
    """固定事件流。"""

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[str, object]] = []
        self.cache = FakeCache()

    def answer_stream(self, question, options=None):
        """meta → delta → done。"""
        self.calls.append((question, options))
        if self.fail:
            raise RuntimeError("boom")
        yield {"type": "meta", "intent": "compliance", "strategy": "direct",
               "cache_hit": False, "query_rewritten": question}
        yield {"type": "delta", "text": "回答"}
        yield {"type": "done", "result": QAResult(
            answer="回答[1]", citations=[{"n": 1, "chunk_id": "c1"}], refused=False,
            intent="compliance", strategy="direct", evidence=[],
            trace={"latency_ms": {"total": 12.5}, "llm_tokens": {"generate": 5},
                   "evidence_by": "rerank_high", "_extra": {"cache_write_id": "new-cache"}},
        )}


def make_client(**overrides) -> tuple[TestClient, dict]:
    """构造测试客户端，默认登录为用户 1。"""
    components = {
        "pipelines": [FakePipeline()],
        "store": FakeStore(),
        "users": FakeUsers(),
        "auth": FakeAuth(),
        "profiles": FakeProfiles(),
        "feedback": FakeFeedback(),
        "quota": FakeQuota(),
        "limiter": FakeLimiter(),
        "health_checks": {"milvus": lambda: True, "decider": lambda: True},
        "web_dist": None,  # 默认不挂前端，测试不依赖 web/dist 有没有打包
    }
    components.update(overrides)
    app = create_app(**components)
    app.dependency_overrides[current_user_id] = lambda: 1
    client = TestClient(app)
    client.app = app
    client.components = components
    return client, components


def parse_sse(text: str) -> list[tuple[str, dict]]:
    """解析 SSE。"""
    events: list[tuple[str, dict]] = []
    for block in text.split("\n\n"):
        lines = [line for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        event = next((line.split(": ", 1)[1] for line in lines if line.startswith("event: ")), "")
        data = next((line.split(": ", 1)[1] for line in lines if line.startswith("data: ")), "{}")
        events.append((event, json.loads(data)))
    return events


AUTH = {"Authorization": "Bearer token"}


# ---------------------------------------------------------------------------
# 账号
# ---------------------------------------------------------------------------


def test_register_and_login() -> None:
    """注册 201，登录返回 token。"""
    client, components = make_client()
    with client:
        response = client.post("/api/auth/register",
                               json={"phone": "13800138000", "password": "abcd1234"})
        assert response.status_code == 201
        assert response.json()["user_id"] == 2

        login = client.post("/api/auth/login", json={"phone": "13800138000", "password": "abcd1234"})
        assert login.status_code == 200
        assert login.json()["access_token"] == "access-token"
        assert login.json()["refresh_token"] == "refresh-token"
        assert login.json()["user"]["plan"] == "free"

        bad = client.post("/api/auth/login", json={"phone": "13800138000", "password": "wrong123"})
        assert bad.status_code == 401 and bad.json()["error"] == "手机号或密码错误"


def test_register_duplicate_returns_409() -> None:
    """重复手机号 409。"""
    class TakenAuth(FakeAuth):
        def register(self, phone, password, nickname=None):
            from xiaomao.user.store import PhoneTaken

            raise PhoneTaken(phone)

    client, _ = make_client(auth=TakenAuth())
    with client:
        response = client.post("/api/auth/register",
                               json={"phone": "13800138000", "password": "abcd1234"})
    assert response.status_code == 409


def test_me_returns_masked_phone() -> None:
    """GET /api/me 打码。"""
    client, _ = make_client()
    with client:
        body = client.get("/api/me", headers=AUTH).json()
    assert body["phone"] == "138****8000"
    assert body["plan"] == "free"
    assert body["daily_quota"] == 20


def test_unauthorized_without_token() -> None:
    """没有 token 401。"""
    client, _ = make_client()
    client.app.dependency_overrides.pop(current_user_id, None)
    with client:
        assert client.post("/api/chat", json={"question": "问题"}).status_code == 401


def test_logout_and_refresh() -> None:
    """登出和刷新。"""
    client, components = make_client()
    with client:
        assert client.post("/api/auth/logout", headers=AUTH,
                           json={"refresh_token": "r"}).status_code == 200
        refreshed = client.post("/api/auth/refresh", json={"refresh_token": "r"})
    assert refreshed.status_code == 200
    assert refreshed.json()["access_token"] == "new-access"
    assert components["auth"].logged_out


# ---------------------------------------------------------------------------
# 画像
# ---------------------------------------------------------------------------


def test_profile_get_and_put() -> None:
    """画像读写，带 options。"""
    client, _ = make_client()
    with client:
        default = client.get("/api/profile", headers=AUTH).json()
        assert default["platforms"] == [] and "options" in default

        saved = client.put("/api/profile", headers=AUTH,
                           json={"platforms": ["amazon"], "sites": ["DE"]}).json()
        assert saved["platforms"] == ["amazon"]
        assert "amazon" in saved["options"]["platforms"]


def test_profile_invalid_returns_422() -> None:
    """非法平台 422。"""
    client, _ = make_client()
    with client:
        response = client.put("/api/profile", headers=AUTH, json={"platforms": ["ebay"]})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# 会话
# ---------------------------------------------------------------------------


def test_conversations_list_and_delete() -> None:
    """列表、删除、别人的 404。"""
    client, components = make_client()
    store = components["store"]
    conversation_id = store.create_conversation(1, "问题")
    with client:
        rows = client.get("/api/conversations", headers=AUTH).json()
        assert rows[0]["id"] == conversation_id

        assert client.delete(f"/api/conversations/{conversation_id}", headers=AUTH).status_code == 204
        assert client.get("/api/conversations", headers=AUTH).json() == []

    other = store.create_conversation(2, "别人的")
    with client:
        assert client.delete(f"/api/conversations/{other}", headers=AUTH).status_code == 404


def test_messages_include_rating() -> None:
    """消息列表带 rating。"""
    client, components = make_client()
    store = components["store"]
    conversation_id = store.create_conversation(1, "问题")
    message_id = store.add_message(1, conversation_id, "assistant", "回答")
    store.messages[-1]["rating"] = 1
    with client:
        rows = client.get(f"/api/conversations/{conversation_id}/messages", headers=AUTH).json()
    assert rows[0]["id"] == message_id
    assert rows[0]["rating"] == 1


def test_foreign_conversation_returns_404() -> None:
    """别人的会话 404。"""
    client, components = make_client()
    foreign = components["store"].create_conversation(2, "别人的")
    with client:
        assert client.get(f"/api/conversations/{foreign}/messages", headers=AUTH).status_code == 404


# ---------------------------------------------------------------------------
# 反馈
# ---------------------------------------------------------------------------


def test_feedback_like_does_not_delete_cache() -> None:
    """点赞不删缓存。"""
    client, components = make_client()
    pipeline = components["pipelines"][0]
    store = components["store"]
    message_id = store.add_message(1, 1, "assistant", "回答")

    with client:
        body = client.post(f"/api/messages/{message_id}/feedback", headers=AUTH,
                           json={"rating": 1}).json()
    assert body["rating"] == 1
    assert body["cache_deleted"] == []
    assert pipeline.cache.deleted == []


def test_feedback_dislike_deletes_cache() -> None:
    """点踩删掉命中的缓存和写进去的缓存。"""
    client, components = make_client()
    pipeline = components["pipelines"][0]
    store = components["store"]
    message_id = store.add_message(1, 1, "assistant", "回答")
    store.traces[message_id] = {
        "cache_hit": 1,
        "decisions": json.dumps({"_extra": {"cache": {"candidate_id": "cache-1"},
                                            "cache_write_id": "new-cache"}}),
    }

    with client:
        body = client.post(f"/api/messages/{message_id}/feedback", headers=AUTH,
                           json={"rating": -1, "reason": "wrong"}).json()
    assert body["cache_deleted"] == ["cache-1", "new-cache"]
    assert pipeline.cache.deleted == ["cache-1", "new-cache"]


def test_feedback_on_user_message_returns_404() -> None:
    """对用户消息反馈 404。"""
    client, components = make_client()
    message_id = components["store"].add_message(1, 1, "user", "问题")
    with client:
        assert client.post(f"/api/messages/{message_id}/feedback", headers=AUTH,
                           json={"rating": 1}).status_code == 404


def test_feedback_invalid_reason_returns_422() -> None:
    """原因不合法 422。"""
    client, components = make_client()
    message_id = components["store"].add_message(1, 1, "assistant", "回答")
    with client:
        assert client.post(f"/api/messages/{message_id}/feedback", headers=AUTH,
                           json={"rating": -1, "reason": "boring"}).status_code == 422


# ---------------------------------------------------------------------------
# 问答、限流配额
# ---------------------------------------------------------------------------


def test_chat_passes_profile_filters() -> None:
    """画像只有一个平台时补进 QAOptions。"""
    client, components = make_client()
    components["profiles"].save(1, {"platforms": ["amazon"], "sites": ["DE"]})
    with client:
        with client.stream("POST", "/api/chat", headers=AUTH, json={"question": "耳机认证"}) as response:
            events = parse_sse("".join(response.iter_text()))

    names = [name for name, _ in events]
    assert names[0] == "meta" and names[-1] == "done"
    options = components["pipelines"][0].calls[0][1]
    assert options.profile == {"platform": "amazon", "site": "DE"}
    assert events[-1][1]["tokens"] == 5


def test_chat_rate_limited() -> None:
    """限流 429 带 Retry-After。"""
    client, _ = make_client(limiter=FakeLimiter(allow=False))
    with client:
        response = client.post("/api/chat", headers=AUTH, json={"question": "问题"})
    assert response.status_code == 429
    assert response.headers["Retry-After"] == "60"
    assert response.json()["error"] == "rate_limited"


def test_chat_quota_exceeded() -> None:
    """配额 429。"""
    client, components = make_client(quota=FakeQuota(allow=False, used=20))
    with client:
        response = client.post("/api/chat", headers=AUTH, json={"question": "问题"})
    assert response.status_code == 429
    body = response.json()
    assert body["error"] == "quota_exceeded" and body["used"] == 20
    assert components["quota"].refunds == 0  # 配额不通过不退款


def test_chat_error_refunds_quota() -> None:
    """链路出错退回配额并放回池子。"""
    client, components = make_client(pipelines=[FakePipeline(fail=True)])
    with client:
        with client.stream("POST", "/api/chat", headers=AUTH, json={"question": "问题"}) as response:
            events = parse_sse("".join(response.iter_text()))
        assert events[0][0] == "error"
        assert components["quota"].refunds == 1


def test_usage_endpoint() -> None:
    """GET /api/usage。"""
    client, _ = make_client()
    with client:
        body = client.get("/api/usage", headers=AUTH).json()
    assert body == {"day": "2026-10-31", "used": 1, "daily_quota": 20,
                    "tokens": 100, "plan": "free"}


def test_health_degraded_and_down() -> None:
    """决策模型挂了 degraded，其他挂了 503。"""
    client, _ = make_client(health_checks={"milvus": lambda: True, "decider": lambda: False})
    with client:
        assert client.get("/api/health").json()["status"] == "degraded"

    client, _ = make_client(health_checks={"milvus": lambda: False, "decider": lambda: True})
    with client:
        assert client.get("/api/health").status_code == 503


def test_request_id_header() -> None:
    """响应带 X-Request-Id。"""
    client, _ = make_client()
    with client:
        response = client.get("/api/health")
    assert response.headers["x-request-id"]


def test_serves_web_app_with_spa_fallback(tmp_path) -> None:
    """打包好的前端：/ 和前端路由返回 index.html，静态文件原样返回，/api 下的未知路径仍是 404 JSON。"""
    (tmp_path / "assets").mkdir()
    (tmp_path / "index.html").write_text("<!doctype html><div id=root></div>", encoding="utf-8")
    (tmp_path / "assets" / "app.js").write_text("console.log(1)", encoding="utf-8")
    (tmp_path / "favicon.svg").write_text("<svg/>", encoding="utf-8")
    client, _ = make_client(web_dist=tmp_path)

    for path in ("/", "/c/12"):
        response = client.get(path)
        assert response.status_code == 200 and "id=root" in response.text
        assert response.headers["cache-control"] == "no-cache"
    assert client.get("/assets/app.js").text == "console.log(1)"
    assert client.get("/favicon.svg").text == "<svg/>"
    assert "id=root" in client.get("/../pyproject.toml").text  # 不会读到 dist 外面的文件
    missing = client.get("/api/nope")
    assert missing.status_code == 404 and missing.json() == {"error": "not found"}
