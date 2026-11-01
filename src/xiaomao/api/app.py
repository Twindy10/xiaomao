"""FastAPI 接口：账号、画像、会话、反馈、流式问答、限流配额、健康检查。"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from xiaomao.api.logging_setup import REQUEST_ID, new_request_id, setup_logging

logger = logging.getLogger("xiaomao.api")
POOL_TIMEOUT = 30
HEALTH_TIMEOUT = 2
RATE_RETRY_AFTER = 60
FEEDBACK_REASONS = {"wrong", "outdated", "off_topic", "other"}
COMMENT_MAX = 500
# 前端打包产物（cd web && npm run build）；没打包时只提供 /api
WEB_DIST = Path(__file__).resolve().parents[3] / "web" / "dist"


class ChatRequest(BaseModel):
    """问答请求体。"""

    question: str
    conversation_id: int | None = None


class RegisterRequest(BaseModel):
    """注册请求体。"""

    phone: str
    password: str
    nickname: str | None = None


class LoginRequest(BaseModel):
    """登录请求体。"""

    phone: str
    password: str


class RefreshRequest(BaseModel):
    """刷新请求体。"""

    refresh_token: str


class LogoutRequest(BaseModel):
    """登出请求体。"""

    refresh_token: str | None = None


class FeedbackRequest(BaseModel):
    """反馈请求体。"""

    rating: int
    reason: str | None = None
    comment: str | None = None


class ProfileRequest(BaseModel):
    """画像请求体（整体覆盖）。"""

    platforms: list[str] = []
    sites: list[str] = []
    categories: list[str] = []
    stage: str | None = None


# ---------------------------------------------------------------------------
# 依赖
# ---------------------------------------------------------------------------


def _bearer(authorization: str | None) -> str:
    """从 Authorization 头取 token。"""
    if not authorization:
        raise HTTPException(status_code=401, detail="unauthorized")
    parts = authorization.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1].strip():
        raise HTTPException(status_code=401, detail="unauthorized")
    return parts[1].strip()


def current_user_id(request: Request, authorization: str | None = Header(default=None)) -> int:
    """验 JWT，返回用户 ID；失败一律 401。"""
    from xiaomao.user.auth import AuthError

    try:
        return request.app.state.auth.verify(_bearer(authorization))
    except AuthError as exc:
        raise HTTPException(status_code=401, detail="unauthorized") from exc


def current_user(request: Request, user_id: int = Depends(current_user_id)) -> dict:
    """返回整行用户信息（带套餐字段），避免重复查库。"""
    user = request.app.state.users.get_by_id(user_id)
    if user is None:
        raise HTTPException(status_code=401, detail="unauthorized")
    return user


def _sse(event: str, data: dict) -> str:
    """拼一个 SSE 事件。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


def _run_check(func: Callable[[], bool]) -> bool:
    """带超时的健康检查。"""
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(func)
        try:
            return bool(future.result(timeout=HEALTH_TIMEOUT))
        except Exception:
            return False


def default_health_checks(pool=None) -> dict[str, Callable[[], bool]]:
    """真实服务的检查项。MySQL 用应用自己的连接池，不要每次检查都新建一个池子。"""
    def milvus() -> bool:
        from pymilvus import MilvusClient

        client = MilvusClient(uri=os.environ["MILVUS_URI"], token=os.environ.get("MILVUS_TOKEN", ""))
        return bool(client.get_server_version())

    def mysql() -> bool:
        with pool.connection() as conn, conn.cursor() as cursor:
            cursor.execute("SELECT 1")
        return True

    def redis_check() -> bool:
        from xiaomao.runtime import redis_from_env

        return bool(redis_from_env().ping())

    def decider() -> bool:
        import httpx

        base = os.environ.get("DECIDER_BASE_URL", "http://127.0.0.1:8765").rstrip("/")
        try:
            response = httpx.get(base + "/healthz", timeout=HEALTH_TIMEOUT, trust_env=False)
            return response.status_code == 200
        except Exception:
            return False

    return {"milvus": milvus, "mysql": mysql, "redis": redis_check, "decider": decider}


def _token_body(pair) -> dict:
    """token 对转响应体。"""
    return {
        "access_token": pair.access_token,
        "refresh_token": pair.refresh_token,
        "expires_in": pair.expires_in,
    }


def create_app(
    *,
    pipelines: list | None = None,
    store=None,
    health_checks: dict[str, Callable[[], bool]] | None = None,
    pool=None,
    users=None,
    auth=None,
    redis_client=None,
    profiles=None,
    feedback=None,
    quota=None,
    limiter=None,
    web_dist: Path | None = WEB_DIST,
) -> FastAPI:
    """构造应用；测试可以注入假组件。"""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        setup_logging()
        if app.state.pool is None:
            from xiaomao.runtime import build_pool

            app.state.pool = build_pool()
        if app.state.redis_client is None:
            from xiaomao.runtime import redis_from_env

            app.state.redis_client = redis_from_env()
        if app.state.users is None:
            from xiaomao.user.store import UserStore

            app.state.users = UserStore(app.state.pool)
        if app.state.auth is None:
            from xiaomao.user.auth import AuthService

            app.state.auth = AuthService(
                app.state.users, app.state.redis_client,
                secret=os.environ["JWT_SECRET"],
                bcrypt_rounds=int(os.environ.get("BCRYPT_ROUNDS", "12")),
            )
        if app.state.store is None:
            from xiaomao.storage.chat_store import ChatStore

            app.state.store = ChatStore(app.state.pool)
        if app.state.profiles is None:
            from xiaomao.user.profile import ProfileStore

            app.state.profiles = ProfileStore(app.state.pool, app.state.redis_client)
        if app.state.feedback is None:
            from xiaomao.storage.feedback_store import FeedbackStore

            app.state.feedback = FeedbackStore(app.state.pool)
        if app.state.quota is None:
            from xiaomao.user.limits import QuotaService

            app.state.quota = QuotaService(app.state.redis_client, app.state.pool)
        if app.state.limiter is None:
            from xiaomao.user.limits import RateLimiter

            app.state.limiter = RateLimiter(
                app.state.redis_client,
                per_minute=int(os.environ.get("RATE_LIMIT_PER_MIN", "10")),
            )
        if app.state.pipelines is None:
            from xiaomao.runtime import build_pipelines

            app.state.pipelines = build_pipelines(4, with_cache=True, with_rewriter=True)
        if app.state.health_checks is None:
            app.state.health_checks = default_health_checks(app.state.pool)
        app.state.pool_queue = queue.Queue()
        for pipeline in app.state.pipelines:
            app.state.pool_queue.put(pipeline)

        stop = threading.Event()
        app.state.sync_stop = stop
        try:
            from xiaomao.user.usage import start_sync_thread

            app.state.sync_thread = start_sync_thread(app.state.redis_client, app.state.pool, stop)
        except Exception:
            logger.warning("用量同步线程没起来", exc_info=True)
            app.state.sync_thread = None
        yield
        stop.set()
        if app.state.sync_thread is not None:
            app.state.sync_thread.join(timeout=5)

    app = FastAPI(title="小贸同学", lifespan=lifespan)
    app.state.pool = pool
    app.state.users = users
    app.state.auth = auth
    app.state.redis_client = redis_client
    app.state.store = store
    app.state.profiles = profiles
    app.state.feedback = feedback
    app.state.quota = quota
    app.state.limiter = limiter
    app.state.pipelines = pipelines
    app.state.health_checks = health_checks
    app.state.pool_queue = queue.Queue()

    @app.exception_handler(HTTPException)
    def http_error(request: Request, exc: HTTPException):
        """所有错误统一成 {"error": ...}。"""
        content = exc.detail if isinstance(exc.detail, dict) else {"error": exc.detail}
        return JSONResponse(status_code=exc.status_code, content=content, headers=exc.headers)

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        request_id = new_request_id()
        token = REQUEST_ID.set(request_id)
        try:
            response = await call_next(request)
        finally:
            REQUEST_ID.reset(token)
        response.headers["X-Request-Id"] = request_id
        return response

    # ------------------------------------------------------------------
    # 账号
    # ------------------------------------------------------------------

    @app.post("/api/auth/register", status_code=201)
    def register(payload: RegisterRequest):
        try:
            user_id = app.state.auth.register(payload.phone, payload.password, payload.nickname)
        except ValueError as exc:
            return JSONResponse(status_code=422, content={"error": str(exc)})
        except Exception as exc:
            if type(exc).__name__ == "PhoneTaken":
                return JSONResponse(status_code=409, content={"error": "手机号已注册"})
            raise
        return {"user_id": user_id}

    @app.post("/api/auth/login")
    def login(payload: LoginRequest):
        from xiaomao.user.auth import AuthError, BannedError

        try:
            pair = app.state.auth.login(payload.phone, payload.password)
        except BannedError as exc:
            return JSONResponse(status_code=403, content={"error": str(exc)})
        except AuthError as exc:
            return JSONResponse(status_code=401, content={"error": str(exc)})
        me = app.state.users.get_by_phone(payload.phone) or {}
        full = app.state.users.get_by_id(int(me.get("id") or 0)) or me
        return {
            **_token_body(pair),
            "user": {
                "id": int(full.get("id") or 0),
                "nickname": full.get("nickname"),
                "plan": full.get("plan_code") or "free",
            },
        }

    @app.post("/api/auth/refresh")
    def refresh(payload: RefreshRequest):
        from xiaomao.user.auth import AuthError

        try:
            pair = app.state.auth.refresh(payload.refresh_token)
        except AuthError as exc:
            return JSONResponse(status_code=401, content={"error": str(exc)})
        return _token_body(pair)

    @app.post("/api/auth/logout")
    def logout(
        payload: LogoutRequest | None = None,
        authorization: str | None = Header(default=None),
        user_id: int = Depends(current_user_id),
    ):
        token = _bearer(authorization)
        app.state.auth.logout(token, payload.refresh_token if payload else None)
        return {"ok": True}

    @app.get("/api/me")
    def me(user: dict = Depends(current_user)):
        from xiaomao.user.auth import mask_phone

        return {
            "id": int(user["id"]),
            "phone": mask_phone(user.get("phone")),
            "nickname": user.get("nickname"),
            "plan": user.get("plan_code") or "free",
            "daily_quota": int(user.get("daily_quota") or 0),
        }

    # ------------------------------------------------------------------
    # 画像
    # ------------------------------------------------------------------

    def _profile_body(profile: dict) -> dict:
        from xiaomao.user.profile import profile_options

        return {**profile, "options": profile_options()}

    @app.get("/api/profile")
    def get_profile(user: dict = Depends(current_user)):
        return _profile_body(app.state.profiles.get(int(user["id"])))

    @app.put("/api/profile")
    def put_profile(payload: ProfileRequest, user: dict = Depends(current_user)):
        try:
            saved = app.state.profiles.save(int(user["id"]), payload.model_dump())
        except ValueError as exc:
            return JSONResponse(status_code=422, content={"error": str(exc)})
        return _profile_body(saved)

    # ------------------------------------------------------------------
    # 会话
    # ------------------------------------------------------------------

    @app.get("/api/conversations")
    def list_conversations(limit: int = 20, before: str | None = None,
                           user: dict = Depends(current_user)):
        limit = max(1, min(int(limit), 100))
        rows = app.state.store.list_conversations(
            int(user["id"]), limit=limit, before=before,
            max_history_days=int(user.get("max_history_days") or 0) or None,
        )
        return [
            {"id": int(row["id"]), "title": row.get("title") or "",
             "updated_at": str(row.get("updated_at"))}
            for row in rows
        ]

    @app.delete("/api/conversations/{conversation_id}", status_code=204)
    def delete_conversation(conversation_id: int, user: dict = Depends(current_user)):
        if not app.state.store.delete_conversation(int(user["id"]), conversation_id):
            raise HTTPException(status_code=404, detail="conversation not found")
        return None

    @app.get("/api/conversations/{conversation_id}/messages")
    def messages(conversation_id: int, user: dict = Depends(current_user)):
        store = app.state.store
        user_id = int(user["id"])
        if store.get_conversation(user_id, conversation_id) is None:
            raise HTTPException(status_code=404, detail="conversation not found")
        return [
            {
                "id": int(row["id"]),
                "role": row["role"],
                "content": row["content"],
                "citations": row.get("citations"),
                "rating": row.get("rating"),
                "created_at": str(row.get("created_at")),
            }
            for row in store.list_messages(user_id, conversation_id)
        ]

    # ------------------------------------------------------------------
    # 反馈
    # ------------------------------------------------------------------

    @app.post("/api/messages/{message_id}/feedback")
    def submit_feedback(message_id: int, payload: FeedbackRequest, user: dict = Depends(current_user)):
        if payload.rating not in (1, -1):
            return JSONResponse(status_code=422, content={"error": "rating 只能是 1 或 -1"})
        reason = payload.reason if payload.rating == -1 else None
        if reason is not None and reason not in FEEDBACK_REASONS:
            return JSONResponse(status_code=422, content={"error": f"reason 取值不合法：{reason}"})
        comment = (payload.comment or "").strip() or None
        if comment is not None and len(comment) > COMMENT_MAX:
            return JSONResponse(status_code=422, content={"error": f"comment 最多 {COMMENT_MAX} 字"})

        user_id = int(user["id"])
        store = app.state.store
        message = store.get_message(user_id, message_id)
        if message is None or message.get("role") != "assistant":
            raise HTTPException(status_code=404, detail="message not found")

        feedback_id = app.state.feedback.upsert(user_id, message_id, payload.rating, reason, comment)
        deleted: list[str] = []
        if payload.rating == -1:
            deleted = _delete_answer_caches(app, store, user_id, message_id)
        return {"feedback_id": feedback_id, "rating": payload.rating, "cache_deleted": deleted}

    # ------------------------------------------------------------------
    # 用量
    # ------------------------------------------------------------------

    @app.get("/api/usage")
    def usage(user: dict = Depends(current_user)):
        today = app.state.quota.today(int(user["id"]))
        daily_quota = int(user.get("daily_quota") or 0)
        return {
            "day": today["day"],
            "used": today["used"],
            "daily_quota": daily_quota,
            "tokens": today["tokens"],
            "plan": user.get("plan_code") or "free",
        }

    # ------------------------------------------------------------------
    # 问答
    # ------------------------------------------------------------------

    @app.post("/api/chat")
    def chat(payload: ChatRequest, user: dict = Depends(current_user)):
        user_id = int(user["id"])

        allowed, _ = app.state.limiter.hit(user_id)
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={"error": "rate_limited", "message": "提问太快了，请稍后再试"},
                headers={"Retry-After": str(RATE_RETRY_AFTER)},
            )

        daily_quota = int(user.get("daily_quota") or 0)
        quota_ok, used = app.state.quota.consume(user_id, daily_quota)
        if not quota_ok:
            return JSONResponse(
                status_code=429,
                content={"error": "quota_exceeded", "message": "今日额度已用完",
                         "used": used, "daily_quota": daily_quota},
            )

        question = (payload.question or "").strip()
        if not 1 <= len(question) <= 1000:
            app.state.quota.refund(user_id)
            return JSONResponse(status_code=422, content={"error": "question 长度必须是 1–1000 字"})

        store = app.state.store
        if payload.conversation_id is None:
            conversation_id = store.create_conversation(user_id, question)
        else:
            conversation_id = int(payload.conversation_id)
            if store.get_conversation(user_id, conversation_id) is None:
                app.state.quota.refund(user_id)
                raise HTTPException(status_code=404, detail="conversation not found")

        history = store.recent_messages(user_id, conversation_id, limit=6)
        store.add_message(user_id, conversation_id, "user", question)

        try:
            pipeline = app.state.pool_queue.get(timeout=POOL_TIMEOUT)
        except queue.Empty:
            app.state.quota.refund(user_id)
            return JSONResponse(status_code=503, content={"error": "服务繁忙，请稍后再试"})

        try:
            from xiaomao.user.profile import profile_filters

            filters = profile_filters(app.state.profiles.get(user_id))
        except Exception:
            logger.warning("画像读取失败，按无画像处理", exc_info=True)
            filters = None

        return StreamingResponse(
            _event_stream(app, pipeline, user_id, conversation_id, question, history, filters),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ------------------------------------------------------------------

    @app.get("/api/health")
    def health():
        checks = app.state.health_checks or {}
        results = {name: _run_check(func) for name, func in checks.items()}
        hard_failed = [name for name, ok in results.items() if not ok and name != "decider"]
        if hard_failed:
            return JSONResponse(status_code=503, content={"status": "down", "checks": results})
        status = "ok" if all(results.values()) else "degraded"
        return {"status": status, "checks": results}

    if web_dist is not None and (web_dist / "index.html").is_file():
        _mount_web(app, web_dist)
    return app


def _mount_web(app: FastAPI, dist: Path) -> None:
    """前端：/assets 是带哈希的静态文件，其余非 /api 路径都返回 index.html（单页应用）。"""
    if (dist / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")
    index = dist / "index.html"
    root = dist.resolve()

    @app.get("/{path:path}", include_in_schema=False)
    def web(path: str):
        if path == "api" or path.startswith("api/"):
            raise HTTPException(status_code=404, detail="not found")
        # public/ 里的文件（favicon.svg 等）原样返回；路径必须落在 dist 里
        target = (dist / path).resolve()
        if path and target.is_file() and target.is_relative_to(root):
            return FileResponse(target)
        # index.html 不缓存：重新打包后文件名哈希会变，浏览器要拿到新的入口
        return FileResponse(index, headers={"Cache-Control": "no-cache"})


def _delete_answer_caches(app, store, user_id: int, message_id: int) -> list[str]:
    """点踩时删掉这条回答用到的缓存（命中过的那条、或后来写进去的那条）。"""
    cache = None
    try:
        cache = getattr(app.state.pipelines[0], "cache", None)
    except Exception:
        cache = None
    if cache is None:
        return []

    candidates: list[str] = []
    try:
        trace = store.get_trace(user_id, message_id)
        if trace:
            decisions = trace.get("decisions")
            if isinstance(decisions, str):
                decisions = json.loads(decisions or "{}")
            extra = (decisions or {}).get("_extra") or {}
            detail = extra.get("cache") or {}
            if int(trace.get("cache_hit") or 0) and detail.get("candidate_id"):
                candidates.append(str(detail["candidate_id"]))
            if extra.get("cache_write_id"):
                candidates.append(str(extra["cache_write_id"]))
    except Exception:
        logger.warning("读 qa_trace 失败，跳过删缓存", extra={"fields": {"message_id": message_id}})

    deleted: list[str] = []
    for cache_id in dict.fromkeys(candidates):
        try:
            cache.delete(cache_id)
            deleted.append(cache_id)
        except Exception:
            logger.warning("删缓存失败", extra={"fields": {"message_id": message_id, "cache_id": cache_id}})
    if deleted:
        logger.info("点踩删缓存", extra={"fields": {"message_id": message_id, "cache_ids": deleted}})
    return deleted


def _event_stream(app, pipeline, user_id: int, conversation_id: int, question: str,
                  history: list[dict], profile: dict | None):
    """把链路事件转成 SSE，并在结束时落库、记 token。"""
    from xiaomao.qa import QAOptions

    store = app.state.store
    started = time.perf_counter()
    first_delta_ms: float | None = None
    result = None
    request_id = REQUEST_ID.get()
    try:
        options = QAOptions(use_cache=True, history=history, profile=profile)
        for event in pipeline.answer_stream(question, options):
            kind = event["type"]
            if kind == "meta":
                yield _sse("meta", {
                    "conversation_id": conversation_id,
                    "intent": event.get("intent"),
                    "strategy": event.get("strategy"),
                    "cache_hit": event.get("cache_hit", False),
                })
            elif kind == "delta":
                if first_delta_ms is None:
                    first_delta_ms = (time.perf_counter() - started) * 1000
                yield _sse("delta", {"text": event["text"]})
            elif kind == "done":
                result = event["result"]
    except GeneratorExit:
        logger.warning("客户端提前断开", extra={"fields": {"user_id": user_id,
                                                       "conversation_id": conversation_id}})
        raise
    except Exception:
        logger.exception("问答失败", extra={"fields": {"user_id": user_id,
                                                     "conversation_id": conversation_id}})
        app.state.quota.refund(user_id)
        yield _sse("error", {"error": "内部错误", "request_id": request_id})
        return
    finally:
        app.state.pool_queue.put(pipeline)

    if result is None:
        app.state.quota.refund(user_id)
        return

    message_id = None
    try:
        message_id = store.add_message(user_id, conversation_id, "assistant",
                                       result.answer, result.citations)
        store.save_trace(user_id, message_id, question, result.trace)
    except Exception:
        logger.exception("落库失败", extra={"fields": {"user_id": user_id,
                                                    "conversation_id": conversation_id}})

    trace = result.trace or {}
    tokens = sum(int(value or 0) for value in (trace.get("llm_tokens") or {}).values())
    try:
        app.state.quota.add_tokens(user_id, tokens)
    except Exception:
        logger.warning("记 token 失败", exc_info=True)

    total_ms = (time.perf_counter() - started) * 1000
    logger.info("问答完成", extra={"fields": {
        "user_id": user_id,
        "conversation_id": conversation_id,
        "message_id": message_id,
        "intent": result.intent,
        "strategy": result.strategy,
        "cache_hit": bool(trace.get("cache_hit")),
        "refused": result.refused,
        "evidence_by": trace.get("evidence_by"),
        "total_ms": round(total_ms),
        "first_token_ms": round(first_delta_ms) if first_delta_ms is not None else None,
        "tokens": tokens,
        "question": question[:50],
    }})

    yield _sse("done", {
        "message_id": message_id,
        "citations": result.citations,
        "refused": result.refused,
        "tokens": tokens,
        "latency_ms": trace.get("latency_ms", {}),
    })


# uvicorn 的入口：xiaomao.api.app:app
app = create_app()
