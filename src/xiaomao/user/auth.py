"""注册、登录、JWT 和 refresh token。"""

from __future__ import annotations

import hashlib
import re
import secrets
import time
import uuid
from dataclasses import dataclass

import bcrypt
import jwt

from xiaomao.user.store import PhoneTaken, UserStore

PHONE_RE = re.compile(r"^1[3-9]\d{9}$")
PASSWORD_MIN, PASSWORD_MAX = 8, 64
NICKNAME_MAX = 32
# 手机号不存在时也对一个假哈希跑一次 checkpw，让两种失败的耗时接近。
# 假哈希的 cost 必须和真实密码一样，否则耗时差几百倍，可以据此探测手机号是否注册过
_DUMMY_HASHES: dict[int, bytes] = {}


def _dummy_hash(rounds: int) -> bytes:
    """按 cost 缓存一个假哈希。"""
    if rounds not in _DUMMY_HASHES:
        _DUMMY_HASHES[rounds] = bcrypt.hashpw(b"dummy-password", bcrypt.gensalt(rounds))
    return _DUMMY_HASHES[rounds]


class AuthError(Exception):
    """登录态无效或过期。"""


class BannedError(AuthError):
    """账号被停用（403，和普通登录失败区分开）。"""


@dataclass
class TokenPair:
    """一对 token。"""

    access_token: str
    refresh_token: str
    expires_in: int


def mask_phone(phone: str | None) -> str:
    """138****1234。"""
    if not phone or len(phone) < 7:
        return str(phone or "")
    return f"{phone[:3]}****{phone[-4:]}"


def validate_phone(phone: str) -> str:
    """校验手机号格式。"""
    if not isinstance(phone, str) or not PHONE_RE.match(phone):
        raise ValueError("手机号格式不对")
    return phone


def validate_password(password: str) -> str:
    """8–64 位，至少含字母和数字。"""
    if not isinstance(password, str) or not PASSWORD_MIN <= len(password) <= PASSWORD_MAX:
        raise ValueError(f"密码长度必须是 {PASSWORD_MIN}–{PASSWORD_MAX} 位")
    if not re.search(r"[A-Za-z]", password) or not re.search(r"\d", password):
        raise ValueError("密码必须同时包含字母和数字")
    return password


def validate_nickname(nickname: str | None) -> str | None:
    """昵称可空，最多 32 字。"""
    if nickname is None:
        return None
    text = str(nickname).strip()
    if len(text) > NICKNAME_MAX:
        raise ValueError(f"昵称最多 {NICKNAME_MAX} 个字")
    return text or None


class AuthService:
    """账号和 token 的全部逻辑。"""

    def __init__(
        self,
        users: UserStore,
        redis_client,
        *,
        secret: str,
        access_ttl: int = 2 * 3600,
        refresh_ttl: int = 7 * 24 * 3600,
        bcrypt_rounds: int = 12,
    ) -> None:
        self.users = users
        self.redis = redis_client
        self.secret = secret
        self.access_ttl = access_ttl
        self.refresh_ttl = refresh_ttl
        self.bcrypt_rounds = bcrypt_rounds

    # ------------------------------------------------------------------
    # 注册 / 登录
    # ------------------------------------------------------------------

    def register(self, phone: str, password: str, nickname: str | None = None) -> int:
        """校验后建号；手机号重复抛 PhoneTaken。"""
        validate_phone(phone)
        validate_password(password)
        name = validate_nickname(nickname)
        password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt(self.bcrypt_rounds)).decode()
        return self.users.create_user(phone, password_hash, name)

    def login(self, phone: str, password: str) -> TokenPair:
        """成功返回 token 对；失败统一报"手机号或密码错误"。"""
        user = self.users.get_by_phone(phone)
        if user is None:
            bcrypt.checkpw(password.encode(), _dummy_hash(self.bcrypt_rounds))
            raise AuthError("手机号或密码错误")
        if not bcrypt.checkpw(password.encode(), str(user["password_hash"]).encode()):
            raise AuthError("手机号或密码错误")
        if int(user.get("status", 1)) == 0:
            raise BannedError("账号已停用")
        self.users.touch_login(int(user["id"]))
        return self._issue(int(user["id"]))

    # ------------------------------------------------------------------
    # token
    # ------------------------------------------------------------------

    def refresh(self, refresh_token: str) -> TokenPair:
        """轮换：旧 refresh 立刻删除，发一对新的。"""
        # GETDEL 原子地取出并删除：同一个 refresh token 并发刷新时只有一个能成功
        raw = self.redis.getdel(self._refresh_key(refresh_token))
        if raw is None:
            raise AuthError("refresh token 无效")
        return self._issue(int(raw))

    def logout(self, access_token: str, refresh_token: str | None) -> None:
        """删 refresh、把 access 的 jti 拉黑。"""
        if refresh_token:
            self.redis.delete(self._refresh_key(refresh_token))
        try:
            payload = jwt.decode(access_token, self.secret, algorithms=["HS256"], options={"verify_exp": False})
        except Exception:
            return
        jti = payload.get("jti")
        if jti:
            remaining = int(payload.get("exp", 0) - time.time())
            self.redis.set(f"xm:jwt:deny:{jti}", "1", ex=max(1, remaining))

    def verify(self, access_token: str) -> int:
        """验签、检查过期和黑名单，再确认用户存在且没被封禁。"""
        if not access_token:
            raise AuthError("unauthorized")
        try:
            payload = jwt.decode(access_token, self.secret, algorithms=["HS256"])
        except Exception as exc:
            raise AuthError("unauthorized") from exc
        if payload.get("typ") != "access":
            raise AuthError("unauthorized")
        jti = payload.get("jti")
        if jti and self.redis.exists(f"xm:jwt:deny:{jti}"):
            raise AuthError("unauthorized")
        user = self.users.get_by_id(int(payload["sub"]))
        if user is None or int(user.get("status", 1)) == 0:
            raise AuthError("unauthorized")
        return int(payload["sub"])

    # ------------------------------------------------------------------

    def _issue(self, user_id: int) -> TokenPair:
        """发一对新 token。"""
        now = int(time.time())
        access = jwt.encode(
            {
                "sub": str(user_id),
                "jti": uuid.uuid4().hex,
                "iat": now,
                "exp": now + self.access_ttl,
                "typ": "access",
            },
            self.secret,
            algorithm="HS256",
        )
        refresh_token = secrets.token_urlsafe(32)
        self.redis.set(self._refresh_key(refresh_token), user_id, ex=self.refresh_ttl)
        return TokenPair(access_token=access, refresh_token=refresh_token, expires_in=self.access_ttl)

    @staticmethod
    def _refresh_key(refresh_token: str) -> str:
        """Redis 里只存 refresh token 的哈希。"""
        digest = hashlib.sha256(refresh_token.encode()).hexdigest()
        return f"xm:refresh:{digest}"
