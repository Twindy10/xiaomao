"""账号测试：注册校验、登录、JWT、登出、刷新轮换。"""

from __future__ import annotations

import time

import jwt
import pytest

from xiaomao.user.auth import AuthError, AuthService, mask_phone
from xiaomao.user.store import PhoneTaken

SECRET = "test-secret"
PHONE = "13800138000"


class FakeRedis:
    """最小 Redis。"""

    def __init__(self) -> None:
        self.data: dict[str, str] = {}
        self.expires: dict[str, int] = {}

    def get(self, key):
        """读。"""
        return self.data.get(key)

    def set(self, key, value, ex=None):
        """写。"""
        self.data[key] = str(value)
        if ex:
            self.expires[key] = ex

    def delete(self, key):
        """删。"""
        self.data.pop(key, None)

    def getdel(self, key):
        """取出并删除。"""
        return self.data.pop(key, None)

    def exists(self, key):
        """是否存在。"""
        return key in self.data


class FakeUsers:
    """内存用户表。"""

    def __init__(self) -> None:
        self.rows: dict[int, dict] = {}
        self._next = 0
        self.touched = 0

    def create_user(self, phone: str, password_hash: str, nickname: str | None) -> int:
        """建号，重复手机号抛 PhoneTaken。"""
        if any(row["phone"] == phone for row in self.rows.values()):
            raise PhoneTaken(phone)
        self._next += 1
        self.rows[self._next] = {
            "id": self._next, "phone": phone, "password_hash": password_hash,
            "nickname": nickname, "status": 1, "plan_id": 1,
        }
        return self._next

    def get_by_phone(self, phone: str):
        """按手机号查。"""
        return next((row for row in self.rows.values() if row["phone"] == phone), None)

    def get_by_id(self, user_id: int):
        """按主键查。"""
        return self.rows.get(user_id)

    def touch_login(self, user_id: int) -> None:
        """记录调用。"""
        self.touched += 1


def build() -> tuple[AuthService, FakeUsers, FakeRedis]:
    """构造服务。"""
    users = FakeUsers()
    redis_client = FakeRedis()
    service = AuthService(users, redis_client, secret=SECRET, bcrypt_rounds=4)
    return service, users, redis_client


def test_register_validation() -> None:
    """手机号和密码格式。"""
    service, _, _ = build()
    with pytest.raises(ValueError):
        service.register("12345", "abcd1234")
    with pytest.raises(ValueError):
        service.register(PHONE, "short1")
    with pytest.raises(ValueError):
        service.register(PHONE, "abcdefgh")  # 没有数字


def test_duplicate_phone() -> None:
    """重复手机号抛 PhoneTaken。"""
    service, _, _ = build()
    service.register(PHONE, "abcd1234")
    with pytest.raises(PhoneTaken):
        service.register(PHONE, "abcd1234")


def test_login_success() -> None:
    """登录成功发一个 access 和一个 refresh。"""
    service, users, redis_client = build()
    service.register(PHONE, "abcd1234", "小贸")
    pair = service.login(PHONE, "abcd1234")

    assert pair.access_token and pair.refresh_token
    assert users.touched == 1
    assert any(key.startswith("xm:refresh:") for key in redis_client.data)


def test_login_failures_look_the_same() -> None:
    """手机号不存在和密码错误返回同一个错误。"""
    service, _, _ = build()
    service.register(PHONE, "abcd1234")

    with pytest.raises(AuthError) as missing:
        service.login("13900139000", "abcd1234")
    with pytest.raises(AuthError) as wrong:
        service.login(PHONE, "wrong1234")
    assert str(missing.value) == str(wrong.value) == "手机号或密码错误"


def test_disabled_account() -> None:
    """status=0 不能登录。"""
    service, users, _ = build()
    service.register(PHONE, "abcd1234")
    users.rows[1]["status"] = 0
    with pytest.raises(AuthError):
        service.login(PHONE, "abcd1234")


def test_verify_ok_and_bad_signature() -> None:
    """正常 token 通过，签名不对报错。"""
    service, _, _ = build()
    service.register(PHONE, "abcd1234")
    pair = service.login(PHONE, "abcd1234")

    assert service.verify(pair.access_token) == 1

    forged = jwt.encode({"sub": "1", "typ": "access", "exp": int(time.time()) + 60}, "other", algorithm="HS256")
    with pytest.raises(AuthError):
        service.verify(forged)


def test_expired_token() -> None:
    """过期 token 报错。"""
    service, _, _ = build()
    service.register(PHONE, "abcd1234")
    expired = jwt.encode(
        {"sub": "1", "typ": "access", "jti": "x", "exp": int(time.time()) - 10}, SECRET, algorithm="HS256"
    )
    with pytest.raises(AuthError):
        service.verify(expired)


def test_logout_denies_access() -> None:
    """登出后 access token 失效。"""
    service, _, _ = build()
    service.register(PHONE, "abcd1234")
    pair = service.login(PHONE, "abcd1234")

    service.logout(pair.access_token, pair.refresh_token)
    with pytest.raises(AuthError):
        service.verify(pair.access_token)


def test_refresh_rotation() -> None:
    """刷新后旧 refresh 不能再用。"""
    service, _, _ = build()
    service.register(PHONE, "abcd1234")
    pair = service.login(PHONE, "abcd1234")

    new_pair = service.refresh(pair.refresh_token)
    assert new_pair.refresh_token != pair.refresh_token
    with pytest.raises(AuthError):
        service.refresh(pair.refresh_token)


def test_mask_phone() -> None:
    """手机号打码。"""
    assert mask_phone("13800138000") == "138****8000"
    assert mask_phone(None) == ""
