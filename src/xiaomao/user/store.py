"""用户存储。除登录外，所有查询都带 id / user_id。"""

from __future__ import annotations

import pymysql
from pymysql.cursors import DictCursor

from xiaomao.storage.db import MySQLPool


class PhoneTaken(Exception):
    """手机号已注册。"""


class UserStore:
    """`user` 表读写。"""

    def __init__(self, pool: MySQLPool) -> None:
        self.pool = pool

    def create_user(self, phone: str, password_hash: str, nickname: str | None) -> int:
        """插入用户；唯一索引冲突转成 PhoneTaken。"""
        try:
            with self.pool.connection() as conn, conn.cursor(DictCursor) as cursor:
                cursor.execute(
                    "INSERT INTO `user` (phone, password_hash, nickname) VALUES (%s, %s, %s)",
                    (phone, password_hash, nickname),
                )
                return int(cursor.lastrowid)
        except pymysql.err.IntegrityError as exc:
            if exc.args and int(exc.args[0]) == 1062:
                raise PhoneTaken(phone) from exc
            raise

    def get_by_phone(self, phone: str) -> dict | None:
        """登录用：按手机号查，天然没有 user_id。"""
        with self.pool.connection() as conn, conn.cursor(DictCursor) as cursor:
            cursor.execute("SELECT * FROM `user` WHERE phone=%s", (phone,))
            return cursor.fetchone()

    def get_by_id(self, user_id: int) -> dict | None:
        """按主键查，连 plan 带出套餐信息。"""
        with self.pool.connection() as conn, conn.cursor(DictCursor) as cursor:
            cursor.execute(
                "SELECT u.*, p.code AS plan_code, p.daily_quota, p.max_history_days "
                "FROM `user` u JOIN `plan` p ON p.id = u.plan_id WHERE u.id=%s",
                (user_id,),
            )
            return cursor.fetchone()

    def touch_login(self, user_id: int) -> None:
        """更新最近登录时间。"""
        with self.pool.connection() as conn, conn.cursor(DictCursor) as cursor:
            cursor.execute("UPDATE `user` SET last_login_at=NOW(3) WHERE id=%s", (user_id,))
