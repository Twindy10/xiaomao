"""卖家画像：校验、MySQL upsert、Redis cache-aside。"""

from __future__ import annotations

import json

from pymysql.cursors import DictCursor

from xiaomao.decision.questions import PLATFORMS, SITES
from xiaomao.storage.db import MySQLPool

MAX_CATEGORIES = 10
CATEGORY_MIN, CATEGORY_MAX = 1, 20
STAGES = {"newbie", "growing"}
PLATFORM_KEYS = [key for key in PLATFORMS if key != "none"]
SITE_KEYS = [key for key in SITES if key != "none"]
EMPTY_PROFILE = {"platforms": [], "sites": [], "categories": [], "stage": None}


def _clean_list(values, allowed: list[str] | None, label: str, limit: int | None = None) -> list[str]:
    """去重、保序、校验取值。"""
    if values is None:
        return []
    if not isinstance(values, list):
        raise ValueError(f"{label}必须是数组")
    cleaned: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text or text in cleaned:
            continue
        if allowed is not None and text not in allowed:
            raise ValueError(f"{label}取值不合法：{text}")
        if limit is not None and not CATEGORY_MIN <= len(text) <= CATEGORY_MAX:
            raise ValueError(f"{label}每项必须是 {CATEGORY_MIN}–{CATEGORY_MAX} 个字")
        cleaned.append(text)
    if limit is not None and len(cleaned) > limit:
        raise ValueError(f"{label}最多 {limit} 个")
    return cleaned


def validate_profile(profile: dict) -> dict:
    """整体校验，不合格抛 ValueError。"""
    if not isinstance(profile, dict):
        raise ValueError("画像必须是对象")
    platforms = _clean_list(profile.get("platforms"), PLATFORM_KEYS, "平台")
    sites = _clean_list(profile.get("sites"), SITE_KEYS, "站点")
    categories = _clean_list(profile.get("categories"), None, "品类", limit=MAX_CATEGORIES)
    stage = profile.get("stage")
    if stage in ("", None):
        stage = None
    elif stage not in STAGES:
        raise ValueError(f"经营阶段取值不合法：{stage}")
    return {"platforms": platforms, "sites": sites, "categories": categories, "stage": stage}


def profile_filters(profile: dict | None) -> dict | None:
    """只主营一个平台 / 站点时补进检索条件；多个不猜。"""
    if not profile:
        return None
    platforms = profile.get("platforms") or []
    sites = profile.get("sites") or []
    platform = platforms[0] if len(platforms) == 1 else None
    site = sites[0] if len(sites) == 1 else None
    if platform is None and site is None:
        return None
    return {"platform": platform, "site": site}


def _as_list(value) -> list:
    """MySQL JSON 列可能返回已解析的 list，也可能返回字符串。"""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, (bytes, str)):
        try:
            parsed = json.loads(value)
        except Exception:
            return []
        return list(parsed) if isinstance(parsed, list) else []
    return []


def profile_options() -> dict:
    """前端渲染选项用。"""
    return {
        "platforms": {key: PLATFORMS[key] for key in PLATFORM_KEYS},
        "sites": {key: SITES[key] for key in SITE_KEYS},
        "stages": {"newbie": "刚起步", "growing": "在增长"},
    }


class ProfileStore:
    """cache-aside：读先 Redis 后 MySQL；写先 MySQL 再删 Redis。"""

    def __init__(self, pool: MySQLPool, redis_client, ttl_seconds: int = 3600) -> None:
        self.pool = pool
        self.redis = redis_client
        self.ttl = ttl_seconds

    @staticmethod
    def _key(user_id: int) -> str:
        """缓存 key。"""
        return f"xm:profile:{user_id}"

    def get(self, user_id: int) -> dict:
        """没填过返回空画像。"""
        cached = self.redis.get(self._key(user_id))
        if cached:
            try:
                return json.loads(cached if isinstance(cached, str) else cached.decode())
            except Exception:
                pass

        with self.pool.connection() as conn, conn.cursor(DictCursor) as cursor:
            cursor.execute(
                "SELECT platforms, sites, categories, stage FROM seller_profile WHERE user_id=%s",
                (user_id,),
            )
            row = cursor.fetchone()
        profile = dict(EMPTY_PROFILE)
        if row:
            profile = {
                "platforms": _as_list(row.get("platforms")),
                "sites": _as_list(row.get("sites")),
                "categories": _as_list(row.get("categories")),
                "stage": row.get("stage"),
            }
        self.redis.set(self._key(user_id), json.dumps(profile, ensure_ascii=False), ex=self.ttl)
        return profile

    def save(self, user_id: int, profile: dict) -> dict:
        """校验后整体覆盖，再删缓存。"""
        cleaned = validate_profile(profile)
        with self.pool.connection() as conn, conn.cursor(DictCursor) as cursor:
            cursor.execute(
                "INSERT INTO seller_profile (user_id, platforms, sites, categories, stage) "
                "VALUES (%s, %s, %s, %s, %s) "
                "ON DUPLICATE KEY UPDATE platforms=VALUES(platforms), sites=VALUES(sites), "
                "categories=VALUES(categories), stage=VALUES(stage)",
                (
                    user_id,
                    json.dumps(cleaned["platforms"], ensure_ascii=False),
                    json.dumps(cleaned["sites"], ensure_ascii=False),
                    json.dumps(cleaned["categories"], ensure_ascii=False),
                    cleaned["stage"],
                ),
            )
        self.redis.delete(self._key(user_id))
        return cleaned
