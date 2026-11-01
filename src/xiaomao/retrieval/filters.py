"""检索过滤条件：日期恒定过滤，平台/站点按需拼。"""

from __future__ import annotations

import re
from datetime import date

# 欧盟 27 个成员国（ISO 3166-1 alpha-2）
EU_MEMBERS: frozenset[str] = frozenset({
    "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE", "GR", "HU",
    "IE", "IT", "LV", "LT", "LU", "MT", "NL", "PL", "PT", "RO", "SK", "SI", "ES", "SE",
})

PLATFORM_RE = re.compile(r"^[a-z_]+$")
SITE_RE = re.compile(r"^[A-Z]{2}$")


def build_filter(
    platform: str | None = None,
    site: str | None = None,
    today: date | None = None,
) -> str:
    """拼 Milvus 的 filter 表达式；日期总是过滤，平台站点给了才加。"""
    current = today or date.today()
    stamp = int(current.strftime("%Y%m%d"))
    parts = [f"effective_date <= {stamp}", f"expire_date > {stamp}"]

    if platform is not None:
        if not PLATFORM_RE.match(platform):
            raise ValueError(f"platform 取值非法：{platform!r}")
        parts.append(f'platform in ["{platform}", ""]')

    if site is not None:
        if not SITE_RE.match(site):
            raise ValueError(f"site 取值非法：{site!r}")
        sites = [site, ""]
        if site in EU_MEMBERS:
            sites.append("EU")  # 德国站的问题要能搜到欧盟法规
        joined = ", ".join(f'"{item}"' for item in sites)
        parts.append(f"site in [{joined}]")

    return " and ".join(parts)
