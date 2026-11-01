"""手动同步一次用量。

    uv run python scripts/sync_usage.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from dotenv import load_dotenv  # noqa: E402


def main() -> int:
    """同步 Redis 里待写回的用量。"""
    load_dotenv(ROOT_DIR / ".env", override=True)

    from xiaomao.runtime import build_pool, redis_from_env
    from xiaomao.user.usage import sync_usage

    pool = build_pool()
    try:
        count = sync_usage(redis_from_env(), pool)
    finally:
        pool.close()
    print(f"同步 {count} 条用量")
    return 0


if __name__ == "__main__":
    sys.exit(main())
