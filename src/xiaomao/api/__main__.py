"""`uv run python -m xiaomao.api` 启动服务。"""

from __future__ import annotations

import sys


def main() -> int:
    """等价于 uvicorn xiaomao.api.app:app --workers 1。"""
    sys.stdout.reconfigure(encoding="utf-8")
    import uvicorn

    uvicorn.run("xiaomao.api.app:app", host="127.0.0.1", port=8000, workers=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
