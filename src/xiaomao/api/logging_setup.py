"""应用日志：JSON 行、request_id、控制台 + 轮转文件。"""

from __future__ import annotations

import contextvars
import json
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

REQUEST_ID = contextvars.ContextVar("request_id", default="-")
_configured = False


class JsonFormatter(logging.Formatter):
    """每行一个 JSON，额外字段通过 extra={"fields": {...}} 传入。"""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "request_id": REQUEST_ID.get(),
            "msg": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            payload.update(fields)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(log_file: str | Path | None = "logs/xiaomao.log", level: int = logging.INFO) -> None:
    """启动时调用一次；重复调用不重复加 handler。"""
    global _configured
    if _configured:
        return
    formatter = JsonFormatter()
    root = logging.getLogger()
    root.setLevel(level)

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8")
        handler.setLevel(level)
        handler.setFormatter(formatter)
        root.addHandler(handler)
    _configured = True


def new_request_id() -> str:
    """生成一个短 request_id。"""
    import uuid

    return uuid.uuid4().hex[:12]
