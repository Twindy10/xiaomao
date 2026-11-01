"""运营复核：把待处理的点踩和链路日志对起来，输出 Markdown。

注意：这是运营脚本，**不按 user_id 过滤**，会把所有用户的点踩都列出来。

    uv run python scripts/feedback_report.py [--limit 50]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

for _name in ("NO_PROXY", "no_proxy"):
    os.environ[_name] = ",".join(filter(None, [os.environ.get(_name), "127.0.0.1,localhost"]))

from dotenv import load_dotenv  # noqa: E402

REASON_TEXT = {"wrong": "答错了", "outdated": "政策过时", "off_topic": "答非所问", "other": "其他"}


def _loads(value, default):
    """把 MySQL 里的 JSON 列解析出来。"""
    if value is None:
        return default
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except Exception:
        return default


def classify(row: dict) -> str:
    """粗分一类，方便运营排优先级。"""
    if int(row.get("cache_hit") or 0):
        return "缓存误命中"
    if row.get("evidence_sufficient") == 0:
        return "拒答"
    scores = _loads(row.get("rerank_scores"), [])
    if scores and float(scores[0]) < 0.5:
        return "证据弱"
    return "其他"


def build_markdown(rows: list[dict]) -> str:
    """每条反馈一小节。"""
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(classify(row), []).append(row)

    lines = [f"# 点踩复核（{datetime.now().strftime('%Y-%m-%d %H:%M')}）", ""]
    lines.append(f"共 {len(rows)} 条待处理，按类型：")
    lines.append("")
    for name, items in sorted(groups.items(), key=lambda item: -len(item[1])):
        lines.append(f"- {name}：{len(items)} 条")
    lines.append("")

    for name, items in groups.items():
        lines.append(f"## {name}（{len(items)} 条）")
        lines.append("")
        for row in items:
            decisions = _loads(row.get("decisions"), {})
            extra = decisions.get("_extra") or {}
            scores = _loads(row.get("rerank_scores"), [])
            citations = _loads(row.get("citations"), [])
            urls = ", ".join(str(item.get("source_url")) for item in citations if item.get("source_url"))
            answer = str(row.get("answer") or "").replace("\n", " ")[:200]

            lines.append(f"### message {row['message_id']}（user {row['user_id']}，"
                         f"{REASON_TEXT.get(row.get('reason') or '', row.get('reason') or '未选原因')}，"
                         f"{row.get('created_at')}）")
            lines.append("")
            lines.append(f"- 原问题：{row.get('query_raw')}")
            lines.append(f"- 改写后：{row.get('query_rewritten')}")
            lines.append(f"- 意图 / 策略：{row.get('intent')} / {row.get('strategy')}")
            lines.append(f"- 命中缓存：{'是' if int(row.get('cache_hit') or 0) else '否'}")
            lines.append(f"- 证据是否充分：{row.get('evidence_sufficient')}")
            lines.append(f"- rerank 第 1 名：{round(float(scores[0]), 3) if scores else '-'}")
            lines.append(f"- 引用：{urls or '-'}")
            lines.append(f"- 备注：{row.get('comment') or '-'}")
            if extra.get("filters"):
                lines.append(f"- 过滤条件：{extra['filters']}")
            lines.append(f"- 回答：{answer}")
            lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """读待处理点踩，写 Markdown。"""
    load_dotenv(ROOT_DIR / ".env", override=True)
    parser = argparse.ArgumentParser(description="点踩复核报告")
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args(argv)

    from xiaomao.runtime import build_pool
    from xiaomao.storage.feedback_store import FeedbackStore

    pool = build_pool()
    try:
        rows = FeedbackStore(pool).list_pending(limit=args.limit)
    finally:
        pool.close()

    out_dir = ROOT_DIR / "eval" / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"feedback_{datetime.now().strftime('%Y%m%d')}.md"
    out_path.write_text(build_markdown(rows), encoding="utf-8")
    print(f"{len(rows)} 条待处理点踩 -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
