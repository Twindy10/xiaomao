"""从 qa_trace 统计线上链路：延迟、命中率、兜底、tokens、每千次成本。

    uv run python scripts/trace_stats.py --run m6 --users eval/runs/replay_m6/users.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from dotenv import load_dotenv  # noqa: E402

STAGES = ("rewrite", "decide", "encode", "cache", "retrieve", "evidence", "generate")
STAGE_KEYS = ("rewrite", "decide", "translate", "retrieve", "generate", "cache_gate")
PATHS = ("缓存命中", "完整生成", "拒答", "闲聊 / 超出范围")


def read_jsonl(path: Path) -> list[dict]:
    """读 JSONL。"""
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_json(value, default=None):
    """MySQL JSON 列可能是字符串。"""
    if value is None:
        return default if default is not None else {}
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except Exception:
        return default if default is not None else {}


def percentile(values: list[float], ratio: float) -> float:
    """线性插值分位数。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * ratio
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def classify(row: dict) -> str:
    """按链路路径分组。"""
    if int(row.get("cache_hit") or 0):
        return "缓存命中"
    if not row.get("strategy"):
        return "闲聊 / 超出范围"
    if row.get("evidence_sufficient") is not None and int(row["evidence_sufficient"]) == 0:
        return "拒答"
    return "完整生成"


def extras(row: dict) -> dict:
    """decisions._extra。"""
    decisions = parse_json(row.get("decisions"))
    return decisions.get("_extra") or {}


def answer_tokens(row: dict) -> dict:
    """一次回答的各步 tokens。"""
    extra = extras(row)
    llm = extra.get("llm_tokens") or {}
    tokens = {key: int(llm.get(key) or 0) for key in STAGE_KEYS}
    tokens["rerank"] = int(extra.get("rerank_tokens") or 0)
    return tokens


def answer_cost(row: dict, price_in: float, price_out: float, price_rerank: float,
                other_input_ratio: float) -> float:
    """单次成本（元）。生成按真实 input/output，其余按比例估。"""
    tokens = answer_tokens(row)
    prompt = int(row.get("prompt_tokens") or 0)
    completion = int(row.get("completion_tokens") or 0)
    cost = prompt / 1_000_000 * price_in + completion / 1_000_000 * price_out

    other = sum(value for key, value in tokens.items() if key not in ("generate", "rerank"))
    cost += other * other_input_ratio / 1_000_000 * price_in
    cost += other * (1 - other_input_ratio) / 1_000_000 * price_out
    cost += tokens["rerank"] / 1_000_000 * price_rerank
    return cost


def summarize(
    rows: list[dict],
    *,
    price_in: float = 0.2,
    price_out: float = 0.8,
    price_rerank: float = 0.5,
    other_input_ratio: float = 0.85,
) -> dict:
    """总量、路径、延迟、兜底、tokens、成本。"""
    total = len(rows)
    if not total:
        return {"total": 0}

    intents: dict[str, int] = {}
    strategies: dict[str, int] = {}
    for row in rows:
        intents[str(row.get("intent") or "-")] = intents.get(str(row.get("intent") or "-"), 0) + 1
        strategies[str(row.get("strategy") or "-")] = strategies.get(str(row.get("strategy") or "-"), 0) + 1

    chat = [row for row in rows if not row.get("strategy") and not int(row.get("cache_hit") or 0)]
    non_chat = [row for row in rows if row not in chat]
    refused = sum(1 for row in rows if row.get("evidence_sufficient") is not None
                  and int(row["evidence_sufficient"]) == 0)
    backtracked = sum(1 for row in rows if int(row.get("backtracked") or 0) == 1)
    cache_hits = sum(1 for row in rows if int(row.get("cache_hit") or 0))

    # 决策兜底
    fell_back = {name: 0 for name in ("intent", "platform", "site", "strategy")}
    decision_errors = 0
    for row in rows:
        decisions = parse_json(row.get("decisions"))
        extra = decisions.get("_extra") or {}
        if extra.get("decision_error"):
            decision_errors += 1
        for name in fell_back:
            answer = decisions.get(name)
            if isinstance(answer, dict) and answer.get("fell_back"):
                fell_back[name] += 1

    # 按路径分组
    paths: dict[str, dict] = {}
    for path in PATHS:
        group = [row for row in rows if classify(row) == path]
        if not group:
            continue
        latencies = [float(row.get("total_latency_ms") or 0) for row in group]
        stages = {}
        for stage in STAGES:
            values = [float(parse_json(row.get("latency_ms")).get(stage) or 0) for row in group]
            stages[stage] = percentile([value for value in values if value], 0.5)
        costs = [answer_cost(row, price_in, price_out, price_rerank, other_input_ratio) for row in group]
        paths[path] = {
            "count": len(group),
            "p50_ms": percentile(latencies, 0.5),
            "p95_ms": percentile(latencies, 0.95),
            "stage_p50": stages,
            "avg_cost": sum(costs) / len(group),
            "cost_per_1k": sum(costs) / len(group) * 1000,
        }

    tokens = {key: 0 for key in STAGE_KEYS}
    tokens["rerank"] = 0
    for row in rows:
        for key, value in answer_tokens(row).items():
            tokens[key] += value
    tokens_avg = {key: value / total for key, value in tokens.items()}

    all_cost = sum(answer_cost(row, price_in, price_out, price_rerank, other_input_ratio) for row in rows)
    full = paths.get("完整生成")
    cache_group = paths.get("缓存命中")
    no_cache_cost = all_cost
    if full and cache_group:
        no_cache_cost = all_cost - cache_group["avg_cost"] * cache_group["count"] + \
            full["avg_cost"] * cache_group["count"]

    return {
        "total": total,
        "intents": intents,
        "strategies": strategies,
        "refusal_rate": refused / total,
        "backtrack_rate": backtracked / total,
        "cache_hit_rate": cache_hits / len(non_chat) if non_chat else None,
        "cache_hits": cache_hits,
        "fell_back": fell_back,
        "decision_errors": decision_errors,
        "paths": paths,
        "tokens_avg": tokens_avg,
        "cost_per_1k_all": all_cost / total * 1000,
        "cost_per_1k_no_cache": no_cache_cost / total * 1000,
        "price": (price_in, price_out, price_rerank, other_input_ratio),
    }


def render(summary: dict, title: str = "线上链路统计") -> str:
    """Markdown 报告。"""
    if not summary.get("total"):
        return f"# {title}\n\n没有数据。\n"
    lines = [f"# {title}", ""]
    lines.append(f"- 回答数：{summary['total']}")
    lines.append(f"- 拒答率：{summary['refusal_rate']:.1%}，Backtrack 触发率：{summary['backtrack_rate']:.1%}")
    cache_rate = summary["cache_hit_rate"]
    lines.append(f"- 缓存命中率：{'-' if cache_rate is None else f'{cache_rate:.1%}'}"
                 f"（{summary['cache_hits']} 次，分母不含闲聊）")
    lines.append(f"- 决策整体降级：{summary['decision_errors']} 次")
    lines.append("")
    lines.append("## 分布")
    lines.append("")
    lines.append("| 意图 | 次数 |")
    lines.append("|---|---|")
    for name, count in sorted(summary["intents"].items(), key=lambda item: -item[1]):
        lines.append(f"| {name} | {count} |")
    lines.append("")
    lines.append("| 策略 | 次数 |")
    lines.append("|---|---|")
    for name, count in sorted(summary["strategies"].items(), key=lambda item: -item[1]):
        lines.append(f"| {name} | {count} |")
    lines.append("")
    lines.append("## 决策兜底")
    lines.append("")
    lines.append("| 问题 | 兜底比例 |")
    lines.append("|---|---|")
    for name, count in summary["fell_back"].items():
        lines.append(f"| {name} | {count / summary['total']:.1%} |")
    lines.append("")
    lines.append("## 延迟（按路径）")
    lines.append("")
    lines.append("| 路径 | 次数 | 总耗时 P50 | 总耗时 P95 | " + " | ".join(STAGES) + " |")
    lines.append("|---" * (3 + len(STAGES)) + "|")
    for path, row in summary["paths"].items():
        stage_cells = " | ".join(f"{row['stage_p50'].get(stage, 0):.0f}" for stage in STAGES)
        lines.append(f"| {path} | {row['count']} | {row['p50_ms']:.0f}ms | {row['p95_ms']:.0f}ms | {stage_cells} |")
    lines.append("")
    lines.append("## tokens（每次回答平均）")
    lines.append("")
    lines.append("| " + " | ".join(tokens_label() ) + " |")
    lines.append("|" + "---|" * len(tokens_label()))
    lines.append("| " + " | ".join(f"{summary['tokens_avg'].get(key, 0):.0f}" for key in tokens_label()) + " |")
    lines.append("")
    price_in, price_out, price_rerank, ratio = summary["price"]
    lines.append("## 成本")
    lines.append("")
    lines.append(f"单价：输入 {price_in} 元/百万、输出 {price_out} 元/百万、rerank {price_rerank} 元/百万；"
                 f"其余 LLM 步骤按 {ratio:.0%} 输入、{1 - ratio:.0%} 输出估算。")
    lines.append("")
    lines.append("| 路径 | 每千次成本（元） |")
    lines.append("|---|---|")
    for path, row in summary["paths"].items():
        lines.append(f"| {path} | {row['cost_per_1k']:.1f} |")
    lines.append(f"| **整体** | **{summary['cost_per_1k_all']:.1f}** |")
    lines.append(f"| 整体（假设没有缓存） | {summary['cost_per_1k_no_cache']:.1f} |")
    return "\n".join(lines)


def tokens_label() -> list[str]:
    """token 列名。"""
    return [*STAGE_KEYS, "rerank"]


def load_rows(since: str | None = None, users_path: Path | None = None) -> list[dict]:
    """从 MySQL 读 qa_trace。"""
    from xiaomao.runtime import build_pool

    sql = "SELECT * FROM qa_trace"
    params: list = []
    conditions: list[str] = []
    if since:
        conditions.append("created_at >= %s")
        params.append(since)
    if users_path and users_path.is_file():
        text = users_path.read_text(encoding="utf-8").strip()
        if text.startswith("["):  # users.json 是数组，也兼容 JSONL
            user_ids = [int(row["user_id"]) for row in json.loads(text)]
        else:
            user_ids = [int(row["user_id"]) for row in read_jsonl(users_path)]
        if user_ids:
            placeholders = ",".join(["%s"] * len(user_ids))
            conditions.append(f"user_id IN ({placeholders})")
            params.extend(user_ids)
    if conditions:
        sql += " WHERE " + " AND ".join(conditions)
    sql += " ORDER BY id"

    pool = build_pool()
    try:
        with pool.connection() as conn, conn.cursor() as cursor:
            cursor.execute(sql, tuple(params))
            columns = [item[0] for item in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
    finally:
        pool.close()


def main(argv: list[str] | None = None) -> int:
    """统计并写报告。"""
    load_dotenv(ROOT_DIR / ".env", override=True)
    parser = argparse.ArgumentParser(description="线上链路统计")
    parser.add_argument("--run", default="m6")
    parser.add_argument("--since", default=None, help="例如 '2026-11-01 00:00:00'")
    parser.add_argument("--users", type=Path, default=None)
    parser.add_argument("--price-in", type=float, default=0.2)
    parser.add_argument("--price-out", type=float, default=0.8)
    parser.add_argument("--price-rerank", type=float, default=0.5)
    parser.add_argument("--other-input-ratio", type=float, default=0.85)
    args = parser.parse_args(argv)

    rows = load_rows(args.since, args.users)
    summary = summarize(
        rows, price_in=args.price_in, price_out=args.price_out,
        price_rerank=args.price_rerank, other_input_ratio=args.other_input_ratio,
    )
    report = render(summary, f"线上链路统计（{args.run}）")
    out_dir = ROOT_DIR / "eval" / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"trace_stats_{args.run}.md"
    out_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n报告 -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
