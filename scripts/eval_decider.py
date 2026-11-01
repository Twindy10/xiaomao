"""决策模型对比：准确率、延迟、校准误差、兜底比例、离线阈值扫描。

    uv run python scripts/eval_decider.py --backend jev_style --run m6
    uv run python scripts/eval_decider.py --backend systemone --name kev --base-url http://127.0.0.1:8008 --model kev-latest --run m6
    uv run python scripts/eval_decider.py --backend fallback --primary systemone --name kev --base-url http://127.0.0.1:8008 --run m6
    uv run python scripts/eval_decider.py compare --run m6
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

for _name in ("NO_PROXY", "no_proxy"):
    os.environ[_name] = ",".join(filter(None, [os.environ.get(_name), "127.0.0.1,localhost"]))

from dotenv import load_dotenv  # noqa: E402

from xiaomao.decision import (  # noqa: E402
    ROUTING_QUESTIONS,
    DeciderError,
    FallbackDecider,
    JevStyleDecider,
    LLMJsonDecider,
    SystemOneDecider,
)

DATA = ROOT_DIR / "data" / "eval" / "intent.jsonl"
NO_RETRIEVAL = {"chitchat", "out_of_scope"}
THRESHOLDS = [index / 10 for index in range(10)]


def read_jsonl(path: Path) -> list[dict]:
    """读 JSONL。"""
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """先写临时文件再改名。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def gold_value(item: dict, name: str) -> str:
    """标注里的 null 对应选项 none。"""
    value = item.get(name)
    return "none" if value in (None, "") else str(value)


def percentile(values: list[float], ratio: float) -> float:
    """线性插值分位数。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * ratio
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def expected_calibration_error(pairs: list[tuple[float, bool]], bins: int = 10) -> float | None:
    """ECE：按第一名概率分 10 个等宽箱，算 |平均置信度 − 准确率| 的加权和。"""
    if not pairs:
        return None
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for confidence, correct in pairs:
        index = min(int(max(0.0, min(1.0, confidence)) * bins), bins - 1)
        buckets[index].append((confidence, correct))

    total = len(pairs)
    error = 0.0
    for bucket in buckets:
        if not bucket:
            continue
        mean_confidence = sum(item[0] for item in bucket) / len(bucket)
        accuracy = sum(1 for item in bucket if item[1]) / len(bucket)
        error += len(bucket) / total * abs(mean_confidence - accuracy)
    return error


def build_decider(args: argparse.Namespace):
    """按 backend 构造决策器。"""
    if args.backend == "jev_style":
        return JevStyleDecider()
    if args.backend == "llm_json":
        return LLMJsonDecider()
    if args.backend == "systemone":
        return SystemOneDecider(
            args.base_url, backend=args.name or "systemone",
            model=args.model or None, api_key=args.api_key or None,
        )
    if args.backend == "fallback":
        if args.primary == "jev_style":
            primary = JevStyleDecider()
        else:
            primary = SystemOneDecider(
                args.base_url, backend=args.name or "systemone",
                model=args.model or None, api_key=args.api_key or None,
            )
        return FallbackDecider(primary, LLMJsonDecider(), threshold=args.threshold)
    raise SystemExit(f"[ERROR] 未知 backend：{args.backend}")


def summarize(rows: list[dict], items: list[dict], names: list[str]) -> dict:
    """算准确率、延迟、ECE、兜底比例。"""
    gold = {item["id"]: item for item in items}
    correct = {name: 0 for name in names}
    total = {name: 0 for name in names}
    routing_correct = 0
    routing_total = 0
    business_missed = 0
    confusion: Counter = Counter()
    latencies: list[float] = []
    tokens: list[int] = []
    fell_back = 0
    calibration: list[tuple[float, bool]] = []

    for row in rows:
        item = gold.get(row["qid"])
        if item is None:
            continue
        latencies.append(float(row.get("latency_ms") or 0))
        tokens.append(int(row.get("tokens") or 0))
        if row.get("fell_back"):
            fell_back += 1

        answers = row.get("answers") or {}
        for name in names:
            answer = answers.get(name)
            if answer is None:
                continue
            expected = gold_value(item, name)
            if name == "strategy" and item.get("strategy") is None:
                continue  # 闲聊/超范围没有策略
            total[name] += 1
            if str(answer["value"]) == expected:
                correct[name] += 1
            elif name == "intent":
                confusion[(expected, str(answer["value"]))] += 1

        intent_answer = answers.get("intent") or {}
        predicted = str(intent_answer.get("value"))
        expected_intent = gold_value(item, "intent")
        routing_total += 1
        if (predicted in NO_RETRIEVAL) == (expected_intent in NO_RETRIEVAL):
            routing_correct += 1
        if expected_intent not in NO_RETRIEVAL and predicted in NO_RETRIEVAL:
            business_missed += 1
        if intent_answer:
            probabilities = intent_answer.get("probabilities") or {}
            if probabilities:
                top = max(float(value) for value in probabilities.values())
            else:  # 老结果没有 probabilities，退回 confidence
                top = float(intent_answer.get("confidence") or 0.0)
            calibration.append((top, predicted == expected_intent))

    return {
        "n": len(rows),
        "accuracy": {
            name: (correct[name] / total[name]) if total[name] else None for name in names
        },
        "routing_accuracy": (routing_correct / routing_total) if routing_total else None,
        "business_missed": business_missed,
        "p50_ms": percentile(latencies, 0.5),
        "p95_ms": percentile(latencies, 0.95),
        "fallback_ratio": (fell_back / len(rows)) if rows else 0.0,
        "avg_tokens": (sum(tokens) / len(tokens)) if tokens else 0.0,
        "ece": expected_calibration_error(calibration),
        "confusion": [[a, b, c] for (a, b), c in confusion.most_common(8)],
    }


def offline_thresholds(
    primary_rows: list[dict],
    fallback_rows: list[dict],
    items: list[dict],
    names: list[str],
    thresholds: list[float] | None = None,
) -> dict:
    """用已落盘的两份结果离线拼出不同阈值下的准确率和兜底比例。"""
    primary = {row["qid"]: row for row in primary_rows}
    fallback = {row["qid"]: row for row in fallback_rows}
    result: dict[str, dict] = {}
    for threshold in thresholds or THRESHOLDS:
        merged: list[dict] = []
        for qid, row in primary.items():
            backup = fallback.get(qid)
            answers = dict(row.get("answers") or {})
            used = False
            if backup is not None:
                for name, answer in answers.items():
                    if name == "strategy":
                        continue  # 策略阈值为 0，不兜底
                    if float(answer.get("confidence") or 0) < threshold:
                        answers[name] = dict(backup["answers"][name], fell_back=True)
                        used = True
            merged.append({"qid": qid, "answers": answers, "fell_back": used,
                           "latency_ms": 0, "tokens": 0})
        summary = summarize(merged, items, names)
        result[f"{threshold:g}"] = {
            "accuracy": summary["accuracy"],
            "fallback_ratio": summary["fallback_ratio"],
        }
    return result


def run_backend(args: argparse.Namespace) -> int:
    """跑一个后端。"""
    load_dotenv(ROOT_DIR / ".env", override=True)
    items = read_jsonl(DATA)
    if args.limit:
        items = items[: args.limit]
    run_dir = ROOT_DIR / "eval" / "runs" / f"decider_{args.run}"
    name = args.name or args.backend
    out_path = run_dir / f"{name}.jsonl"
    done = {row["qid"] for row in read_jsonl(out_path)}

    decider = build_decider(args)
    rows: list[dict] = []
    for index, item in enumerate(items, start=1):
        if item["id"] in done:
            continue
        started = time.perf_counter()
        try:
            answers = decider.decide(item["query"], ROUTING_QUESTIONS)
        except DeciderError as exc:
            print(f"[WARN] {item['id']} 决策失败：{exc}")
            continue
        elapsed = (time.perf_counter() - started) * 1000
        rows.append({
            "qid": item["id"],
            "answers": {
                key: {
                    "value": answer.value,
                    "confidence": answer.confidence,
                    "probabilities": answer.probabilities,
                    "fell_back": answer.fell_back,
                }
                for key, answer in answers.items()
            },
            "fell_back": any(answer.fell_back for answer in answers.values()),
            "latency_ms": elapsed,
            "tokens": int(getattr(decider, "last_tokens", 0) or 0),
        })
        if index % 50 == 0:
            print(f"  已完成 {index}/{len(items)}")

    write_jsonl(out_path, read_jsonl(out_path) + rows)
    all_rows = read_jsonl(out_path)
    names = [question.name for question in ROUTING_QUESTIONS]
    summary = summarize(all_rows, items, names)
    (run_dir / f"summary_{name}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n{name}（{args.backend}）：{summary['n']} 条")
    for question_name in names:
        value = summary["accuracy"][question_name]
        print(f"  {question_name:<10} {'-' if value is None else f'{value:.1%}'}")
    ece = summary["ece"]
    print(f"  该不该检索 {summary['routing_accuracy']:.1%}，业务问题被判成不检索 {summary['business_missed']} 条")
    print(f"  P50 {summary['p50_ms']:.0f}ms / P95 {summary['p95_ms']:.0f}ms，"
          f"兜底 {summary['fallback_ratio']:.1%}，平均 {summary['avg_tokens']:.0f} tokens，"
          f"ECE {'-' if ece is None else f'{ece:.3f}'}")
    if summary["confusion"]:
        print("  意图混淆 Top：" + "，".join(f"{a}->{b}x{c}" for a, b, c in summary["confusion"][:5]))

    fallback_path = run_dir / "llm_json.jsonl"
    if fallback_path.is_file() and name != "llm_json":
        offline = offline_thresholds(all_rows, read_jsonl(fallback_path), items, names)
        (run_dir / f"thresholds_{name}.json").write_text(
            json.dumps(offline, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("\n离线阈值扫描（和 llm_json 拼接，不重复调用）：")
        print("| 阈值 | 意图 | 平台 | 站点 | 走 LLM 比例 |")
        print("|---|---|---|---|---|")
        for threshold, row in offline.items():
            accuracy = row["accuracy"]
            print(f"| {threshold} | {accuracy['intent']:.1%} | {accuracy['platform']:.1%} | "
                  f"{accuracy['site']:.1%} | {row['fallback_ratio']:.1%} |")
    elif not fallback_path.is_file():
        print(f"\n[提示] {fallback_path.name} 不存在，跳过离线阈值扫描")
    return 0


def compare(args: argparse.Namespace) -> int:
    """读同一 run 下所有后端的结果，输出对比表。"""
    run_dir = ROOT_DIR / "eval" / "runs" / f"decider_{args.run}"
    items = read_jsonl(DATA)
    if args.limit:
        items = items[: args.limit]
    names = [question.name for question in ROUTING_QUESTIONS]

    rows = []
    for path in sorted(run_dir.glob("*.jsonl")):
        name = path.stem
        summary = summarize(read_jsonl(path), items, names)
        rows.append((name, summary))
    if not rows:
        print(f"[ERROR] {run_dir} 下没有结果文件")
        return 1

    def pct(value):
        return "-" if value is None else f"{value * 100:.1f}%"

    lines = ["# 决策模型对比", ""]
    lines.append("| 后端 | 意图 | 平台 | 站点 | 该不该检索 | 策略 | P50 | P95 | ECE | 走 LLM 比例 | 每千次 tokens |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for name, summary in rows:
        accuracy = summary["accuracy"]
        ece = summary["ece"]
        ece_text = "-" if ece is None else f"{ece:.3f}"
        lines.append(
            f"| {name} | {pct(accuracy['intent'])} | {pct(accuracy['platform'])} | {pct(accuracy['site'])} | "
            f"{pct(summary['routing_accuracy'])} | {pct(accuracy['strategy'])} | "
            f"{summary['p50_ms']:.0f}ms | {summary['p95_ms']:.0f}ms | {ece_text} | "
            f"{summary['fallback_ratio']:.1%} | {summary['avg_tokens'] * 1000:.0f} |"
        )
    lines.append("")
    (run_dir / "compare.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n对比表 -> {run_dir / 'compare.md'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """解析子命令。"""
    parser = argparse.ArgumentParser(description="决策模型对比")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="跑一个后端")
    run.add_argument("--backend", choices=["jev_style", "llm_json", "systemone", "fallback"], required=True)
    run.add_argument("--primary", choices=["jev_style", "systemone"], default="jev_style")
    run.add_argument("--name", default="")
    run.add_argument("--base-url", default="http://127.0.0.1:8008")
    run.add_argument("--model", default="")
    run.add_argument("--api-key", default="")
    run.add_argument("--threshold", type=float, default=0.5)
    run.add_argument("--run", default="m6")
    run.add_argument("--limit", type=int, default=0)
    run.set_defaults(func=run_backend)

    cmp_parser = sub.add_parser("compare", help="输出对比表")
    cmp_parser.add_argument("--run", default="m6")
    cmp_parser.add_argument("--limit", type=int, default=0)
    cmp_parser.set_defaults(func=compare)

    args = parser.parse_args(argv)
    if args.command == "run":
        return run_backend(args)
    return compare(args)


if __name__ == "__main__":
    sys.exit(main())
