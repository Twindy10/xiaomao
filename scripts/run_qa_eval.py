"""整条链路评估：跑问答、评 Faithfulness、出报告。

    uv run python scripts/run_qa_eval.py run    --run m4 --config rerank
    uv run python scripts/run_qa_eval.py judge  --run m4 --config rerank
    uv run python scripts/run_qa_eval.py grade  --run m4 --config rerank
    uv run python scripts/run_qa_eval.py report --run m4
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import median

sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

for _name in ("NO_PROXY", "no_proxy"):
    os.environ[_name] = ",".join(filter(None, [os.environ.get(_name), "127.0.0.1,localhost"]))

from dotenv import load_dotenv  # noqa: E402

from xiaomao.textnorm import normalize_for_match  # noqa: E402

DATA_DIR = ROOT_DIR / "data"
RUNS_ROOT = ROOT_DIR / "eval" / "runs"

CONFIGS = {
    "hybrid": {"strategy": "direct", "use_rerank": False, "profile": False},
    "rerank": {"strategy": "direct", "use_rerank": True, "profile": False},
    "routing": {"strategy": "auto", "use_rerank": True, "profile": False},
    "profile": {"strategy": "auto", "use_rerank": True, "profile": True},
    "no_gate": {"strategy": "auto", "use_rerank": True, "profile": False, "judge_evidence": False},
    "no_translate": {"strategy": "auto", "use_rerank": True, "profile": False, "translate": False},
}

# 不开思考时评委会把原文照抄的句子也判成没依据（3 题里错 2 题）；要求给出最接近的原文、开思考后 3 题全对
JUDGE_SYSTEM = (
    "你是严格的事实核查员。逐条检查回答里的事实性结论（数字、日期、条件、适用范围、做法），"
    "在资料里找支撑它的原文。措辞不同、翻译、合理概括都算有依据；"
    "资料里找不到、或和资料矛盾的才算没有依据。先找原文再下结论，不要凭印象。"
    "只输出 JSON：{\"unsupported\": [{\"claim\": \"回答里的原句\", \"closest\": \"资料里最接近的原文，没有就写空\"}], "
    "\"faithful\": true/false}。unsupported 为空时 faithful 为 true。"
)


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


def percentile(values: list[float], ratio: float) -> float:
    """线性插值分位数。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * ratio
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


# ---------------------------------------------------------------------------
# 数据与指标
# ---------------------------------------------------------------------------


def load_reference(data_dir: Path = DATA_DIR) -> tuple[list[dict], dict[str, dict]]:
    """题目 + chunk_id -> 片段。"""
    items = read_jsonl(data_dir / "eval" / "retrieval_qa.jsonl")
    chunks = {row["chunk_id"]: row for row in read_jsonl(data_dir / "processed" / "chunks.jsonl")}
    return items, chunks


def targets_for(item: dict, chunks_by_doc: dict[str, list[dict]]) -> dict[str, set[str]]:
    """每条证据 quote 落在哪些片段（doc_key → chunk_id 集合）。"""
    targets: dict[str, set[str]] = {}
    for entry in item.get("evidence") or []:
        quote = normalize_for_match(entry["quote"])
        for chunk in chunks_by_doc.get(entry["doc_key"], []):
            if quote in normalize_for_match(chunk["text"]):
                targets.setdefault(entry["doc_key"], set()).add(chunk["chunk_id"])
    return targets


def rank_of(ids: list[str], targets: set[str], limit: int) -> int | None:
    """第一个目标片段的名次。"""
    for position, chunk_id in enumerate(ids[:limit], start=1):
        if chunk_id in targets:
            return position
    return None


def evaluate(rows: list[dict], items: list[dict], chunks: dict[str, dict]) -> dict:
    """汇总一份 run 的指标。"""
    chunks_by_doc: dict[str, list[dict]] = {}
    for chunk in chunks.values():
        chunks_by_doc.setdefault(chunk["doc_key"], []).append(chunk)
    by_qid = {item["id"]: item for item in items}

    answerable = [row for row in rows if by_qid.get(row["qid"], {}).get("answerable") is True]
    unanswerable = [row for row in rows if by_qid.get(row["qid"], {}).get("answerable") is False]

    hit5 = mrr5 = all_hit = 0
    single_total = single_hit = 0
    multi_total = multi_hit = 0
    citation_total = citation_ok = 0
    latencies: list[float] = []
    tokens: list[int] = []
    intent_missed = 0
    backtracked = 0
    strategy_counts: dict[str, int] = {}

    for row in answerable:
        item = by_qid[row["qid"]]
        targets = targets_for(item, chunks_by_doc)
        flat = {chunk_id for ids in targets.values() for chunk_id in ids}
        ids = [entry["chunk_id"] for entry in row.get("evidence") or []]
        rank = rank_of(ids, flat, 5)
        if rank:
            hit5 += 1
            mrr5 += 1.0 / rank
        if len(targets) >= 2:
            multi_total += 1
            if rank:
                multi_hit += 1
            if all(any(chunk_id in ids for chunk_id in doc_targets) for doc_targets in targets.values()):
                all_hit += 1
        else:
            single_total += 1
            if rank:
                single_hit += 1
        if row.get("citations"):
            citation_total += 1
            target_docs = set(targets)
            cited_docs = {chunks[entry["chunk_id"]]["doc_key"] for entry in row["citations"]
                          if entry["chunk_id"] in chunks}
            if cited_docs & target_docs:
                citation_ok += 1
        if str(row.get("intent")) in {"chitchat", "out_of_scope"}:
            intent_missed += 1
        if row.get("trace", {}).get("backtracked"):
            backtracked += 1
        strategy = str(row.get("strategy"))
        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        latency = row.get("trace", {}).get("latency_ms", {}).get("total")
        if latency is not None:
            latencies.append(float(latency))
        trace = row.get("trace", {})
        tokens.append(sum((trace.get("llm_tokens") or {}).values())
                      or int(trace.get("prompt_tokens") or 0) + int(trace.get("completion_tokens") or 0))

    hard_answered = 0
    hard_answered_qids: list[str] = []
    for row in unanswerable:
        if not row.get("refused") and not row.get("error"):
            hard_answered += 1
            if len(hard_answered_qids) < 20:
                hard_answered_qids.append(
                    f"{row['qid']}（{str(row.get('answer') or '')[:60]}）"
                )

    refused_answerable = sum(1 for row in answerable if row.get("refused"))
    refused_unanswerable = sum(1 for row in unanswerable if row.get("refused"))
    return {
        "n": len(rows),
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
        "hit5": hit5 / len(answerable) if answerable else 0.0,
        "mrr5": mrr5 / len(answerable) if answerable else 0.0,
        "all_hit5": (all_hit / multi_total) if multi_total else None,
        "single_hit5": (single_hit / single_total) if single_total else None,
        "multi_hit5": (multi_hit / multi_total) if multi_total else None,
        "refusal_rate": (refused_unanswerable / len(unanswerable)) if unanswerable else None,
        "false_refusal_rate": (refused_answerable / len(answerable)) if answerable else None,
        "intent_missed": intent_missed,
        "citation_accuracy": (citation_ok / citation_total) if citation_total else None,
        "avg_tokens": (sum(tokens) / len(tokens)) if tokens else 0.0,
        "p50_ms": percentile(latencies, 0.5),
        "p95_ms": percentile(latencies, 0.95),
        "backtracked": backtracked,
        "strategies": strategy_counts,
        "hard_answered": hard_answered,
        "hard_answered_qids": hard_answered_qids,
    }


def correctness(graded: list[dict]) -> dict | None:
    """答对率：有答案的题按 correct / partial / wrong 计；无答案的题看是否拒答。"""
    answerable = [row for row in graded if row.get("answerable")]
    unanswerable = [row for row in graded if not row.get("answerable")]
    if not answerable:
        return None
    count = lambda grade: sum(1 for row in answerable if row.get("grade") == grade)
    return {
        "n": len(answerable),
        "correct": count("correct") / len(answerable),
        "partial": count("partial") / len(answerable),
        "wrong": count("wrong") / len(answerable),
        "refused_ok": (sum(1 for row in unanswerable if row.get("grade") == "correct") / len(unanswerable))
        if unanswerable else None,
    }


def faithfulness(rows: list[dict], judged: list[dict]) -> dict | None:
    """Faithfulness：只算有回答、也有评审结果的题。"""
    flags = {row["qid"]: row.get("faithful") for row in judged}
    values = [flags[row["qid"]] for row in rows if not row.get("refused") and row["qid"] in flags]
    if not values:
        return None
    return {
        "n": len(values),
        "faithful": sum(1 for value in values if value) / len(values),
    }


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


from xiaomao.runtime import LockedRetriever, build_pipelines  # noqa: E402


def result_row(item: dict, result) -> dict:
    """一题的落盘结构。"""
    return {
        "qid": item["id"],
        "query": item["query"],
        "answer": result.answer,
        "citations": result.citations,
        "refused": result.refused,
        "intent": result.intent,
        "strategy": result.strategy,
        "evidence": [
            {"chunk_id": candidate.chunk_id, "score": candidate.score}
            for candidate in result.evidence
        ],
        "trace": result.trace,
    }


def cmd_run(args: argparse.Namespace, pipeline=None) -> int:
    """跑一组配置。--workers > 1 时并发（延迟会因排队检索略偏高，看延迟用 workers=1 的结果）。"""
    load_dotenv(ROOT_DIR / ".env", override=True)
    from xiaomao.qa import QAOptions

    settings = CONFIGS[args.config]
    run_dir = args.runs_dir / f"qa_{args.run}"
    out_path = run_dir / f"{args.config}.jsonl"
    done = {row["qid"] for row in read_jsonl(out_path)}

    items = read_jsonl(args.data_dir / "eval" / "retrieval_qa.jsonl")
    if args.limit:
        items = items[: args.limit]
    todo = [item for item in items if item["id"] not in done]

    workers = max(1, getattr(args, "workers", 1))
    pipelines = [pipeline] if pipeline is not None else build_pipelines(workers)
    pool_of_pipelines: queue.Queue = queue.Queue()
    for each in pipelines:
        pool_of_pipelines.put(each)

    def answer_one(item: dict) -> dict:
        options = QAOptions(
            strategy=settings["strategy"],
            use_rerank=settings["use_rerank"],
            profile=({"platform": item.get("platform"), "site": item.get("site")}
                     if settings["profile"] else None),
            judge_evidence=settings.get("judge_evidence", True),
            translate=settings.get("translate", True),
        )
        worker = pool_of_pipelines.get()
        try:
            return result_row(item, worker.answer(item["query"], options))
        except Exception as exc:  # 单题失败不影响整批
            return {"qid": item["id"], "query": item["query"], "error": f"{type(exc).__name__}: {exc}"}
        finally:
            pool_of_pipelines.put(worker)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    # 逐题追加，中断后可续跑
    with out_path.open("a", encoding="utf-8", newline="\n") as handle, \
            ThreadPoolExecutor(max_workers=len(pipelines)) as executor:
        for row in executor.map(answer_one, todo):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            written += 1
            if written % 20 == 0:
                print(f"  已完成 {written}/{len(todo)}", flush=True)

    print(f"写入 {written} 条 -> {out_path}")
    return 0


# ---------------------------------------------------------------------------
# judge
# ---------------------------------------------------------------------------


def cmd_judge(args: argparse.Namespace) -> int:
    """评 Faithfulness。"""
    load_dotenv(ROOT_DIR / ".env", override=True)
    from xiaomao.llm import LLMClient

    run_dir = args.runs_dir / f"qa_{args.run}"
    rows = read_jsonl(run_dir / f"{args.config}.jsonl")
    out_path = run_dir / f"{args.config}_judge.jsonl"
    done = {row["qid"] for row in read_jsonl(out_path)}
    _, chunks = load_reference(args.data_dir)

    client = LLMClient()
    client.model = "qwen3.7-plus"  # 评委用更强的模型
    client.enable_thinking = True
    pending: list[tuple[str, str]] = []
    for row in rows:
        if row["qid"] in done or row.get("refused") or row.get("error"):
            continue
        # 对照生成时给的全部资料（Top-5）；没标引用的回答也要评
        context = "\n\n".join(
            f"[{index}] " + str(chunks[entry["chunk_id"]].get("text") or "")[:1500]
            for index, entry in enumerate(row.get("evidence") or [], start=1)
            if entry["chunk_id"] in chunks
        )
        if not context:
            continue
        prompt = (
            f"问题：{row['query']}\n\n回答：{row['answer']}\n\n生成时给的资料：\n{context}\n\n"
            "请逐条核查，输出 JSON。"
        )
        pending.append((row["qid"], prompt))
    if args.limit:
        pending = pending[: args.limit]

    def judge_one(task: tuple[str, str]) -> dict | None:
        qid, prompt = task
        try:
            payload = client.chat_json(prompt, JUDGE_SYSTEM, max_tokens=1500, temperature=0.0)
        except Exception as exc:
            print(f"[WARN] {qid} 评审失败：{exc}")
            return None
        return {"qid": qid, "faithful": bool(payload.get("faithful")),
                "unsupported": payload.get("unsupported") or []}

    judged: list[dict] = []
    with ThreadPoolExecutor(max_workers=8) as pool:  # 开思考后单条 10 秒以上，并发跑
        for result in pool.map(judge_one, pending):
            if result is not None:
                judged.append(result)
                if len(judged) % 20 == 0:
                    print(f"  已评审 {len(judged)}/{len(pending)} 条", flush=True)

    write_jsonl(out_path, read_jsonl(out_path) + judged)
    print(f"评审 {len(judged)} 条 -> {out_path}")
    return 0


# ---------------------------------------------------------------------------
# grade：对照参考答案评答对率
# ---------------------------------------------------------------------------


GRADE_SYSTEM = (
    "你是严格的阅卷老师。对照参考答案，判断系统回答是否答对了用户的问题。"
    "先列出参考答案里回答问题所必需的要点，再逐条看系统回答有没有覆盖、有没有说错。"
    "措辞不同、顺序不同、多给了不矛盾的补充信息，都不扣分；数字、条件、适用范围说错算错。"
    "评级：correct = 必需要点全部覆盖且没有错误；partial = 覆盖了一部分要点，或有次要错误；"
    "wrong = 没答到要点、答非所问、有关键错误，或者明明有答案却说资料里没有。"
    "只输出 JSON：{\"key_points\": [\"...\"], \"missing\": [\"...\"], \"errors\": [\"...\"], "
    "\"grade\": \"correct|partial|wrong\"}。"
)


def cmd_grade(args: argparse.Namespace) -> int:
    """评答对率：有答案的题交给评委对照参考答案；无答案的题拒答即算对，不调评委。"""
    load_dotenv(ROOT_DIR / ".env", override=True)
    from xiaomao.llm import LLMClient

    run_dir = args.runs_dir / f"qa_{args.run}"
    rows = read_jsonl(run_dir / f"{args.config}.jsonl")
    out_path = run_dir / f"{args.config}_grade.jsonl"
    done = {row["qid"] for row in read_jsonl(out_path)}
    items = {item["id"]: item for item in read_jsonl(args.data_dir / "eval" / "retrieval_qa.jsonl")}

    client = LLMClient()
    client.model = "qwen3.7-plus"
    client.enable_thinking = True

    def grade_one(row: dict) -> dict | None:
        item = items[row["qid"]]
        if not item.get("answerable"):
            return {"qid": row["qid"], "answerable": False, "grade": "correct" if row.get("refused") else "wrong"}
        if row.get("refused") or row.get("error"):
            return {"qid": row["qid"], "answerable": True, "grade": "wrong", "missing": ["拒答或报错"]}
        prompt = (f"用户问题：{item['query']}\n\n参考答案：{item['reference_answer']}\n\n"
                  f"系统回答：{row['answer']}\n\n请评判，输出 JSON。")
        try:
            payload = client.chat_json(prompt, GRADE_SYSTEM, max_tokens=1500, temperature=0.0)
        except Exception as exc:
            print(f"[WARN] {row['qid']} 评判失败：{exc}")
            return None
        return {"qid": row["qid"], "answerable": True, "grade": str(payload.get("grade")),
                "missing": payload.get("missing") or [], "errors": payload.get("errors") or []}

    todo = [row for row in rows if row["qid"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    with ThreadPoolExecutor(max_workers=8) as executor:
        graded = [result for result in executor.map(grade_one, todo) if result is not None]
    write_jsonl(out_path, read_jsonl(out_path) + graded)
    print(f"评判 {len(graded)} 条 -> {out_path}")
    return 0


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def _pct(value: float | None) -> str:
    """百分比，保留一位小数。"""
    return "-" if value is None else f"{value * 100:.1f}%"


def cmd_report(args: argparse.Namespace) -> int:
    """汇总成 report.md。"""
    load_dotenv(ROOT_DIR / ".env", override=True)
    run_dir = args.runs_dir / f"qa_{args.run}"
    items, chunks = load_reference(args.data_dir)
    by_qid = {item["id"]: item for item in items}

    summaries: dict[str, dict] = {}
    rows_by_config: dict[str, list[dict]] = {}
    for config in CONFIGS:
        rows = read_jsonl(run_dir / f"{config}.jsonl")
        if not rows:
            continue
        rows_by_config[config] = rows
        summary = evaluate(rows, items, chunks)
        summary["faithfulness"] = faithfulness(rows, read_jsonl(run_dir / f"{config}_judge.jsonl"))
        summary["correctness"] = correctness(read_jsonl(run_dir / f"{config}_grade.jsonl"))
        summaries[config] = summary

    if not summaries:
        print(f"[ERROR] {run_dir} 里没有结果文件")
        return 1

    lines = ["# M4a 整条链路评估", ""]
    lines.append(f"- 题目：{len(items)} 道（有答案 {sum(1 for i in items if i.get('answerable'))}，"
                 f"无答案 {sum(1 for i in items if not i.get('answerable'))}）")
    lines.append(f"- 生成模型：{os.environ.get('LLM_MODEL', '-')}；rerank：{os.environ.get('RERANK_API_MODEL', '-')}")
    lines.append("")
    lines.append("| config | Hit@5 | MRR@5 | AllHit@5 | 拒答率 | 误拒率 | 引用准确率 | Faithfulness | 答对 / 部分答对 | 平均 tokens | P50 总延迟 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for config, summary in summaries.items():
        grade = summary["correctness"] or {}
        lines.append(
            f"| {config} | {_pct(summary['hit5'])} | {summary['mrr5']:.3f} | {_pct(summary['all_hit5'])} | "
            f"{_pct(summary['refusal_rate'])} | {_pct(summary['false_refusal_rate'])} | "
            f"{_pct(summary['citation_accuracy'])} | {_pct((summary['faithfulness'] or {}).get('faithful'))} | "
            f"{_pct(grade.get('correct'))} / {_pct(grade.get('partial'))} | "
            f"{summary['avg_tokens']:.0f} | {summary['p50_ms']:.0f} ms |"
        )
    lines.append("")

    lines.append("## 分组（Hit@5）")
    lines.append("")
    lines.append("| config | 单证据 | 多证据 | 判成闲聊/超范围 | Backtrack 触发 | 无答案题被硬答 |")
    lines.append("|---|---|---|---|---|---|")
    for config, summary in summaries.items():
        lines.append(
            f"| {config} | {_pct(summary['single_hit5'])} | {_pct(summary['multi_hit5'])} | "
            f"{summary['intent_missed']} | {summary['backtracked']} | {summary.get('hard_answered', 0)} |"
        )
    lines.append("")

    lines.append("## 没命中 / 误拒 / 不忠实")
    lines.append("")
    for config, rows in rows_by_config.items():
        bad = [row for row in rows
               if row.get("error") or (row.get("refused") and by_qid.get(row["qid"], {}).get("answerable"))]
        if bad:
            lines.append(f"**{config}：误拒 {len(bad)} 条**")
            lines.append("")
            for row in bad[:10]:
                lines.append(f"- `{row['qid']}` {row['query']}")
            lines.append("")

    lines.append("## 无答案题被硬答")
    lines.append("")
    for config, summary in summaries.items():
        qids = summary.get("hard_answered_qids") or []
        if not qids:
            continue
        lines.append(f"**{config}**：{summary['hard_answered']} 条")
        lines.append("")
        for item in qids:
            lines.append(f"- {item}")
        lines.append("")

    out_path = run_dir / "report.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"报告 -> {out_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """解析子命令。"""
    parser = argparse.ArgumentParser(description="整条链路评估")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-dir", type=Path, default=DATA_DIR)
    common.add_argument("--runs-dir", type=Path, default=RUNS_ROOT)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", parents=[common], help="跑问答")
    run.add_argument("--run", required=True)
    run.add_argument("--config", choices=list(CONFIGS), required=True)
    run.add_argument("--limit", type=int, default=0)
    run.add_argument("--workers", type=int, default=1)
    run.set_defaults(func=cmd_run)

    judge = sub.add_parser("judge", parents=[common], help="评 Faithfulness")
    judge.add_argument("--run", required=True)
    judge.add_argument("--config", choices=list(CONFIGS), required=True)
    judge.add_argument("--limit", type=int, default=0)
    judge.set_defaults(func=cmd_judge)

    grade = sub.add_parser("grade", parents=[common], help="对照参考答案评答对率")
    grade.add_argument("--run", required=True)
    grade.add_argument("--config", choices=list(CONFIGS), required=True)
    grade.add_argument("--limit", type=int, default=0)
    grade.set_defaults(func=cmd_grade)

    report = sub.add_parser("report", parents=[common], help="出报告")
    report.add_argument("--run", required=True)
    report.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
