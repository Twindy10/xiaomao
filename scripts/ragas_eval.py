"""RAGAS 评估：在隔离环境里跑，不能 import xiaomao。

    uv run --no-project --python 3.12 --with ragas==0.4.3 --with "langchain-community<0.4" \
        --with langchain-openai --with python-dotenv python scripts/ragas_eval.py --run m4v4 --config routing

只评「有答案且没被拒答」的题；拒答率和误拒率由 run_qa_eval.py 负责。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parents[1]
CONTEXT_LIMIT = 1500
BATCH_SIZE = 10


def read_jsonl(path: Path) -> list[dict]:
    """读 JSONL。"""
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, rows: list[dict]) -> None:
    """追加写一行，立即落盘。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_samples(
    rows: list[dict],
    items: list[dict],
    chunks: dict[str, dict],
    limit: int = 0,
) -> tuple[list[dict], dict]:
    """构造 RAGAS 样本；顺便统计被过滤掉的题数。"""
    by_qid = {item["id"]: item for item in items}
    samples: list[dict] = []
    dropped = {"refused": 0, "unanswerable": 0, "missing": 0}

    for row in rows:
        item = by_qid.get(row.get("qid"))
        if item is None:
            dropped["missing"] += 1
            continue
        if item.get("answerable") is not True:
            dropped["unanswerable"] += 1
            continue
        if row.get("refused"):
            dropped["refused"] += 1
            continue
        contexts = []
        for entry in row.get("evidence") or []:
            chunk = chunks.get(entry["chunk_id"])
            if chunk:
                contexts.append(str(chunk.get("text") or "")[:CONTEXT_LIMIT])
        samples.append({
            "qid": row["qid"],
            "user_input": row["query"],
            "response": row.get("answer") or "",
            "retrieved_contexts": contexts,
            "reference": item.get("reference_answer") or "",
        })
        if limit and len(samples) >= limit:
            break
    return samples, dropped


def done_qids(path: Path) -> set[str]:
    """已有结果的 qid，断点续跑用。"""
    return {row["qid"] for row in read_jsonl(path) if row.get("qid")}


def is_nan(value) -> bool:
    """None 和 NaN 都算缺失。"""
    if value is None:
        return True
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return True


def contingency(
    ragas_scores: dict[str, float],
    judge_flags: dict[str, bool],
    threshold: float = 0.9,
) -> dict:
    """RAGAS 忠实度和自己评委的 2×2 列联表。"""
    table = {"both_faithful": 0, "ragas_only": 0, "judge_only": 0, "neither": 0}
    mismatches: list[dict] = []
    total = 0

    for qid, score in ragas_scores.items():
        if qid not in judge_flags or is_nan(score):
            continue
        total += 1
        ragas_faithful = float(score) >= threshold
        judge_faithful = bool(judge_flags[qid])
        if ragas_faithful and judge_faithful:
            table["both_faithful"] += 1
        elif ragas_faithful:
            table["ragas_only"] += 1
        elif judge_faithful:
            table["judge_only"] += 1
        else:
            table["neither"] += 1
        if ragas_faithful != judge_faithful:
            mismatches.append({"qid": qid, "ragas": float(score), "judge": judge_faithful})

    agree = table["both_faithful"] + table["neither"]
    return {
        "table": table,
        "total": total,
        "agreement": (agree / total) if total else None,
        "mismatches": mismatches,
    }


def average(values: list[float]) -> float | None:
    """NaN 不计入。"""
    clean = [float(value) for value in values if not is_nan(value)]
    return sum(clean) / len(clean) if clean else None


def summarize(
    scores_by_config: dict[str, list[dict]],
    judge_by_config: dict[str, dict[str, bool]],
    metric_names: list[str],
    threshold: float = 0.9,
) -> str:
    """Markdown 汇总。"""
    lines = ["# RAGAS 评估", ""]
    header = "| 配置 | 样本 | " + " | ".join(metric_names) + " | NaN 数 |"
    lines.append(header)
    lines.append("|" + "---|" * (len(metric_names) + 3))

    for config, rows in scores_by_config.items():
        cells = []
        nan_count = 0
        for name in metric_names:
            values = [row.get(name) for row in rows]
            nan_count += sum(1 for value in values if is_nan(value))
            mean = average(values)
            cells.append("-" if mean is None else f"{mean:.3f}")
        lines.append(f"| {config} | {len(rows)} | " + " | ".join(cells) + f" | {nan_count} |")

    lines.append("")
    lines.append(f"## 和自建评委的一致性（忠实度 ≥ {threshold:.2f} 算忠实）")
    lines.append("")
    for config, rows in scores_by_config.items():
        judges = judge_by_config.get(config) or {}
        if not judges:
            lines.append(f"- {config}：没有评委结果，跳过")
            continue
        ragas_scores = {row["qid"]: row.get("faithfulness") for row in rows}
        result = contingency(ragas_scores, judges, threshold)
        table = result["table"]
        agreement = result["agreement"]
        lines.append(f"**{config}**：一致率 {'-' if agreement is None else f'{agreement:.1%}'}"
                     f"（{result['total']} 题）")
        lines.append("")
        lines.append("| | 评委：忠实 | 评委：不忠实 |")
        lines.append("|---|---|---|")
        lines.append(f"| RAGAS：忠实 | {table['both_faithful']} | {table['ragas_only']} |")
        lines.append(f"| RAGAS：不忠实 | {table['judge_only']} | {table['neither']} |")
        lines.append("")
        if result["mismatches"]:
            lines.append("不一致的题（最多 15 条）：")
            for item in result["mismatches"][:15]:
                lines.append(f"- `{item['qid']}`：RAGAS {item['ragas']:.3f}，评委 faithful={item['judge']}")
            lines.append("")
    return "\n".join(lines)


def load_inputs(run: str, config: str) -> tuple[list[dict], list[dict], dict[str, dict]]:
    """读跑分、评估集、片段。"""
    rows = read_jsonl(ROOT_DIR / "eval" / "runs" / f"qa_{run}" / f"{config}.jsonl")
    items = read_jsonl(ROOT_DIR / "data" / "eval" / "retrieval_qa.jsonl")
    chunks = {row["chunk_id"]: row for row in read_jsonl(ROOT_DIR / "data" / "processed" / "chunks.jsonl")}
    return rows, items, chunks


def judge_flags(run: str, config: str) -> dict[str, bool]:
    """自己评委的结论。"""
    rows = read_jsonl(ROOT_DIR / "eval" / "runs" / f"qa_{run}" / f"{config}_judge.jsonl")
    return {row["qid"]: bool(row.get("faithful")) for row in rows}


def main(argv: list[str] | None = None) -> int:
    """跑 RAGAS。"""
    from dotenv import load_dotenv

    load_dotenv(ROOT_DIR / ".env", override=True)

    parser = argparse.ArgumentParser(description="RAGAS 评估")
    parser.add_argument("--run", default="m4v4")
    parser.add_argument("--config", action="append", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--with-recall", action="store_true")
    parser.add_argument("--faithful-threshold", type=float, default=0.9)
    args = parser.parse_args(argv)
    configs = args.config or ["routing"]

    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )
    from ragas.run_config import RunConfig

    judge = ChatOpenAI(
        model=os.environ.get("JUDGE_MODEL", "qwen3.7-plus"),
        base_url=os.environ["LLM_BASE_URL"],
        api_key=os.environ["LLM_API_KEY"],
        max_tokens=4096,
        timeout=120,
        extra_body={"enable_thinking": False},
    )
    embeddings = OpenAIEmbeddings(
        model=os.environ.get("RAGAS_EMBED_MODEL", "text-embedding-v4"),
        base_url=os.environ["LLM_BASE_URL"],
        api_key=os.environ["LLM_API_KEY"],
        check_embedding_ctx_length=False,
    )
    llm = LangchainLLMWrapper(judge)
    embedder = LangchainEmbeddingsWrapper(embeddings)

    metrics = [Faithfulness(), ResponseRelevancy(strictness=1), LLMContextPrecisionWithReference()]
    if args.with_recall:
        metrics.append(LLMContextRecall())

    scores_by_config: dict[str, list[dict]] = {}
    judge_by_config: dict[str, dict[str, bool]] = {}
    for config in configs:
        rows, items, chunks = load_inputs(args.run, config)
        samples, dropped = build_samples(rows, items, chunks, args.limit)
        out_path = ROOT_DIR / "eval" / "runs" / f"ragas_{args.run}" / f"{config}.jsonl"
        done = done_qids(out_path)
        pending = [sample for sample in samples if sample["qid"] not in done]
        print(f"{config}: 样本 {len(samples)}，已跑 {len(samples) - len(pending)}，"
              f"跳过拒答 {dropped['refused']} / 无答案 {dropped['unanswerable']}")

        for start in range(0, len(pending), BATCH_SIZE):
            batch = pending[start:start + BATCH_SIZE]
            dataset = EvaluationDataset(samples=[
                SingleTurnSample(
                    user_input=sample["user_input"],
                    response=sample["response"],
                    retrieved_contexts=sample["retrieved_contexts"],
                    reference=sample["reference"],
                )
                for sample in batch
            ])
            try:
                result = evaluate(
                    dataset=dataset, metrics=metrics, llm=llm, embeddings=embedder,
                    run_config=RunConfig(max_workers=8, timeout=180), raise_exceptions=False,
                )
                score_rows = result.scores
            except Exception as exc:
                append_jsonl(out_path, [
                    {"qid": sample["qid"], "error": f"{type(exc).__name__}: {exc}"} for sample in batch
                ])
                print(f"  批次失败：{exc}")
                continue

            out_rows = []
            for sample, score_row in zip(batch, score_rows):
                row = {"qid": sample["qid"]}
                error = None
                for metric in metrics:
                    value = (score_row or {}).get(metric.name)
                    if value is None or is_nan(value):
                        row[metric.name] = None
                        error = error or "nan"
                    else:
                        row[metric.name] = float(value)
                if error:
                    row["error"] = error
                out_rows.append(row)
            append_jsonl(out_path, out_rows)
            print(f"  完成 {min(start + BATCH_SIZE, len(pending))}/{len(pending)}")

        scores_by_config[config] = read_jsonl(out_path)
        judge_by_config[config] = judge_flags(args.run, config)

    metric_names = [metric.name for metric in metrics]
    summary = summarize(scores_by_config, judge_by_config, metric_names, args.faithful_threshold)
    out_dir = ROOT_DIR / "eval" / "runs" / f"ragas_{args.run}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.md").write_text(summary, encoding="utf-8")
    print("\n" + summary)
    print(f"\n汇总 -> {out_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
