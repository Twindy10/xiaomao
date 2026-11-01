"""检索评估：召回落盘 → 消融 → rerank 对比 → 出报告。

    uv run python scripts/run_eval.py routes   --run m3 --filter none
    uv run python scripts/run_eval.py ablation --run m3
    uv run python scripts/run_eval.py rerank   --run m3 --config dense+bm25 --backend api
    uv run python scripts/run_eval.py report   --run m3 --price-per-1k-tokens 0.0005
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date
from pathlib import Path
from statistics import median

sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

for _name in ("NO_PROXY", "no_proxy"):
    os.environ[_name] = ",".join(filter(None, [os.environ.get(_name), "127.0.0.1,localhost"]))

from dotenv import load_dotenv  # noqa: E402

from xiaomao.ingest.index import bm25_text  # noqa: E402
from xiaomao.retrieval.filters import build_filter  # noqa: E402
from xiaomao.retrieval.hybrid import fuse, parse_config  # noqa: E402
from xiaomao.retrieval.rerank import NoopReranker, get_reranker  # noqa: E402
from xiaomao.retrieval.search import (  # noqa: E402
    DEFAULT_COLLECTIONS,
    ROUTES,
    Candidate,
    encode_queries,
    search_route,
)
from xiaomao.textnorm import normalize_for_match  # noqa: E402

DEFAULT_CONFIGS = [
    "dense", "bm25", "sparse_m3",
    "dense+sparse_m3", "dense+bm25",
    "dense+sparse_m3+bm25", "dense:2+bm25", "dense:2+sparse_m3+bm25",
]

DATA_DIR = ROOT_DIR / "data"
RUNS_ROOT = ROOT_DIR / "eval" / "runs"


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


def write_json(path: Path, payload: dict) -> None:
    """原子写 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# 评估数据
# ---------------------------------------------------------------------------


def load_chunks(data_dir: Path = DATA_DIR) -> tuple[dict[str, list[dict]], dict[str, dict]]:
    """chunks.jsonl → (按 doc_key 分组, 按 chunk_id 索引)。"""
    rows = read_jsonl(data_dir / "processed" / "chunks.jsonl")
    by_doc: dict[str, list[dict]] = {}
    by_id: dict[str, dict] = {}
    for chunk in rows:
        by_doc.setdefault(chunk["doc_key"], []).append(chunk)
        by_id[chunk["chunk_id"]] = chunk
    return by_doc, by_id


def evidence_targets(item: dict, chunks_by_doc: dict[str, list[dict]]) -> dict[str, set[str]]:
    """每条证据 quote 落在哪些片段：doc_key → 目标 chunk_id 集合。"""
    targets: dict[str, set[str]] = {}
    for entry in item.get("evidence") or []:
        quote = normalize_for_match(entry["quote"])
        for chunk in chunks_by_doc.get(entry["doc_key"], []):
            if quote in normalize_for_match(chunk["text"]):
                targets.setdefault(entry["doc_key"], set()).add(chunk["chunk_id"])
    return targets


def load_questions(data_dir: Path = DATA_DIR) -> list[dict]:
    """有答案的题，附带目标片段和分组信息。"""
    chunks_by_doc, _ = load_chunks(data_dir)
    documents = {
        doc["doc_key"]: doc for doc in read_jsonl(data_dir / "processed" / "documents.jsonl")
    }
    questions: list[dict] = []
    for item in read_jsonl(data_dir / "eval" / "retrieval_qa.jsonl"):
        if item.get("answerable") is not True:
            continue
        targets = evidence_targets(item, chunks_by_doc)
        flat = {chunk_id for ids in targets.values() for chunk_id in ids}
        if not flat:
            print(f"[WARN] 找不到目标片段，跳过：{item['id']}")
            continue
        collections = {
            documents[entry["doc_key"]]["collection"]
            for entry in item["evidence"]
            if entry["doc_key"] in documents
        }
        questions.append(
            {
                "qid": item["id"],
                "item": item,
                "targets": flat,
                "targets_by_doc": targets,
                "multi_doc": len(targets) >= 2,
                "collection": "both" if len(collections) > 1 else next(iter(collections), "unknown"),
                "intent": item.get("intent"),
            }
        )
    return questions


def variants(item: dict) -> list[tuple[int, str]]:
    """原问题 + 同义改写。"""
    queries = [(0, item["query"])]
    for index, text in enumerate(item.get("paraphrases") or [], start=1):
        if text:
            queries.append((index, text))
    return queries


# ---------------------------------------------------------------------------
# 指标
# ---------------------------------------------------------------------------


def ranks_of_targets(ids: list[str], targets: set[str], limit: int) -> int | None:
    """第一个目标出现的名次（1 起），超出 limit 返回 None。"""
    for position, chunk_id in enumerate(ids[:limit], start=1):
        if chunk_id in targets:
            return position
    return None


def score_question(ids: list[str], question: dict) -> dict:
    """单题指标。"""
    hit5 = ranks_of_targets(ids, question["targets"], 5) is not None
    rank = ranks_of_targets(ids, question["targets"], 10)
    recall20 = ranks_of_targets(ids[:20], question["targets"], 20) is not None
    all_hit = None
    if question["multi_doc"]:
        all_hit = all(
            any(chunk_id in ids[:5] for chunk_id in doc_targets)
            for doc_targets in question["targets_by_doc"].values()
        )
    return {"hit5": hit5, "mrr10": 1.0 / rank if rank else 0.0, "recall20": recall20, "all_hit5": all_hit}


def aggregate(rows: list[dict]) -> dict:
    """把单题指标汇总成平均。"""
    if not rows:
        return {"n": 0, "hit5": 0.0, "mrr10": 0.0, "recall20": 0.0, "all_hit5": None}
    multi = [row["all_hit5"] for row in rows if row.get("all_hit5") is not None]
    return {
        "n": len(rows),
        "hit5": sum(1 for row in rows if row["hit5"]) / len(rows),
        "mrr10": sum(row["mrr10"] for row in rows) / len(rows),
        "recall20": sum(1 for row in rows if row["recall20"]) / len(rows),
        "all_hit5": (sum(1 for flag in multi if flag) / len(multi)) if multi else None,
    }


# ---------------------------------------------------------------------------
# routes：查 Milvus 并落盘
# ---------------------------------------------------------------------------


def cmd_routes(args: argparse.Namespace) -> int:
    """把每道题（原问题 + 改写）的三路召回结果落盘。"""
    load_dotenv(ROOT_DIR / ".env", override=True)
    from pymilvus import MilvusClient

    from xiaomao.ingest.embed import Embedder

    run_dir = args.runs_dir / args.run
    routes_path = run_dir / "routes.jsonl"
    existing = {(row["qid"], row["variant"]) for row in read_jsonl(routes_path)}

    questions = load_questions(args.data_dir)
    if args.limit:
        questions = questions[: args.limit]

    pending: list[dict] = []
    for question in questions:
        for variant_index, query in variants(question["item"]):
            if (question["qid"], variant_index) in existing:
                continue
            pending.append({"question": question, "variant": variant_index, "query": query})
    if not pending:
        print("全部题目都跑过了")
        return 0

    client = MilvusClient(uri=os.environ["MILVUS_URI"], token=os.environ.get("MILVUS_TOKEN", ""))
    embedder = Embedder(model_name=os.environ.get("EMBED_MODEL"), max_length=512)
    vectors = encode_queries(embedder, [row["query"] for row in pending])

    rows: list[dict] = []
    started = time.perf_counter()
    for index, row in enumerate(pending, start=1):
        item = row["question"]["item"]
        if args.filter == "gold":
            filter_expr = build_filter(item.get("platform"), item.get("site"))
        else:
            filter_expr = build_filter()
        routes: dict[str, list[list]] = {}
        for route in ROUTES:
            merged: dict[str, Candidate] = {}
            for collection in args.collections:
                for candidate in search_route(
                    client, collection, route, row["query"], vectors[index - 1],
                    filter_expr=filter_expr, limit=args.candidates,
                ):
                    merged[candidate.chunk_id] = candidate
            ordered = sorted(merged.values(), key=lambda candidate: -candidate.score)
            routes[route] = [[candidate.chunk_id, candidate.score] for candidate in ordered]
        rows.append(
            {
                "qid": row["question"]["qid"],
                "variant": row["variant"],
                "query": row["query"],
                "filter": filter_expr,
                "routes": routes,
            }
        )
        if index % 50 == 0:
            print(f"  已完成 {index}/{len(pending)}")

    write_jsonl(routes_path, read_jsonl(routes_path) + rows)
    write_json(
        run_dir / "meta.json",
        {
            "date": date.today().isoformat(),
            "filter": args.filter,
            "collections": list(args.collections),
            "candidates": args.candidates,
            "questions": len(questions),
            "embed_model": Path(os.environ.get("EMBED_MODEL", "")).name,
            "counts": {name: client.get_collection_stats(name)["row_count"] for name in args.collections},
        },
    )
    print(f"写入 {len(rows)} 条 -> {routes_path}，耗时 {time.perf_counter() - started:.1f}s")
    return 0


# ---------------------------------------------------------------------------
# ablation：读 routes 算各种融合组合
# ---------------------------------------------------------------------------


def _slug(config: str) -> str:
    """config 名转成文件名：dense+bm25:0.2@z → dense_bm25x0.2@z（Windows 文件名不能有冒号）。"""
    return config.replace("+", "_").replace(":", "x")


def _candidates_from_route(pairs: list) -> list[Candidate]:
    """把落盘的 [chunk_id, score] 还原成 Candidate。"""
    return [
        Candidate(chunk_id=str(chunk_id), collection="", score=float(score), rank=index, fields={})
        for index, (chunk_id, score) in enumerate(pairs, start=1)
    ]


def cmd_ablation(args: argparse.Namespace) -> int:
    """对每份召回结果重放多种融合组合，算指标。"""
    run_dir = args.runs_dir / args.run
    routes = read_jsonl(run_dir / "routes.jsonl")
    if not routes:
        print(f"[ERROR] {run_dir} 里没有 routes.jsonl，先跑 routes")
        return 1
    questions = {question["qid"]: question for question in load_questions(args.data_dir)}

    configs = args.configs or DEFAULT_CONFIGS
    result: dict[str, dict] = {}
    for name in configs:
        config = parse_config(name)
        original_rows: list[dict] = []
        paraphrase_rows: list[dict] = []
        groups: dict[str, dict[str, list[dict]]] = {"intent": {}, "collection": {}, "multi": {}}
        for row in routes:
            question = questions.get(row["qid"])
            if question is None:
                continue
            fused = fuse({route: _candidates_from_route(pairs) for route, pairs in row["routes"].items()},
                             config, limit=20)
            ids = [candidate.chunk_id for candidate in fused]
            metrics = score_question(ids, question)
            if row["variant"] == 0:
                original_rows.append(metrics)
                groups["intent"].setdefault(question["intent"] or "unknown", []).append(metrics)
                groups["collection"].setdefault(question["collection"], []).append(metrics)
                groups["multi"].setdefault("multi" if question["multi_doc"] else "single", []).append(metrics)
            else:
                paraphrase_rows.append(metrics)
        result[name] = {
            "original": aggregate(original_rows),
            "paraphrase": aggregate(paraphrase_rows),
            "groups": {
                kind: {key: aggregate(rows) for key, rows in mapping.items()}
                for kind, mapping in groups.items()
            },
        }

    write_json(run_dir / "ablation.json", result)

    print(f"{'config':<26}{'Hit@5':>8}{'MRR@10':>9}{'Recall@20':>11}{'AllHit@5':>10}{'改写 Hit@5':>12}")
    for name in configs:
        row = result[name]
        all_hit = row["original"]["all_hit5"]
        print(
            f"{name:<26}{row['original']['hit5']:>8.1%}{row['original']['mrr10']:>9.3f}"
            f"{row['original']['recall20']:>11.1%}"
            f"{(all_hit if all_hit is not None else 0):>10.1%}"
            f"{row['paraphrase']['hit5']:>12.1%}"
        )
    return 0


# ---------------------------------------------------------------------------
# rerank：对 Top-20 重排
# ---------------------------------------------------------------------------


def _chunk_texts(data_dir: Path = DATA_DIR) -> dict[str, str]:
    """chunk_id -> 送进 reranker 的文本（与 Milvus text 一致）。"""
    _, by_id = load_chunks(data_dir)
    return {chunk_id: bm25_text(chunk) for chunk_id, chunk in by_id.items()}


def cmd_rerank(args: argparse.Namespace) -> int:
    """对每条题目的 Top-20 做重排并统计。"""
    run_dir = args.runs_dir / args.run
    routes = {row["qid"]: row for row in read_jsonl(run_dir / "routes.jsonl") if row["variant"] == 0}
    if not routes:
        print(f"[ERROR] {run_dir} 里没有原问题的 routes，先跑 routes")
        return 1

    out_path = run_dir / f"rerank_{_slug(args.config)}_{args.backend}.jsonl"
    done = {row["qid"] for row in read_jsonl(out_path)}
    questions = {question["qid"]: question for question in load_questions(args.data_dir)}
    texts = _chunk_texts(args.data_dir)

    if args.backend == "none":
        reranker = NoopReranker()
    else:
        load_dotenv(ROOT_DIR / ".env", override=True)
        if args.backend == "local":
            from xiaomao.retrieval.rerank import LocalReranker

            reranker = LocalReranker(batch_size=args.batch_size)
        else:
            reranker = get_reranker("api")

    config = parse_config(args.config)
    rows: list[dict] = []
    peak_vram = None
    for qid, row in routes.items():
        if qid in done or qid not in questions:
            continue
        fused = fuse({route: _candidates_from_route(pairs) for route, pairs in row["routes"].items()},
                         config, limit=20)
        candidates = [
            Candidate(chunk_id=candidate.chunk_id, collection=candidate.collection,
                      score=candidate.score, rank=candidate.rank,
                      fields={"text": texts.get(candidate.chunk_id, "")})
            for candidate in fused
        ]
        result = reranker.rerank(row["query"], candidates, top_n=None)
        if hasattr(reranker, "peak_vram_mb"):
            peak_vram = reranker.peak_vram_mb()
        rows.append(
            {
                "qid": qid,
                "query": row["query"],
                "skipped": result.skipped,
                "latency_ms": result.latency_ms,
                "tokens": result.tokens,
                "error": result.error,
                "ranked": [[candidate.chunk_id, candidate.score] for candidate in result.candidates],
            }
        )
        if len(rows) % 20 == 0:
            print(f"  已重排 {len(rows)} 条")

    write_jsonl(out_path, read_jsonl(out_path) + rows)

    all_rows = read_jsonl(out_path)
    scored = [row for row in all_rows if row["qid"] in questions]
    metrics = [
        score_question([item[0] for item in row["ranked"]], questions[row["qid"]])
        for row in scored
    ]
    latencies = sorted(row["latency_ms"] for row in scored) or [0.0]
    summary = aggregate(metrics)
    summary.update(
        {
            "p50_ms": latencies[len(latencies) // 2],
            "p95_ms": latencies[int(len(latencies) * 0.95) - 1] if len(latencies) > 1 else latencies[0],
            "avg_tokens": sum(row["tokens"] for row in scored) / len(scored) if scored else 0.0,
            "skipped": sum(1 for row in scored if row["skipped"]),
            "peak_vram_mb": peak_vram,
        }
    )
    write_json(run_dir / f"summary_{_slug(args.config)}_{args.backend}.json", summary)

    print(f"{args.config} / {args.backend}: Hit@5 {summary['hit5']:.1%}，MRR@10 {summary['mrr10']:.3f}，"
          f"P50 {summary['p50_ms']:.0f}ms，P95 {summary['p95_ms']:.0f}ms，"
          f"平均 {summary['avg_tokens']:.0f} tokens，跳过 {summary['skipped']} 条")
    return 0


# ---------------------------------------------------------------------------
# report：汇总成 markdown
# ---------------------------------------------------------------------------


def _pct(value: float | None) -> str:
    """百分比，保留一位小数。"""
    return "-" if value is None else f"{value * 100:.1f}%"


def cmd_report(args: argparse.Namespace) -> int:
    """把 meta、ablation、rerank 结果汇总成 report.md。"""
    run_dir = args.runs_dir / args.run
    load_dotenv(ROOT_DIR / ".env", override=True)
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    ablation = json.loads((run_dir / "ablation.json").read_text(encoding="utf-8"))
    questions = {question["qid"]: question for question in load_questions(args.data_dir)}
    price = args.price_per_1k_tokens

    lines = ["# M3 检索报告", ""]
    lines.append(f"- 日期：{meta.get('date')}")
    lines.append(f"- filter 模式：{meta.get('filter')}")
    lines.append(f"- embedding：{meta.get('embed_model')}")
    lines.append(f"- rerank API 模型：{os.environ.get('RERANK_API_MODEL', '-')}")
    lines.append(f"- 题数：{meta.get('questions')}（原问题），candidates={meta.get('candidates')}")
    lines.append("- 集合条数：" + "，".join(f"{k}={v}" for k, v in (meta.get("counts") or {}).items()))
    lines.append("")

    lines.append("## 检索消融")
    lines.append("")
    lines.append("| config | Hit@5 | MRR@10 | Recall@20 | AllHit@5 | 同义改写 Hit@5 |")
    lines.append("|---|---|---|---|---|---|")
    for name, row in ablation.items():
        original = row["original"]
        lines.append(
            f"| {name} | {_pct(original['hit5'])} | {original['mrr10']:.3f} | "
            f"{_pct(original['recall20'])} | {_pct(original.get('all_hit5'))} | "
            f"{_pct(row['paraphrase']['hit5'])} |"
        )
    lines.append("")

    best = max(ablation, key=lambda name: (ablation[name]["original"]["hit5"], ablation[name]["original"]["mrr10"]))
    lines.append(f"## 最好配置 `{best}` 的分组")
    lines.append("")
    for kind, mapping in ablation[best]["groups"].items():
        lines.append(f"**{kind}**")
        lines.append("")
        lines.append("| 分组 | 题数 | Hit@5 | MRR@10 |")
        lines.append("|---|---|---|---|")
        for key, row in sorted(mapping.items()):
            lines.append(f"| {key} | {row['n']} | {_pct(row['hit5'])} | {row['mrr10']:.3f} |")
        lines.append("")

    lines.append("## rerank 后端对比")
    lines.append("")
    lines.append("| 融合 / 后端 | Hit@5 | MRR@10 | P50 | P95 | 平均 tokens | 每千次成本 | 显存 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for path in sorted(run_dir.glob("summary_*.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        label = path.stem.removeprefix("summary_")
        # price 是每千 tokens 的单价：一千次请求 = 平均 tokens × 1000 次 / 1000 × price
        cost = f"{summary['avg_tokens'] * price:.2f} 元" if price else "待填"
        vram = f"{summary['peak_vram_mb']:.0f} MB" if summary.get("peak_vram_mb") else "-"
        lines.append(
            f"| {label} | {_pct(summary['hit5'])} | {summary['mrr10']:.3f} | "
            f"{summary['p50_ms']:.0f} ms | {summary['p95_ms']:.0f} ms | "
            f"{summary['avg_tokens']:.0f} | {cost} | {vram} |"
        )
    lines.append("")

    def hits(rows: list[dict]) -> list[tuple[dict, bool]]:
        return [(row, score_question([item[0] for item in row["ranked"]], questions[row["qid"]])["hit5"])
                for row in rows if row["qid"] in questions]

    # 没命中清单取 Hit@5 最高的那份 rerank 结果
    scored_files = [(path, hits(read_jsonl(path))) for path in run_dir.glob("rerank_*.jsonl")]
    if scored_files:
        best_path, best_rows = max(scored_files, key=lambda item: sum(flag for _, flag in item[1]))
        _, by_id = load_chunks(args.data_dir)
        title = lambda chunk_id: by_id.get(chunk_id, {}).get("title_path", chunk_id)  # noqa: E731
        misses = [row for row, flag in best_rows if not flag]
        lines.append(f"## 没命中（{best_path.stem}，{len(misses)} 条）")
        lines.append("")
        for row in misses[: args.max_misses]:
            question = questions[row["qid"]]
            first = row["ranked"][0][0] if row["ranked"] else "-"
            lines.append(f"- `{row['qid']}` {row['query']}")
            lines.append(f"  - 目标片段：{'；'.join(sorted({title(c) for c in question['targets']}))[:200]}")
            lines.append(f"  - 实际第 1 名：{title(first)}")
        lines.append("")

    out_path = run_dir / "report.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"报告 -> {out_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """解析子命令。"""
    parser = argparse.ArgumentParser(description="检索评估")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-dir", type=Path, default=DATA_DIR)
    common.add_argument("--runs-dir", type=Path, default=RUNS_ROOT)
    sub = parser.add_subparsers(dest="command", required=True)

    routes = sub.add_parser("routes", help="跑三路召回并落盘", parents=[common])
    routes.add_argument("--run", required=True)
    routes.add_argument("--filter", choices=["none", "gold"], default="none")
    routes.add_argument("--collections", nargs="+", default=list(DEFAULT_COLLECTIONS))
    routes.add_argument("--candidates", type=int, default=20)
    routes.add_argument("--limit", type=int, default=0)
    routes.set_defaults(func=cmd_routes)

    ablation = sub.add_parser("ablation", parents=[common], help="重放融合组合")
    ablation.add_argument("--run", required=True)
    ablation.add_argument("--configs", nargs="+", default=None)
    ablation.set_defaults(func=cmd_ablation)

    rerank = sub.add_parser("rerank", parents=[common], help="对 Top-20 重排")
    rerank.add_argument("--run", required=True)
    rerank.add_argument("--config", default="dense+bm25:0.2@z")
    rerank.add_argument("--backend", choices=["none", "api", "local"], default="api")
    rerank.add_argument("--batch-size", type=int, default=8)
    rerank.add_argument("--limit", type=int, default=0)
    rerank.set_defaults(func=cmd_rerank)

    report = sub.add_parser("report", parents=[common], help="生成 markdown 报告")
    report.add_argument("--run", required=True)
    report.add_argument("--price-per-1k-tokens", type=float, default=0.0)
    report.add_argument("--max-misses", type=int, default=20)
    report.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
