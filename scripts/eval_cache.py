"""语义缓存评估：复现「τ = 0.80 + LLM 把关」这一行。

    uv run python scripts/eval_cache.py [--gate llm|none] [--tau 0.80] [--run m4b]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

for _name in ("NO_PROXY", "no_proxy"):
    os.environ[_name] = ",".join(filter(None, [os.environ.get(_name), "127.0.0.1,localhost"]))

from dotenv import load_dotenv  # noqa: E402

DATA_DIR = ROOT_DIR / "data"
PREFIX = "xm:evalcache:"
INDEX_NAME = "xm:idx:evalcache"

# Claude 2026-10-31 逐条判断：这 8 对"命中别的问题"不能复用；其余命中别的问题的都是重复题，可以复用
BAD_REUSE = {
    ("亚马逊美国站FBA的仓储费和附加费具体是怎么计算的？", "qa-0105"),
    ("英国大型组织在满足包装回收义务时，关于PRNs和PERNs的使用有哪些具体要求？", "qa-0162"),
    ("日本亚马逊对新卖家有什么补贴？我应该卖什么产品比较好？", "qa-0012"),
    ("亚马逊为新入驻的品牌注册卖家提供了哪些返利和资金支持？", "qa-0061"),
    ("参与低价商品物流计划的卖家，每单配送费能便宜多少美元？", "qa-0087"),
    ("申请亚马逊低价商品物流费率需要做什么操作吗？", "qa-0014"),
    ("如何判断我的商品是否符合亚马逊低价物流的费率标准？", "qa-0014"),
    ("做英国自配送需要查商品编码吗？为什么？", "qa-0106"),
}


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


def main(argv: list[str] | None = None) -> int:
    """写缓存、跑同义改写和无答案题，输出对比表。"""
    load_dotenv(ROOT_DIR / ".env", override=True)
    parser = argparse.ArgumentParser(description="语义缓存评估")
    parser.add_argument("--run", default="m4b")
    parser.add_argument("--answers", type=Path, default=ROOT_DIR / "eval/runs/qa_m4v4/routing.jsonl")
    parser.add_argument("--gate", choices=["llm", "none"], default="llm")
    parser.add_argument("--tau", type=float, default=0.80)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)

    from xiaomao.cache.semantic import SemanticCache
    from xiaomao.decision.llm_json import LLMJsonDecider
    from xiaomao.ingest.embed import Embedder
    from xiaomao.retrieval.search import encode_queries
    from xiaomao.runtime import redis_from_env

    items = read_jsonl(DATA_DIR / "eval" / "retrieval_qa.jsonl")
    answers = {row["qid"]: row for row in read_jsonl(args.answers)}
    cached = [item for item in items if item.get("answerable") is True and item["id"] in answers]
    unanswerable = [item for item in items if item.get("answerable") is False]
    if args.limit:
        cached = cached[: args.limit]

    queries: list[dict] = []
    for item in cached:
        for paraphrase in item.get("paraphrases") or []:
            queries.append({"qid": item["id"], "query": paraphrase, "answerable": True})
    for item in unanswerable:
        queries.append({"qid": item["id"], "query": item["query"], "answerable": False})

    texts = [item["query"] for item in cached] + [row["query"] for row in queries]
    embedder = Embedder(model_name=os.environ.get("EMBED_MODEL"), max_length=512)
    vectors = encode_queries(embedder, texts)
    cache_vectors = vectors[: len(cached)]
    query_vectors = vectors[len(cached):]

    gate = LLMJsonDecider() if args.gate == "llm" else None
    cache = SemanticCache(
        redis_from_env(), gate=gate, tau=args.tau, prefix=PREFIX, index_name=INDEX_NAME,
    )
    cache.reset()

    # 写入：138 道有答案题当作"已缓存"
    for item, vector in zip(cached, cache_vectors):
        row = answers[item["id"]]
        cache.write(
            question=item["query"],
            vector=vector.dense,
            platform=item.get("platform"),
            site=item.get("site"),
            intent=item.get("intent") or "",
            answer=row.get("answer") or "",
            citations=row.get("citations") or [],
            doc_ids=[],
        )

    indexed = cache.wait_until_indexed(len(cached))
    print(f"索引就绪：{indexed}/{len(cached)} 条")

    by_id = {item["id"]: item for item in items}
    rows: list[dict] = []
    lookup_ms: list[float] = []
    gate_ms: list[float] = []
    gate_tokens: list[int] = []
    for query, vector in zip(queries, query_vectors):
        item = by_id[query["qid"]]
        hit, detail = cache.lookup(
            query["query"], vector.dense,
            platform=item.get("platform"), site=item.get("site"),
        )
        if detail.get("lookup_ms") is not None:
            lookup_ms.append(float(detail["lookup_ms"]))
        if detail.get("gate_ms"):
            gate_ms.append(float(detail["gate_ms"]))
        if detail.get("gate_tokens"):
            gate_tokens.append(int(detail["gate_tokens"]))

        hit_qid = None
        if hit is not None:
            hit_qid = next((key for key, value in by_id.items() if value["query"] == hit.question), None)
            hit_qid = hit_qid or detail.get("candidate_id")
        rows.append({
            "qid": query["qid"],
            "query": query["query"],
            "answerable": query["answerable"],
            "hit": hit is not None,
            "hit_question": hit.question if hit else None,
            "hit_qid": hit_qid,
            "similarity": detail.get("similarity"),
            "gate_p": detail.get("gate_p"),
        })

    # 命中判定：命中自己算正确；命中别的问题时按 BAD_REUSE 清单判对错
    paraphrase_rows = [row for row in rows if row["answerable"]]
    for row in paraphrase_rows:
        if not row["hit"]:
            row["verdict"] = "miss"
        elif row["hit_qid"] == row["qid"]:
            row["verdict"] = "correct"
        elif (row["query"], row["hit_qid"]) in BAD_REUSE:
            row["verdict"] = "wrong"
        else:
            row["verdict"] = "equivalent"

    hits = [row for row in paraphrase_rows if row["hit"]]
    correct = sum(1 for row in hits if row["verdict"] == "correct")
    equivalent = sum(1 for row in hits if row["verdict"] == "equivalent")
    wrong = sum(1 for row in hits if row["verdict"] == "wrong")
    total = len(paraphrase_rows)
    bad_hits = sum(1 for row in hits if row["verdict"] == "wrong")
    false_hits = sum(1 for row in rows if not row["answerable"] and row["hit"])

    out_path = ROOT_DIR / "eval" / "runs" / f"cache_{args.run}.jsonl"
    write_jsonl(out_path, rows)

    print(f"\nτ={args.tau}，把关={args.gate}，缓存 {len(cached)} 条，同义改写 {total} 条，"
          f"无答案题 {len(unanswerable)} 条")
    print(f"  命中率 {len(hits) / total:.1%}（{len(hits)}/{total}）")
    print(f"  命中中：正确 {correct}、等价 {equivalent}、错误 {wrong}")
    print(f"  BAD_REUSE 清单放行 {bad_hits} 对（验收 ≤ 4）")
    print(f"  无答案题误命中 {false_hits}")
    print(f"  查找 P50 {percentile(lookup_ms, 0.5):.1f}ms / P95 {percentile(lookup_ms, 0.95):.1f}ms，"
          f"把关 P50 {percentile(gate_ms, 0.5):.0f}ms，平均 {sum(gate_tokens) / len(gate_tokens):.0f} tokens"
          if gate_tokens else f"  查找 P50 {percentile(lookup_ms, 0.5):.1f}ms")
    print(f"  明细 -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
