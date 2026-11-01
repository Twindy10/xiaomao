"""入库后的检索抽查：拿检索评估集里有答案的题去搜，看证据片段能不能进 Top-K。

只做 M2 验收用的粗查（证据文档在哪个 collection 就搜哪个，不做意图路由、不加过滤、不 rerank），
正式的检索评估在 M3。

    uv run python scripts/search_sanity.py              # 随机 10 道
    uv run python scripts/search_sanity.py --all        # 全部有答案的题，看三路各自的命中率
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

from dotenv import load_dotenv

from xiaomao.textnorm import normalize_for_match

ROOT_DIR = Path(__file__).resolve().parents[1]

ROUTES = ("dense", "bm25", "sparse_m3", "hybrid")
CANDIDATES = 20  # 每一路先取 Top-20，再用 RRF 合并


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def evidence_targets(item: dict, chunks_by_doc: dict[str, list[dict]]) -> set[str]:
    """证据引文所在的片段 id（与 validate_data.py 的匹配规则一致）。"""
    targets: set[str] = set()
    for entry in item["evidence"]:
        quote = normalize_for_match(entry["quote"])
        for chunk in chunks_by_doc.get(entry["doc_key"], []):
            if quote in normalize_for_match(chunk["text"]):
                targets.add(chunk["chunk_id"])
    return targets


class Searcher:
    """一个 BGE-M3 + 一个 MilvusClient，按四种方式检索。"""

    def __init__(self) -> None:
        from FlagEmbedding import BGEM3FlagModel
        from pymilvus import MilvusClient

        self.model = BGEM3FlagModel(os.environ["EMBED_MODEL"], use_fp16=True, devices="cuda:0")
        self.client = MilvusClient(uri=os.environ["MILVUS_URI"], token=os.environ.get("MILVUS_TOKEN", ""))

    def search(self, collection: str, query: str, k: int) -> dict[str, list[str]]:
        from pymilvus import AnnSearchRequest, RRFRanker

        out = self.model.encode([query], max_length=512, return_dense=True, return_sparse=True)
        dense = [float(x) for x in out["dense_vecs"][0]]
        sparse = {int(key): float(value) for key, value in out["lexical_weights"][0].items() if value > 0}
        requests = {
            "dense": AnnSearchRequest([dense], "dense", {"metric_type": "COSINE", "params": {"ef": 64}}, CANDIDATES),
            "bm25": AnnSearchRequest([query], "sparse_bm25", {"metric_type": "BM25"}, CANDIDATES),
            "sparse_m3": AnnSearchRequest([sparse], "sparse_m3", {"metric_type": "IP"}, CANDIDATES),
        }
        results: dict[str, list[str]] = {}
        for route, request in requests.items():
            hits = self.client.search(collection, data=request.data, anns_field=request.anns_field,
                                      search_params=request.param, limit=k)[0]
            results[route] = [hit["chunk_id"] for hit in hits]
        hybrid = self.client.hybrid_search(collection, [requests["dense"], requests["bm25"]],
                                           RRFRanker(60), limit=k)[0]
        results["hybrid"] = [hit["chunk_id"] for hit in hybrid]
        return results


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(ROOT_DIR / ".env", override=True)
    for name in ("NO_PROXY", "no_proxy"):
        os.environ[name] = ",".join(filter(None, [os.environ.get(name), "127.0.0.1,localhost"]))

    parser = argparse.ArgumentParser(description="入库后的检索抽查")
    parser.add_argument("--n", type=int, default=10, help="随机抽几道（默认 10）")
    parser.add_argument("--all", action="store_true", help="跑全部有答案的题")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261031)
    parser.add_argument("--show-misses", action="store_true", help="打印 hybrid 没命中的题")
    args = parser.parse_args()

    documents = {doc["doc_key"]: doc for doc in read_jsonl(ROOT_DIR / "data/processed/documents.jsonl")}
    chunks_by_doc: dict[str, list[dict]] = {}
    for chunk in read_jsonl(ROOT_DIR / "data/processed/chunks.jsonl"):
        chunks_by_doc.setdefault(chunk["doc_key"], []).append(chunk)

    items = []
    for item in read_jsonl(ROOT_DIR / "data/eval/retrieval_qa.jsonl"):
        if item.get("answerable") is not True:
            continue
        collections = {documents[e["doc_key"]]["collection"] for e in item["evidence"] if e["doc_key"] in documents}
        if len(collections) != 1:
            continue  # 证据跨两个 collection 的题留给 M3 的路由去测
        items.append((item, collections.pop()))
    if not args.all:
        items = random.Random(args.seed).sample(items, min(args.n, len(items)))

    searcher = Searcher()
    hits = {route: 0 for route in ROUTES}
    misses: list[str] = []
    for item, collection in items:
        targets = evidence_targets(item, chunks_by_doc)
        results = searcher.search(collection, item["query"], args.k)
        for route in ROUTES:
            if targets & set(results[route]):
                hits[route] += 1
        if not targets & set(results["hybrid"]):
            misses.append(f"  {item['id']} [{collection}] {item['query']}")

    print(f"抽查 {len(items)} 道（有答案、证据在同一个 collection），Hit@{args.k}：")
    for route in ROUTES:
        print(f"  {route:<10} {hits[route]:>3}/{len(items)}  {hits[route] / len(items):.0%}")
    if misses and (args.show_misses or not args.all):
        print("hybrid 没命中：")
        print("\n".join(misses))
    return 0


if __name__ == "__main__":
    sys.exit(main())
