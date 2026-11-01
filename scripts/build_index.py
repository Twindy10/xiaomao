"""向量化入库入口：规划 → 编码 → 写 Milvus / MySQL → 对账。

用法：
    uv run python scripts/build_index.py [--collection xm_faq ...] [--dry-run] [--rebuild]
                                          [--expire-missing] [--batch-size 16]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")  # Windows 控制台默认 GBK

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))


def _add_no_proxy() -> None:
    """本地服务不走系统代理（gRPC 会读系统代理）。"""
    for name in ("NO_PROXY", "no_proxy"):
        current = os.environ.get(name, "")
        parts = [item for item in current.split(",") if item]
        for host in ("127.0.0.1", "localhost"):
            if host not in parts:
                parts.append(host)
        os.environ[name] = ",".join(parts)


_add_no_proxy()

from dotenv import load_dotenv  # noqa: E402

from xiaomao.ingest.embed import Embedder  # noqa: E402
from xiaomao.ingest.incremental import RunStats, apply_plan, plan_collection, reconcile  # noqa: E402
from xiaomao.ingest.index import COLLECTIONS, ensure_collection  # noqa: E402
from xiaomao.ingest.kb_store import IngestLock, KbDocumentStore, connect_mysql  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    """读 JSONL。"""
    if not path.is_file():
        raise SystemExit(f"[ERROR] 找不到 {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def group_chunks(chunks: list[dict]) -> dict[str, list[dict]]:
    """按 doc_key 分组，并按 chunk_index 排好序。"""
    grouped: dict[str, list[dict]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk["doc_key"], []).append(chunk)
    for rows in grouped.values():
        rows.sort(key=lambda item: item["chunk_index"])
    return grouped


def check_alignment(documents: list[dict], chunks_by_doc: dict[str, list[dict]]) -> None:
    """chunks 的 doc_key 必须都能在 documents 里找到，且 collection 一致。"""
    doc_by_key = {doc["doc_key"]: doc for doc in documents}
    for doc_key, rows in chunks_by_doc.items():
        document = doc_by_key.get(doc_key)
        if document is None:
            raise SystemExit(f"[ERROR] 片段引用了不存在的文档：{doc_key}")
        mismatched = {row["collection"] for row in rows} - {document["collection"]}
        if mismatched:
            raise SystemExit(f"[ERROR] {doc_key} 的片段 collection 与文档不一致：{sorted(mismatched)}")


def print_plan(collection: str, actions: list, limit: int = 3) -> None:
    """dry-run 时打印规划结果。"""
    counts: dict[str, int] = {}
    for action in actions:
        counts[action.kind] = counts.get(action.kind, 0) + 1
    summary = "，".join(f"{kind}={value}" for kind, value in sorted(counts.items()))
    print(f"[{collection}] 共 {len(actions)} 篇：{summary}")
    for action in actions[:limit]:
        title = (action.doc or action.old or {}).get("title") or ""
        print(f"    - {action.kind}: {str(title)[:40]}")


def main(argv: list[str] | None = None) -> int:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="向量化入库")
    parser.add_argument("--collection", action="append", default=None, choices=list(COLLECTIONS))
    parser.add_argument("--dry-run", action="store_true", help="只做规划，不写 Milvus、不加载模型")
    parser.add_argument("--rebuild", action="store_true", help="删掉 collection 重建，所有文档重写")
    parser.add_argument("--expire-missing", action="store_true", help="jsonl 里消失的文档标记 expired")
    parser.add_argument("--docs", type=Path, default=ROOT_DIR / "data/processed/documents.jsonl")
    parser.add_argument("--chunks", type=Path, default=ROOT_DIR / "data/processed/chunks.jsonl")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args(argv)

    load_dotenv(ROOT_DIR / ".env", override=True)

    documents = read_jsonl(args.docs)
    chunks_by_doc = group_chunks(read_jsonl(args.chunks))
    check_alignment(documents, chunks_by_doc)

    selected = args.collection or list(COLLECTIONS)
    docs_by_collection: dict[str, list[dict]] = {name: [] for name in selected}
    for document in documents:
        if document["collection"] in docs_by_collection:
            docs_by_collection[document["collection"]].append(document)

    store = KbDocumentStore(connect_mysql())
    milvus = None
    redis_client = None
    if not args.dry_run:
        from pymilvus import MilvusClient
        from redis import Redis

        milvus = MilvusClient(uri=os.environ["MILVUS_URI"], token=os.environ.get("MILVUS_TOKEN", ""))
        redis_client = Redis.from_url(os.environ["REDIS_URL"], socket_timeout=10)

    holder: dict[str, Embedder] = {}

    def embedder_factory() -> Embedder:
        """整个进程只加载一个 BGE-M3。"""
        if "embedder" not in holder:
            holder["embedder"] = Embedder(batch_size=args.batch_size)
        return holder["embedder"]

    exit_code = 0
    total_chunks = 0
    started = time.perf_counter()

    for collection in selected:
        print(f"\n=== {collection} ===")
        if args.dry_run:
            actions = plan_collection(collection, docs_by_collection[collection], store)
            print_plan(collection, actions)
            continue

        with IngestLock(redis_client, collection):
            created = ensure_collection(milvus, collection, drop=args.rebuild)
            if created:
                print("  新建 collection" if not args.rebuild else "  重建 collection")
            actions = plan_collection(collection, docs_by_collection[collection], store, rebuild=args.rebuild)
            stats: RunStats = apply_plan(
                collection,
                actions,
                chunks_by_doc,
                embedder_factory=embedder_factory,
                milvus=milvus,
                store=store,
                redis_client=redis_client,
                expire_missing=args.expire_missing or args.rebuild,
            )
            problems = reconcile(
                collection, docs_by_collection[collection], chunks_by_doc, actions,
                milvus=milvus, store=store,
            )

        summary = "，".join(f"{kind}={value}" for kind, value in sorted(stats.actions.items()))
        print(f"  动作：{summary}")
        print(f"  写入片段 {stats.chunks_written}（编码 {stats.embedded} 条，"
              f"编码 {stats.encode_seconds:.1f}s，写入 {stats.write_seconds:.1f}s）")
        print(f"  缓存失效 {stats.cache_invalidated} 条")
        if problems:
            exit_code = 1
            for problem in problems:
                print(f"  [对账失败] {problem}")
        else:
            print("  对账一致")
        total_chunks += stats.chunks_written

    elapsed = time.perf_counter() - started
    print(f"\n合计：写入 {total_chunks} 条片段，耗时 {elapsed:.1f}s")
    if "embedder" in holder:
        peak = holder["embedder"].peak_vram_mb()
        print(f"峰值显存：{peak:.0f} MB" if peak is not None else "峰值显存：无 CUDA")
    else:
        print("峰值显存：模型未加载")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
