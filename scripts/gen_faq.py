"""FAQ 生成：根据政策与法规文档，生成新手卖家常问的 FAQ。

用法：
    python scripts/gen_faq.py --docs data/processed/documents.jsonl \
        --out data/processed/faq_items.jsonl [--target 500] [--doc-keys k1,k2] [--limit N] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

from xiaomao.llm import LLMClient
from xiaomao.textnorm import normalize_for_match

sys.stdout.reconfigure(encoding="utf-8")  # Windows 控制台默认 GBK

ROOT_DIR = Path(__file__).resolve().parents[1]
FAQ_COLLECTIONS = {"xm_platform_policy", "xm_compliance"}
INTENTS = {
    "platform_policy", "compliance", "logistics", "tax", "ops", "account", "chitchat", "out_of_scope",
}
SEGMENT_LIMIT = 4000
MIN_QUOTE, MAX_QUOTE = 10, 200
MIN_ANSWER, MAX_ANSWER = 20, 400
HEADING_RE = re.compile(r"^#{1,6}\s")

SYSTEM_PROMPT = (
    "你是一名资深跨境电商运营与合规专家，正在为中国中小卖家编写 FAQ。"
    "你只能使用给出的原文信息，不能引入原文以外的知识；原文不足以支撑完整问答时宁可少出。"
    "始终输出合法的 JSON 对象。"
)


def doc_quota(text_length: int) -> int:
    """每篇文档的 FAQ 配额：按字数估算，2–6 条。"""
    return min(6, max(2, math.ceil(text_length / 1500)))


def split_segments(text: str, limit: int = SEGMENT_LIMIT) -> list[str]:
    """按段落贪心合并成不超过 limit 字符的段，尽量在标题行前断开。"""
    segments: list[str] = []
    current: list[str] = []
    size = 0

    def flush() -> None:
        nonlocal current, size
        if current:
            segments.append("\n".join(current).strip())
        current, size = [], 0

    for line in text.split("\n"):
        is_heading = bool(HEADING_RE.match(line))
        if current and size + len(line) + 1 > limit and (is_heading or size > limit * 0.6):
            flush()
        current.append(line)
        size += len(line) + 1
    flush()
    return [segment for segment in segments if segment]


def allocate(total: int, segments: list[str]) -> list[int]:
    """按各段字数比例分配配额，每段至少 1 条。"""
    if not segments:
        return []
    sizes = [len(segment) for segment in segments]
    whole = sum(sizes) or 1
    counts = [max(1, round(total * size / whole)) for size in sizes]
    while sum(counts) > total and max(counts) > 1:
        counts[counts.index(max(counts))] -= 1
    return counts


def build_prompt(document: dict, segment: str, count: int) -> str:
    """构造一次 FAQ 生成的 prompt。"""
    return f"""文档标题：{document['title']}
所属平台：{document.get('platform') or '不适用'}
适用站点：{document.get('site') or '通用'}

原文：
\"\"\"
{segment}
\"\"\"

请基于上面的原文，生成 {count} 条新手卖家 FAQ，输出 JSON：
{{"faqs": [{{"question": "...", "answer": "...", "quote": "...", "intent": "..."}}]}}

要求：
1. 问题用中国中小跨境卖家的口吻，口语化、可以不规范，可以自然带上"亚马逊""英国"这类背景，但不要强行加。
2. 答案用中文写 2–5 句话（{MIN_ANSWER}–{MAX_ANSWER} 字），只能用原文里的信息，不能补充原文以外的知识。
3. quote 从原文逐字复制**一到两句话**，长度必须落在 {MIN_QUOTE}–{MAX_QUOTE} 个字符之间（含标点，按字符数算，不要复制整段）；原文是英文就复制英文；不要复制标题行。超过 {MAX_QUOTE} 字符会被丢弃。
4. intent 从这 8 类里选一个：platform_policy（平台规则）、compliance（合规认证）、logistics（物流）、tax（税务）、ops（运营技巧）、account（账号问题）、chitchat（闲聊）、out_of_scope（超出范围）。
5. 同一段里的几个问题要问不同的点，不要换个说法问同一件事。
6. 原文不足以支撑完整问答时，少出几条，不要硬凑。"""


def _text_stats() -> dict[str, int]:
    """丢弃原因计数器。"""
    return {
        "json_error": 0,
        "quote_mismatch": 0,
        "quote_length": 0,
        "answer_length": 0,
        "duplicate": 0,
        "bad_fields": 0,
    }


def validate_faqs(
    payload: dict,
    document: dict,
    seen: set[str],
    stats: dict[str, int],
) -> list[dict]:
    """校验模型输出：引文、长度、去重、枚举；返回通过校验的条目。"""
    items = payload.get("faqs")
    if not isinstance(items, list):
        stats["json_error"] += 1
        return []

    body = normalize_for_match(document["text"])
    kept: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            stats["json_error"] += 1
            continue
        question = str(item.get("question") or "").strip()
        answer = str(item.get("answer") or "").strip()
        quote = str(item.get("quote") or "").strip()
        intent = str(item.get("intent") or "").strip()

        if not question or _key(question) in seen:
            stats["duplicate"] += 1
            continue
        if not MIN_QUOTE <= len(quote) <= MAX_QUOTE:
            stats["quote_length"] += 1
            continue
        if normalize_for_match(quote) not in body:
            stats["quote_mismatch"] += 1
            continue
        if not MIN_ANSWER <= len(answer) <= MAX_ANSWER:
            stats["answer_length"] += 1
            continue
        if intent not in INTENTS:
            stats["bad_fields"] += 1
            continue

        seen.add(_key(question))
        kept.append(
            {
                "question": question,
                "answer": answer,
                "intent": intent,
                "platform": document.get("platform"),
                "site": document.get("site"),
                "category": document.get("category"),
                "lang": "zh",
                "effective_date": document.get("effective_date"),
                "evidence": [{"doc_key": document["doc_key"], "quote": quote}],
                "source": "llm_generated",
                "reviewed": False,
                "reviewed_by": None,
                "note": None,
            }
        )
    return kept


def _key(question: str) -> str:
    """去重用的问题指纹：去掉空白和标点。"""
    return re.sub(r"[\s\W_]+", "", question).lower()


def read_jsonl(path: Path) -> list[dict]:
    """读 JSONL，跳过空行。"""
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


def load_progress(path: Path) -> dict:
    """读断点续跑进度。"""
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"done_doc_keys": [], "next_faq_id": 1}


def save_progress(path: Path, progress: dict) -> None:
    """写断点续跑进度。"""
    path.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None, client=None) -> int:
    """按文档配额生成 FAQ，写进 faq_items.jsonl。"""
    parser = argparse.ArgumentParser(description="生成 FAQ")
    parser.add_argument("--docs", type=Path, default=ROOT_DIR / "data/processed/documents.jsonl")
    parser.add_argument("--out", type=Path, default=ROOT_DIR / "data/processed/faq_items.jsonl")
    parser.add_argument("--target", type=int, default=500, help="总条数达到这个值就停止")
    parser.add_argument("--doc-keys", default="", help="只处理这几篇（逗号分隔）")
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 篇文档")
    parser.add_argument("--dry-run", action="store_true", help="只打印第一个请求的 prompt")
    args = parser.parse_args(argv)

    documents = [
        doc for doc in read_jsonl(args.docs) if doc.get("collection") in FAQ_COLLECTIONS
    ]
    if args.doc_keys:
        wanted = {key.strip() for key in args.doc_keys.split(",") if key.strip()}
        documents = [doc for doc in documents if doc["doc_key"] in wanted]
    if args.limit:
        documents = documents[: args.limit]
    if not documents:
        print("[ERROR] 没有可处理的文档")
        return 1

    progress_path = args.out.with_suffix(".progress.json")
    progress = load_progress(progress_path)
    done = set(progress.get("done_doc_keys") or [])
    documents = [doc for doc in documents if doc["doc_key"] not in done]
    if not documents:
        print("全部文档都处理过了，跳过")
        return 0

    existing = read_jsonl(args.out)
    seen = {_key(row["question"]) for row in existing}
    stats = _text_stats()

    # 按文档配额估算要处理几篇，凑够 --target 就停
    budget = max(0, args.target - len(existing))
    selected: list[dict] = []
    planned = 0
    for document in documents:
        selected.append(document)
        planned += doc_quota(len(document["text"]))
        if planned >= budget:
            break

    tasks: list[tuple[dict, str, int]] = []
    for document in selected:
        quota = doc_quota(len(document["text"]))
        segments = split_segments(document["text"])
        for segment, count in zip(segments, allocate(quota, segments)):
            tasks.append((document, segment, count))

    if args.dry_run:
        document, segment, count = tasks[0]
        print(f"[dry-run] 共 {len(tasks)} 个请求；第一个：{document['title']}（{count} 条）\n")
        print(build_prompt(document, segment, count))
        return 0

    client = client or LLMClient()

    def worker(task: tuple[dict, str, int]) -> tuple[dict, dict]:
        document, segment, count = task
        return document, client.chat_json(build_prompt(document, segment, count), SYSTEM_PROMPT)

    results: list[dict] = []
    for document, payload in client.map(worker, tasks):
        results.extend(validate_faqs(payload, document, seen, stats))

    next_id = max(progress.get("next_faq_id") or 1, len(existing) + 1)
    for offset, row in enumerate(results):
        row["faq_id"] = f"faq-{next_id + offset:04d}"

    write_jsonl(args.out, existing + results)
    progress["next_faq_id"] = next_id + len(results)
    progress["done_doc_keys"] = sorted(done | {doc["doc_key"] for doc in selected})
    save_progress(progress_path, progress)

    doc_collection = {doc["doc_key"]: doc["collection"] for doc in selected}
    by_collection: dict[str, int] = {}
    by_intent: dict[str, int] = {}
    for row in results:
        collection = doc_collection.get(row["evidence"][0]["doc_key"], "未知")
        by_collection[collection] = by_collection.get(collection, 0) + 1
        by_intent[row["intent"]] = by_intent.get(row["intent"], 0) + 1

    print(f"\n新增 {len(results)} 条，文件共 {len(existing) + len(results)} 条 -> {args.out}")
    for collection in sorted(by_collection):
        print(f"  {collection}: {by_collection[collection]} 条")
    print("  意图分布：" + "，".join(f"{key}={value}" for key, value in sorted(by_intent.items())))
    dropped = {key: value for key, value in stats.items() if value}
    if dropped:
        print("丢弃原因：" + "，".join(f"{key}={value}" for key, value in sorted(dropped.items())))
    print(client.usage.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
