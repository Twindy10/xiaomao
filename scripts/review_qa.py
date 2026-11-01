"""检索评估集审核（用户委托 Claude 审，结果标 reviewed_by=claude）：模型筛查 + Claude 复核。

流程同 review_faq.py：
    1. screen：可回答的题，让审核模型对照证据文档判断参考答案、问题、改写、标注；
       不可回答的题，先用 BGE-M3 在全部片段里检索最像的 8 块，再让审核模型判断知识库是不是其实能答
    2. report：汇总并列出需要复核的条目
    3. apply：读入 Claude 的决定，写回 retrieval_qa.jsonl；删除的移到 rejected.jsonl

用法：
    uv run python scripts/review_qa.py screen [--ids qa-0001,qa-0002] [--model qwen3.7-plus]
    uv run python scripts/review_qa.py report [--show flagged|all|none]
    uv run python scripts/review_qa.py apply [--decisions data/processed/qa_decisions.jsonl]

决定文件一行一条：{"id": "...", "action": "keep|edit|drop", "query"?, "reference_answer"?,
"paraphrases"?, "intent"?, "platform"?, "site"?, "reason"?}。platform/site 写 "" 表示改成 null。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

from xiaomao.llm import LLMClient
from xiaomao.textnorm import normalize_for_match

sys.stdout.reconfigure(encoding="utf-8")  # Windows 控制台默认 GBK

ROOT_DIR = Path(__file__).resolve().parents[1]
PROCESSED = ROOT_DIR / "data" / "processed"
QA_PATH = ROOT_DIR / "data" / "eval" / "retrieval_qa.jsonl"
REJECTED_PATH = ROOT_DIR / "data" / "eval" / "rejected.jsonl"
REVIEW_PATH = PROCESSED / "qa_review.jsonl"

INTENTS = ("platform_policy", "compliance", "logistics", "tax", "ops", "account", "chitchat", "out_of_scope")
PLATFORMS = ("amazon", "temu", "tiktok_shop", "shopee")
LEVELS = ("yes", "partial", "no")
FULL_DOC_LIMIT = 12000  # 不超过这个长度的证据文档整篇给审核模型
WINDOW_LIMIT = 8000
TOP_K = 8  # 不可回答的题检索多少块给审核模型看

SYSTEM_PROMPT = (
    "你是一名严格的检索评估集审核员。评估集用来衡量一个跨境电商知识库问答系统的检索和回答质量，"
    "标错一题就会让指标失真。你只根据给出的材料判断，不用自己的知识补全。始终输出合法的 JSON 对象。"
)

LABEL_RULES = """意图只能是这些英文键之一：platform_policy（平台规则、政策、违规处罚）、compliance（产品合规、认证、法规、EPR）、
logistics（物流、仓储、配送）、tax（税务、VAT、关税）、ops（运营、推广、选品、开店福利、工具使用）、
account（开店注册、账户、申诉）、chitchat、out_of_scope。
platform 只标问题里**明确提到**的平台（amazon / temu / tiktok_shop / shopee），没提就是 null，不要按常识推测；
site 同理，只标明确提到的国家或站点，用两位大写代码（US、UK、DE、JP、EU 等），没提就是 null。"""

ANSWERABLE_PROMPT = """审核下面这道"可回答"的检索评估题。

## 证据所在的原文
{context}

## 题目
问题：{query}
证据引文：{quotes}
参考答案：{answer}
同义改写：{paraphrases}
当前标注：intent={intent}，platform={platform}，site={site}

## 审核项
1. answerable：只凭上面的原文，这个问题能否被完整回答。"yes" 能；"partial" 只能答一部分；"no" 原文里没有答案。
2. faithful：参考答案的每个事实（数字、日期、条件、机构名）是否都能在原文找到依据。"yes"/"partial"/"no"，
   原文找不到依据的说法原样摘出放进 unsupported。
3. bad_paraphrases：意思和问题不一样（问了别的点、范围变了、多了或少了条件）的改写，原样列出；都没问题给空列表。
4. needs_all_docs：{multi_hint}
5. intent / platform / site：按下面的规则给出正确的标注。
{label_rules}
6. comment：一句话说明主要问题；没有问题写"无"。

只输出 JSON：{{"answerable": "yes|partial|no", "faithful": "yes|partial|no", "unsupported": [], "bad_paraphrases": [],
"needs_all_docs": true, "intent": "...", "platform": null, "site": null, "comment": "..."}}"""

UNANSWERABLE_PROMPT = """审核下面这道标成"知识库无法回答"的检索评估题。它用来测试系统能否老实说"知识库未覆盖"，
所以知识库里**确实不能有答案**。下面是在整个知识库里检索出的最相近的 {k} 个片段。

## 最相近的片段
{chunks}

## 题目
问题：{query}
同义改写：{paraphrases}
当前标注：intent={intent}，platform={platform}，site={site}

## 审核项
1. covered：这些片段能否回答这个问题。"yes" 能完整回答；"partial" 能回答一部分或者答案高度相关；"no" 不能。
   如果是 yes 或 partial，在 covering 里写出是第几个片段。
2. bad_paraphrases：意思和问题不一样的改写，原样列出；都没问题给空列表。
3. intent / platform / site：按下面的规则给出正确的标注。
{label_rules}
4. comment：一句话说明。

只输出 JSON：{{"covered": "yes|partial|no", "covering": [], "bad_paraphrases": [], "intent": "...",
"platform": null, "site": null, "comment": "..."}}"""


def _read_jsonl(path: Path) -> list[dict]:
    """读 JSONL，跳过空行。"""
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    """先写临时文件再改名。"""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _doc_context(doc: dict, quotes: list[str], chunks: list[dict]) -> str:
    """短文档给全文；长文档给每条引文所在片段的前后各 2 块。"""
    header = f"【{doc['title']}】"
    if len(doc["text"]) <= FULL_DOC_LIMIT:
        return f"{header}\n{doc['text']}"
    picked: set[int] = set()
    for quote in quotes:
        norm = normalize_for_match(quote)
        for index, chunk in enumerate(chunks):
            if norm in normalize_for_match(chunk["text"]):
                picked.update(range(max(0, index - 2), min(len(chunks), index + 3)))
    text = "\n\n".join(chunks[index]["text"] for index in sorted(picked))
    return f"{header}（节选）\n{text[:WINDOW_LIMIT]}"


def _normalize_label(value: object) -> str | None:
    """模型给的 platform/site 可能是 "null"、""、大小写不一，统一一下。"""
    if value is None:
        return None
    text = str(value).strip()
    return None if text.lower() in ("", "null", "none") else text


def _labels(result: dict) -> dict:
    platform = _normalize_label(result.get("platform"))
    site = _normalize_label(result.get("site"))
    return {
        "intent": result.get("intent"),
        "platform": platform.lower() if platform else None,
        "site": site.upper() if site else None,
    }


class Embedder:
    """BGE-M3 dense，只给不可回答的题找相近片段用。"""

    def __init__(self, chunks: list[dict]) -> None:
        import numpy as np
        from dotenv import load_dotenv
        from FlagEmbedding import BGEM3FlagModel

        load_dotenv(ROOT_DIR / ".env", override=True)
        self.np = np
        self.model = BGEM3FlagModel(os.getenv("EMBED_MODEL", "BAAI/bge-m3"), use_fp16=True)
        self.chunks = chunks
        texts = [f"{chunk['title_path']}\n{chunk['text']}" for chunk in chunks]
        self.matrix = self.model.encode(texts, batch_size=8, max_length=1024)["dense_vecs"]

    def top(self, text: str, k: int) -> list[dict]:
        vector = self.model.encode([text], max_length=512)["dense_vecs"][0]
        scores = self.matrix @ vector
        return [self.chunks[index] for index in self.np.argsort(-scores)[:k]]


def judge(client: LLMClient, row: dict, docs: dict, chunks_by_doc: dict, embedder: Embedder | None) -> dict:
    """审一题，返回结构化结果。"""
    common = dict(
        query=row["query"],
        paraphrases=json.dumps(row.get("paraphrases") or [], ensure_ascii=False),
        intent=row.get("intent"),
        platform=row.get("platform"),
        site=row.get("site"),
        label_rules=LABEL_RULES,
    )
    if row.get("answerable"):
        evidence = row["evidence"]
        by_doc: dict[str, list[str]] = {}
        for entry in evidence:
            by_doc.setdefault(entry["doc_key"], []).append(entry["quote"])
        context = "\n\n".join(
            _doc_context(docs[key], quotes, chunks_by_doc.get(key, [])) for key, quotes in by_doc.items()
        )
        multi_hint = (
            "这道题的证据来自两篇不同的文档。只用其中一篇能否完整回答？需要两篇都用到才能答全写 true，一篇就够写 false。"
            if len(by_doc) > 1
            else "单文档题，写 true。"
        )
        prompt = ANSWERABLE_PROMPT.format(
            context=context,
            quotes=json.dumps([entry["quote"] for entry in evidence], ensure_ascii=False),
            answer=row.get("reference_answer"),
            multi_hint=multi_hint,
            **common,
        )
        result = client.chat_json(prompt, SYSTEM_PROMPT, max_tokens=900, temperature=0)
        if result.get("answerable") not in LEVELS or result.get("faithful") not in LEVELS:
            raise ValueError(f"审核结果结构不对：{result}")
        return {
            "kind": "answerable",
            "answerable": result["answerable"],
            "faithful": result["faithful"],
            "unsupported": [str(item) for item in result.get("unsupported") or []],
            "bad_paraphrases": [str(item) for item in result.get("bad_paraphrases") or []],
            "needs_all_docs": bool(result.get("needs_all_docs", True)),
            "multi_doc": len(by_doc) > 1,
            **_labels(result),
            "comment": str(result.get("comment") or ""),
        }

    assert embedder is not None
    neighbours = embedder.top(row["query"], TOP_K)
    listing = "\n\n".join(
        f"[{index}]（{chunk['title_path']}）\n{chunk['text'][:1200]}" for index, chunk in enumerate(neighbours, 1)
    )
    prompt = UNANSWERABLE_PROMPT.format(k=TOP_K, chunks=listing, **common)
    result = client.chat_json(prompt, SYSTEM_PROMPT, max_tokens=600, temperature=0)
    if result.get("covered") not in LEVELS:
        raise ValueError(f"审核结果结构不对：{result}")
    covering = []
    for index in result.get("covering") or []:
        if isinstance(index, int) and 1 <= index <= len(neighbours):
            covering.append(neighbours[index - 1]["title_path"])
    return {
        "kind": "unanswerable",
        "covered": result["covered"],
        "covering": covering,
        "bad_paraphrases": [str(item) for item in result.get("bad_paraphrases") or []],
        **_labels(result),
        "comment": str(result.get("comment") or ""),
    }


def needs_attention(row: dict, review: dict) -> list[str]:
    """需要 Claude 复核的原因；空列表表示直接通过。"""
    reasons = []
    if review["kind"] == "answerable":
        if review["answerable"] != "yes":
            reasons.append(f"证据答不全={review['answerable']}")
        if review["faithful"] != "yes":
            reasons.append(f"参考答案 faithful={review['faithful']}")
        if review["multi_doc"] and not review["needs_all_docs"]:
            reasons.append("多文档题其实一篇就够")
    elif review["covered"] != "no":
        reasons.append(f"知识库其实能答={review['covered']}")
    if review["bad_paraphrases"]:
        reasons.append(f"改写偏题 {len(review['bad_paraphrases'])} 条")
    for key in ("intent", "platform", "site"):
        if review[key] != row.get(key):
            reasons.append(f"{key} {row.get(key)}→{review[key]}")
    return reasons


def cmd_screen(args: argparse.Namespace) -> int:
    """逐题筛查，已经筛过的跳过。"""
    rows = _read_jsonl(QA_PATH)
    docs = {row["doc_key"]: row for row in _read_jsonl(PROCESSED / "documents.jsonl")}
    chunks = _read_jsonl(PROCESSED / "chunks.jsonl")
    chunks_by_doc: dict[str, list[dict]] = {}
    for chunk in chunks:
        chunks_by_doc.setdefault(chunk["doc_key"], []).append(chunk)

    done = {row["id"] for row in _read_jsonl(REVIEW_PATH)}
    todo = [row for row in rows if row["id"] not in done]
    if args.ids:
        wanted = set(args.ids.split(","))
        todo = [row for row in todo if row["id"] in wanted]
    print(f"检索题 {len(rows)} 条，已筛查 {len(done)} 条，本次 {len(todo)} 条，审核模型 {args.model}")

    embedder = None
    if any(not row.get("answerable") for row in todo):
        print("加载 BGE-M3，给全部片段编码……")
        embedder = Embedder(chunks)
    client = LLMClient()
    client.model = args.model
    errors: Counter[str] = Counter()

    def work(row: dict) -> dict | None:
        try:
            return {"id": row["id"], "model": args.model, **judge(client, row, docs, chunks_by_doc, embedder)}
        except Exception as exc:  # 单条失败不中断，下次续跑会重试
            errors[type(exc).__name__] += 1
            return None

    # 不可回答的题要用本地模型编码问题，放在主线程里串行做，避免多线程同时用 GPU
    unanswerable = [row for row in todo if not row.get("answerable")]
    answerable = [row for row in todo if row.get("answerable")]
    with REVIEW_PATH.open("a", encoding="utf-8", newline="\n") as handle:
        for row in unanswerable:
            result = work(row)
            if result:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        handle.flush()
        for start in range(0, len(answerable), 20):
            for result in client.map(work, answerable[start : start + 20]):
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"  可回答题已完成 {min(start + 20, len(answerable))}/{len(answerable)}")

    if errors:
        print("失败（下次续跑会重试）：" + "，".join(f"{key}={value}" for key, value in errors.items()))
    print(client.usage.summary())
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """汇总筛查结果。"""
    rows = {row["id"]: row for row in _read_jsonl(QA_PATH)}
    reviews = {row["id"]: row for row in _read_jsonl(REVIEW_PATH) if row["id"] in rows}
    print(f"检索题 {len(rows)} 条，已筛查 {len(reviews)} 条")
    flagged = 0
    for qa_id, review in sorted(reviews.items()):
        row = rows[qa_id]
        reasons = needs_attention(row, review)
        flagged += bool(reasons)
        if args.show == "all" or (args.show == "flagged" and reasons):
            print(f"\n--- {qa_id} [{'；'.join(reasons) or '通过'}]")
            print(f"Q: {row['query']}")
            if row.get("reference_answer"):
                print(f"A: {row['reference_answer']}")
            for claim in review.get("unsupported", []):
                print(f"  ✗ {claim}")
            for item in review["bad_paraphrases"]:
                print(f"  ≠ {item}")
            if review.get("covering"):
                print(f"  覆盖片段：{' | '.join(review['covering'])}")
            print(f"  审核意见：{review['comment']}")
    print(f"\n需要复核 {flagged} 条，直接通过 {len(reviews) - flagged} 条")
    return 0


def cmd_apply(args: argparse.Namespace) -> int:
    """把决定写回 retrieval_qa.jsonl。"""
    rows = _read_jsonl(QA_PATH)
    reviews = {row["id"]: row for row in _read_jsonl(REVIEW_PATH)}
    decisions = {row["id"]: row for row in _read_jsonl(args.decisions)}
    rejected = _read_jsonl(REJECTED_PATH)

    kept: list[dict] = []
    stats: Counter[str] = Counter()
    for row in rows:
        if row.get("reviewed_by") == "human":
            kept.append(row)
            continue
        review = reviews.get(row["id"])
        decision = decisions.get(row["id"])
        if decision is None:
            if review is None or needs_attention(row, review):
                stats["未决定，保持未审核"] += 1
                kept.append(row)
                continue
            decision = {"action": "keep"}
        action = decision["action"]
        if action == "drop":
            rejected.append({"set": "检索集", "reason": f"Claude 审核删除：{decision.get('reason') or '未注明'}", "row": row})
            stats["drop"] += 1
            continue
        if action not in ("keep", "edit"):
            raise ValueError(f"{row['id']} 的 action 不合法：{action}")
        for key in ("query", "reference_answer", "paraphrases", "intent"):
            if decision.get(key):
                row[key] = decision[key]
        for key in ("platform", "site"):
            if key in decision:
                row[key] = decision[key] or None
        if row.get("intent") not in INTENTS or (row.get("platform") and row["platform"] not in PLATFORMS):
            raise ValueError(f"{row['id']} 的标注不合法：{row.get('intent')} / {row.get('platform')}")
        if len(row.get("paraphrases") or []) < 2:
            raise ValueError(f"{row['id']} 的改写少于 2 条")
        row["reviewed"], row["reviewed_by"] = True, "claude"
        if action == "edit":
            row["note"] = "；".join(part for part in (row.get("note"), f"Claude 审核修改：{decision.get('reason') or ''}") if part)
        kept.append(row)
        stats[action] += 1

    _write_jsonl(QA_PATH, kept)
    _write_jsonl(REJECTED_PATH, rejected)
    print("写回 " + "，".join(f"{key} {value}" for key, value in stats.items()))
    reviewed = [row for row in kept if row.get("reviewed")]
    print(f"已审核 {len(reviewed)} 条，其中不可回答 {sum(1 for row in reviewed if not row.get('answerable'))} 条、"
          f"多文档 {sum(1 for row in reviewed if len({e['doc_key'] for e in row.get('evidence') or []}) >= 2)} 条")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="检索评估集审核：模型筛查 + Claude 复核")
    sub = parser.add_subparsers(dest="command", required=True)

    screen = sub.add_parser("screen")
    screen.add_argument("--ids", default="")
    screen.add_argument("--model", default="qwen3.7-plus")
    screen.set_defaults(func=cmd_screen)

    report = sub.add_parser("report")
    report.add_argument("--show", choices=("flagged", "all", "none"), default="flagged")
    report.set_defaults(func=cmd_report)

    apply = sub.add_parser("apply")
    apply.add_argument("--decisions", type=Path, default=PROCESSED / "qa_decisions.jsonl")
    apply.set_defaults(func=cmd_apply)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
