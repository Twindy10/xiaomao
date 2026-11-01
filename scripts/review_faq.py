"""C4 FAQ 初审：模型筛查 + Claude 复核。

流程：
    1. screen：用比生成模型更强的模型（默认 qwen3.7-plus）对照原文逐条判断，
       结果写进 data/processed/faq_review.jsonl，可断点续跑
    2. report：汇总，并列出需要复核的条目
    3. apply：读入 Claude 复核后的决定，写回 faq_items.jsonl 的 reviewed / reviewed_by / note

用法：
    uv run python scripts/review_faq.py screen [--ids faq-0001,faq-0002] [--limit N] [--model qwen3.7-plus]
    uv run python scripts/review_faq.py report [--show all|flagged]
    uv run python scripts/review_faq.py apply --decisions data/processed/faq_decisions.jsonl

决定文件一行一条：{"faq_id": "...", "action": "keep|edit|drop", "question"?: "...",
"answer"?: "...", "intent"?: "...", "reason"?: "..."}。没写进决定文件、并且筛查通过的条目按 keep 处理。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from xiaomao.llm import LLMClient
from xiaomao.textnorm import normalize_for_match

sys.stdout.reconfigure(encoding="utf-8")  # Windows 控制台默认 GBK

ROOT_DIR = Path(__file__).resolve().parents[1]
PROCESSED = ROOT_DIR / "data" / "processed"
FAQ_PATH = PROCESSED / "faq_items.jsonl"
REVIEW_PATH = PROCESSED / "faq_review.jsonl"

INTENTS = ("platform_policy", "compliance", "logistics", "tax", "ops", "account", "chitchat", "out_of_scope")
FULL_DOC_LIMIT = 12000  # 不超过这个长度的文档整篇交给审核模型（129 篇里 118 篇）
CONTEXT_LIMIT = 8000  # 更长的文档只给引文前后各 2 块，最多这么多字符
FAITHFUL = ("yes", "partial", "no")

SYSTEM_PROMPT = (
    "你是一名严格的知识库审核员，负责审核给中国跨境电商卖家用的 FAQ。"
    "你只根据给出的原文判断，不使用自己的知识补全。始终输出合法的 JSON 对象。"
)

PROMPT_TEMPLATE = """请对照原文审核下面这条 FAQ。

## 原文（文档：{title}；标题路径：{title_path}）
{context}

## FAQ
问题：{question}
答案：{answer}
引文：{quote}
当前意图标注：{intent}

## 审核项
1. faithful：答案里的每一个事实（数字、日期、条件、机构名、步骤）是否都能在原文里找到依据。
   - "yes"：全部有依据；"partial"：大部分有依据，但有个别说法原文没有或说得更绝对；"no"：主要内容原文没有或与原文矛盾。
   - 把原文找不到依据的说法逐条列进 unsupported（原样摘出答案里的那句话），没有就给空列表。
2. answers_question：答案是否真正回答了问题（不是答非所问、也不是只答了一小部分）。
3. standalone_question：把问题改写成不看原文也能看懂的版本：把"这个功能""该计划""它"这类指代换成具体对象，
   缺平台、国家、法规名的补上。保持原来的口语风格，不要改问的点。已经能看懂的原样返回。
4. intent：按下面的定义重新判断这条 FAQ 的意图，只能是其中一个英文键：
   platform_policy（平台规则、政策、违规处罚）、compliance（产品合规、认证、法规、EPR）、logistics（物流、仓储、配送）、
   tax（税务、VAT、关税）、ops（运营、推广、选品、开店福利、工具使用）、account（开店注册、账户、申诉）、
   chitchat（闲聊、行业前景、和具体操作无关的泛泛之谈）、out_of_scope（与跨境电商无关）。
5. comment：一句话说明主要问题；没有问题写"无"。

只输出 JSON：{{"faithful": "yes|partial|no", "unsupported": [], "answers_question": true,
"standalone_question": "...", "intent": "...", "comment": "..."}}"""

# 问题能否独立看懂要单独判断：审核模型看着原文时知道"这个"指什么，会一律判成能看懂
QUESTION_PROMPT = """下面是一位跨境电商卖家在知识库搜索框里输入的一句话。你没有看过任何上下文。

问题：{question}

判断 self_contained：只看这句话，能否明确知道他问的是哪个平台功能、哪项法规、哪个服务或哪个对象？
- 用"这个功能""这个码""该计划""它""这个标准"指代，而这句话本身没说清是什么的，算 false。
- 句子里已经点名了对象（如"AWD这个仓怎么收费""英国卖化妆品要找负责人吗"），算 true。
- 只是没提平台或国家、但问的对象本身明确的（如"FBA 有啥重量限制"），算 true。

只输出 JSON：{{"self_contained": true, "missing": "缺的是什么对象；能看懂就写空字符串"}}"""


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


def build_context(faq: dict, chunks_by_doc: dict[str, list[dict]], docs: dict[str, dict]) -> tuple[str, str]:
    """短文档给全文；长文档找到引文所在的片段，带上前后各 2 块。

    只给引文附近几块时，答案里出自同一篇其他位置的说法会被误判成"原文没有"。
    """
    evidence = faq["evidence"][0]
    doc_key = evidence["doc_key"]
    text = docs[doc_key]["text"]
    if len(text) <= FULL_DOC_LIMIT:
        return text, "（全文）"
    quote = normalize_for_match(evidence["quote"])
    chunks = chunks_by_doc.get(doc_key, [])
    for index, chunk in enumerate(chunks):
        if quote in normalize_for_match(chunk["text"]):
            window = chunks[max(0, index - 2) : index + 3]
            context = "\n\n".join(item["text"] for item in window)
            return context[:CONTEXT_LIMIT], chunk["title_path"]
    return docs[doc_key]["text"][:CONTEXT_LIMIT], "（引文不在单个片段里，给出文档开头）"


def judge(client: LLMClient, faq: dict, context: str, title: str, title_path: str) -> dict:
    """让审核模型判断一条 FAQ，并校验返回结构。"""
    prompt = PROMPT_TEMPLATE.format(
        title=title,
        title_path=title_path,
        context=context,
        question=faq["question"],
        answer=faq["answer"],
        quote=faq["evidence"][0]["quote"],
        intent=faq["intent"],
    )
    result = client.chat_json(prompt, SYSTEM_PROMPT, max_tokens=800, temperature=0)
    if result.get("faithful") not in FAITHFUL or result.get("intent") not in INTENTS:
        raise ValueError(f"审核结果结构不对：{result}")
    standalone = client.chat_json(
        QUESTION_PROMPT.format(question=faq["question"]), SYSTEM_PROMPT, max_tokens=200, temperature=0
    )
    self_contained = bool(standalone.get("self_contained", True))
    rewritten = str(result.get("standalone_question") or "").strip()
    return {
        "faithful": result["faithful"],
        "unsupported": [str(item) for item in result.get("unsupported") or []],
        "answers_question": bool(result.get("answers_question")),
        "self_contained": self_contained,
        "missing": str(standalone.get("missing") or ""),
        # 只有判成看不懂时才采用改写，避免模型顺手改动本来就好的问题
        "suggested_question": rewritten if not self_contained and rewritten != faq["question"] else None,
        "intent": result["intent"],
        "comment": str(result.get("comment") or ""),
    }


def needs_attention(faq: dict, review: dict) -> list[str]:
    """筛查结果里需要 Claude 复核的原因；空列表表示直接通过。"""
    reasons = []
    if review["faithful"] != "yes":
        reasons.append(f"faithful={review['faithful']}")
    if not review["answers_question"]:
        reasons.append("答非所问")
    if not review["self_contained"]:
        reasons.append("问题缺主语")
    if review["intent"] != faq["intent"]:
        reasons.append(f"意图 {faq['intent']}→{review['intent']}")
    return reasons


def cmd_screen(args: argparse.Namespace) -> int:
    """逐条筛查，已经筛过的跳过。"""
    faqs = _read_jsonl(FAQ_PATH)
    docs = {row["doc_key"]: row for row in _read_jsonl(PROCESSED / "documents.jsonl")}
    chunks_by_doc: dict[str, list[dict]] = {}
    for chunk in _read_jsonl(PROCESSED / "chunks.jsonl"):
        chunks_by_doc.setdefault(chunk["doc_key"], []).append(chunk)

    done = {row["faq_id"] for row in _read_jsonl(REVIEW_PATH)}
    todo = [faq for faq in faqs if faq["faq_id"] not in done]
    if args.ids:
        wanted = set(args.ids.split(","))
        todo = [faq for faq in todo if faq["faq_id"] in wanted]
    if args.limit:
        todo = todo[: args.limit]
    print(f"FAQ {len(faqs)} 条，已筛查 {len(done)} 条，本次 {len(todo)} 条，审核模型 {args.model}")

    client = LLMClient()
    client.model = args.model
    errors: Counter[str] = Counter()

    def work(faq: dict) -> dict | None:
        doc = docs[faq["evidence"][0]["doc_key"]]
        context, title_path = build_context(faq, chunks_by_doc, docs)
        try:
            review = judge(client, faq, context, doc["title"], title_path)
        except Exception as exc:  # 单条失败不中断，下次续跑会重试
            errors[type(exc).__name__] += 1
            return None
        return {"faq_id": faq["faq_id"], "model": args.model, **review}

    # 每 20 条落一次盘，中断后不丢太多
    with REVIEW_PATH.open("a", encoding="utf-8", newline="\n") as handle:
        for start in range(0, len(todo), 20):
            for row in client.map(work, todo[start : start + 20]):
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"  已完成 {min(start + 20, len(todo))}/{len(todo)}")

    if errors:
        print("失败（下次续跑会重试）：" + "，".join(f"{key}={value}" for key, value in errors.items()))
    print(client.usage.summary())
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """汇总筛查结果，列出需要复核的条目。"""
    faqs = {row["faq_id"]: row for row in _read_jsonl(FAQ_PATH)}
    reviews = {row["faq_id"]: row for row in _read_jsonl(REVIEW_PATH)}
    print(f"FAQ {len(faqs)} 条，已筛查 {len(reviews)} 条")
    print("faithful：" + "，".join(f"{k}={v}" for k, v in Counter(r["faithful"] for r in reviews.values()).items()))
    print(f"答非所问 {sum(not r['answers_question'] for r in reviews.values())}，"
          f"问题缺主语 {sum(not r['self_contained'] for r in reviews.values())}，"
          f"意图改动 {sum(r['intent'] != faqs[i]['intent'] for i, r in reviews.items())}")

    flagged = 0
    for faq_id, review in sorted(reviews.items()):
        faq = faqs[faq_id]
        reasons = needs_attention(faq, review)
        if reasons:
            flagged += 1
        if args.show == "all" or (args.show == "flagged" and reasons):
            print(f"\n--- {faq_id} [{'；'.join(reasons) or '通过'}]")
            print(f"Q: {faq['question']}")
            print(f"A: {faq['answer']}")
            print(f"E: {faq['evidence'][0]['quote'][:200]}")
            for claim in review["unsupported"]:
                print(f"  ✗ {claim}")
            if review["suggested_question"]:
                print(f"  → 建议问题：{review['suggested_question']}")
            print(f"  审核意见：{review['comment']}")
    print(f"\n需要复核 {flagged} 条，直接通过 {len(reviews) - flagged} 条")
    return 0


def cmd_apply(args: argparse.Namespace) -> int:
    """把复核决定写回 faq_items.jsonl。"""
    faqs = _read_jsonl(FAQ_PATH)
    reviews = {row["faq_id"]: row for row in _read_jsonl(REVIEW_PATH)}
    decisions = {row["faq_id"]: row for row in _read_jsonl(args.decisions)}

    stats: Counter[str] = Counter()
    for faq in faqs:
        review = reviews.get(faq["faq_id"])
        decision = decisions.get(faq["faq_id"])
        if decision is None:
            if review is None or needs_attention(faq, review):
                stats["未决定，保持未审核"] += 1  # 没筛过、或者筛出问题但还没复核的不放行
                continue
            decision = {"action": "keep"}

        action = decision["action"]
        if action == "drop":
            faq["reviewed"], faq["reviewed_by"] = False, None
            faq["note"] = f"Claude 初审淘汰：{decision.get('reason') or '未注明'}"
        elif action in ("keep", "edit"):
            for field in ("question", "answer", "intent"):
                if decision.get(field):
                    faq[field] = decision[field]
            if decision.get("intent") and decision["intent"] not in INTENTS:
                raise ValueError(f"{faq['faq_id']} 的意图不合法：{decision['intent']}")
            faq["reviewed"], faq["reviewed_by"] = True, "claude"
            if action == "edit":
                faq["note"] = f"Claude 初审修改：{decision.get('reason') or '见历史'}"
        else:
            raise ValueError(f"{faq['faq_id']} 的 action 不合法：{action}")
        stats[action] += 1

    _write_jsonl(FAQ_PATH, faqs)
    print("写回 " + "，".join(f"{key} {value}" for key, value in stats.items()))
    print(f"reviewed=true 共 {sum(1 for faq in faqs if faq.get('reviewed'))} 条")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="C4 FAQ 初审：模型筛查 + Claude 复核")
    sub = parser.add_subparsers(dest="command", required=True)

    screen = sub.add_parser("screen", help="用审核模型逐条筛查")
    screen.add_argument("--ids", default="", help="只筛这几条，逗号分隔")
    screen.add_argument("--limit", type=int, default=0)
    screen.add_argument("--model", default="qwen3.7-plus", help="审核模型，默认比生成用的 flash 更强")
    screen.set_defaults(func=cmd_screen)

    report = sub.add_parser("report", help="汇总并列出需要复核的条目")
    report.add_argument("--show", choices=("flagged", "all", "none"), default="flagged")
    report.set_defaults(func=cmd_report)

    apply = sub.add_parser("apply", help="把复核决定写回 faq_items.jsonl")
    apply.add_argument("--decisions", type=Path, default=PROCESSED / "faq_decisions.jsonl")
    apply.set_defaults(func=cmd_apply)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
