"""数据校验：按 docs/data_format.md 检查 M1 产出的所有 JSONL。

用法：
    python scripts/validate_data.py --stage processed|eval|all [--allow-unreviewed] [--data-dir data]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

from xiaomao.textnorm import normalize_for_match

sys.stdout.reconfigure(encoding="utf-8")  # Windows 控制台默认 GBK

ROOT_DIR = Path(__file__).resolve().parents[1]
MAX_ERRORS_SHOWN = 10
MIN_QUOTE, MAX_QUOTE = 10, 200

COLLECTIONS = {"xm_platform_policy", "xm_compliance", "xm_faq"}
PLATFORMS = {"amazon", "temu", "tiktok_shop", "shopee", None}
DOC_TYPES = {"policy", "guide", "regulation", "announcement", "faq"}
LANGS = {"zh", "en", "de", "ja", "fr", "es", "it"}
INTENTS = {
    "platform_policy", "compliance", "logistics", "tax", "ops", "account", "chitchat", "out_of_scope",
}
STRATEGIES = {"direct", "subquery", "hyde", None}
SOURCES = {"handwritten", "llm_generated"}
REVIEWERS = {"human", "claude", None}
SPOT_CHECK_MARK = "[人工抽检通过]"  # 和 scripts/review_sheet.py 的 FAQ_CHECKED 一致

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
SITE_RE = re.compile(r"^[A-Z]{2}$")
CHUNK_ID_RE = re.compile(r"^[0-9a-f]{16}-\d{4}$")
DOC_KEY_RE = re.compile(r"^[0-9a-f]{16}$")
HASH_RE = re.compile(r"^[0-9a-f]{64}$")

FIELDS = {
    "manifest": [
        "doc_key", "collection", "source_url", "title", "platform", "site", "doc_type", "lang",
        "category", "effective_date", "expire_date", "fetched_at", "raw_path", "content_type",
        "http_status",
    ],
    "faq_items": [
        "faq_id", "question", "answer", "intent", "platform", "site", "category", "lang",
        "effective_date", "evidence", "source", "reviewed", "reviewed_by", "note",
    ],
    "documents": [
        "doc_key", "collection", "source_url", "title", "platform", "site", "doc_type", "lang",
        "category", "effective_date", "expire_date", "content_hash", "fetched_at", "text",
        "question", "answer",
    ],
    "chunks": [
        "chunk_id", "doc_key", "chunk_index", "collection", "platform", "site", "category",
        "doc_type", "lang", "effective_date", "expire_date", "source_url", "title_path", "text",
        "question", "answer", "token_count",
    ],
    "intent": [
        "id", "query", "intent", "strategy", "platform", "site", "source", "reviewed",
        "reviewed_by", "note",
    ],
    "retrieval_qa": [
        "id", "query", "intent", "platform", "site", "answerable", "evidence",
        "reference_answer", "paraphrases", "source", "reviewed", "reviewed_by", "note",
    ],
}

# 允许存在但不在必需清单里的可选字段
OPTIONAL_FIELDS = {"faq_items": {"generated_at"}}

# 数量下限（--allow-unreviewed 时跳过）
THRESHOLDS = {
    "documents_policy_min": 70,  # 2026-10-29 从 80 降到 70：平台页只能手动保存，低价值页面不凑数
    "documents_compliance_min": 40,
    "faq_reviewed_min": 300,
    "intent_human_min": 300,
    "intent_per_class_min": 25,
    "intent_chitchat_min": 20,
    "intent_out_of_scope_min": 20,
    "strategy_subquery_min": 40,
    "strategy_hyde_min": 40,
    "qa_reviewed_min": 150,
    # 检索集由 Claude 审的部分，用户至少抽检这个比例（note 里带 SPOT_CHECK_MARK）
    "qa_spot_check_ratio": 0.10,
    "qa_unanswerable_ratio": (0.10, 0.20),
    # 2026-10-30 下限 15% → 12%：审核删掉 6 道证据有问题的多文档题后剩 24 条（14.3%），差 1–2 条不值得再生成一轮
    "qa_multi_evidence_ratio": (0.12, 0.25),
}


class Reporter:
    """收集错误和警告，按文件分组输出。"""

    def __init__(self) -> None:
        self.errors: list[tuple[str, int, str]] = []
        self.warnings: list[tuple[str, int, str]] = []
        self.files: list[str] = []

    def error(self, file: str, line: int, message: str) -> None:
        self.errors.append((file, line, message))

    def warn(self, file: str, line: int, message: str) -> None:
        self.warnings.append((file, line, message))

    def report(self) -> int:
        for file in self.files:
            file_errors = [item for item in self.errors if item[0] == file]
            file_warnings = [item for item in self.warnings if item[0] == file]
            print(f"\n[{file}] 错误 {len(file_errors)}，警告 {len(file_warnings)}")
            for _, line, message in file_errors[:MAX_ERRORS_SHOWN]:
                print(f"  第 {line} 行: {message}")
            for _, line, message in file_warnings[:MAX_ERRORS_SHOWN]:
                print(f"  第 {line} 行（警告）: {message}")
        if self.errors:
            print(f"\n校验失败：{len(self.errors)} 个错误")
            return 1
        print(f"\n校验通过（警告 {len(self.warnings)} 条）")
        return 0


def read_jsonl(path: Path, reporter: Reporter, name: str, required: bool = True) -> list[tuple[int, dict]]:
    """读 JSONL；缺文件按 required 记错误或警告，坏行记错误并跳过。"""
    rows: list[tuple[int, dict]] = []
    reporter.files.append(name)
    if not path.is_file():
        (reporter.error if required else reporter.warn)(name, 0, f"文件不存在：{path}")
        return rows
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            reporter.error(name, number, f"不是合法 JSON：{exc.msg}")
            continue
        if not isinstance(item, dict):
            reporter.error(name, number, "每一行必须是 JSON 对象")
            continue
        rows.append((number, item))
    return rows


def is_date(value: object) -> bool:
    """日期是 YYYY-MM-DD 或 null。"""
    return value is None or (isinstance(value, str) and bool(DATE_RE.match(value)))


def check_common(name: str, line: int, item: dict, reporter: Reporter) -> None:
    """检查字段齐全、无多余字段、枚举和日期格式。"""
    required = set(FIELDS[name])
    missing = required - set(item)
    if missing:
        reporter.error(name, line, f"缺少字段：{', '.join(sorted(missing))}")
    allowed = required | OPTIONAL_FIELDS.get(name, set())
    extra = set(item) - allowed
    if extra:
        reporter.warn(name, line, f"约定以外的字段：{', '.join(sorted(extra))}")

    if "collection" in item and item["collection"] not in COLLECTIONS:
        reporter.error(name, line, f"collection 取值非法：{item['collection']!r}")
    if "platform" in item and item["platform"] not in PLATFORMS:
        reporter.error(name, line, f"platform 取值非法：{item['platform']!r}")
    if "site" in item:
        site = item["site"]
        if site is not None and not (SITE_RE.match(str(site)) or site == "EU"):
            reporter.error(name, line, f"site 取值非法：{site!r}")
    if "doc_type" in item and item["doc_type"] not in DOC_TYPES:
        reporter.error(name, line, f"doc_type 取值非法：{item['doc_type']!r}")
    if "lang" in item and item["lang"] not in LANGS:
        reporter.error(name, line, f"lang 取值非法：{item['lang']!r}")
    if "intent" in item and item["intent"] not in INTENTS:
        reporter.error(name, line, f"intent 取值非法：{item['intent']!r}")
    if "strategy" in item and item["strategy"] not in STRATEGIES:
        reporter.error(name, line, f"strategy 取值非法：{item['strategy']!r}")
    if "source" in item and item["source"] not in SOURCES:
        reporter.error(name, line, f"source 取值非法：{item['source']!r}")
    if "reviewed_by" in item and item["reviewed_by"] not in REVIEWERS:
        reporter.error(name, line, f"reviewed_by 取值非法：{item['reviewed_by']!r}")
    for field in ("effective_date", "expire_date"):
        if field in item and not is_date(item[field]):
            reporter.error(name, line, f"{field} 不是 YYYY-MM-DD 或 null：{item[field]!r}")
    if item.get("reviewed") is True and item.get("reviewed_by") is None:
        reporter.error(name, line, "reviewed=true 但 reviewed_by 是 null")


def check_evidence(
    name: str,
    line: int,
    item: dict,
    documents: dict[str, dict],
    reporter: Reporter,
) -> None:
    """evidence 里的 doc_key 必须存在，quote 必须能在原文里原样找到。"""
    evidence = item.get("evidence")
    if not isinstance(evidence, list):
        reporter.error(name, line, "evidence 必须是数组")
        return
    for entry in evidence:
        if not isinstance(entry, dict) or "doc_key" not in entry or "quote" not in entry:
            reporter.error(name, line, "evidence 元素必须含 doc_key 和 quote")
            continue
        document = documents.get(entry["doc_key"])
        if document is None:
            reporter.error(name, line, f"evidence 引用了不存在的 doc_key：{entry['doc_key']}")
            continue
        raw_quote = str(entry["quote"])
        if not MIN_QUOTE <= len(raw_quote) <= MAX_QUOTE:
            reporter.error(name, line, f"quote 长度必须 {MIN_QUOTE}–{MAX_QUOTE} 个字符：{len(raw_quote)}")
            continue
        quote = normalize_for_match(raw_quote)
        body = normalize_for_match(str(document.get("text") or ""))
        if quote and quote not in body:
            reporter.error(name, line, f"quote 在原文中找不到：{raw_quote[:40]!r}")


def check_quotes_in_chunks(
    rows: list[tuple[int, dict]], chunk_index: dict[str, list[str]], reporter: Reporter
) -> None:
    """retrieval_qa 的 quote 必须完整落在同一篇文档的某一个片段里。"""
    if not chunk_index:
        reporter.warn("retrieval_qa", 0, "没有 chunks.jsonl，跳过“证据落在单个片段”检查")
        return
    for line, item in rows:
        if item.get("answerable") is not True:
            continue
        for entry in item.get("evidence") or []:
            if not isinstance(entry, dict):
                continue
            doc_key = entry.get("doc_key")
            quote = normalize_for_match(str(entry.get("quote") or ""))
            if not quote or doc_key not in chunk_index:
                continue
            if not any(quote in normalize_for_match(text) for text in chunk_index[doc_key]):
                reporter.error(
                    "retrieval_qa", line, "quote 没有完整落在任何单个片段里（Hit@K 不可能命中）"
                )


def check_documents(rows: list[tuple[int, dict]], reporter: Reporter) -> dict[str, dict]:
    """documents 自身的检查，返回 doc_key -> 文档。"""
    by_key: dict[str, dict] = {}
    for line, item in rows:
        check_common("documents", line, item, reporter)
        doc_key = item.get("doc_key")
        if isinstance(doc_key, str):
            if not DOC_KEY_RE.match(doc_key):
                reporter.error("documents", line, f"doc_key 格式非法：{doc_key!r}")
            expected = hashlib.sha256(
                f"{item.get('collection')}|{item.get('source_url')}".encode("utf-8")
            ).hexdigest()[:16]
            if doc_key != expected:
                reporter.error("documents", line, "doc_key 与 collection|source_url 算出的值不一致")
            if doc_key in by_key:
                reporter.error("documents", line, f"doc_key 重复：{doc_key}")
            by_key[doc_key] = item
        text = str(item.get("text") or "")
        content_hash = item.get("content_hash")
        if isinstance(content_hash, str):
            if not HASH_RE.match(content_hash):
                reporter.error("documents", line, "content_hash 不是 64 位十六进制")
            elif content_hash != hashlib.sha256(text.encode("utf-8")).hexdigest():
                reporter.error("documents", line, "content_hash 与 text 不一致")
        if item.get("collection") == "xm_faq":
            question, answer = item.get("question"), item.get("answer")
            if not question or not answer:
                reporter.error("documents", line, "FAQ 文档缺少 question 或 answer")
            elif text != f"{question}\n{answer}":
                reporter.error("documents", line, "FAQ 文档的 text 不等于 question + 换行 + answer")
    return by_key


def check_chunks(
    rows: list[tuple[int, dict]], documents: dict[str, dict], reporter: Reporter
) -> None:
    """chunks 自身的检查，以及与 documents 的一致性。"""
    seen: set[str] = set()
    per_doc: dict[str, list[int]] = {}
    for line, item in rows:
        check_common("chunks", line, item, reporter)
        chunk_id = item.get("chunk_id")
        if isinstance(chunk_id, str):
            if not CHUNK_ID_RE.match(chunk_id):
                reporter.error("chunks", line, f"chunk_id 格式非法：{chunk_id!r}")
            if chunk_id in seen:
                reporter.error("chunks", line, f"chunk_id 重复：{chunk_id}")
            seen.add(chunk_id)
        doc_key = item.get("doc_key")
        document = documents.get(doc_key)
        if document is None:
            reporter.error("chunks", line, f"doc_key 在 documents 里找不到：{doc_key}")
        else:
            for field in ("collection", "platform", "site", "category", "doc_type", "lang",
                          "effective_date", "expire_date", "source_url"):
                if item.get(field) != document.get(field):
                    reporter.error("chunks", line, f"{field} 与所属文档不一致")
        index = item.get("chunk_index")
        if isinstance(index, int):
            per_doc.setdefault(str(doc_key), []).append(index)
        token_count = item.get("token_count")
        if not isinstance(token_count, int):
            reporter.error("chunks", line, "token_count 必须是整数")
        elif token_count > 800:
            reporter.error("chunks", line, f"token_count 超过 800：{token_count}")
        if len(str(item.get("text") or "")) > 1500:
            reporter.error("chunks", line, "text 超过 1500 字符")

    for doc_key, indexes in per_doc.items():
        if sorted(indexes) != list(range(len(indexes))):
            reporter.error("chunks", 0, f"doc_key {doc_key} 的 chunk_index 不是从 0 开始连续：{sorted(indexes)}")

    for doc_key, document in documents.items():
        if document.get("collection") != "xm_faq" and doc_key not in per_doc:
            reporter.error("chunks", 0, f"非 FAQ 文档没有任何片段：{doc_key}")


def check_eval_common(name: str, rows: list[tuple[int, dict]], allow_unreviewed: bool,
                      reporter: Reporter, require_strategy: bool = True) -> None:
    """评估集：必须有 intent；intent.jsonl 还要检查 strategy，未审核时按参数决定是否报错。"""
    for line, item in rows:
        check_common(name, line, item, reporter)
        if "intent" in item and item["intent"] not in INTENTS:
            continue
        if require_strategy:
            if item.get("intent") in {"chitchat", "out_of_scope"}:
                if item.get("strategy") is not None:
                    reporter.error(name, line, "chitchat / out_of_scope 的 strategy 必须是 null")
            elif item.get("strategy") is None:
                reporter.error(name, line, "非闲聊/超范围的问题必须给 strategy")
        if allow_unreviewed:
            continue
        # 意图集必须人审；检索集允许用户委托 Claude 审（再由用户抽检，见 _check_eval_counts）
        allowed = {"human"} if name == "intent" else {"human", "claude"}
        if item.get("reviewed") is not True or item.get("reviewed_by") not in allowed:
            reporter.error(
                name, line,
                f"评估集必须 reviewed=true 且 reviewed_by ∈ {sorted(allowed)}，当前是 {item.get('reviewed_by')!r}",
            )


def check_qa(rows: list[tuple[int, dict]], documents: dict[str, dict], reporter: Reporter) -> None:
    """retrieval_qa 的 answerable / evidence / paraphrases 一致性。"""
    for line, item in rows:
        check_evidence("retrieval_qa", line, item, documents, reporter)
        evidence = item.get("evidence") or []
        if item.get("answerable") is True:
            if not evidence:
                reporter.error("retrieval_qa", line, "answerable=true 但 evidence 为空")
            if not item.get("reference_answer"):
                reporter.error("retrieval_qa", line, "answerable=true 但 reference_answer 为空")
        elif item.get("answerable") is False:
            if evidence:
                reporter.error("retrieval_qa", line, "answerable=false 但 evidence 非空")
            if item.get("reference_answer") is not None:
                reporter.error("retrieval_qa", line, "answerable=false 但 reference_answer 非 null")
        else:
            reporter.error("retrieval_qa", line, "answerable 必须是布尔值")
        if len(item.get("paraphrases") or []) < 2:
            reporter.error("retrieval_qa", line, "paraphrases 至少 2 条")


def check_counts(
    documents: list[tuple[int, dict]],
    faq_items: list[tuple[int, dict]],
    intents: list[tuple[int, dict]],
    qa_rows: list[tuple[int, dict]],
    allow_unreviewed: bool,
    reporter: Reporter,
    want_processed: bool = True,
    want_eval: bool = True,
) -> None:
    """数量达标检查；--allow-unreviewed 时整体跳过，只检查本次 --stage 读入的文件。"""
    if allow_unreviewed:
        return
    if want_processed:
        _check_processed_counts(documents, faq_items, reporter)
    if want_eval:
        _check_eval_counts(intents, qa_rows, reporter)


def _check_processed_counts(
    documents: list[tuple[int, dict]], faq_items: list[tuple[int, dict]], reporter: Reporter
) -> None:
    """文档和 FAQ 的数量。"""
    policy = sum(1 for _, item in documents if item.get("collection") == "xm_platform_policy")
    compliance = sum(1 for _, item in documents if item.get("collection") == "xm_compliance")
    if policy < THRESHOLDS["documents_policy_min"]:
        reporter.error("documents", 0, f"xm_platform_policy 只有 {policy} 篇，要求 ≥ {THRESHOLDS['documents_policy_min']}")
    if compliance < THRESHOLDS["documents_compliance_min"]:
        reporter.error("documents", 0, f"xm_compliance 只有 {compliance} 篇，要求 ≥ {THRESHOLDS['documents_compliance_min']}")

    reviewed_faq = sum(1 for _, item in faq_items if item.get("reviewed") is True)
    if reviewed_faq < THRESHOLDS["faq_reviewed_min"]:
        reporter.error("faq_items", 0, f"reviewed=true 的 FAQ 只有 {reviewed_faq} 条，要求 ≥ {THRESHOLDS['faq_reviewed_min']}")


def _check_eval_counts(
    intents: list[tuple[int, dict]], qa_rows: list[tuple[int, dict]], reporter: Reporter
) -> None:
    """评估集的人审数量和比例。"""
    human = [item for _, item in intents if item.get("reviewed_by") == "human"]
    if len(human) < THRESHOLDS["intent_human_min"]:
        reporter.error("intent", 0, f"reviewed_by=human 只有 {len(human)} 条，要求 ≥ {THRESHOLDS['intent_human_min']}")
    for intent in INTENTS:
        minimum = THRESHOLDS["intent_per_class_min"]
        if intent == "chitchat":
            minimum = THRESHOLDS["intent_chitchat_min"]
        elif intent == "out_of_scope":
            minimum = THRESHOLDS["intent_out_of_scope_min"]
        count = sum(1 for item in human if item.get("intent") == intent)
        if count < minimum:
            reporter.error("intent", 0, f"意图 {intent} 只有 {count} 条，要求 ≥ {minimum}")
    for strategy in ("subquery", "hyde"):
        count = sum(1 for item in human if item.get("strategy") == strategy)
        if count < THRESHOLDS[f"strategy_{strategy}_min"]:
            reporter.error("intent", 0, f"策略 {strategy} 只有 {count} 条，要求 ≥ {THRESHOLDS[f'strategy_{strategy}_min']}")

    qa_human = [
        item for _, item in qa_rows
        if item.get("reviewed") is True and item.get("reviewed_by") in ("human", "claude")
    ]
    if len(qa_human) < THRESHOLDS["qa_reviewed_min"]:
        reporter.error("retrieval_qa", 0, f"审核通过只有 {len(qa_human)} 条，要求 ≥ {THRESHOLDS['qa_reviewed_min']}")
    by_claude = [item for item in qa_human if item.get("reviewed_by") == "claude"]
    checked = sum(1 for item in by_claude if SPOT_CHECK_MARK in (item.get("note") or ""))
    need = math.ceil(len(by_claude) * THRESHOLDS["qa_spot_check_ratio"])
    if checked < need:
        reporter.error("retrieval_qa", 0, f"Claude 审核的 {len(by_claude)} 条里人工抽检通过 {checked} 条，要求 ≥ {need} 条")
    if qa_human:
        unanswerable = sum(1 for item in qa_human if item.get("answerable") is False) / len(qa_human)
        low, high = THRESHOLDS["qa_unanswerable_ratio"]
        if not low <= unanswerable <= high:
            reporter.error("retrieval_qa", 0, f"answerable=false 占比 {unanswerable:.1%}，应在 {low:.0%}–{high:.0%}")
        multi = sum(1 for item in qa_human if len(item.get("evidence") or []) >= 2) / len(qa_human)
        low, high = THRESHOLDS["qa_multi_evidence_ratio"]
        if not low <= multi <= high:
            reporter.error("retrieval_qa", 0, f"多文档证据占比 {multi:.1%}，应在 {low:.0%}–{high:.0%}")


def main(argv: list[str] | None = None) -> int:
    """按 --stage 检查数据文件，返回退出码。"""
    parser = argparse.ArgumentParser(description="M1 数据校验")
    parser.add_argument("--stage", choices=["processed", "eval", "all"], default="all")
    parser.add_argument("--allow-unreviewed", action="store_true", help="开发期间跳过人工审核和数量检查")
    parser.add_argument("--data-dir", type=Path, default=ROOT_DIR / "data")
    args = parser.parse_args(argv)

    reporter = Reporter()
    data_dir = args.data_dir
    want_processed = args.stage in ("processed", "all")
    want_eval = args.stage in ("eval", "all")

    documents: list[tuple[int, dict]] = []
    faq_items: list[tuple[int, dict]] = []
    intents: list[tuple[int, dict]] = []
    qa_rows: list[tuple[int, dict]] = []

    if want_processed:
        read_jsonl(data_dir / "raw" / "manifest.jsonl", reporter, "manifest")
        faq_items = read_jsonl(
            data_dir / "processed" / "faq_items.jsonl", reporter, "faq_items",
            required=not args.allow_unreviewed,  # 开发期间 FAQ 可能还没生成
        )
        documents = read_jsonl(data_dir / "processed" / "documents.jsonl", reporter, "documents")
        chunks = read_jsonl(data_dir / "processed" / "chunks.jsonl", reporter, "chunks")
        doc_index = check_documents(documents, reporter)
        check_chunks(chunks, doc_index, reporter)
        for line, item in faq_items:
            check_common("faq_items", line, item, reporter)
            check_evidence("faq_items", line, item, doc_index, reporter)
    else:
        # 只校验评估集时，仍然需要 documents 的全文和 chunks 来做 quote 匹配
        documents = read_jsonl(data_dir / "processed" / "documents.jsonl", reporter, "documents")
        doc_index = {item["doc_key"]: item for _, item in documents if "doc_key" in item}
        chunks = read_jsonl(data_dir / "processed" / "chunks.jsonl", reporter, "chunks", required=False)

    chunk_index: dict[str, list[str]] = {}
    for _, item in chunks:
        if "doc_key" in item and "text" in item:
            chunk_index.setdefault(item["doc_key"], []).append(str(item["text"]))

    if want_eval:
        intents = read_jsonl(data_dir / "eval" / "intent.jsonl", reporter, "intent")
        qa_rows = read_jsonl(data_dir / "eval" / "retrieval_qa.jsonl", reporter, "retrieval_qa")
        check_eval_common("intent", intents, args.allow_unreviewed, reporter)
        check_eval_common("retrieval_qa", qa_rows, args.allow_unreviewed, reporter, require_strategy=False)
        check_qa(qa_rows, doc_index, reporter)
        check_quotes_in_chunks(qa_rows, chunk_index, reporter)

    check_counts(
        documents, faq_items, intents, qa_rows, args.allow_unreviewed, reporter,
        want_processed=want_processed, want_eval=want_eval,
    )
    return reporter.report()


if __name__ == "__main__":
    sys.exit(main())
