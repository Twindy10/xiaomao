"""评估集生成：意图标注集和检索问答集，产出供人工审核的候选数据。

用法：
    python scripts/gen_eval.py intent --out data/eval/intent.jsonl [--scale 1.3] [--limit N] [--dry-run]
    python scripts/gen_eval.py qa --docs ... --chunks ... --faq ... --out data/eval/retrieval_qa.jsonl \
        [--n 200] [--limit N] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from pathlib import Path

from xiaomao.textnorm import normalize_for_match

sys.stdout.reconfigure(encoding="utf-8")  # Windows 控制台默认 GBK

ROOT_DIR = Path(__file__).resolve().parents[1]

INTENTS = {
    "platform_policy", "compliance", "logistics", "tax", "ops", "account", "chitchat", "out_of_scope",
}
INTENT_LABELS = {
    "platform_policy": "平台规则（入驻、店铺政策、平台收费）",
    "compliance": "合规认证（产品认证、法规、标签）",
    "logistics": "物流（发货、头程、仓配、FBA 入仓）",
    "tax": "税务（VAT、关税、申报）",
    "ops": "运营技巧（listing、广告、选品）",
    "account": "账号问题（封店、申诉、侵权）",
    "chitchat": "闲聊（打招呼、无关寒暄）",
    "out_of_scope": "超出范围（与跨境电商无关）",
}
STRATEGY_LABELS = {
    "direct": "只问一个明确的点",
    "subquery": "一句话里问了 2–3 个不同的点",
    "hyde": "描述了一个具体情况再求助，比如“我的链接被下架了……怎么办”",
}

# 数据格式第 4 节的最低数量
INTENT_MINIMUMS = {intent: 25 for intent in INTENTS}
INTENT_MINIMUMS["chitchat"] = 20
INTENT_MINIMUMS["out_of_scope"] = 20
STRATEGY_MINIMUMS = {"subquery": 40, "hyde": 40}

# 平台 / 站点别名表：用来校验模型标注是否在问题文本里有依据
PLATFORM_ALIASES = {
    "amazon": ["亚马逊", "amazon", "amz", "亚麻"],
    "temu": ["temu", "特木"],
    "tiktok_shop": ["tiktok", "tik tok", "抖音小店", "英国小店"],
    "shopee": ["shopee", "虾皮"],
}
SITE_ALIASES = {
    "US": ["美国", "美站", "us", "usa", "美国站"],
    "UK": ["英国", "英站", "uk", "英国站"],
    "DE": ["德国", "德站", "de", "德国站"],
    "JP": ["日本", "日站", "jp", "日本站"],
    "EU": ["欧盟", "欧洲", "eu", "欧洲站"],
    "AU": ["澳洲", "澳大利亚", "au", "澳洲站"],
    "AE": ["中东", "阿联酋", "ae", "迪拜"],
}

INTENT_KEYS = "platform_policy / compliance / logistics / tax / ops / account / chitchat / out_of_scope"

MIN_QUOTE, MAX_QUOTE = 10, 200
FAQ_SIMILARITY_LIMIT = 0.90

SYSTEM_PROMPT = (
    "你是一名资深的跨境电商数据标注专家，负责构造高质量的评估集。"
    "你只能依据给出的资料，不能编造事实；输出必须是合法的 JSON 对象。"
)


def _key(text: str) -> str:
    """去重指纹：去掉空白和标点。"""
    return re.sub(r"[\s\W_]+", "", text).lower()


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


# ---------------------------------------------------------------------------
# 平台 / 站点标注校验
# ---------------------------------------------------------------------------


def infer_platform(query: str, claimed: str | None) -> str | None:
    """问题文本里找不到任何别名时，把平台改成 null。"""
    text = query.lower()
    if claimed and claimed in PLATFORM_ALIASES:
        if any(alias in text for alias in PLATFORM_ALIASES[claimed]):
            return claimed
    return None


def infer_site(query: str, claimed: str | None) -> str | None:
    """站点同理：文本里没有依据就置 null。"""
    text = query.lower()
    if claimed and claimed.upper() in SITE_ALIASES:
        code = claimed.upper()
        if any(alias in text for alias in SITE_ALIASES[code]):
            return code
    return None


# ---------------------------------------------------------------------------
# 意图标注集
# ---------------------------------------------------------------------------


def intent_quotas(scale: float) -> dict[str, int]:
    """按最低数量乘 scale 分配各类配额。"""
    return {intent: math.ceil(minimum * scale) for intent, minimum in INTENT_MINIMUMS.items()}


def plan_requests(
    quotas: dict[str, int], scale: float, batch: int = 10
) -> list[tuple[str, str | None, int]]:
    """把每类配额拆成「意图 + 策略 + 本批条数」。"""
    chat_intents = {"chitchat", "out_of_scope"}
    non_chat = [(intent, count) for intent, count in quotas.items() if intent not in chat_intents]
    total_non_chat = sum(count for _, count in non_chat) or 1
    targets = {name: math.ceil(value * scale) for name, value in STRATEGY_MINIMUMS.items()}

    allocation: dict[str, dict[str, int]] = {}
    for intent, count in non_chat:
        subquery = min(count, round(count * targets["subquery"] / total_non_chat))
        hyde = min(count - subquery, round(count * targets["hyde"] / total_non_chat))
        allocation[intent] = {"subquery": subquery, "hyde": hyde, "direct": count - subquery - hyde}

    # 四舍五入可能少给一点，从 direct 里补足
    for strategy in ("subquery", "hyde"):
        gap = targets[strategy] - sum(row[strategy] for row in allocation.values())
        for intent in allocation:
            while gap > 0 and allocation[intent]["direct"] > 0:
                allocation[intent]["direct"] -= 1
                allocation[intent][strategy] += 1
                gap -= 1
            if gap <= 0:
                break

    requests: list[tuple[str, str | None, int]] = []
    for intent, count in quotas.items():
        if intent in chat_intents:
            remaining = count
            while remaining > 0:
                size = min(batch, remaining)
                requests.append((intent, None, size))
                remaining -= size
            continue
        for strategy in ("direct", "subquery", "hyde"):
            remaining = allocation[intent][strategy]
            while remaining > 0:
                size = min(batch, remaining)
                requests.append((intent, strategy, size))
                remaining -= size
    return requests


def build_intent_prompt(intent: str, strategy: str | None, combo: dict[str, str], batch: int) -> str:
    """构造一次意图标注生成请求。"""
    strategy_line = (
        f"- 检索策略：{strategy}（{STRATEGY_LABELS[strategy]}）"
        if strategy
        else "- 这个问题是闲聊或超出范围，没有检索策略"
    )
    return f"""请生成 {batch} 个中国跨境卖家可能问的问题，用于意图分类评估。

约束：
- 意图：{intent}（{INTENT_LABELS[intent]}）
{strategy_line}
- 卖家画像：{combo['profile']}
- 平台：{combo['platform']}；站点：{combo['site']}
- 语气：{combo['tone']}

输出 JSON：
{{"items": [{{"query": "...", "platform": "amazon|temu|tiktok_shop|shopee|null", "site": "US|UK|DE|JP|EU|null"}}]}}

要求：
1. query 是卖家的原话，口语化，可以有错别字。
2. platform 和 site 只在问题里**明确提到**时才标注，否则为 null，不要根据常识推测。
3. 每个问题都要符合上面的意图和策略。"""


def validate_intent_payload(
    payload: dict, intent: str, strategy: str | None, seen: set[str], stats: dict[str, int]
) -> list[dict]:
    """校验并规范化一批意图标注。"""
    items = payload.get("items")
    if not isinstance(items, list):
        stats["json_error"] += 1
        return []

    kept: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            stats["json_error"] += 1
            continue
        query = str(item.get("query") or "").strip()
        if not query or _key(query) in seen:
            stats["duplicate"] += 1
            continue
        platform = infer_platform(query, str(item.get("platform") or "") or None)
        site = infer_site(query, str(item.get("site") or "") or None)
        if platform is None and item.get("platform"):
            stats["alias_mismatch"] += 1
        if site is None and item.get("site"):
            stats["alias_mismatch"] += 1
        seen.add(_key(query))
        kept.append(
            {
                "query": query,
                "intent": intent,
                "strategy": strategy,
                "platform": platform,
                "site": site,
                "source": "llm_generated",
                "reviewed": False,
                "reviewed_by": None,
                "note": None,
            }
        )
    return kept


def run_intent(args: argparse.Namespace, client, embed=None) -> int:
    """生成意图标注集候选数据。"""
    quotas = intent_quotas(args.scale)
    planned = plan_requests(quotas, args.scale)
    if args.limit:
        planned = planned[: args.limit]

    existing = read_jsonl(args.out)
    seen = {_key(row.get("query", "")) for row in existing}

    if args.dry_run:
        intent, strategy, size = planned[0]
        combo = {"profile": "新手", "platform": "Amazon", "site": "美国", "tone": "很口语"}
        total = sum(size for _, _, size in planned)
        print(f"[dry-run] 共 {len(planned)} 个请求、约 {total} 条；第一个：{intent} / {strategy}（{size} 条）\n")
        print(build_intent_prompt(intent, strategy, combo, size))
        return 0

    rng = random.Random(args.seed)
    combos = {
        "profile": ["刚入驻的新手", "做了两年的有经验卖家"],
        "platform": ["提到 Amazon", "提到 Temu", "提到 TikTok Shop", "提到 Shopee", "不提平台"],
        "site": ["提到美国站", "提到德国站", "提到英国站", "提到日本站", "不提站点"],
        "tone": ["很口语，可以有错别字", "比较正式的书面表达", "口语但带具体场景"],
    }
    tasks = []
    for intent, strategy, size in planned:
        combo = {key: rng.choice(values) for key, values in combos.items()}
        tasks.append((intent, strategy, combo, size))

    stats = {key: 0 for key in ("json_error", "duplicate", "alias_mismatch")}

    def worker(task: tuple[str, str | None, dict[str, str], int]) -> tuple[str, str | None, dict]:
        intent, strategy, combo, size = task
        payload = client.chat_json(build_intent_prompt(intent, strategy, combo, size), SYSTEM_PROMPT)
        return intent, strategy, payload

    rows: list[dict] = []
    for intent, strategy, payload in client.map(worker, tasks):
        rows.extend(validate_intent_payload(payload, intent, strategy, seen, stats))

    next_id = len(existing) + 1
    for offset, row in enumerate(rows):
        row["id"] = f"int-{next_id + offset:04d}"
    write_jsonl(args.out, existing + rows)

    by_intent: dict[str, int] = {}
    for row in rows:
        by_intent[row["intent"]] = by_intent.get(row["intent"], 0) + 1
    print(f"\n新增 {len(rows)} 条，文件共 {len(existing) + len(rows)} 条 -> {args.out}")
    print("  意图分布：" + "，".join(f"{key}={value}" for key, value in sorted(by_intent.items())))
    dropped = {key: value for key, value in stats.items() if value}
    if dropped:
        print("丢弃/修正：" + "，".join(f"{key}={value}" for key, value in sorted(dropped.items())))
    print(client.usage.summary())
    return 0


# ---------------------------------------------------------------------------
# 检索问答集
# ---------------------------------------------------------------------------


def quotes_in_single_chunk(quote: str, doc_key: str, chunk_index: dict[str, list[str]]) -> bool:
    """证据必须完整落在同一篇文档的某一个片段里，否则 Hit@K 永远不可能命中。"""
    normalized = normalize_for_match(quote)
    return any(normalized in normalize_for_match(chunk) for chunk in chunk_index.get(doc_key, []))


def build_single_prompt(document: dict, excerpt: str) -> str:
    """单文档问题。"""
    return f"""资料标题：{document['title']}
平台：{document.get('platform') or '不适用'}；站点：{document.get('site') or '通用'}

资料原文：
\"\"\"
{excerpt}
\"\"\"

请基于资料出 1 个中文问题，输出 JSON：
{{"query": "...", "reference_answer": "...", "evidence": [{{"quote": "..."}}],
  "paraphrases": ["...", "..."], "intent": "...", "platform": "amazon|null", "site": "US|null"}}

要求：
1. query、paraphrases、reference_answer 都用中文；资料是英文时也一样（这是为了评估跨语言检索）。
2. reference_answer 用 2–5 句话回答，只能用资料里的信息。
3. evidence 给 1 条 quote，从原文逐字复制、{MIN_QUOTE}–{MAX_QUOTE} 个字符，不要复制标题行，也要保证这句话完整落在原文的同一个段落里。
4. paraphrases 给 2–3 条意思相同的不同说法。
5. intent 只能取这 8 个英文值之一：{INTENT_KEYS}；platform / site 只在问题里明确提到时才标注，否则 null。"""


def build_multi_prompt(left: dict, right: dict) -> str:
    """多文档问题：先判断能不能出题，再生成。"""
    return f"""有两份资料：
A《{left['title']}》（站点：{left.get('site') or '通用'}）：{(left['text'][:2000])}
B《{right['title']}》（站点：{right.get('site') or '通用'}）：{(right['text'][:2000])}

请判断能否提出一个**同时用到 A 和 B 才能回答完整**的自然问题。
参考：A 讲平台操作、B 讲法规要求时，通常可以问"某站点做某事，平台和法规各有什么要求"。
输出 JSON：
{{"possible": true/false, "query": "...", "reference_answer": "...",
  "evidence": [{{"doc": "A", "quote": "..."}}, {{"doc": "B", "quote": "..."}}],
  "paraphrases": ["...", "..."], "intent": "...", "platform": "amazon|null", "site": "US|null"}}

要求：
1. 只要两篇资料都与这个问题相关、回答时需要同时引用两篇里的信息就可以，不要求严格到"缺一不可"；只有确实毫无关系时才返回 possible=false。
2. 问题用中文，口语化；reference_answer 用中文 2–5 句，需要同时用到两篇的信息。
3. 两篇各给 1 条 quote，逐字复制、{MIN_QUOTE}–{MAX_QUOTE} 个字符。
4. intent 只能取这 8 个英文值之一：{INTENT_KEYS}；platform / site 只在问题里明确提到时才标注，否则 null。"""


def build_out_of_scope_prompt(coverage: str, count: int) -> str:
    """知识库外问题。"""
    return f"""当前知识库覆盖情况：
{coverage}

请生成 {count} 个**明显不在上述覆盖范围内**的跨境电商问题，输出 JSON：
{{"items": [{{"query": "..."}}]}}

要求：
1. 优先选完全没有资料的平台（Temu、TikTok Shop、Shopee、eBay、Walmart）和法规（韩国 KC、日本 PSE、美国 FCC）。
2. 不要选 Amazon 的边缘话题，因为知识库可能部分覆盖，人工很难判断。
3. 问题用中文，口语化。"""


def collect_coverage(documents: list[dict]) -> str:
    """按平台、站点、文档类型汇总覆盖范围，交给模型。"""
    platforms: dict[str, int] = {}
    sites: dict[str, int] = {}
    for document in documents:
        platforms[document.get("platform") or "通用"] = platforms.get(document.get("platform") or "通用", 0) + 1
        sites[document.get("site") or "不限"] = sites.get(document.get("site") or "不限", 0) + 1
    lines = [f"共 {len(documents)} 篇文档"]
    lines.append("平台：" + "，".join(f"{key}({value}篇)" for key, value in sorted(platforms.items())))
    lines.append("站点：" + "，".join(f"{key}({value}篇)" for key, value in sorted(sites.items())))
    lines.append("标题示例：" + "；".join(doc["title"] for doc in documents[:10]))
    return "\n".join(lines)


TOPIC_KEYWORDS = (
    "vat", "epr", "包装", "packaging", "gpsr", "安全", "safety", "ce", "red", "weee",
    "商标", "trademark", "进口", "import", "税", "tax", "合规", "compliance",
    "fba", "物流", "logistics", "ce认证", "注册", "register",
)


def _topic_score(left: dict, right: dict) -> int:
    """配对打分：站点匹配 + 主题关键词重合数，越高越可能出综合题。"""
    text_left = (left["title"] + left["text"][:800]).lower()
    text_right = (right["title"] + right["text"][:800]).lower()
    score = sum(1 for word in TOPIC_KEYWORDS if word in text_left and word in text_right)
    a, b = left.get("site"), right.get("site")
    if a is None or b is None or a == b or {a, b} == {"DE", "EU"}:
        score += 2
    else:
        score -= 10
    return score


def pair_candidates(documents: list[dict], rng: random.Random) -> list[tuple[dict, dict]]:
    """按规则配对：跨集合且站点匹配；或者同集合同品类。按主题相关度从高到低排。"""
    by_collection: dict[str, list[dict]] = {}
    for document in documents:
        by_collection.setdefault(document["collection"], []).append(document)

    pairs: list[tuple[dict, dict]] = []
    platforms = list(by_collection.get("xm_platform_policy", []))
    compliances = list(by_collection.get("xm_compliance", []))
    rng.shuffle(platforms)
    rng.shuffle(compliances)
    for platform_doc in platforms:
        for compliance_doc in compliances:
            pairs.append((platform_doc, compliance_doc))

    for collection_docs in by_collection.values():
        groups: dict[str, list[dict]] = {}
        for document in collection_docs:
            if document.get("category"):
                groups.setdefault(document["category"], []).append(document)
        for group in groups.values():
            for index in range(0, len(group) - 1, 2):
                pairs.append((group[index], group[index + 1]))

    # 同集合、同站点的文档（比如同为英国法规的 WEEE 和电池要求），主题往往天然相关
    for collection_docs in by_collection.values():
        same_site: dict[str, list[dict]] = {}
        for document in collection_docs:
            if document.get("site"):
                same_site.setdefault(document["site"], []).append(document)
        for group in same_site.values():
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    pairs.append((group[i], group[j]))

    # 站点不匹配的组合直接去掉（任务书的配对规则），不只是排到后面
    pairs = [pair for pair in pairs if _sites_compatible(*pair)]
    # 主题最相关的先试，同分打乱
    rng.shuffle(pairs)
    pairs.sort(key=lambda item: _topic_score(*item), reverse=True)
    return pairs


def _sites_compatible(left: dict, right: dict) -> bool:
    """站点相同、任一方不限站点、或者 DE 和 EU 互相匹配。"""
    a, b = left.get("site"), right.get("site")
    return a is None or b is None or a == b or {a, b} == {"DE", "EU"}


def _extract_evidence(payload: dict, documents: list[dict]) -> list[dict]:
    """把模型给的 quote（带 doc 标记或不带）映射到 doc_key。"""
    evidence: list[dict] = []
    for item in payload.get("evidence") or []:
        if not isinstance(item, dict):
            continue
        quote = str(item.get("quote") or "").strip()
        if not quote:
            continue
        tag = str(item.get("doc") or "").upper()
        index = 1 if tag == "B" else 0
        if index >= len(documents):
            index = 0
        evidence.append({"doc_key": documents[index]["doc_key"], "quote": quote})
    return evidence


def validate_qa_payload(
    payload: dict,
    documents: list[dict],
    chunk_index: dict[str, list[str]],
    seen: set[str],
    stats: dict[str, int],
    allow_answerable: bool = True,
) -> dict | None:
    """校验一条 qa 候选；不通过返回 None 并记原因。"""
    query = str(payload.get("query") or "").strip()
    if not query or _key(query) in seen:
        stats["duplicate"] += 1
        return None

    paraphrases = [str(item).strip() for item in (payload.get("paraphrases") or []) if str(item).strip()]
    if len(paraphrases) < 2:
        stats["paraphrase_short"] += 1
        return None

    intent = str(payload.get("intent") or "").strip()
    if intent not in INTENTS:
        stats["bad_intent"] += 1
        return None

    evidence = _extract_evidence(payload, documents)
    if not evidence:
        stats["quote_mismatch"] += 1
        return None

    text_by_key = {document["doc_key"]: document["text"] for document in documents}
    for entry in evidence:
        quote = entry["quote"]
        if not MIN_QUOTE <= len(quote) <= MAX_QUOTE:
            stats["quote_length"] += 1
            return None
        if normalize_for_match(quote) not in normalize_for_match(text_by_key.get(entry["doc_key"], "")):
            stats["quote_mismatch"] += 1
            return None
        if not quotes_in_single_chunk(quote, entry["doc_key"], chunk_index):
            stats["quote_split"] += 1
            return None

    seen.add(_key(query))
    return {
        "query": query,
        "intent": intent,
        "platform": infer_platform(query, str(payload.get("platform") or "") or None),
        "site": infer_site(query, str(payload.get("site") or "") or None),
        "answerable": allow_answerable,
        "evidence": evidence,
        "reference_answer": str(payload.get("reference_answer") or "").strip() or None,
        "paraphrases": paraphrases[:3],
        "source": "llm_generated",
        "reviewed": False,
        "reviewed_by": None,
        "note": None,
    }


def _cosine(left: list[float], right: list[float]) -> float:
    """余弦相似度。"""
    dot = sum(a * b for a, b in zip(left, right))
    norm_left = math.sqrt(sum(a * a for a in left)) or 1.0
    norm_right = math.sqrt(sum(b * b for b in right)) or 1.0
    return dot / (norm_left * norm_right)


def is_leaked(query: str, faq_vectors: list[list[float]], embed) -> bool:
    """和任意一条 FAQ 问题的相似度 > 0.90 就算泄题。"""
    if not faq_vectors or embed is None:
        return False
    vector = embed([query])[0]
    return any(_cosine(vector, other) > FAQ_SIMILARITY_LIMIT for other in faq_vectors)


def run_qa(args: argparse.Namespace, client, embed=None) -> int:
    """生成检索问答集候选数据。"""
    documents = read_jsonl(args.docs)
    if not documents:
        print("[ERROR] 没有文档，先跑 parse")
        return 1
    chunk_rows = read_jsonl(args.chunks)
    chunk_index: dict[str, list[str]] = {}
    for row in chunk_rows:
        chunk_index.setdefault(row["doc_key"], []).append(row["text"])

    faq_rows = read_jsonl(args.faq) if args.faq else []
    if not faq_rows:
        print("[WARN] 没有提供 faq_items.jsonl，跳过泄题检查")
    faq_vectors: list[list[float]] = []
    if faq_rows and embed is not None:
        faq_vectors = embed([row["question"] for row in faq_rows])

    existing = read_jsonl(args.out)
    seen = {_key(row.get("query", "")) for row in existing}
    used_single = {entry["doc_key"] for row in existing for entry in row.get("evidence", [])}

    rng = random.Random(args.seed)
    total = args.n
    # 目标按总数算；已有的部分要从缺口里扣掉，支持续跑补齐
    existing_single = sum(
        1 for row in existing if row.get("answerable") and len(row.get("evidence") or []) == 1
    )
    existing_multi = sum(1 for row in existing if len(row.get("evidence") or []) >= 2)
    existing_out = sum(1 for row in existing if row.get("answerable") is False)
    single_count = max(0, int(total * 0.65) - existing_single)
    multi_count = max(0, int(total * 0.20) - existing_multi)
    out_count = max(0, int(total * 0.15) - existing_out)

    stats = {
        key: 0
        for key in (
            "duplicate", "paraphrase_short", "bad_intent", "quote_mismatch",
            "quote_length", "quote_split", "leak", "pair_failed", "json_error",
        )
    }

    if args.dry_run:
        document = documents[0]
        print(f"[dry-run] 单文档问题示例：{document['title']}\n")
        print(build_single_prompt(document, document["text"][:1200]))
        return 0

    rows: list[dict] = []
    by_collection: dict[str, list[dict]] = {}
    for document in documents:
        by_collection.setdefault(document["collection"], []).append(document)

    # 1) 单文档问题：按集合分层抽样
    singles: list[dict] = []
    for collection_docs in by_collection.values():
        singles.extend(collection_docs)
    rng.shuffle(singles)
    candidates = [doc for doc in singles if doc["doc_key"] not in used_single]

    def single_worker(document: dict) -> tuple[list[dict], dict]:
        payload = client.chat_json(
            build_single_prompt(document, document["text"][:4000]), SYSTEM_PROMPT
        )
        return [document], payload

    for _ in range(2):  # 两轮：第一轮每篇一题，第二轮换一个点
        if len(rows) >= single_count:
            break
        batch = candidates[: max(0, single_count - len(rows))]
        if not batch:
            break
        for meta, payload in client.map(single_worker, batch):
            row = validate_qa_payload(payload, meta, chunk_index, seen, stats)
            if row is None:
                continue
            if is_leaked(row["query"], faq_vectors, embed):
                stats["leak"] += 1
                continue
            rows.append(row)

    # 2) 多文档问题
    pairs = pair_candidates(documents, rng)
    attempts = 0
    max_attempts = max(10, multi_count * 5)  # 配不出来就换，最多换 5 次/题
    used_multi: set[str] = {
        entry["doc_key"]
        for row in existing
        if len(row.get("evidence") or []) >= 2
        for entry in row["evidence"]
    }
    for left, right in pairs:
        if len(rows) >= single_count + multi_count or attempts >= max_attempts:
            break
        if left["doc_key"] in used_multi or right["doc_key"] in used_multi:
            continue
        attempts += 1
        payload = client.chat_json(build_multi_prompt(left, right), SYSTEM_PROMPT)
        if not payload.get("possible"):
            stats["pair_failed"] += 1
            continue
        row = validate_qa_payload(payload, [left, right], chunk_index, seen, stats)
        if row is None:
            continue
        if len({entry["doc_key"] for entry in row["evidence"]}) < 2:
            stats["pair_failed"] += 1
            continue
        if is_leaked(row["query"], faq_vectors, embed):
            stats["leak"] += 1
            continue
        used_multi.update({left["doc_key"], right["doc_key"]})
        rows.append(row)
    if out_count > 0:
        payload = client.chat_json(
            build_out_of_scope_prompt(collect_coverage(documents), out_count), SYSTEM_PROMPT
        )
        for item in payload.get("items") or []:
            query = str(item.get("query") or "").strip() if isinstance(item, dict) else ""
            if not query or _key(query) in seen:
                stats["duplicate"] += 1
                continue
            seen.add(_key(query))
            rows.append(
                {
                    "query": query,
                    "intent": "out_of_scope",
                    "platform": None,
                    "site": None,
                    "answerable": False,
                    "evidence": [],
                    "reference_answer": None,
                    "paraphrases": [query + "？", query + " 怎么处理"],
                    "source": "llm_generated",
                    "reviewed": False,
                    "reviewed_by": None,
                    "note": "待人工确认知识库确实未覆盖",
                }
            )

    next_id = len(existing) + 1
    for offset, row in enumerate(rows):
        row["id"] = f"qa-{next_id + offset:04d}"
    write_jsonl(args.out, existing + rows)

    print(
        f"\n新增 {len(rows)} 条（目标：单文档 {single_count}、多文档 {multi_count}、库外 {out_count}），"
        f"文件共 {len(existing) + len(rows)} 条 -> {args.out}"
    )
    dropped = {key: value for key, value in stats.items() if value}
    if dropped:
        print("丢弃原因：" + "，".join(f"{key}={value}" for key, value in sorted(dropped.items())))
    print(client.usage.summary())
    return 0


# ---------------------------------------------------------------------------
# 命令行
# ---------------------------------------------------------------------------


def load_embedder():
    """加载 BGE-M3 作为泄题检查的编码器；加载失败时退化为跳过检查。"""
    try:
        import os

        from dotenv import load_dotenv
        from FlagEmbedding import BGEM3FlagModel

        load_dotenv(ROOT_DIR / ".env", override=True)
        model = BGEM3FlagModel(os.getenv("EMBED_MODEL", "BAAI/bge-m3"), use_fp16=True)

        def encode(texts: list[str]) -> list[list[float]]:
            output = model.encode(texts, batch_size=8, max_length=512)["dense_vecs"]
            return [list(map(float, vector)) for vector in output]

        return encode
    except Exception as exc:  # 模型没装或显存不够时不阻断生成
        print(f"[WARN] 无法加载 embedding，跳过泄题检查：{type(exc).__name__}: {exc}")
        return None


def main(argv: list[str] | None = None, client=None, embed=None) -> int:
    """解析子命令并分发。"""
    parser = argparse.ArgumentParser(description="生成评估集")
    sub = parser.add_subparsers(dest="command", required=True)

    intent_parser = sub.add_parser("intent", help="生成意图标注集")
    intent_parser.add_argument("--out", type=Path, default=ROOT_DIR / "data/eval/intent.jsonl")
    intent_parser.add_argument("--scale", type=float, default=1.3)
    intent_parser.add_argument("--seed", type=int, default=42)
    intent_parser.add_argument("--limit", type=int, default=0)
    intent_parser.add_argument("--dry-run", action="store_true")

    qa_parser = sub.add_parser("qa", help="生成检索问答集")
    qa_parser.add_argument("--docs", type=Path, default=ROOT_DIR / "data/processed/documents.jsonl")
    qa_parser.add_argument("--chunks", type=Path, default=ROOT_DIR / "data/processed/chunks.jsonl")
    qa_parser.add_argument("--faq", type=Path, default=ROOT_DIR / "data/processed/faq_items.jsonl")
    qa_parser.add_argument("--out", type=Path, default=ROOT_DIR / "data/eval/retrieval_qa.jsonl")
    qa_parser.add_argument("--n", type=int, default=200)
    qa_parser.add_argument("--seed", type=int, default=42)
    qa_parser.add_argument("--limit", type=int, default=0)
    qa_parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(argv)
    if args.limit and args.command == "qa":
        args.n = min(args.n, args.limit)

    if client is None and not args.dry_run:
        from xiaomao.llm import LLMClient

        client = LLMClient()
    # 只有 qa 的泄题检查需要向量模型；intent 不用加载
    if args.command == "qa" and embed is None and not args.dry_run:
        embed = load_embedder()

    if args.command == "intent":
        return run_intent(args, client, embed)
    return run_qa(args, client, embed)


if __name__ == "__main__":
    sys.exit(main())
