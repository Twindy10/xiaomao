"""gen_eval.py 的测试：mock 掉 LLM 和向量模型。"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from types import ModuleType

GEN_EVAL_PATH = Path(__file__).resolve().parents[1] / "scripts" / "gen_eval.py"


def load_gen_eval() -> ModuleType:
    """按路径加载 scripts/gen_eval.py。"""
    spec = importlib.util.spec_from_file_location("gen_eval", GEN_EVAL_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeUsage:
    """替代 LLMClient.usage。"""

    def summary(self) -> str:
        return "调用 0 次"


class FakeClient:
    """按 prompt 内容返回预设 JSON 的假客户端。"""

    def __init__(self) -> None:
        self.calls = 0
        self.usage = FakeUsage()

    def chat_json(self, prompt: str, system: str, max_tokens: int = 3000) -> dict:
        self.calls += 1
        if "请判断能否提出" in prompt:
            quote = "这是所有文档都包含的一句通用正文内容。"
            return {
                "possible": True,
                "query": "德国站开店既要满足平台规则又要符合 GPSR，应该先做哪些准备？",
                "reference_answer": "先完成平台入驻和资质审核，再按 GPSR 准备安全信息和欧盟责任人。",
                "evidence": [{"doc": "A", "quote": quote}, {"doc": "B", "quote": quote}],
                "paraphrases": ["德国站开店合规准备有哪些", "德国站开店要先办什么"],
                "intent": "compliance",
                "platform": "amazon",
                "site": "DE",
            }
        if "明显不在上述覆盖范围内" in prompt:
            return {"items": [{"query": "TikTok Shop 英国小店怎么开通？"}]}
        if "请生成" in prompt and "意图分类评估" in prompt:
            return {
                "items": [
                    {"query": "亚马逊美国站发 FBA 有啥重量限制", "platform": "amazon", "site": "US"},
                ]
            }
        return {
            "query": "德国站卖耳机需要哪些认证？",
            "reference_answer": "需要 CE-RED 认证，并完成包装法注册。",
            "evidence": [{"quote": "这是所有文档都包含的一句通用正文内容。"}],
            "paraphrases": ["德国站耳机要什么认证", "在德国卖耳机需要哪些资质"],
            "intent": "compliance",
            "platform": None,
            "site": "DE",
        }

    def map(self, func, items):
        return [func(item) for item in items]


def make_document(index: int, collection: str, site: str | None, category: str | None = None) -> dict:
    """构造一篇最小文档。"""
    body = "这是所有文档都包含的一句通用正文内容。其余内容只用于凑长度。" * 5
    return {
        "doc_key": f"{index:016x}",
        "collection": collection,
        "source_url": f"https://example.com/{index}",
        "title": f"文档{index}",
        "platform": "amazon" if collection == "xm_platform_policy" else None,
        "site": site,
        "category": category,
        "doc_type": "guide" if collection == "xm_platform_policy" else "regulation",
        "lang": "zh",
        "effective_date": None,
        "expire_date": None,
        "content_hash": "x" * 64,
        "fetched_at": "2026-10-29T10:00:00+08:00",
        "text": body,
        "question": None,
        "answer": None,
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """写 JSONL。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def test_infer_platform_and_site() -> None:
    """标注必须有文本依据，否则置 null。"""
    gen_eval = load_gen_eval()
    assert gen_eval.infer_platform("亚马逊美国站怎么发货", "amazon") == "amazon"
    assert gen_eval.infer_platform("亚马逊美国站怎么发货", "temu") is None
    assert gen_eval.infer_site("德国站卖耳机要认证吗", "DE") == "DE"
    assert gen_eval.infer_site("美国站卖耳机要认证吗", "DE") is None
    assert gen_eval.infer_site("欧盟卖耳机要认证吗", "EU") == "EU"


def test_intent_quotas_scale() -> None:
    """配额 = 最低数量 × scale，向上取整。"""
    gen_eval = load_gen_eval()
    quotas = gen_eval.intent_quotas(1.3)
    assert quotas["chitchat"] == math.ceil(20 * 1.3)
    assert quotas["compliance"] == math.ceil(25 * 1.3)
    assert sum(quotas.values()) == 6 * math.ceil(25 * 1.3) + 2 * math.ceil(20 * 1.3)


def test_plan_requests_meets_strategy_minimums() -> None:
    """subquery / hyde 达到最低数量，闲聊和超范围没有策略，批大小不超过 10。"""
    gen_eval = load_gen_eval()
    planned = gen_eval.plan_requests(gen_eval.intent_quotas(1.3), 1.3)

    subquery = sum(size for _, strategy, size in planned if strategy == "subquery")
    hyde = sum(size for _, strategy, size in planned if strategy == "hyde")
    assert subquery >= math.ceil(40 * 1.3)
    assert hyde >= math.ceil(40 * 1.3)
    assert all(strategy is None for intent, strategy, _ in planned if intent in ("chitchat", "out_of_scope"))
    assert all(1 <= size <= 10 for _, _, size in planned)
    assert sum(size for _, _, size in planned) == sum(gen_eval.intent_quotas(1.3).values())


def test_validate_intent_payload_dedupes_and_fixes_alias() -> None:
    """重复问题丢弃；标了平台但文本没有依据时置 null 并记原因。"""
    gen_eval = load_gen_eval()
    stats = {"json_error": 0, "duplicate": 0, "alias_mismatch": 0}
    seen: set[str] = set()
    payload = {
        "items": [
            {"query": "亚马逊美国站怎么发货", "platform": "amazon", "site": "US"},
            {"query": "亚马逊美国站怎么发货？", "platform": "amazon", "site": "US"},
            {"query": "包装法怎么注册", "platform": "temu", "site": "DE"},
        ]
    }
    kept = gen_eval.validate_intent_payload(payload, "logistics", "direct", seen, stats)

    assert len(kept) == 2
    assert kept[0]["platform"] == "amazon"
    assert kept[0]["site"] == "US"
    assert kept[1]["platform"] is None and kept[1]["site"] is None
    assert stats["duplicate"] == 1
    assert stats["alias_mismatch"] == 2  # 平台和站点各修正一次


def test_quotes_in_single_chunk() -> None:
    """证据必须完整落在某一个片段里。"""
    gen_eval = load_gen_eval()
    chunks = {"a" * 16: ["第一段内容。这是完整的一句话。", "第二段内容。"]}

    assert gen_eval.quotes_in_single_chunk("这是完整的一句话。", "a" * 16, chunks)
    assert not gen_eval.quotes_in_single_chunk("第一段内容。第二段内容。", "a" * 16, chunks)


def test_pair_candidates_rules() -> None:
    """配对：跨集合站点匹配（DE 算 EU）、null 站点通配、同品类。"""
    gen_eval = load_gen_eval()
    import random

    docs = [
        make_document(1, "xm_platform_policy", "DE"),
        make_document(2, "xm_compliance", "EU"),
        make_document(3, "xm_compliance", "US"),
        make_document(4, "xm_platform_policy", None, category="3C"),
        make_document(5, "xm_platform_policy", "JP", category="3C"),
    ]
    pairs = gen_eval.pair_candidates(docs, random.Random(42))
    keys = {(left["doc_key"], right["doc_key"]) for left, right in pairs}

    assert (docs[0]["doc_key"], docs[1]["doc_key"]) in keys  # DE + EU
    assert (docs[0]["doc_key"], docs[2]["doc_key"]) not in keys  # DE + US 不匹配
    assert any({a, b} == {docs[3]["doc_key"], docs[4]["doc_key"]} for a, b in keys)  # 同品类


def test_validate_qa_payload_drops_split_quote() -> None:
    """引文跨片段的证据直接丢弃。"""
    gen_eval = load_gen_eval()
    left = make_document(1, "xm_compliance", "EU")
    stats = {key: 0 for key in ("duplicate", "paraphrase_short", "bad_intent", "quote_mismatch",
                                "quote_length", "quote_split", "leak", "pair_failed", "json_error")}
    payload = {
        "query": "欧盟卖耳机要什么认证？",
        "reference_answer": "需要 CE-RED。",
        "evidence": [{"quote": "这是所有文档都包含的一句通用正文内容。"}],
        "paraphrases": ["欧盟耳机认证要求", "在欧盟卖耳机的合规要求"],
        "intent": "compliance",
        "platform": None,
        "site": "EU",
    }
    split = {left["doc_key"]: ["这是所有文档都", "包含的一句通用正文内容。"]}
    assert gen_eval.validate_qa_payload(payload, [left], split, set(), stats) is None
    assert stats["quote_split"] == 1

    whole = {left["doc_key"]: [left["text"]]}
    row = gen_eval.validate_qa_payload(payload, [left], whole, set(), stats)
    assert row is not None and row["answerable"] is True


def test_is_leaked_uses_cosine_threshold() -> None:
    """和 FAQ 问题相似度 > 0.90 判为泄题。"""
    gen_eval = load_gen_eval()
    faq_vectors = [[1.0, 0.0]]

    def embed(texts: list[str]) -> list[list[float]]:
        return [[0.99, 0.01] for _ in texts]

    assert gen_eval.is_leaked("问题", faq_vectors, embed)
    assert not gen_eval.is_leaked("问题", [], embed)


def test_run_qa_end_to_end(tmp_path: Path) -> None:
    """端到端：三部分比例、编号、answerable 分布。"""
    gen_eval = load_gen_eval()
    docs = [
        make_document(1, "xm_platform_policy", "DE"),
        make_document(2, "xm_compliance", "EU"),
        make_document(3, "xm_compliance", "EU"),
        make_document(4, "xm_platform_policy", None),
        make_document(5, "xm_compliance", "US"),
    ]
    docs_path = tmp_path / "documents.jsonl"
    chunks_path = tmp_path / "chunks.jsonl"
    write_jsonl(docs_path, docs)
    write_jsonl(
        chunks_path,
        [{"doc_key": doc["doc_key"], "text": doc["text"], "chunk_index": 0} for doc in docs],
    )

    client = FakeClient()
    code = gen_eval.main(
        [
            "qa",
            "--docs", str(docs_path),
            "--chunks", str(chunks_path),
            "--faq", str(tmp_path / "none.jsonl"),
            "--out", str(tmp_path / "qa.jsonl"),
            "--n", "8",
        ],
        client=client,
        embed=lambda texts: [[1.0, 0.0] for _ in texts],
    )

    assert code == 0
    rows = [json.loads(line) for line in (tmp_path / "qa.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["id"] for row in rows] == [f"qa-{index:04d}" for index in range(1, len(rows) + 1)]
    assert any(row["answerable"] is False for row in rows)
    assert sum(1 for row in rows if row["answerable"]) >= 1


def test_intent_subcommand_end_to_end(tmp_path: Path) -> None:
    """端到端：写文件、编号、只生成 limit 条。"""
    gen_eval = load_gen_eval()
    client = FakeClient()
    out = tmp_path / "intent.jsonl"

    code = gen_eval.main(
        ["intent", "--out", str(out), "--limit", "3", "--scale", "1.0"], client=client, embed=None
    )

    assert code == 0
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) <= 3 * 10
    assert rows[0]["id"] == "int-0001"
    assert rows[0]["source"] == "llm_generated"
    assert rows[0]["reviewed_by"] is None
