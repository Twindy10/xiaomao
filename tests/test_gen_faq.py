"""gen_faq.py 的测试：mock 掉 LLM 调用，不依赖网络和 .env。"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from types import ModuleType

GEN_FAQ_PATH = Path(__file__).resolve().parents[1] / "scripts" / "gen_faq.py"


def load_gen_faq() -> ModuleType:
    """按路径加载 scripts/gen_faq.py。"""
    spec = importlib.util.spec_from_file_location("gen_faq", GEN_FAQ_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeUsage:
    """替代 LLMClient.usage。"""

    def summary(self) -> str:
        return "调用 0 次"


class FakeClient:
    """按顺序返回预设 JSON 的假客户端。"""

    def __init__(self, payloads: list[dict]) -> None:
        self.payloads = list(payloads)
        self.prompts: list[str] = []
        self.usage = FakeUsage()

    def chat_json(self, prompt: str, system: str, max_tokens: int = 3000) -> dict:
        self.prompts.append(prompt)
        return self.payloads.pop(0)

    def map(self, func, items):
        return [func(item) for item in items]


def make_document(text: str, **overrides: object) -> dict:
    """构造一篇最小文档。"""
    document = {
        "doc_key": "a" * 16,
        "collection": "xm_platform_policy",
        "source_url": "https://example.com/a",
        "title": "德国站合规指南",
        "platform": "amazon",
        "site": "DE",
        "category": "3C",
        "doc_type": "guide",
        "lang": "zh",
        "effective_date": "2026-01-01",
        "expire_date": None,
        "content_hash": "x" * 64,
        "fetched_at": "2026-10-29T10:00:00+08:00",
        "text": text,
        "question": None,
        "answer": None,
    }
    document.update(overrides)
    return document


def test_doc_quota_by_length() -> None:
    """配额按字数：2–6 条。"""
    gen_faq = load_gen_faq()
    assert gen_faq.doc_quota(500) == 2
    assert gen_faq.doc_quota(3000) == 2
    assert gen_faq.doc_quota(4500) == 3
    assert gen_faq.doc_quota(20000) == 6


def test_allocate_sums_and_at_least_one() -> None:
    """按段落字数分配配额，每段至少 1 条，总数不超。"""
    gen_faq = load_gen_faq()
    counts = gen_faq.allocate(5, ["a" * 1000, "b" * 300, "c" * 50])
    assert sum(counts) <= 5
    assert all(count >= 1 for count in counts)


def test_split_segments_respects_limit() -> None:
    """长文档切段，每段不超过上限。"""
    gen_faq = load_gen_faq()
    text = "\n".join(f"## 小节{i}\n\n" + "正文内容。" * 200 for i in range(10))
    segments = gen_faq.split_segments(text, limit=2000)
    assert len(segments) > 1
    assert all(len(segment) <= 2100 for segment in segments)


def test_quote_without_markdown_passes() -> None:
    """原文带 ** 加粗，模型抄成纯文本也要能通过引文校验。"""
    gen_faq = load_gen_faq()
    document = make_document("卖家需要在 **LUCID** 注册包装，并填写注册号。")
    payload = {
        "faqs": [
            {
                "question": "德国站包装怎么注册？",
                "answer": "需要在 LUCID 注册包装，并填写注册号，注册号还要填到平台后台。",
                "quote": "卖家需要在 LUCID 注册包装，并填写注册号。",
                "intent": "compliance",
            }
        ]
    }
    stats = gen_faq._text_stats()
    kept = gen_faq.validate_faqs(payload, document, set(), stats)

    assert len(kept) == 1
    assert kept[0]["evidence"][0]["doc_key"] == document["doc_key"]
    assert kept[0]["platform"] == "amazon"
    assert kept[0]["site"] == "DE"
    assert kept[0]["lang"] == "zh"
    assert kept[0]["reviewed"] is False


def test_bad_quotes_are_dropped() -> None:
    """引文对不上、太长、答案太短都要丢弃并计数。"""
    gen_faq = load_gen_faq()
    document = make_document("卖家需要在 LUCID 注册包装，并填写注册号。")
    payload = {
        "faqs": [
            {"question": "问一？", "answer": "答案足够长的中文回答，需要超过二十个字符才算合格。",
             "quote": "原文里根本没有出现过这一句话。", "intent": "compliance"},
            {"question": "问二？", "answer": "答案足够长的中文回答，需要超过二十个字符才算合格。",
             "quote": "太短", "intent": "compliance"},
            {"question": "问三？", "answer": "短", "quote": "卖家需要在 LUCID 注册包装，并填写注册号。",
             "intent": "compliance"},
        ]
    }
    stats = gen_faq._text_stats()
    kept = gen_faq.validate_faqs(payload, document, set(), stats)

    assert kept == []
    assert stats["quote_mismatch"] == 1
    assert stats["quote_length"] == 1
    assert stats["answer_length"] == 1


def test_duplicate_question_is_dropped() -> None:
    """去掉标点空白后相同的问题算重复。"""
    gen_faq = load_gen_faq()
    document = make_document("卖家需要在 LUCID 注册包装，并填写注册号。")
    payload = {
        "faqs": [
            {"question": "德国站包装怎么注册？",
             "answer": "需要在 LUCID 注册包装，并填写注册号，注册号还要填到平台后台。",
             "quote": "卖家需要在 LUCID 注册包装，并填写注册号。", "intent": "compliance"}
        ]
    }
    seen: set[str] = set()
    stats = gen_faq._text_stats()

    assert len(gen_faq.validate_faqs(payload, document, seen, stats)) == 1
    assert len(gen_faq.validate_faqs(payload, document, seen, stats)) == 0
    assert stats["duplicate"] == 1


def test_main_writes_faq_and_continues_ids(tmp_path: Path) -> None:
    """端到端：写文件、复制元数据、已有编号时接着编。"""
    gen_faq = load_gen_faq()
    document = make_document("卖家需要在 LUCID 注册包装，并填写注册号，注册号要填到平台后台。" * 3)
    docs_path = tmp_path / "documents.jsonl"
    docs_path.write_text(json.dumps(document, ensure_ascii=False) + "\n", encoding="utf-8")
    out_path = tmp_path / "faq_items.jsonl"
    out_path.write_text(
        json.dumps({"faq_id": "faq-0001", "question": "旧问题？"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    payload = {
        "faqs": [
            {
                "question": "德国站包装要怎么注册？",
                "answer": "需要在 LUCID 注册包装，然后把注册号填到平台后台，否则会被判定不合规。",
                "quote": "卖家需要在 LUCID 注册包装，并填写注册号",
                "intent": "compliance",
            }
        ]
    }
    client = FakeClient([payload])
    code = gen_faq.main(
        ["--docs", str(docs_path), "--out", str(out_path), "--target", "5"], client=client
    )

    assert code == 0
    rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["faq_id"] == "faq-0001"
    assert rows[1]["faq_id"] == "faq-0002"
    assert rows[1]["question"] == "德国站包装要怎么注册？"
    assert rows[1]["evidence"][0]["doc_key"] == document["doc_key"]


def test_dry_run_does_not_call_model(tmp_path: Path, capsys) -> None:
    """--dry-run 只打印 prompt。"""
    gen_faq = load_gen_faq()
    document = make_document("正文。" * 800)
    docs_path = tmp_path / "documents.jsonl"
    docs_path.write_text(json.dumps(document, ensure_ascii=False) + "\n", encoding="utf-8")

    code = gen_faq.main(
        ["--docs", str(docs_path), "--out", str(tmp_path / "out.jsonl"), "--dry-run"]
    )

    assert code == 0
    assert "[dry-run]" in capsys.readouterr().out


def test_resume_skips_done_documents(tmp_path: Path) -> None:
    """断点续跑：已经处理过的文档跳过。"""
    gen_faq = load_gen_faq()
    document = make_document("正文。" * 800)
    docs_path = tmp_path / "documents.jsonl"
    docs_path.write_text(json.dumps(document, ensure_ascii=False) + "\n", encoding="utf-8")
    out_path = tmp_path / "faq_items.jsonl"
    (tmp_path / "faq_items.progress.json").write_text(
        json.dumps({"done_doc_keys": [document["doc_key"]], "next_faq_id": 3}), encoding="utf-8"
    )

    code = gen_faq.main(["--docs", str(docs_path), "--out", str(out_path)], client=FakeClient([]))

    assert code == 0
    assert not out_path.exists() or out_path.read_text(encoding="utf-8") == ""


def test_tokens_math() -> None:
    """配额计算不超过 6，也不少于一。"""
    gen_faq = load_gen_faq()
    for length in (0, 1, 1500, 1501, 100000):
        assert 1 <= gen_faq.doc_quota(length) <= 6
        assert gen_faq.doc_quota(length) == min(6, max(2, math.ceil(length / 1500)))
