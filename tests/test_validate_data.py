"""validate_data.py 的测试：在临时目录里造最小合法数据集，再逐项改坏。"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

VALIDATOR_PATH = Path(__file__).resolve().parents[1] / "scripts" / "validate_data.py"


def load_validator() -> ModuleType:
    """按文件路径加载 scripts/validate_data.py（scripts 不是包）。"""
    spec = importlib.util.spec_from_file_location("validate_data", VALIDATOR_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def doc_key(collection: str, url: str) -> str:
    """按约定算 doc_key。"""
    return hashlib.sha256(f"{collection}|{url}".encode("utf-8")).hexdigest()[:16]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """写 JSONL。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8"
    )


def build_dataset(root: Path) -> dict[str, object]:
    """造一套最小合法数据，返回关键 id 方便测试改坏。"""
    policy_text = "欧盟法规正文：蓝牙耳机需要符合 RED 指令。"
    faq_question, faq_answer = "德国站耳机要认证吗？", "需要 CE-RED 认证。"
    key_policy = doc_key("xm_compliance", "https://example.com/a")
    key_faq = doc_key("xm_faq", "faq://generated/faq-0001")

    documents = [
        {
            "doc_key": key_policy,
            "collection": "xm_compliance",
            "source_url": "https://example.com/a",
            "title": "欧盟法规",
            "platform": None,
            "site": "EU",
            "doc_type": "regulation",
            "lang": "zh",
            "category": None,
            "effective_date": "2026-01-01",
            "expire_date": None,
            "content_hash": hashlib.sha256(policy_text.encode("utf-8")).hexdigest(),
            "fetched_at": "2026-10-28T10:00:00+08:00",
            "text": policy_text,
            "question": None,
            "answer": None,
        },
        {
            "doc_key": key_faq,
            "collection": "xm_faq",
            "source_url": "faq://generated/faq-0001",
            "title": faq_question,
            "platform": None,
            "site": "DE",
            "doc_type": "faq",
            "lang": "zh",
            "category": None,
            "effective_date": None,
            "expire_date": None,
            "content_hash": hashlib.sha256(f"{faq_question}\n{faq_answer}".encode("utf-8")).hexdigest(),
            "fetched_at": "2026-10-28T10:00:00+08:00",
            "text": f"{faq_question}\n{faq_answer}",
            "question": faq_question,
            "answer": faq_answer,
        },
    ]

    def chunk(document: dict, index: int) -> dict:
        return {
            "chunk_id": f"{document['doc_key']}-{index:04d}",
            "doc_key": document["doc_key"],
            "chunk_index": index,
            "collection": document["collection"],
            "platform": document["platform"],
            "site": document["site"],
            "category": document["category"],
            "doc_type": document["doc_type"],
            "lang": document["lang"],
            "effective_date": document["effective_date"],
            "expire_date": document["expire_date"],
            "source_url": document["source_url"],
            "title_path": document["title"],
            "text": document["text"],
            "question": document["question"],
            "answer": document["answer"],
            "token_count": 20,
        }

    chunks = [chunk(document, 0) for document in documents]

    manifest = [
        {
            "doc_key": key_policy,
            "collection": "xm_compliance",
            "source_url": "https://example.com/a",
            "title": "欧盟法规",
            "platform": None,
            "site": "EU",
            "doc_type": "regulation",
            "lang": "zh",
            "category": None,
            "effective_date": "2026-01-01",
            "expire_date": None,
            "fetched_at": "2026-10-28T10:00:00+08:00",
            "raw_path": "data/raw/a.html",
            "content_type": "text/html",
            "http_status": 200,
        }
    ]

    faq_items = [
        {
            "faq_id": "faq-0001",
            "question": faq_question,
            "answer": faq_answer,
            "intent": "compliance",
            "platform": None,
            "site": "DE",
            "category": None,
            "lang": "zh",
            "effective_date": None,
            "evidence": [{"doc_key": key_policy, "quote": policy_text}],
            "source": "llm_generated",
            "reviewed": True,
            "reviewed_by": "claude",
            "note": None,
        }
    ]

    intents = [
        {
            "id": "int-0001",
            "query": "德国站卖蓝牙耳机要什么认证",
            "intent": "compliance",
            "strategy": "direct",
            "platform": None,
            "site": "DE",
            "source": "handwritten",
            "reviewed": True,
            "reviewed_by": "human",
            "note": None,
        }
    ]

    qa_rows = [
        {
            "id": "qa-0001",
            "query": "德国站卖蓝牙耳机需要哪些认证",
            "intent": "compliance",
            "platform": None,
            "site": "DE",
            "answerable": True,
            "evidence": [{"doc_key": key_policy, "quote": policy_text}],
            "reference_answer": "需要符合 RED 指令。",
            "paraphrases": ["德国站蓝牙耳机认证要求", "在德国卖耳机要什么认证"],
            "source": "handwritten",
            "reviewed": True,
            "reviewed_by": "human",
            "note": None,
        }
    ]

    write_jsonl(root / "raw" / "manifest.jsonl", manifest)
    write_jsonl(root / "processed" / "faq_items.jsonl", faq_items)
    write_jsonl(root / "processed" / "documents.jsonl", documents)
    write_jsonl(root / "processed" / "chunks.jsonl", chunks)
    write_jsonl(root / "eval" / "intent.jsonl", intents)
    write_jsonl(root / "eval" / "retrieval_qa.jsonl", qa_rows)
    return {"policy_key": key_policy, "faq_key": key_faq, "documents": documents,
            "chunks": chunks, "qa_rows": qa_rows}


def read_rows(path: Path) -> list[dict]:
    """读回 JSONL。"""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run(root: Path, *extra: str) -> int:
    """用指定数据目录跑校验。"""
    validator = load_validator()
    argv = ["--data-dir", str(root), *extra]
    return validator.main(argv)


def test_valid_dataset_passes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """最小合法数据集应当校验通过。"""
    build_dataset(tmp_path)
    code = run(tmp_path, "--stage", "all", "--allow-unreviewed")

    assert code == 0
    assert "校验通过" in capsys.readouterr().out


def test_broken_chunk_id_is_reported(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """chunk_id 格式错误要报出来。"""
    build_dataset(tmp_path)
    chunks = read_rows(tmp_path / "processed" / "chunks.jsonl")
    chunks[0]["chunk_id"] = "bad-id"
    write_jsonl(tmp_path / "processed" / "chunks.jsonl", chunks)

    assert run(tmp_path, "--stage", "processed", "--allow-unreviewed") == 1
    assert "chunk_id 格式非法" in capsys.readouterr().out


def test_broken_quote_is_reported(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """quote 在原文里找不到要报出来。"""
    build_dataset(tmp_path)
    rows = read_rows(tmp_path / "eval" / "retrieval_qa.jsonl")
    rows[0]["evidence"][0]["quote"] = "这句话在原文里根本找不到的。"
    write_jsonl(tmp_path / "eval" / "retrieval_qa.jsonl", rows)

    assert run(tmp_path, "--stage", "eval", "--allow-unreviewed") == 1
    assert "quote 在原文中找不到" in capsys.readouterr().out


def test_broken_content_hash_is_reported(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """content_hash 与 text 不一致要报出来。"""
    build_dataset(tmp_path)
    documents = read_rows(tmp_path / "processed" / "documents.jsonl")
    documents[0]["text"] = documents[0]["text"] + "被篡改了"
    write_jsonl(tmp_path / "processed" / "documents.jsonl", documents)

    assert run(tmp_path, "--stage", "processed", "--allow-unreviewed") == 1
    assert "content_hash 与 text 不一致" in capsys.readouterr().out


def test_unreviewed_eval_is_reported(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """不带 --allow-unreviewed 时，评估集不是 human 审核要报错。"""
    build_dataset(tmp_path)
    rows = read_rows(tmp_path / "eval" / "intent.jsonl")
    rows[0]["reviewed_by"] = "claude"
    write_jsonl(tmp_path / "eval" / "intent.jsonl", rows)

    assert run(tmp_path, "--stage", "eval") == 1
    assert "reviewed_by ∈ ['human']" in capsys.readouterr().out


def test_qa_reviewed_by_claude_needs_spot_check(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """检索集允许 reviewed_by=claude，但 Claude 审的部分要有 ≥10% 带人工抽检标记。"""
    build_dataset(tmp_path)
    rows = read_rows(tmp_path / "eval" / "retrieval_qa.jsonl")
    for row in rows:
        row["reviewed"], row["reviewed_by"], row["note"] = True, "claude", None
    write_jsonl(tmp_path / "eval" / "retrieval_qa.jsonl", rows)

    run(tmp_path, "--stage", "eval")
    output = capsys.readouterr().out
    assert "reviewed_by ∈" not in output.split("[retrieval_qa]")[1]  # 逐行检查接受 claude
    assert "人工抽检通过 0 条" in output

    validator = load_validator()
    rows[0]["note"] = validator.SPOT_CHECK_MARK
    write_jsonl(tmp_path / "eval" / "retrieval_qa.jsonl", rows)
    run(tmp_path, "--stage", "eval")
    assert "人工抽检通过" not in capsys.readouterr().out


def test_missing_file_is_reported_without_crashing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """缺文件只报错，不影响其他文件检查。"""
    build_dataset(tmp_path)
    (tmp_path / "processed" / "chunks.jsonl").unlink()

    assert run(tmp_path, "--stage", "processed", "--allow-unreviewed") == 1
    output = capsys.readouterr().out
    assert "文件不存在" in output
    assert "documents" in output


def test_chunk_index_must_be_continuous(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """chunk_index 不连续要报出来。"""
    build_dataset(tmp_path)
    chunks = read_rows(tmp_path / "processed" / "chunks.jsonl")
    chunks[0]["chunk_index"] = 3
    write_jsonl(tmp_path / "processed" / "chunks.jsonl", chunks)

    assert run(tmp_path, "--stage", "processed", "--allow-unreviewed") == 1
    assert "chunk_index 不是从 0 开始连续" in capsys.readouterr().out


def test_extra_field_is_warning_not_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """多出来的字段只报警告。"""
    build_dataset(tmp_path)
    rows = read_rows(tmp_path / "processed" / "chunks.jsonl")
    rows[0]["unexpected"] = 1
    write_jsonl(tmp_path / "processed" / "chunks.jsonl", rows)

    code = run(tmp_path, "--stage", "processed", "--allow-unreviewed")
    output = capsys.readouterr().out
    assert code == 0
    assert "约定以外的字段" in output


def test_missing_faq_items_is_only_warning_in_dev_mode(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """开发模式下 faq_items 还没生成只报警告；正式校验时报错。"""
    build_dataset(tmp_path)
    (tmp_path / "processed" / "faq_items.jsonl").unlink()

    assert run(tmp_path, "--stage", "processed", "--allow-unreviewed") == 0
    assert run(tmp_path, "--stage", "processed") == 1
