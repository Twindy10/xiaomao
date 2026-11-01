"""RAGAS 脚本纯函数测试（不导入 ragas）。"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ragas_eval.py"


def load_script() -> ModuleType:
    """按路径加载脚本。"""
    spec = importlib.util.spec_from_file_location("ragas_eval_script", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sample_inputs():
    """两道有答案、一道无答案、一道被拒答。"""
    items = [
        {"id": "a", "answerable": True, "reference_answer": "参考 A"},
        {"id": "b", "answerable": True, "reference_answer": "参考 B"},
        {"id": "c", "answerable": False, "reference_answer": None},
    ]
    rows = [
        {"qid": "a", "query": "问题 A", "answer": "回答 A", "refused": False,
         "evidence": [{"chunk_id": "c2"}, {"chunk_id": "c1"}]},
        {"qid": "b", "query": "问题 B", "answer": "回答 B", "refused": True,
         "evidence": [{"chunk_id": "c1"}]},
        {"qid": "c", "query": "无答案", "answer": "拒答", "refused": False,
         "evidence": [{"chunk_id": "c1"}]},
    ]
    chunks = {
        "c1": {"chunk_id": "c1", "text": "第一段"},
        "c2": {"chunk_id": "c2", "text": "第二段" * 1000},
    }
    return rows, items, chunks


def test_build_samples_filters_and_orders() -> None:
    """只保留有答案且没拒答的题，上下文按 evidence 顺序并截断 1500 字。"""
    script = load_script()
    rows, items, chunks = sample_inputs()

    samples, dropped = script.build_samples(rows, items, chunks)

    assert [sample["qid"] for sample in samples] == ["a"]
    assert dropped == {"refused": 1, "unanswerable": 1, "missing": 0}
    assert samples[0]["reference"] == "参考 A"
    assert samples[0]["retrieved_contexts"][0].startswith("第二段")
    assert len(samples[0]["retrieved_contexts"][0]) == 1500
    assert samples[0]["retrieved_contexts"][1] == "第一段"


def test_build_samples_limit() -> None:
    """--limit 只取前 N 道。"""
    script = load_script()
    rows, items, chunks = sample_inputs()
    rows.append({"qid": "b", "query": "问题 B", "answer": "回答", "refused": False,
                 "evidence": [{"chunk_id": "c1"}]})
    items[1]["answerable"] = True

    samples, _ = script.build_samples(rows, items, chunks, limit=1)
    assert len(samples) == 1


def test_done_qids_skips_existing(tmp_path: Path) -> None:
    """断点续跑：已有结果的 qid 不再跑。"""
    script = load_script()
    path = tmp_path / "config.jsonl"
    script.append_jsonl(path, [{"qid": "a", "faithfulness": 0.9}, {"qid": "b", "error": "nan"}])

    assert script.done_qids(path) == {"a", "b"}
    assert script.done_qids(tmp_path / "missing.jsonl") == set()


def test_contingency_table() -> None:
    """2×2 列联表和一致率。"""
    script = load_script()
    ragas_scores = {"a": 0.95, "b": 0.2, "c": 0.91, "d": 0.1, "e": float("nan")}
    judge_flags = {"a": True, "b": True, "c": False, "d": False, "e": True}

    result = script.contingency(ragas_scores, judge_flags, threshold=0.9)

    assert result["table"] == {"both_faithful": 1, "ragas_only": 1, "judge_only": 1, "neither": 1}
    assert result["total"] == 4
    assert result["agreement"] == 0.5
    assert {item["qid"] for item in result["mismatches"]} == {"b", "c"}


def test_contingency_missing_judge_rows_are_skipped() -> None:
    """评委没跑到的题不计入。"""
    script = load_script()
    result = script.contingency({"a": 0.95}, {}, threshold=0.9)
    assert result["total"] == 0
    assert result["agreement"] is None


def test_average_ignores_nan() -> None:
    """NaN 不计入平均。"""
    script = load_script()
    assert script.average([1.0, float("nan"), 0.0]) == 0.5
    assert script.average([float("nan")]) is None


def test_summarize_contains_agreement() -> None:
    """汇总里有指标均值和一致率。"""
    script = load_script()
    text = script.summarize(
        {"routing": [{"qid": "a", "faithfulness": 0.95, "answer_relevancy": 0.8}]},
        {"routing": {"a": True}},
        ["faithfulness", "answer_relevancy"],
    )
    assert "routing" in text
    assert "0.950" in text
    assert "100.0%" in text
