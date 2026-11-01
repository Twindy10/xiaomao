"""评估脚本测试：目标匹配、指标、消融重放、rerank 断点续跑。"""

from __future__ import annotations

import importlib.util
import json
from argparse import Namespace
from pathlib import Path
from types import ModuleType

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_eval.py"


def load_run_eval() -> ModuleType:
    """按路径加载 scripts/run_eval.py。"""
    spec = importlib.util.spec_from_file_location("run_eval", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """写 JSONL。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def build_data(root: Path) -> None:
    """造一套最小评估数据。"""
    write_jsonl(
        root / "processed" / "documents.jsonl",
        [{"doc_key": "d1", "collection": "xm_compliance", "source_url": "https://example.com/1",
          "title": "文档一"}],
    )
    write_jsonl(
        root / "processed" / "chunks.jsonl",
        [
            {"chunk_id": "d1-0000", "doc_key": "d1", "chunk_index": 0, "collection": "xm_compliance",
             "title_path": "标题", "text": "**包装法**要求在 LUCID 注册。", "platform": None,
             "site": "DE", "doc_type": "regulation", "lang": "zh", "effective_date": None,
             "expire_date": None, "source_url": "https://example.com/1", "question": None,
             "answer": None, "token_count": 20},
            {"chunk_id": "d1-0001", "doc_key": "d1", "chunk_index": 1, "collection": "xm_compliance",
             "title_path": "标题", "text": "另一段无关内容。", "platform": None, "site": "DE",
             "doc_type": "regulation", "lang": "zh", "effective_date": None, "expire_date": None,
             "source_url": "https://example.com/1", "question": None, "answer": None, "token_count": 10},
        ],
    )
    write_jsonl(
        root / "eval" / "retrieval_qa.jsonl",
        [
            {"id": "qa-0001", "query": "包装法怎么注册", "intent": "compliance", "platform": None,
             "site": "DE", "answerable": True,
             "evidence": [{"doc_key": "d1", "quote": "包装法要求在 LUCID 注册。"}],
             "reference_answer": "要在 LUCID 注册。",
             "paraphrases": ["德国包装法怎么注册"],
             "source": "handwritten", "reviewed": True, "reviewed_by": "human", "note": None},
        ],
    )


def build_routes(runs: Path) -> Path:
    """造一份 routes.jsonl：dense 里第 1 名是目标。"""
    run = runs / "t"
    write_jsonl(
        run / "routes.jsonl",
        [
            {"qid": "qa-0001", "variant": 0, "query": "包装法怎么注册",
             "filter": "effective_date <= 20261031",
             "routes": {"dense": [["x-1", 0.9], ["d1-0000", 0.8]],
                        "bm25": [["x-1", 5.0], ["x-2", 4.0]]}},
            {"qid": "qa-0001", "variant": 1, "query": "德国包装法怎么注册",
             "filter": "effective_date <= 20261031",
             "routes": {"dense": [["d1-0000", 0.9]],
                        "bm25": [["x-1", 5.0]]}},
        ],
    )
    (run / "meta.json").write_text(json.dumps({"date": "2026-10-31", "filter": "none",
                                               "collections": ["xm_compliance"], "candidates": 20,
                                               "questions": 1, "embed_model": "bge-m3",
                                               "counts": {"xm_compliance": 2}}), encoding="utf-8")
    return run


def test_evidence_targets_matches_ignoring_markdown(tmp_path: Path) -> None:
    """原文带 ** 时，引文不带也能匹配到目标片段。"""
    run_eval = load_run_eval()
    chunks_by_doc = {"d1": [{"chunk_id": "c1", "text": "**包装法**要求在 LUCID 注册。"}]}
    item = {"evidence": [{"doc_key": "d1", "quote": "包装法要求在 LUCID 注册。"}]}

    targets = run_eval.evidence_targets(item, chunks_by_doc)
    assert targets == {"d1": {"c1"}}


def test_metrics_single_document() -> None:
    """单文档题：Hit@5、MRR@10、Recall@20。"""
    run_eval = load_run_eval()
    question = {"targets": {"c1"}, "targets_by_doc": {"d1": {"c1"}}, "multi_doc": False}

    assert run_eval.score_question(["c1", "x"], question) == {
        "hit5": True, "mrr10": 1.0, "recall20": True, "all_hit5": None,
    }
    miss = run_eval.score_question(["x1", "x2"], question)
    assert miss["hit5"] is False and miss["mrr10"] == 0.0 and miss["recall20"] is False

    second = run_eval.score_question(["x1", "c1"], question)
    assert second["mrr10"] == 0.5


def test_metrics_multi_document_all_hit() -> None:
    """多文档题要每篇证据都有片段进前 5。"""
    run_eval = load_run_eval()
    question = {
        "targets": {"a1", "b1"},
        "targets_by_doc": {"d1": {"a1"}, "d2": {"b1"}},
        "multi_doc": True,
    }

    assert run_eval.score_question(["a1", "b1", "x"], question)["all_hit5"] is True
    assert run_eval.score_question(["a1", "x", "y"], question)["all_hit5"] is False


def test_aggregate_averages() -> None:
    """汇总平均，多文档指标只在多文档题上平均。"""
    run_eval = load_run_eval()
    rows = [
        {"hit5": True, "mrr10": 1.0, "recall20": True, "all_hit5": None},
        {"hit5": False, "mrr10": 0.0, "recall20": True, "all_hit5": True},
        {"hit5": True, "mrr10": 0.5, "recall20": True, "all_hit5": False},
    ]
    summary = run_eval.aggregate(rows)

    assert summary["n"] == 3
    assert abs(summary["hit5"] - 2 / 3) < 1e-9
    assert abs(summary["mrr10"] - 0.5) < 1e-9
    assert summary["recall20"] == 1.0
    assert summary["all_hit5"] == 0.5


def test_ablation_end_to_end(tmp_path: Path) -> None:
    """从 routes.jsonl 重放出消融结果。"""
    run_eval = load_run_eval()
    data_dir = tmp_path / "data"
    runs = tmp_path / "runs"
    build_data(data_dir)
    run = build_routes(runs)

    code = run_eval.cmd_ablation(
        Namespace(run="t", configs=["dense", "bm25"], data_dir=data_dir, runs_dir=runs)
    )

    assert code == 0
    result = json.loads((run / "ablation.json").read_text(encoding="utf-8"))
    assert result["dense"]["original"]["hit5"] == 1.0  # 第 2 名是目标
    assert result["dense"]["original"]["mrr10"] == 0.5
    assert result["bm25"]["original"]["hit5"] == 0.0
    assert result["dense"]["paraphrase"]["hit5"] == 1.0


def test_rerank_writes_and_resumes(tmp_path: Path) -> None:
    """rerank 落盘；再跑一次不重复处理。"""
    run_eval = load_run_eval()
    data_dir = tmp_path / "data"
    runs = tmp_path / "runs"
    build_data(data_dir)
    run = build_routes(runs)

    args = Namespace(run="t", config="dense+bm25", backend="none", batch_size=8,
                     data_dir=data_dir, runs_dir=runs)
    assert run_eval.cmd_rerank(args) == 0
    out = run / "rerank_dense_bm25_none.jsonl"
    first = out.read_text(encoding="utf-8").splitlines()
    assert len(first) == 1  # 只跑原问题

    assert run_eval.cmd_rerank(args) == 0
    assert len(out.read_text(encoding="utf-8").splitlines()) == len(first)


def test_report_generates_markdown(tmp_path: Path) -> None:
    """report 写出表格和结论。"""
    run_eval = load_run_eval()
    data_dir = tmp_path / "data"
    runs = tmp_path / "runs"
    build_data(data_dir)
    run = build_routes(runs)
    run_eval.cmd_ablation(Namespace(run="t", configs=["dense"], data_dir=data_dir, runs_dir=runs))
    run_eval.cmd_rerank(Namespace(run="t", config="dense", backend="none", batch_size=8,
                                  data_dir=data_dir, runs_dir=runs))

    code = run_eval.cmd_report(Namespace(run="t", price_per_1k_tokens=0.0005, max_misses=5,
                                         data_dir=data_dir, runs_dir=runs))

    assert code == 0
    report = (run / "report.md").read_text(encoding="utf-8")
    assert "检索消融" in report
    assert "rerank 后端对比" in report
    assert "100.0%" in report


def test_report_cost_is_per_thousand_requests(tmp_path: Path) -> None:
    """每千次成本 = 平均 tokens × 单价（元 / 千 tokens）：7000 tokens × 0.0005 = 3.50 元。"""
    run_eval = load_run_eval()
    data_dir = tmp_path / "data"
    runs = tmp_path / "runs"
    build_data(data_dir)
    run = build_routes(runs)
    run_eval.cmd_ablation(Namespace(run="t", configs=["dense"], data_dir=data_dir, runs_dir=runs))
    run_eval.cmd_rerank(Namespace(run="t", config="dense", backend="none", batch_size=8,
                                  data_dir=data_dir, runs_dir=runs))
    summary_path = next(run.glob("summary_*.json"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["avg_tokens"] = 7000
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    run_eval.cmd_report(Namespace(run="t", price_per_1k_tokens=0.0005, max_misses=5,
                                  data_dir=data_dir, runs_dir=runs))
    assert "3.50 元" in (run / "report.md").read_text(encoding="utf-8")
