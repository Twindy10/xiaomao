"""整条链路评估脚本测试：指标计算和断点续跑。"""

from __future__ import annotations

import importlib.util
import json
from argparse import Namespace
from pathlib import Path
from types import ModuleType

from xiaomao.qa import QAResult

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_qa_eval.py"


def load_script() -> ModuleType:
    """按路径加载 scripts/run_qa_eval.py。"""
    spec = importlib.util.spec_from_file_location("run_qa_eval", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """写 JSONL。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def build_data(root: Path) -> None:
    """两条题：一道有答案、一道无答案。"""
    write_jsonl(
        root / "processed" / "chunks.jsonl",
        [
            {"chunk_id": "c1", "doc_key": "d1", "chunk_index": 0, "collection": "xm_compliance",
             "title_path": "包装法", "text": "包装法要求在 LUCID 注册。", "platform": None,
             "site": "DE", "doc_type": "regulation", "lang": "zh", "effective_date": None,
             "expire_date": None, "source_url": "https://example.com/1", "question": None,
             "answer": None, "token_count": 10},
            {"chunk_id": "c2", "doc_key": "d2", "chunk_index": 0, "collection": "xm_compliance",
             "title_path": "无关", "text": "另一段内容。", "platform": None, "site": "DE",
             "doc_type": "regulation", "lang": "zh", "effective_date": None, "expire_date": None,
             "source_url": "https://example.com/2", "question": None, "answer": None, "token_count": 10},
        ],
    )
    write_jsonl(
        root / "eval" / "retrieval_qa.jsonl",
        [
            {"id": "qa-0001", "query": "包装法怎么注册", "intent": "compliance", "platform": None,
             "site": "DE", "answerable": True,
             "evidence": [{"doc_key": "d1", "quote": "包装法要求在 LUCID 注册。"}],
             "reference_answer": "在 LUCID 注册。", "paraphrases": ["德国包装法怎么注册"],
             "source": "handwritten", "reviewed": True, "reviewed_by": "human", "note": None},
            {"id": "qa-0002", "query": "TikTok 英国小店怎么开通", "intent": "out_of_scope",
             "platform": None, "site": None, "answerable": False, "evidence": [],
             "reference_answer": None, "paraphrases": ["英国 TikTok 开店流程", "tiktok 小店入驻"],
             "source": "handwritten", "reviewed": True, "reviewed_by": "human", "note": None},
        ],
    )


class FakePipeline:
    """按题返回预设结果。"""

    def __init__(self, refused_ids: set[str] | None = None) -> None:
        self.refused_ids = refused_ids or set()
        self.calls: list[str] = []

    def answer(self, query: str, options=None) -> QAResult:
        """构造 QAResult。"""
        self.calls.append(query)
        qid = "qa-0001" if "包装法" in query else "qa-0002"
        refused = qid in self.refused_ids or qid == "qa-0002"
        evidence = [] if refused else [type("C", (), {"chunk_id": "c1", "score": 0.9})()]
        return QAResult(
            answer="拒答" if refused else "要在 LUCID 注册[1]",
            citations=[] if refused else [{"n": 1, "chunk_id": "c1", "collection": "xm_compliance",
                                           "source_url": "https://example.com/1",
                                           "title_path": "包装法", "effective_date": None, "rank": 1}],
            refused=refused,
            intent="out_of_scope" if qid == "qa-0002" else "compliance",
            strategy="direct",
            evidence=evidence,
            trace={"latency_ms": {"total": 100.0}, "prompt_tokens": 100, "completion_tokens": 20,
                   "backtracked": 0},
        )


def run_args(tmp_path: Path, config: str = "rerank") -> Namespace:
    """构造命令行参数。"""
    return Namespace(run="t", config=config, limit=0,
                     data_dir=tmp_path / "data", runs_dir=tmp_path / "runs")


def test_run_writes_and_resumes(tmp_path: Path) -> None:
    """落盘 + 断点续跑。"""
    run_eval = load_script()
    build_data(tmp_path / "data")
    pipeline = FakePipeline()

    assert run_eval.cmd_run(run_args(tmp_path), pipeline=pipeline) == 0
    out = tmp_path / "runs" / "qa_t" / "rerank.jsonl"
    first = out.read_text(encoding="utf-8").splitlines()
    assert len(first) == 2

    assert run_eval.cmd_run(run_args(tmp_path), pipeline=pipeline) == 0
    assert len(out.read_text(encoding="utf-8").splitlines()) == 2
    assert len(pipeline.calls) == 2  # 第二次全跳过


def test_metrics(tmp_path: Path) -> None:
    """Hit@5、拒答率、误拒率、引用准确率。"""
    run_eval = load_script()
    build_data(tmp_path / "data")
    items, chunks = run_eval.load_reference(tmp_path / "data")
    rows = [
        {"qid": "qa-0001", "refused": False, "intent": "compliance", "strategy": "direct",
         "evidence": [{"chunk_id": "c1", "score": 0.9}],
         "citations": [{"chunk_id": "c1"}],
         "trace": {"latency_ms": {"total": 100.0}, "prompt_tokens": 10, "completion_tokens": 5}},
        {"qid": "qa-0002", "refused": True, "intent": "out_of_scope", "strategy": "direct",
         "evidence": [], "citations": [],
         "trace": {"latency_ms": {"total": 50.0}, "prompt_tokens": 0, "completion_tokens": 0}},
    ]

    summary = run_eval.evaluate(rows, items, chunks)

    assert summary["hit5"] == 1.0
    assert summary["mrr5"] == 1.0
    assert summary["refusal_rate"] == 1.0
    assert summary["false_refusal_rate"] == 0.0
    assert summary["citation_accuracy"] == 1.0
    assert summary["n"] == 2


def test_false_refusal_counts(tmp_path: Path) -> None:
    """有答案的题被拒，算误拒。"""
    run_eval = load_script()
    build_data(tmp_path / "data")
    items, chunks = run_eval.load_reference(tmp_path / "data")
    rows = [{"qid": "qa-0001", "refused": True, "intent": "compliance", "strategy": "direct",
             "evidence": [], "citations": [],
             "trace": {"latency_ms": {"total": 10.0}}}]

    summary = run_eval.evaluate(rows, items, chunks)
    assert summary["false_refusal_rate"] == 1.0
    assert summary["hit5"] == 0.0


def test_faithfulness_ratio() -> None:
    """只统计有回答且被评审过的题。"""
    run_eval = load_script()
    rows = [
        {"qid": "a", "refused": False},
        {"qid": "b", "refused": False},
        {"qid": "c", "refused": True},
    ]
    judged = [{"qid": "a", "faithful": True}, {"qid": "b", "faithful": False},
              {"qid": "c", "faithful": False}]

    summary = run_eval.faithfulness(rows, judged)
    assert summary["n"] == 2
    assert summary["faithful"] == 0.5


def test_correctness_ratio() -> None:
    """答对率只算有答案的题；无答案的题单独看拒答。"""
    run_eval = load_script()
    graded = [
        {"qid": "a", "answerable": True, "grade": "correct"},
        {"qid": "b", "answerable": True, "grade": "partial"},
        {"qid": "c", "answerable": True, "grade": "wrong"},
        {"qid": "d", "answerable": True, "grade": "correct"},
        {"qid": "e", "answerable": False, "grade": "correct"},
    ]

    summary = run_eval.correctness(graded)
    assert summary["n"] == 4
    assert summary["correct"] == 0.5
    assert summary["partial"] == 0.25
    assert summary["refused_ok"] == 1.0


def test_report_writes_markdown(tmp_path: Path) -> None:
    """report 生成表格。"""
    run_eval = load_script()
    build_data(tmp_path / "data")
    pipeline = FakePipeline()
    run_eval.cmd_run(run_args(tmp_path), pipeline=pipeline)

    assert run_eval.cmd_report(Namespace(run="t", data_dir=tmp_path / "data",
                                          runs_dir=tmp_path / "runs")) == 0
    report = (tmp_path / "runs" / "qa_t" / "report.md").read_text(encoding="utf-8")
    assert "整条链路评估" in report
    assert "Hit@5" in report
    assert "100.0%" in report
