"""决策对比脚本测试：ECE 和离线阈值拼接。"""

from __future__ import annotations

import importlib.util

import pytest
from pathlib import Path
from types import ModuleType

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "eval_decider.py"


def load_script() -> ModuleType:
    """按路径加载脚本（不会执行 main）。"""
    spec = importlib.util.spec_from_file_location("eval_decider_script", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ece_hand_computed() -> None:
    """手算：3 条 0.9（2 对 1 错）+ 1 条 0.1（错）。"""
    script = load_script()
    pairs = [(0.9, True), (0.9, True), (0.9, False), (0.1, False)]
    # 0.75 * |0.9 - 2/3| + 0.25 * |0.1 - 0| = 0.175 + 0.025
    assert script.expected_calibration_error(pairs) == pytest.approx(0.2)


def test_ece_empty_and_perfect() -> None:
    """没有样本返回 None，全对且置信度 1 时为 0。"""
    script = load_script()
    assert script.expected_calibration_error([]) is None
    assert script.expected_calibration_error([(1.0, True), (1.0, True)]) == pytest.approx(0.0)


def test_ece_underconfident() -> None:
    """置信度 0.2 但全对：误差 0.8。"""
    script = load_script()
    assert script.expected_calibration_error([(0.2, True)]) == pytest.approx(0.8)


def make_row(qid: str, intent: str, confidence: float, correct: bool) -> dict:
    """构造一条结果。"""
    value = intent if correct else "ops"
    return {
        "qid": qid,
        "answers": {
            "intent": {
                "value": value,
                "confidence": confidence,
                "probabilities": {value: confidence},
                "fell_back": False,
            }
        },
        "fell_back": False,
        "latency_ms": 10,
        "tokens": 0,
    }


def test_offline_threshold_merges_two_files() -> None:
    """阈值越高，走 LLM 的比例越高，准确率按拼接结果变化。"""
    script = load_script()
    items = [
        {"id": "1", "intent": "compliance"},
        {"id": "2", "intent": "compliance"},
    ]
    primary = [make_row("1", "compliance", 0.2, True), make_row("2", "compliance", 0.9, True)]
    fallback = [make_row("1", "compliance", 1.0, False), make_row("2", "compliance", 1.0, False)]
    names = ["intent"]

    result = script.offline_thresholds(primary, fallback, items, names, thresholds=[0.0, 0.5, 0.95])

    assert result["0"]["accuracy"]["intent"] == 1.0
    assert result["0"]["fallback_ratio"] == 0.0
    assert result["0.5"]["accuracy"]["intent"] == 0.5
    assert result["0.5"]["fallback_ratio"] == 0.5
    assert result["0.95"]["accuracy"]["intent"] == 0.0
    assert result["0.95"]["fallback_ratio"] == 1.0


def test_offline_threshold_ignores_strategy() -> None:
    """策略问题阈值为 0，永远不兜底。"""
    script = load_script()
    items = [{"id": "1", "intent": "ops", "strategy": "direct"}]
    primary = [{
        "qid": "1",
        "answers": {
            "intent": {"value": "ops", "confidence": 1.0, "probabilities": {"ops": 1.0}},
            "strategy": {"value": "direct", "confidence": 0.0, "probabilities": {"direct": 1.0}},
        },
    }]
    fallback = [{
        "qid": "1",
        "answers": {
            "intent": {"value": "ops", "confidence": 1.0, "probabilities": {"ops": 1.0}},
            "strategy": {"value": "hyde", "confidence": 1.0, "probabilities": {"hyde": 1.0}},
        },
    }]

    result = script.offline_thresholds(primary, fallback, items, ["intent", "strategy"], thresholds=[0.9])
    assert result["0.9"]["accuracy"]["strategy"] == 1.0
