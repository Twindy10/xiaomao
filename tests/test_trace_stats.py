"""链路统计测试：分组、分位数、兜底、成本。"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "trace_stats.py"


def load_script() -> ModuleType:
    """按路径加载脚本。"""
    spec = importlib.util.spec_from_file_location("trace_stats_script", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row(path: str, **overrides) -> dict:
    """构造一行 qa_trace。"""
    base = {
        "id": 1, "intent": "compliance", "strategy": "direct", "cache_hit": 0,
        "evidence_sufficient": 1, "backtracked": 0, "total_latency_ms": 8000,
        "latency_ms": '{"decide": 40, "retrieve": 600, "generate": 3000}',
        "prompt_tokens": 2000, "completion_tokens": 200,
        "decisions": '{"intent": {"fell_back": false}, "_extra": {"llm_tokens": {"decide": 300, "generate": 2200}, "rerank_tokens": 7000}}',
    }
    base.update(overrides)
    return base


def test_classify_paths() -> None:
    """四种路径分组。"""
    script = load_script()
    assert script.classify(row("x", cache_hit=1)) == "缓存命中"
    assert script.classify(row("x", strategy=None, evidence_sufficient=None)) == "闲聊 / 超出范围"
    assert script.classify(row("x", evidence_sufficient=0)) == "拒答"
    assert script.classify(row("x")) == "完整生成"


def test_percentile() -> None:
    """线性插值。"""
    script = load_script()
    assert script.percentile([1, 2, 3, 4], 0.5) == 2.5
    assert script.percentile([], 0.5) == 0.0


def test_cost_hand_computed() -> None:
    """生成按 input/output，其余按 85/15 估，rerank 单独算。"""
    script = load_script()
    cost = script.answer_cost(row("x"), 0.2, 0.8, 0.5, 0.85)
    expected = (
        2000 / 1e6 * 0.2
        + 200 / 1e6 * 0.8
        + 300 * 0.85 / 1e6 * 0.2
        + 300 * 0.15 / 1e6 * 0.8
        + 7000 / 1e6 * 0.5
    )
    assert cost == pytest.approx(expected)


def test_summarize_paths_and_rates() -> None:
    """路径计数、拒答率、命中率（分母不含闲聊）。"""
    script = load_script()
    rows = [
        row("cache", cache_hit=1, total_latency_ms=1000,
            latency_ms='{"cache": 500, "decide": 200}',
            prompt_tokens=0, completion_tokens=0,
            decisions='{"_extra": {"llm_tokens": {"cache_gate": 500}}}'),
        row("full"),
        row("refused", evidence_sufficient=0, total_latency_ms=3000),
        row("chat", strategy=None, evidence_sufficient=None, total_latency_ms=500, intent="chitchat"),
    ]

    summary = script.summarize(rows)

    assert summary["total"] == 4
    assert summary["paths"]["缓存命中"]["count"] == 1
    assert summary["paths"]["完整生成"]["count"] == 1
    assert summary["paths"]["拒答"]["count"] == 1
    assert summary["paths"]["闲聊 / 超出范围"]["count"] == 1
    assert summary["refusal_rate"] == pytest.approx(0.25)
    assert summary["cache_hit_rate"] == pytest.approx(1 / 3)
    assert summary["paths"]["完整生成"]["p50_ms"] == pytest.approx(8000)


def test_fallback_counts() -> None:
    """兜底和整体降级次数。"""
    script = load_script()
    decisions = {
        "intent": {"fell_back": True},
        "platform": {"fell_back": False},
        "_extra": {"decision_error": True},
    }
    summary = script.summarize([row("x", decisions=__import__("json").dumps(decisions))])
    assert summary["fell_back"]["intent"] == 1
    assert summary["decision_errors"] == 1


def test_no_cache_cost_uses_full_generation_average() -> None:
    """没有缓存时，缓存命中按完整生成的平均成本算。"""
    script = load_script()
    cache_row = row("cache", cache_hit=1, total_latency_ms=1000,
                    prompt_tokens=0, completion_tokens=0,
                    decisions='{"_extra": {"llm_tokens": {}}}')
    full_row = row("full")
    rows = [cache_row, full_row]

    summary = script.summarize(rows)
    full_cost = script.answer_cost(full_row, 0.2, 0.8, 0.5, 0.85)
    cache_cost = script.answer_cost(cache_row, 0.2, 0.8, 0.5, 0.85)
    expected = (cache_cost + full_cost - cache_cost + full_cost) / 2 * 1000

    assert summary["cost_per_1k_no_cache"] == pytest.approx(expected)
    assert summary["cost_per_1k_all"] == pytest.approx((cache_cost + full_cost) / 2 * 1000)


def test_render_contains_sections() -> None:
    """报告里有主要小节。"""
    script = load_script()
    text = script.render(script.summarize([row("x")]))
    for section in ("回答数", "延迟（按路径）", "tokens", "成本", "每千次成本"):
        assert section in text
