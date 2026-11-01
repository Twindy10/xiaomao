"""通用 systemone 后端测试：请求格式、confidence 重算、错误处理。"""

from __future__ import annotations

import json

import httpx
import pytest

from xiaomao.decision import FallbackDecider, Question, SystemOneDecider, get_decider
from xiaomao.decision.base import DeciderError
from xiaomao.decision.questions import EVIDENCE_QUESTION, INTENT_QUESTION, ROUTING_QUESTIONS, SITE_QUESTION

SAMPLE = {
    "answers": {
        "intent": {"type": "choice", "choice": "compliance", "confidence": 0.11,
                   "probabilities": {"compliance": 0.354, "platform_policy": 0.247, "ops": 0.2,
                                     "account": 0.199}},
        "strategy": {"type": "choice", "choice": "direct", "confidence": 0.2,
                     "probabilities": {"direct": 0.5, "subquery": 0.3, "hyde": 0.2}},
        "platform": {"type": "choice", "choice": "amazon", "confidence": 0.9,
                     "probabilities": {"amazon": 1.0}},
        "site": {"type": "choice", "choice": "DE", "confidence": 0.8,
                 "probabilities": {"DE": 0.9, "US": 0.1}},
        "evidence": {"type": "noul", "noul": 0.82},
        "level": {"type": "score", "score": 0.78, "confidence": 0.3,
                  "probabilities": {"0": 0.28, "1": 0.66, "2": 0.06}},
    },
    "usage": {"input_tokens": 190},
    "latency_ms": 200,
}


def make_decider(handler, **kwargs) -> tuple[SystemOneDecider, dict]:
    """用 MockTransport 构造，返回 decider 和捕获的请求体。"""
    captured: dict = {}

    def wrapped(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.read())
        captured["headers"] = dict(request.headers)
        return handler(request)

    client = httpx.Client(transport=httpx.MockTransport(wrapped), trust_env=False)
    decider = SystemOneDecider("http://127.0.0.1:8008", backend="kev", client=client, **kwargs)
    return decider, captured


def ok(request: httpx.Request) -> httpx.Response:
    """正常返回。"""
    return httpx.Response(200, json=SAMPLE)


def test_request_body_without_model_and_noul() -> None:
    """不传 model；noul 不带 criteria。"""
    level = Question(name="level", type="score", instructions="多急？", criteria=["不急", "一般", "很急"])
    decider, captured = make_decider(ok)
    decider.decide("德国站耳机要认证吗", [*ROUTING_QUESTIONS, EVIDENCE_QUESTION, level])

    body = captured["body"]
    assert "model" not in body
    assert body["state"] == "德国站耳机要认证吗"
    assert "criteria" not in body["questions"]["evidence"]
    assert body["questions"]["level"]["criteria"] == ["不急", "一般", "很急"]
    assert body["questions"]["intent"]["type"] == "choice"


def test_request_body_with_model() -> None:
    """传了 model 就写进请求体。"""
    decider, captured = make_decider(ok, model="kev-latest")
    decider.decide("问题", [INTENT_QUESTION])
    assert captured["body"]["model"] == "kev-latest"


def test_api_key_header() -> None:
    """有 api_key 时带 Bearer。"""
    decider, captured = make_decider(ok, api_key="secret-key")
    decider.decide("问题", [INTENT_QUESTION])
    assert captured["headers"]["authorization"] == "Bearer secret-key"


def test_confidence_recomputed_from_probabilities() -> None:
    """choice 用第一名减第二名，不用服务返回的 confidence。"""
    decider, _ = make_decider(ok)
    answers = decider.decide("问题", [*ROUTING_QUESTIONS, EVIDENCE_QUESTION])

    intent = answers["intent"]
    assert intent.value == "compliance"
    assert intent.confidence == pytest.approx(0.354 - 0.247, abs=1e-4)
    assert answers["site"].confidence == pytest.approx(0.9 - 0.1, abs=1e-4)
    assert answers["evidence"].confidence == pytest.approx(0.64)
    assert answers["evidence"].value == pytest.approx(0.82)


def test_normalizes_probabilities_that_do_not_sum_to_one() -> None:
    """概率四舍五入后先归一化再算。"""
    def rounded(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "answers": {
                "intent": {"type": "choice", "choice": "compliance",
                           "probabilities": {"compliance": 0.6, "ops": 0.6}},
            },
        })

    decider, _ = make_decider(rounded)
    answer = decider.decide("问题", [INTENT_QUESTION])["intent"]
    assert sum(answer.probabilities.values()) == pytest.approx(1.0)
    assert answer.confidence == pytest.approx(0.0)


def test_single_option_has_full_confidence() -> None:
    """只有一个选项时置信度是 1.0。"""
    def single(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "answers": {"platform": {"type": "choice", "choice": "amazon",
                                     "probabilities": {"amazon": 0.7}}},
        })

    decider, _ = make_decider(single)
    answer = decider.decide("问题", [ROUTING_QUESTIONS[2]])["platform"]
    assert answer.confidence == 1.0
    assert answer.value == "amazon"


def test_score_confidence_margin() -> None:
    """score 也用第一名减第二名。"""
    level = Question(name="level", type="score", instructions="多急？", criteria=["不急", "一般", "很急"])
    decider, _ = make_decider(ok)
    answer = decider.decide("问题", [level])["level"]
    assert answer.value == pytest.approx(1.0)
    assert answer.confidence == pytest.approx(0.66 - 0.28, abs=1e-4)


def test_http_error_raises() -> None:
    """非 200 抛 DeciderError。"""
    decider, _ = make_decider(lambda request: httpx.Response(500))
    with pytest.raises(DeciderError):
        decider.decide("问题", [INTENT_QUESTION])


def test_timeout_raises() -> None:
    """超时抛 DeciderError。"""
    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timeout", request=request)

    decider, _ = make_decider(timeout)
    with pytest.raises(DeciderError):
        decider.decide("问题", [INTENT_QUESTION])


def test_missing_answer_raises() -> None:
    """缺答案抛 DeciderError。"""
    def missing(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"answers": {}})

    decider, _ = make_decider(missing)
    with pytest.raises(DeciderError):
        decider.decide("问题", [INTENT_QUESTION])


def test_unknown_type_raises() -> None:
    """不认识的问题类型抛 DeciderError。"""
    def unknown(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"answers": {"intent": {"type": "ranking"}}})

    decider, _ = make_decider(unknown)
    with pytest.raises(DeciderError):
        decider.decide("问题", [INTENT_QUESTION])


def test_get_decider_kev_wraps_fallback(monkeypatch) -> None:
    """DECIDER_BACKEND=kev 时返回包着 SystemOneDecider 的 FallbackDecider。"""
    monkeypatch.setenv("DECIDER_BACKEND", "kev")
    monkeypatch.setenv("DECIDER_BASE_URL", "http://127.0.0.1:8008")
    monkeypatch.setenv("DECIDER_MODEL", "kev-latest")
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:1/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_MODEL", "test-model")

    decider = get_decider()
    assert isinstance(decider, FallbackDecider)
    assert isinstance(decider.primary, SystemOneDecider)
    assert decider.primary.backend == "kev"
    assert decider.primary.model == "kev-latest"
