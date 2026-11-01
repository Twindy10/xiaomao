"""决策层测试：jev-style 解析、LLM JSON、置信度兜底。"""

from __future__ import annotations

import json

import httpx
import pytest

from xiaomao.decision import (
    Answer,
    DeciderError,
    FallbackDecider,
    JevStyleDecider,
    LLMJsonDecider,
    Question,
)
from xiaomao.decision.questions import EVIDENCE_QUESTION, INTENT_QUESTION, ROUTING_QUESTIONS

SAMPLE = {
    "answers": {
        "intent": {"type": "choice", "choice": "compliance", "confidence": 0.98,
                   "probabilities": {"compliance": 0.99, "logistics": 0.01}},
        "strategy": {"type": "choice", "choice": "direct", "confidence": 0.42,
                     "probabilities": {"direct": 0.5, "subquery": 0.3, "hyde": 0.2}},
        "platform": {"type": "choice", "choice": "amazon", "confidence": 0.9,
                     "probabilities": {"amazon": 0.9}},
        "site": {"type": "choice", "choice": "DE", "confidence": 0.8,
                 "probabilities": {"DE": 0.8, "none": 0.2}},
        "evidence": {"type": "noul", "noul": 0.82},
        "level": {"type": "score", "score": 0.78, "confidence": 0.5,
                  "legend": {"0": "不急", "1": "一般", "2": "很急"},
                  "probabilities": {"0": 0.28, "1": 0.66, "2": 0.06}},
    },
    "usage": {"input_tokens": 190},
    "latency_ms": 297.6,
}


def test_jev_style_request_and_parse() -> None:
    """请求体里 noul 不带 criteria、score 的 criteria 是数组；三种类型都解析对。"""
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.read()))
        return httpx.Response(200, json=SAMPLE)

    client = httpx.Client(transport=httpx.MockTransport(handler), trust_env=False)
    decider = JevStyleDecider(base_url="http://127.0.0.1:8765", client=client)

    level = Question(name="level", type="score", instructions="有多急？", criteria=["不急", "一般", "很急"])
    questions = [*ROUTING_QUESTIONS, EVIDENCE_QUESTION, level]
    answers = decider.decide("德国站耳机要认证吗", questions)

    assert captured["state"] == "德国站耳机要认证吗"
    assert captured["questions"]["intent"]["type"] == "choice"
    assert "criteria" not in captured["questions"]["evidence"]  # noul
    assert captured["questions"]["site"]["criteria"]["DE"] == "德国站"
    assert captured["questions"]["level"]["criteria"] == ["不急", "一般", "很急"]

    assert answers["intent"].value == "compliance"
    assert answers["intent"].confidence == pytest.approx(0.98)
    assert answers["site"].value == "DE"
    assert answers["evidence"].value == pytest.approx(0.82)
    assert answers["evidence"].probabilities == {"yes": 0.82, "no": pytest.approx(0.18)}
    assert answers["evidence"].confidence == pytest.approx(0.64)
    assert answers["level"].value == pytest.approx(0.78)


def test_jev_style_http_error_raises() -> None:
    """非 200 抛 DeciderError。"""
    client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(500)), trust_env=False)
    decider = JevStyleDecider(base_url="http://127.0.0.1:8765", client=client)
    with pytest.raises(DeciderError):
        decider.decide("问题", [INTENT_QUESTION])


def test_jev_style_timeout_raises() -> None:
    """超时抛 DeciderError。"""
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timeout", request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler), trust_env=False)
    decider = JevStyleDecider(base_url="http://127.0.0.1:8765", client=client)
    with pytest.raises(DeciderError):
        decider.decide("问题", [INTENT_QUESTION])


class FakeUsage:
    """假 usage。"""

    def __init__(self) -> None:
        self.total_tokens = 0


class FakeLLM:
    """按顺序返回预设 JSON。"""

    def __init__(self, payloads: list[dict]) -> None:
        self.payloads = list(payloads)
        self.prompts: list[str] = []
        self.usage = FakeUsage()
        self.model = "fake"

    def chat_json(self, prompt, system, max_tokens=3000, temperature=0.7):
        """返回下一个预设结果。"""
        self.prompts.append(prompt)
        self.usage.total_tokens += 100
        return self.payloads.pop(0)


def test_llm_json_valid_payload() -> None:
    """合法的 JSON 能解析，noul 置信度用 |2p-1|。"""
    client = FakeLLM([{
        "intent": "compliance", "strategy": "direct", "platform": "none",
        "site": "DE", "evidence": 0.8,
    }])
    decider = LLMJsonDecider(client=client)
    answers = decider.decide("德国站耳机认证", [*ROUTING_QUESTIONS, EVIDENCE_QUESTION])

    assert answers["intent"].value == "compliance"
    assert answers["intent"].probabilities == {"compliance": 1.0}
    assert answers["site"].value == "DE"
    assert answers["evidence"].confidence == pytest.approx(0.6)
    assert decider.last_tokens == 100


def test_llm_json_invalid_choice_raises() -> None:
    """选项不在枚举里抛 DeciderError。"""
    client = FakeLLM([{"intent": "shipping", "strategy": "direct", "platform": "none", "site": "none"}])
    with pytest.raises(DeciderError):
        LLMJsonDecider(client=client).decide("问题", ROUTING_QUESTIONS)


def test_llm_json_invalid_number_raises() -> None:
    """noul 不是数字抛错。"""
    client = FakeLLM([{"intent": "compliance", "strategy": "direct", "platform": "none",
                       "site": "none", "evidence": "很高"}])
    with pytest.raises(DeciderError):
        LLMJsonDecider(client=client).decide("问题", [*ROUTING_QUESTIONS, EVIDENCE_QUESTION])


class StubDecider:
    """可编排的假决策器。"""

    def __init__(self, answers: dict[str, Answer] | None = None, error: Exception | None = None) -> None:
        self.answers = answers or {}
        self.error = error
        self.calls: list[list[str]] = []

    def decide(self, state, questions):
        """返回预设答案或抛错。"""
        self.calls.append([question.name for question in questions])
        if self.error is not None:
            raise self.error
        return {question.name: self.answers[question.name] for question in questions}


def make_answers(confidence: float, value: str = "compliance") -> dict[str, Answer]:
    """构造四个问题的答案。"""
    return {
        "intent": Answer(value, {value: 1.0}, confidence, "primary"),
        "strategy": Answer("direct", {"direct": 1.0}, confidence, "primary"),
        "platform": Answer("none", {"none": 1.0}, confidence, "primary"),
        "site": Answer("none", {"none": 1.0}, confidence, "primary"),
    }


def test_fallback_all_on_primary_error() -> None:
    """主后端出错时全部交给兜底。"""
    primary = StubDecider(error=DeciderError("挂了"))
    fallback = StubDecider(make_answers(1.0))
    answers = FallbackDecider(primary, fallback).decide("问题", ROUTING_QUESTIONS)

    assert all(answer.fell_back for answer in answers.values())
    assert fallback.calls == [["intent", "strategy", "platform", "site"]]


def test_fallback_only_low_confidence_questions() -> None:
    """只有低置信度的问题合并成一次兜底请求。"""
    primary = StubDecider(make_answers(0.2))
    fallback = StubDecider(make_answers(1.0))
    answers = FallbackDecider(primary, fallback).decide("问题", ROUTING_QUESTIONS)

    # strategy 的阈值是 0，不兜底
    assert fallback.calls == [["intent", "platform", "site"]]
    assert answers["intent"].fell_back is True
    assert answers["strategy"].fell_back is False


def test_fallback_threshold_zero_never_falls_back() -> None:
    """阈值为 0 的问题从不兜底。"""
    primary = StubDecider(make_answers(0.0))
    fallback = StubDecider(make_answers(1.0))
    answers = FallbackDecider(primary, fallback, thresholds={"strategy": 0.0, "intent": 0.0}).decide(
        "问题", ROUTING_QUESTIONS
    )

    assert "intent" not in fallback.calls[0]
    assert answers["intent"].fell_back is False


def test_fallback_raises_when_both_fail() -> None:
    """兜底也失败时抛错。"""
    primary = StubDecider(error=DeciderError("挂了"))
    fallback = StubDecider(error=DeciderError("也挂了"))
    with pytest.raises(DeciderError):
        FallbackDecider(primary, fallback).decide("问题", ROUTING_QUESTIONS)


def test_fallback_last_tokens_reset_when_not_used() -> None:
    """上一次兜底的 tokens 不能算到下一次没兜底的调用上。"""
    primary = StubDecider(make_answers(0.2))
    fallback = StubDecider(make_answers(1.0))
    fallback.last_tokens = 300
    decider = FallbackDecider(primary, fallback)
    decider.decide("问题", ROUTING_QUESTIONS)
    assert decider.last_tokens == 300

    primary.answers = make_answers(0.9)
    decider.decide("问题", ROUTING_QUESTIONS)
    assert decider.last_tokens == 0
