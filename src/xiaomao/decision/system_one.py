"""通用 `/v1/systemone` 后端：Kev、laya-multilingual 等。

三家的 confidence 口径不同，所以一律用返回的 probabilities 按统一口径重算：
choice / score 取第一名减第二名，noul 取 |2p − 1|。
"""

from __future__ import annotations

import time

import httpx

from xiaomao.decision.base import Answer, DeciderError, Question

ENDPOINT = "/v1/systemone"


def normalize(probabilities: dict[str, float]) -> dict[str, float]:
    """概率可能被四舍五入，加起来不等于 1，先归一化。"""
    total = sum(value for value in probabilities.values() if value > 0)
    if total <= 0:
        return {}
    return {key: value / total for key, value in probabilities.items() if value > 0}


def top_margin(probabilities: dict[str, float]) -> float:
    """第一名减第二名；只有一个选项时为 1.0。"""
    ordered = sorted(probabilities.values(), reverse=True)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return 1.0
    return max(0.0, ordered[0] - ordered[1])


class SystemOneDecider:
    """和后端无关的 systemone 客户端。"""

    def __init__(
        self,
        base_url: str,
        *,
        backend: str,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = 5.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.backend = backend
        self.model = model
        self.api_key = api_key
        self.client = client or httpx.Client(timeout=timeout, trust_env=False)
        self.last_latency_ms = 0.0
        self.last_usage: dict = {}

    def decide(self, state: str, questions: list[Question]) -> dict[str, Answer]:
        """一次请求问完所有问题。"""
        body: dict = {
            "state": state,
            "questions": {
                question.name: self._question_payload(question) for question in questions
            },
        }
        if self.model:
            body["model"] = self.model
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else None

        started = time.perf_counter()
        try:
            response = self.client.post(self.base_url + ENDPOINT, json=body, headers=headers)
            self.last_latency_ms = (time.perf_counter() - started) * 1000
            if response.status_code != 200:
                raise DeciderError(f"HTTP {response.status_code}")
            payload = response.json()
            raw_answers = payload["answers"]
            self.last_usage = payload.get("usage") or {}
        except DeciderError:
            raise
        except Exception as exc:
            self.last_latency_ms = (time.perf_counter() - started) * 1000
            raise DeciderError(f"{type(exc).__name__}: {exc}") from exc

        answers: dict[str, Answer] = {}
        for question in questions:
            raw = raw_answers.get(question.name)
            if not isinstance(raw, dict):
                raise DeciderError(f"缺少问题 {question.name} 的答案")
            answers[question.name] = self._parse(question, raw)
        return answers

    @staticmethod
    def _question_payload(question: Question) -> dict:
        """noul 不传 criteria；score 的 criteria 是有序数组。"""
        payload: dict = {"type": question.type, "instructions": question.instructions}
        if question.criteria is not None and question.type != "noul":
            payload["criteria"] = question.criteria
        return payload

    def _parse(self, question: Question, raw: dict) -> Answer:
        """只信 probabilities，自己算 confidence。"""
        kind = raw.get("type", question.type)
        if kind in ("choice", "score"):
            probabilities = normalize({
                str(key): float(value) for key, value in (raw.get("probabilities") or {}).items()
            })
            if not probabilities:
                raise DeciderError(f"问题 {question.name} 没有概率")
            value = max(probabilities, key=probabilities.get)
            if kind == "score":
                return Answer(
                    value=float(value), probabilities=probabilities,
                    confidence=top_margin(probabilities), backend=self.backend,
                )
            return Answer(
                value=str(value), probabilities=probabilities,
                confidence=top_margin(probabilities), backend=self.backend,
            )
        if kind == "noul":
            probability = float(raw.get("noul"))
            return Answer(
                value=probability,
                probabilities={"yes": probability, "no": 1 - probability},
                confidence=abs(2 * probability - 1),
                backend=self.backend,
            )
        raise DeciderError(f"未知的问题类型：{kind}")
