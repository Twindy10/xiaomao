"""本地 jev-style 决策模型后端。"""

from __future__ import annotations

import os

import httpx

from xiaomao.decision.base import Answer, DeciderError, Question

ENDPOINT = "/v1/systemone"


class JevStyleDecider:
    """调 `jev-style serve` 的 /v1/systemone。"""

    backend = "jev_style"

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = 5.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = (base_url or os.environ.get("DECIDER_BASE_URL") or "").rstrip("/")
        if not self.base_url:
            raise RuntimeError("DECIDER_BASE_URL 未配置")
        self.client = client or httpx.Client(timeout=timeout, trust_env=False)

    def decide(self, state: str, questions: list[Question]) -> dict[str, Answer]:
        """一次请求问完所有问题。"""
        body = {
            "state": state,
            "questions": {
                question.name: self._question_payload(question) for question in questions
            },
        }
        try:
            response = self.client.post(self.base_url + ENDPOINT, json=body)
            if response.status_code != 200:
                raise DeciderError(f"HTTP {response.status_code}")
            payload = response.json()
            raw_answers = payload["answers"]
        except DeciderError:
            raise
        except Exception as exc:
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
        """noul 不传 criteria；score 的 criteria 必须是有序数组。"""
        payload: dict = {"type": question.type, "instructions": question.instructions}
        if question.criteria is not None and question.type != "noul":
            payload["criteria"] = question.criteria
        return payload

    def _parse(self, question: Question, raw: dict) -> Answer:
        """把服务返回转成统一 Answer。"""
        kind = raw.get("type", question.type)
        if kind == "choice":
            probabilities = {str(key): float(value) for key, value in (raw.get("probabilities") or {}).items()}
            return Answer(
                value=str(raw["choice"]),
                probabilities=probabilities,
                confidence=float(raw.get("confidence", 0.0)),
                backend=self.backend,
            )
        if kind == "noul":
            probability = float(raw["noul"])
            return Answer(
                value=probability,
                probabilities={"yes": probability, "no": 1 - probability},
                confidence=abs(2 * probability - 1),
                backend=self.backend,
            )
        if kind == "score":
            probabilities = {str(key): float(value) for key, value in (raw.get("probabilities") or {}).items()}
            return Answer(
                value=float(raw["score"]),
                probabilities=probabilities,
                confidence=float(raw.get("confidence", 0.0)),
                backend=self.backend,
            )
        raise DeciderError(f"未知的问题类型：{kind}")
