"""用生成模型输出 JSON 的兜底决策后端。"""

from __future__ import annotations

import json

from xiaomao.decision.base import Answer, DeciderError, Question

SYSTEM_PROMPT = (
    "你是一个分类器。只回答被问到的问题，严格按要求的 JSON 结构输出，"
    "不要解释，不要输出 JSON 之外的内容。"
)


class LLMJsonDecider:
    """一次请求回答全部问题，靠文字 prompt 约束格式。"""

    backend = "llm_json"

    def __init__(self, client=None, model: str | None = None) -> None:
        if client is None:
            from xiaomao.llm import LLMClient

            client = LLMClient()
        self.client = client
        if model:
            self.client.model = model
        self.last_tokens = 0

    def decide(self, state: str, questions: list[Question]) -> dict[str, Answer]:
        """调模型并校验返回值。"""
        prompt = self._build_prompt(state, questions)
        before = getattr(self.client.usage, "total_tokens", 0)
        try:
            payload = self.client.chat_json(prompt, SYSTEM_PROMPT, max_tokens=300, temperature=0.0)
        except Exception as exc:
            raise DeciderError(f"{type(exc).__name__}: {exc}") from exc
        self.last_tokens = getattr(self.client.usage, "total_tokens", 0) - before

        answers: dict[str, Answer] = {}
        for question in questions:
            if question.name not in payload:
                raise DeciderError(f"缺少问题 {question.name} 的答案")
            answers[question.name] = self._parse(question, payload[question.name])
        return answers

    @staticmethod
    def _build_prompt(state: str, questions: list[Question]) -> str:
        """把问题列表写成 prompt。"""
        lines = [f"用户问题：{state}", "", "请回答下面的问题，只输出一个 JSON 对象，键是问题名，值是答案："]
        for index, question in enumerate(questions, start=1):
            lines.append(f"{index}. {question.name}（{question.type}）：{question.instructions}")
            if question.type == "choice" and isinstance(question.criteria, dict):
                options = "；".join(f"{key}={label}" for key, label in question.criteria.items())
                lines.append(f"   只能从这些选项里选一个：{options}")
            elif question.type == "noul":
                lines.append("   输出 0 到 1 之间的概率（小数）")
            elif question.type == "score" and isinstance(question.criteria, list):
                options = "；".join(f"{index}={label}" for index, label in enumerate(question.criteria))
                lines.append(f"   等级：{options}；输出等级序号")
        lines.append("")
        lines.append('示例：{"intent": "compliance", "evidence": 0.8}')
        return "\n".join(lines)

    def _parse(self, question: Question, value) -> Answer:
        """校验取值合法性。"""
        if question.type == "choice":
            text = str(value)
            if not isinstance(question.criteria, dict) or text not in question.criteria:
                raise DeciderError(f"{question.name} 的取值不在选项里：{value!r}")
            return Answer(
                value=text,
                probabilities={text: 1.0},
                confidence=1.0,
                backend=self.backend,
            )
        if question.type == "noul":
            try:
                probability = float(value)
            except (TypeError, ValueError) as exc:
                raise DeciderError(f"{question.name} 不是数字：{value!r}") from exc
            if not 0.0 <= probability <= 1.0:
                raise DeciderError(f"{question.name} 超出 0–1：{probability}")
            return Answer(
                value=probability,
                probabilities={"yes": probability, "no": 1 - probability},
                confidence=abs(2 * probability - 1),
                backend=self.backend,
            )
        if question.type == "score":
            try:
                score = float(value)
            except (TypeError, ValueError) as exc:
                raise DeciderError(f"{question.name} 不是数字：{value!r}") from exc
            return Answer(value=score, probabilities={}, confidence=1.0, backend=self.backend)
        raise DeciderError(f"未知的问题类型：{question.type}")
