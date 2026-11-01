"""多轮改写：把带指代的问题补成能独立看懂的问题。"""

from __future__ import annotations

SYSTEM_PROMPT = (
    "你是对话助手，负责把用户当前的问题改写成能独立看懂的问题。"
    "只补全指代和省略（平台、站点、产品），不改变问题意图，不回答问题。"
    "当前问题本来就能独立看懂时原样返回。只输出 JSON：{\"question\": \"...\"}。"
)

RECENT_MESSAGES = 6
ASSISTANT_TRUNCATE = 300


class Rewriter:
    """用独立的 LLM 客户端做改写，tokens 单独统计。"""

    def __init__(self, client=None) -> None:
        if client is None:
            from xiaomao.llm import LLMClient

            client = LLMClient()
        self.client = client
        self.last_tokens = 0

    def __call__(self, query: str, history: list[dict]) -> str:
        """没有历史就不调模型；出错退回原问题。"""
        self.last_tokens = 0
        if not history:
            return query

        recent = history[-RECENT_MESSAGES:]
        lines = []
        for message in recent:
            role = "用户" if message.get("role") == "user" else "助手"
            content = str(message.get("content") or "")
            if role == "助手":
                content = content[:ASSISTANT_TRUNCATE]
            lines.append(f"{role}：{content}")
        prompt = (
            "对话历史：\n" + "\n".join(lines) + f"\n\n当前问题：{query}\n\n请输出改写后的问题。"
        )

        before = int(getattr(getattr(self.client, "usage", None), "total_tokens", 0) or 0)
        try:
            payload = self.client.chat_json(prompt, SYSTEM_PROMPT, max_tokens=200, temperature=0.0)
        except Exception:
            return query
        self.last_tokens = int(getattr(getattr(self.client, "usage", None), "total_tokens", 0) or 0) - before

        rewritten = str(payload.get("question") or "").strip()
        return rewritten or query
