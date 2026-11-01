"""多轮改写测试。"""

from __future__ import annotations

from xiaomao.router.rewrite import Rewriter


class FakeUsage:
    """假 usage。"""

    def __init__(self) -> None:
        self.total_tokens = 0


class FakeLLM:
    """返回预设 JSON。"""

    def __init__(self, payload: dict | None = None, error: Exception | None = None) -> None:
        self.payload = payload or {"question": "法国站蓝牙耳机需要哪些认证"}
        self.error = error
        self.prompts: list[str] = []
        self.usage = FakeUsage()

    def chat_json(self, prompt, system, max_tokens=3000, temperature=0.7):
        """记录调用并返回。"""
        self.prompts.append(prompt)
        self.usage.total_tokens += 60
        if self.error:
            raise self.error
        return self.payload


def test_no_history_skips_model() -> None:
    """没有历史就不调模型。"""
    client = FakeLLM()
    rewriter = Rewriter(client=client)
    assert rewriter("德国站耳机要认证吗", []) == "德国站耳机要认证吗"
    assert client.prompts == []


def test_history_triggers_rewrite() -> None:
    """有历史时用模型结果，并记录 tokens。"""
    client = FakeLLM()
    rewriter = Rewriter(client=client)
    history = [
        {"role": "user", "content": "德国站蓝牙耳机需要哪些认证"},
        {"role": "assistant", "content": "需要 CE-RED。" * 100},
    ]

    assert rewriter("那法国站呢", history) == "法国站蓝牙耳机需要哪些认证"
    assert rewriter.last_tokens == 60
    assert len(client.prompts) == 1


def test_error_returns_original() -> None:
    """出错退回原问题。"""
    rewriter = Rewriter(client=FakeLLM(error=RuntimeError("boom")))
    history = [{"role": "user", "content": "德国站耳机"}]
    assert rewriter("那法国站呢", history) == "那法国站呢"


def test_empty_answer_returns_original() -> None:
    """模型返回空字符串时用原问题。"""
    rewriter = Rewriter(client=FakeLLM(payload={"question": "  "}))
    assert rewriter("那法国站呢", [{"role": "user", "content": "x"}]) == "那法国站呢"
