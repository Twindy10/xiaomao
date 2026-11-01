"""问题翻译成英文，作为附加改写去召回英文合规资料。

合规库大多是英国、欧盟、美国政府网站的英文页面。中文问题配英文文档时，BM25 基本匹配不上，
dense 也会被中文的平台政策片段挤下去（M3 报告第 2 节）。离线实验里，加一路英文译文、合并后统一用
原问题 rerank，Hit@5 从 87.7% 升到 90.6%，多证据 Recall@5 从 33% 升到 42%（M4 报告第 6 节）。
"""

from __future__ import annotations

TRANSLATE_SYSTEM = (
    "把跨境电商卖家的中文问题翻译成英文，用于检索英国、欧盟、美国政府网站上的英文法规和指南。"
    "只翻译，不扩写，不要加原问题没有的法规、术语或条件；平台名、法规缩写、标准编号、型号保持原样。"
    "只输出 JSON：{\"en\": \"...\"}。"
)


class Translator:
    """用独立的 LLM 客户端翻译，方便和决策调用并发、单独统计 tokens。"""

    def __init__(self, client=None) -> None:
        if client is None:
            from xiaomao.llm import LLMClient

            client = LLMClient()
        self.client = client
        self.last_tokens = 0

    def __call__(self, query: str) -> str:
        """返回英文译文；失败时返回空串（只是少一路召回，不影响主流程）。"""
        usage = getattr(self.client, "usage", None)
        before = int(getattr(usage, "total_tokens", 0) or 0)
        try:
            payload = self.client.chat_json(f"中文问题：{query}", TRANSLATE_SYSTEM, max_tokens=200, temperature=0.0)
            text = str(payload.get("en") or "").strip()
        except Exception:
            text = ""
        self.last_tokens = int(getattr(usage, "total_tokens", 0) or 0) - before
        return text
