"""带引用的回答生成：只依据检索资料，标注 [n] 出处。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from xiaomao.retrieval.search import Candidate

CITATION_RE = re.compile(r"\[(\d+)\]")

SYSTEM_PROMPT = (
    "你是面向中国中小跨境卖家的助手，只能根据给出的资料回答问题。\n"
    "1. 先判断资料能回答问题的哪些部分。资料不足以完整回答时，只回答有依据的部分，"
    "不要用常识、推测或其他平台、站点的规定去补。只有用户明确问到、而资料没有覆盖的部分，"
    "才用一句话说明“资料里没有提到 XX”；用户没问的方面不要提，资料已经答全时不要加这类说明。\n"
    "2. 数字、费用、日期、条件必须和资料原文一致；不同资料、不同站点的数字不要混用。\n"
    "3. 资料写明了适用站点或国家的要说清楚；没写的不要推断适用范围，也不要根据链接域名推断。\n"
    "4. 每条结论后面用 [n] 标注出处，n 是资料的编号；找不到出处的句子删掉。"
    "不要加资料里没有的建议、提醒或补充说明。\n"
    "5. 用中文回答，英文资料要翻成中文要点。"
)

# 面向用户的固定回复：像产品文案一样只说事实和下一步，不解释系统多可靠。
# 前端按拒答文案的开头识别历史消息（web/src/lib/chat.ts 的 REFUSAL_HINT），改开头要同步
REFUSAL_TEMPLATE = "暂未找到与该问题相关的资料（检索范围：{scope}）。可补充平台、站点或产品信息后重试。"

CHITCHAT_REPLY = "你好，有什么可以帮你？可咨询平台规则、产品认证与合规、物流、税务、运营和账号相关问题。"

OUT_OF_SCOPE_REPLY = "该问题不在支持范围内。目前支持跨境电商相关问题，如平台规则、合规认证、物流、税务和运营。"

PLATFORM_NAMES = {"amazon": "亚马逊", "temu": "Temu", "tiktok_shop": "TikTok Shop", "shopee": "Shopee"}
SITE_NAMES = {
    "US": "美国站", "UK": "英国站", "DE": "德国站", "JP": "日本站", "FR": "法国站",
    "IT": "意大利站", "ES": "西班牙站", "EU": "欧盟",
}


@dataclass
class GeneratedAnswer:
    """一次生成的结果。"""

    text: str
    citations: list[dict] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0


def build_messages(query: str, candidates: list[Candidate]) -> list[dict]:
    """构造生成用的消息。"""
    lines = [f"用户问题：{query}", "", "资料："]
    for index, candidate in enumerate(candidates, start=1):
        fields = candidate.fields
        title_path = str(fields.get("title_path") or "").strip()
        text = str(fields.get("text") or "").strip()
        source = str(fields.get("source_url") or "")
        effective = fields.get("effective_date")
        effective_text = f"，生效日期：{effective}" if effective not in (None, "", 0) else ""
        lines.append(f"[{index}] {title_path}")
        lines.append(text)
        lines.append(f"来源：{source}{effective_text}")
        lines.append("")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(lines)},
    ]


def parse_citations(text: str, candidates: list[Candidate]) -> tuple[str, list[dict]]:
    """解析正文里的 [n]：越界的从正文删掉，只保留真正被引用的资料。"""
    used: list[int] = []

    def replace_match(match: re.Match) -> str:
        index = int(match.group(1))
        if 1 <= index <= len(candidates):
            if index not in used:
                used.append(index)
            return match.group(0)
        return ""

    cleaned = CITATION_RE.sub(replace_match, text)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()
    citations = [
        {
            "n": index,
            "chunk_id": candidates[index - 1].chunk_id,
            "collection": candidates[index - 1].collection,
            "source_url": candidates[index - 1].fields.get("source_url"),
            "title_path": candidates[index - 1].fields.get("title_path"),
            "effective_date": candidates[index - 1].fields.get("effective_date"),
            "rank": candidates[index - 1].rank,
            "doc_id": candidates[index - 1].fields.get("doc_id"),
        }
        for index in sorted(used)
    ]
    return cleaned, citations


def generate(generator, query: str, candidates: list[Candidate]) -> GeneratedAnswer:
    """调生成模型并解析引用。"""
    result = generator.complete(build_messages(query, candidates), max_tokens=1200, temperature=0.3)
    text, citations = parse_citations(result.text, candidates)
    return GeneratedAnswer(
        text=text,
        citations=citations,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
    )


def refusal_text(platform: str | None, site: str | None) -> str:
    """拒答文案，说明检索范围。"""
    scope = []
    if platform:
        scope.append(PLATFORM_NAMES.get(platform, platform))
    if site:
        scope.append(SITE_NAMES.get(site, site))
    return REFUSAL_TEMPLATE.format(scope=" · ".join(scope) if scope else "全部平台和站点")
