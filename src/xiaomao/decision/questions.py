"""决策问题定义：选项写法在 326 条标注集上实测过，原样使用。"""

from __future__ import annotations

from xiaomao.decision.base import Question

INTENTS = {
    "platform_policy": "平台规则：入驻开店、listing 上架、FBA 规则与费用、店铺政策",
    "compliance": "合规认证：CE、FCC、PSE、UKCA、GPSR、EPR、WEEE、电池法等产品认证与法规",
    "logistics": "物流：头程、海外仓、清关、带电/液体等运输限制",
    "tax": "税务：VAT、OSS、销售税、关税、税号注册与申报",
    "ops": "运营技巧：选品、定价、广告、促销、利润测算",
    "account": "账号问题：注册审核、封号、申诉、侵权投诉、账户健康",
    "chitchat": "闲聊、问候、感谢，和业务无关的寒暄",
    "out_of_scope": "和跨境电商无关的问题（天气、编程、娱乐等）",
}

STRATEGIES = {
    "direct": "问题明确、只问一个点，直接检索即可",
    "subquery": "一句话里问了两个或更多不同的点，需要拆成子问题分别检索",
    "hyde": "描述了一个复杂情况，或者问得很模糊、口语化，需要先推测答案再检索",
}

PLATFORMS = {
    "amazon": "明确提到亚马逊 / Amazon",
    "temu": "明确提到 Temu",
    "tiktok_shop": "明确提到 TikTok Shop / 抖音海外",
    "shopee": "明确提到 Shopee / 虾皮",
    "none": "没有提到任何平台",
}

SITES = {
    "US": "美国站", "UK": "英国站", "DE": "德国站", "JP": "日本站", "FR": "法国站",
    "IT": "意大利站", "ES": "西班牙站", "EU": "欧盟整体", "none": "没有提到任何国家或站点",
}

INTENT_QUESTION = Question(
    name="intent",
    type="choice",
    instructions="这个跨境电商卖家的问题属于哪一类？",
    criteria=INTENTS,
)

STRATEGY_QUESTION = Question(
    name="strategy",
    type="choice",
    instructions="检索这个问题该用哪种策略？",
    criteria=STRATEGIES,
)

PLATFORM_QUESTION = Question(
    name="platform",
    type="choice",
    instructions="问题里明确提到了哪个电商平台？",
    criteria=PLATFORMS,
)

SITE_QUESTION = Question(
    name="site",
    type="choice",
    instructions="问题里明确提到了哪个国家或站点？",
    criteria=SITES,
)

EVIDENCE_QUESTION = Question(
    name="evidence",
    type="noul",
    instructions="检索到的资料里包含能直接回答用户问题的具体信息（不是只沾边）",
)

ROUTING_QUESTIONS: list[Question] = [
    INTENT_QUESTION, STRATEGY_QUESTION, PLATFORM_QUESTION, SITE_QUESTION,
]
