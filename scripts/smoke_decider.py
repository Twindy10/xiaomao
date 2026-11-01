"""M0 决策模型冒烟测试：对本地 Jev 式服务发几个跨境电商问题，看判断是否靠谱、单次多快。

用法：先启动 jev-style serve，再运行 python scripts/smoke_decider.py
"""

import os
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

INTENTS = {
    "platform_policy": "平台规则：入驻、listing、FBA、费用、店铺政策",
    "compliance": "合规认证：CE、FCC、PSE、GPSR、EPR 等产品认证与法规",
    "logistics": "物流：头程、海外仓、带电/液体运输限制",
    "tax": "税务：VAT、销售税、关税",
    "ops": "运营技巧：选品、定价、广告、利润测算",
    "account": "账号问题：封号、申诉、侵权投诉",
    "chitchat": "闲聊或问候",
    "out_of_scope": "和跨境电商无关的问题",
}
STRATEGIES = {
    "direct": "问题明确、只问一个点，直接检索即可",
    "subquery": "一句话里问了多个不同的点，需要拆成子问题",
    "hyde": "描述了一个复杂情况或问题很模糊，需要先推测答案再检索",
}

CASES = [
    ("Amazon 美国站 FBA 单件重量上限是多少？", "platform_policy", "direct"),
    ("德国站卖蓝牙耳机需要哪些认证？VAT 怎么注册？EPR 包装法要做什么？", "compliance", "subquery"),
    ("我的 listing 被投诉侵犯外观专利下架了，现在该怎么办？", "account", "hyde"),
    ("带锂电池的产品发美国海外仓有什么限制？", "logistics", None),
    ("你好呀", "chitchat", None),
    ("今天北京天气怎么样？", "out_of_scope", None),
]


def main() -> None:
    base_url = os.environ.get("DECIDER_BASE_URL", "http://127.0.0.1:8765")
    # 本地服务绕过系统代理
    client = httpx.Client(base_url=base_url, timeout=60, trust_env=False)
    hits = {"intent": 0, "strategy": 0}
    total_strategy = 0
    latencies = []
    for question, want_intent, want_strategy in CASES:
        body = {
            "state": question,
            "questions": {
                "intent": {"type": "choice", "instructions": "这个卖家问题属于哪一类？", "criteria": INTENTS},
                "strategy": {"type": "choice", "instructions": "检索这个问题该用哪种策略？", "criteria": STRATEGIES},
                "about_amazon": {"type": "noul", "instructions": "这个问题明确提到了 Amazon 平台"},
            },
        }
        started = time.perf_counter()
        resp = client.post("/v1/systemone", json=body)
        latencies.append((time.perf_counter() - started) * 1000)
        resp.raise_for_status()
        data = resp.json()
        answers = data.get("answers", data)
        intent, strategy = answers["intent"], answers["strategy"]
        hits["intent"] += intent["choice"] == want_intent
        if want_strategy:
            total_strategy += 1
            hits["strategy"] += strategy["choice"] == want_strategy
        mark_i = "✓" if intent["choice"] == want_intent else "✗"
        mark_s = "" if not want_strategy else ("✓" if strategy["choice"] == want_strategy else "✗")
        print(
            f"{latencies[-1]:6.0f}ms  意图 {intent['choice']:<15}{mark_i} ({intent['confidence']:.2f})  "
            f"策略 {strategy['choice']:<9}{mark_s:<2}  Amazon={answers['about_amazon']['noul']:.2f}  {question}"
        )
    warm = sorted(latencies[1:]) if len(latencies) > 1 else latencies
    print(
        f"\n意图 {hits['intent']}/{len(CASES)}，策略 {hits['strategy']}/{total_strategy}；"
        f"首次 {latencies[0]:.0f}ms，之后中位数 {warm[len(warm) // 2]:.0f}ms"
    )


if __name__ == "__main__":
    main()
