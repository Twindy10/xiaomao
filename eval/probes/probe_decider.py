"""M4 设计前的探针脚本（2026-10-31 跑过，结果写在 docs/deepseek/M4_tasks.md 开头）。
探针：Jev-Style-0.8B 在意图标注集上的准确率、置信度分布和延迟；可选再跑 LLM JSON 做对比。"""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env", override=True)
OUT = ROOT / "eval/runs/probe_decider.jsonl"

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
    "US": "美国站", "UK": "英国站", "DE": "德国站", "JP": "日本站",
    "FR": "法国站", "IT": "意大利站", "ES": "西班牙站", "EU": "欧盟整体",
    "none": "没有提到任何国家或站点",
}


def ask(client: httpx.Client, query: str) -> tuple[dict, float]:
    body = {
        "state": query,
        "questions": {
            "intent": {"type": "choice", "instructions": "这个跨境电商卖家的问题属于哪一类？", "criteria": INTENTS},
            "strategy": {"type": "choice", "instructions": "检索这个问题该用哪种策略？", "criteria": STRATEGIES},
            "platform": {"type": "choice", "instructions": "问题里明确提到了哪个电商平台？", "criteria": PLATFORMS},
            "site": {"type": "choice", "instructions": "问题里明确提到了哪个国家或站点？", "criteria": SITES},
        },
    }
    started = time.perf_counter()
    resp = client.post("/v1/systemone", json=body)
    resp.raise_for_status()
    data = resp.json()
    return data.get("answers", data), (time.perf_counter() - started) * 1000


def main() -> None:
    rows = [json.loads(l) for l in (ROOT / "data/eval/intent.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    client = httpx.Client(base_url=os.environ.get("DECIDER_BASE_URL", "http://127.0.0.1:8765"), timeout=60, trust_env=False)
    ask(client, "你好")  # 预热
    results = []
    for row in rows:
        answers, ms = ask(client, row["query"])
        results.append({"id": row["id"], "query": row["query"], "ms": ms,
                        "gold": {k: row[k] for k in ("intent", "strategy", "platform", "site")},
                        "pred": {k: {"choice": v["choice"], "confidence": v["confidence"]} for k, v in answers.items()}})
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results), encoding="utf-8")

    lat = sorted(r["ms"] for r in results)
    print(f"{len(results)} 条，延迟 P50 {lat[len(lat) // 2]:.0f}ms P95 {lat[int(len(lat) * 0.95)]:.0f}ms")
    for key in ("intent", "strategy", "platform", "site"):
        subset = [r for r in results if not (key == "strategy" and r["gold"]["strategy"] is None)]
        norm = lambda v: "none" if v is None else v
        ok = [norm(r["gold"][key]) == r["pred"][key]["choice"] for r in subset]
        line = f"{key:<9} 准确率 {sum(ok) / len(ok):.1%}（{len(ok)} 条）"
        for th in (0.3, 0.5, 0.7):
            kept = [o for r, o in zip(subset, ok) if r["pred"][key]["confidence"] >= th]
            if kept:
                line += f" | conf≥{th}: 覆盖 {len(kept) / len(ok):.0%}、准确 {sum(kept) / len(kept):.1%}"
        print(line)
    # 意图混淆
    from collections import Counter
    confusion = Counter((r["gold"]["intent"], r["pred"]["intent"]["choice"]) for r in results
                        if r["gold"]["intent"] != r["pred"]["intent"]["choice"])
    print("意图错得最多的：", confusion.most_common(8))


if __name__ == "__main__":
    main()
