"""对比：生成模型输出 JSON 做同样四个判断。"""
import json, sys, time
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))
from probe_decider import INTENTS, STRATEGIES, PLATFORMS, SITES, ROOT
from xiaomao.llm import LLMClient

def opts(d): return "\n".join(f"- {k}：{v}" for k, v in d.items())
PROMPT = """判断下面这个跨境电商卖家问题，只输出 JSON：{{"intent": ..., "strategy": ..., "platform": ..., "site": ...}}

intent（问题属于哪一类）：
{i}

strategy（检索策略；intent 是 chitchat 或 out_of_scope 时也随便选一个）：
{s}

platform（问题里明确提到的平台，没提就 none，不要猜）：
{p}

site（问题里明确提到的国家或站点，没提就 none，不要猜）：
{t}

问题：{q}"""

rows = [json.loads(l) for l in (ROOT / "data/eval/intent.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
client = LLMClient(max_workers=4)
def one(row):
    t = time.perf_counter()
    out = client.chat_json(PROMPT.format(i=opts(INTENTS), s=opts(STRATEGIES), p=opts(PLATFORMS), t=opts(SITES), q=row["query"]),
                           system="你是跨境电商客服系统里的分类器。", max_tokens=200, temperature=0.0)
    return out, (time.perf_counter() - t) * 1000
res = client.map(one, rows)
lat = sorted(ms for _, ms in res)
print(f"{client.model}：延迟 P50 {lat[len(lat)//2]:.0f}ms P95 {lat[int(len(lat)*0.95)]:.0f}ms；{client.usage.summary()}")
for key in ("intent", "strategy", "platform", "site"):
    pairs = [(row, out) for row, (out, _) in zip(rows, res) if not (key == "strategy" and row["strategy"] is None)]
    ok = sum(("none" if row[key] is None else row[key]) == str(out.get(key)) for row, out in pairs)
    print(f"{key:<9} 准确率 {ok/len(pairs):.1%}（{len(pairs)} 条）")
# 0.8B 低置信时交给 LLM 的组合效果
probe = {json.loads(l)["id"]: json.loads(l) for l in (ROOT / "eval/runs/probe_decider.jsonl").read_text(encoding="utf-8").splitlines()}
for key in ("intent", "platform", "site"):
    for th in (0.3, 0.5, 0.7):
        ok = fb = 0
        for row, (out, _) in zip(rows, res):
            p = probe[row["id"]]["pred"][key]
            gold = "none" if row[key] is None else row[key]
            if p["confidence"] >= th:
                ok += p["choice"] == gold
            else:
                fb += 1; ok += str(out.get(key)) == gold
        print(f"  组合 {key} 阈值 {th}：准确率 {ok/len(rows):.1%}，交给 LLM {fb/len(rows):.0%}")
