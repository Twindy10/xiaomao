"""探针：检索 Top-5 后，决策模型 / LLM 判断"证据是否足以回答"，在有答案和无答案的题上的区分度。"""
import json, os, random, sys, time
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
for n in ("NO_PROXY", "no_proxy"): os.environ[n] = "127.0.0.1,localhost"
from dotenv import load_dotenv
ROOT = Path(__file__).resolve().parents[2]; load_dotenv(ROOT / ".env", override=True)
import httpx
from pymilvus import MilvusClient
from xiaomao.ingest.embed import Embedder
from xiaomao.retrieval.hybrid import Retriever
from xiaomao.retrieval.rerank import get_reranker
from xiaomao.llm import LLMClient

qa = [json.loads(l) for l in (ROOT / "data/eval/retrieval_qa.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
rng = random.Random(7)
items = [q for q in qa if not q["answerable"]] + rng.sample([q for q in qa if q["answerable"]], 30)
retriever = Retriever(MilvusClient(uri=os.environ["MILVUS_URI"], token=os.environ.get("MILVUS_TOKEN", "")),
                      Embedder(max_length=512))
reranker = get_reranker("api")
dec = httpx.Client(base_url=os.environ["DECIDER_BASE_URL"], timeout=60, trust_env=False)
llm = LLMClient(max_workers=1)

def state(query, top):
    docs = "\n\n".join(f"[{i}] {c.fields['text'][:1200]}" for i, c in enumerate(top, 1))
    return f"用户问题：{query}\n\n检索到的资料：\n{docs}"

out = []
for q in items:
    top = reranker.rerank(q["query"], retriever.search(q["query"]), top_n=5).candidates
    s = state(q["query"], top)
    t = time.perf_counter()
    r = dec.post("/v1/systemone", json={"state": s, "questions": {"enough": {"type": "noul",
        "instructions": "检索到的资料里包含能直接回答用户问题的具体信息（不是只沾边）"}}})
    ms = (time.perf_counter() - t) * 1000
    p = r.json().get("answers", r.json())["enough"]["noul"] if r.status_code == 200 else None
    j = llm.chat_json(s + "\n\n这些资料能否直接回答用户问题？只输出 JSON：{\"enough\": true/false}", system="你是跨境电商问答系统里的证据审核员。", max_tokens=50, temperature=0.0)
    out.append({"answerable": q["answerable"], "p": p, "llm": bool(j.get("enough")), "ms": ms, "status": r.status_code,
                "top_score": top[0].score if top else 0.0, "chars": len(s)})
print("状态码", {o["status"] for o in out}, "state 字符数中位数", sorted(o["chars"] for o in out)[len(out)//2],
      "决策延迟中位数", round(sorted(o["ms"] for o in out)[len(out)//2]), "ms")
A = [o for o in out if o["answerable"]]; U = [o for o in out if not o["answerable"]]
for th in (0.3, 0.5, 0.7):
    print(f"0.8B p≥{th}：有答案判充分 {sum(o['p']>=th for o in A)}/{len(A)}，无答案判不充分 {sum(o['p']<th for o in U)}/{len(U)}")
print(f"LLM：有答案判充分 {sum(o['llm'] for o in A)}/{len(A)}，无答案判不充分 {sum(not o['llm'] for o in U)}/{len(U)}")
for th in (0.3, 0.5, 0.7):
    print(f"rerank 第 1 名分数 ≥{th}：有答案 {sum(o['top_score']>=th for o in A)}/{len(A)}，无答案 <{th} {sum(o['top_score']<th for o in U)}/{len(U)}")
