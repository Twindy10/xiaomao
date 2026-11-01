"""探针：语义缓存。

1. Redis 8 向量索引能不能用：HASH + FLOAT16 向量 + TAG 过滤的 KNN 查询。
2. 阈值 τ：把 138 道有答案的原问题当成"已缓存的问题"，用它们的同义改写去查，
   画 P-R：命中自己原问题的算对，命中别的问题的算错；无答案题的 29 道原问题拿来查，命中就是误命中。
3. （--gate）中间段相似度交给决策模型判断"缓存答案能否回答当前问题"，看能不能把错的拦下来。
   需要决策模型服务，答案取 eval/runs/qa_m4v4/routing.jsonl。
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")
os.environ["NO_PROXY"] = "127.0.0.1,localhost"

from dotenv import load_dotenv

load_dotenv(ROOT / ".env", override=True)
OUT = ROOT / "eval" / "runs" / "probe_cache.jsonl"


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def redis_check(vectors: np.ndarray) -> None:
    """建一个临时索引，写 3 条，按 TAG 过滤做 KNN。"""
    import redis
    from redis.commands.search.field import TagField, TextField, VectorField
    from redis.commands.search.index_definition import IndexDefinition, IndexType
    from redis.commands.search.query import Query

    client = redis.Redis.from_url(os.environ["REDIS_URL"])
    prefix, index = "xm:probe:cache:", "xm:probe:idx"
    try:
        client.ft(index).dropindex(delete_documents=True)
    except Exception:
        pass
    client.ft(index).create_index(
        [TextField("question"), TagField("platform"), TagField("site"),
         VectorField("vec", "HNSW", {"TYPE": "FLOAT16", "DIM": vectors.shape[1], "DISTANCE_METRIC": "COSINE"})],
        definition=IndexDefinition(prefix=[prefix], index_type=IndexType.HASH),
    )
    rows = [("a", "amazon", "US"), ("b", "amazon", "DE"), ("c", "none", "none")]
    for (name, platform, site), vec in zip(rows, vectors[:3]):
        client.hset(prefix + name, mapping={"question": name, "platform": platform, "site": site,
                                            "vec": vec.astype(np.float16).tobytes()})
    query = (Query("(@platform:{amazon} @site:{DE})=>[KNN 1 @vec $v AS dist]")
             .sort_by("dist").return_fields("question", "dist").dialect(2))
    result = client.ft(index).search(query, query_params={"v": vectors[0].astype(np.float16).tobytes()})
    print("Redis KNN + TAG：", [(d.question, round(1 - float(d.dist), 4)) for d in result.docs],
          "（应只返回 b，相似度 = 第 1 条和第 2 条向量的余弦）")
    print("  对照余弦：", round(float(vectors[0] @ vectors[1]), 4))
    client.ft(index).dropindex(delete_documents=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", action="store_true")
    args = parser.parse_args()

    from xiaomao.ingest.embed import Embedder

    items = read_jsonl(ROOT / "data/eval/retrieval_qa.jsonl")
    cached = [i for i in items if i["answerable"]]
    unanswerable = [i for i in items if not i["answerable"]]
    queries = [(i["id"], p, True) for i in cached for p in (i.get("paraphrases") or [])]
    queries += [(i["id"], i["query"], False) for i in unanswerable]

    embedder = Embedder(model_name=os.environ.get("EMBED_MODEL"), max_length=512)
    texts = [i["query"] for i in cached] + [q for _, q, _ in queries]
    dense = np.array([e.dense for e in embedder.encode(texts)], dtype=np.float32)
    dense /= np.linalg.norm(dense, axis=1, keepdims=True)
    cache_vec, query_vec = dense[: len(cached)], dense[len(cached):]

    redis_check(cache_vec)

    # 模拟按平台、站点 TAG 过滤：用标注的平台和站点，只在一致的缓存里找最近的一条
    def tag(item):
        return (item.get("platform") or "none", item.get("site") or "none")

    by_id = {i["id"]: i for i in items}
    cache_tags = [tag(i) for i in cached]
    rows = []
    for (qid, text, answerable), vec in zip(queries, query_vec):
        sims = cache_vec @ vec
        mask = np.array([t == tag(by_id[qid]) for t in cache_tags])
        sims = np.where(mask, sims, -1.0)
        best = int(np.argmax(sims))
        rows.append({"qid": qid, "query": text, "answerable": answerable,
                     "hit_qid": cached[best]["id"], "sim": float(sims[best]),
                     "correct": answerable and cached[best]["id"] == qid})
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")

    para = [r for r in rows if r["answerable"]]
    neg = [r for r in rows if not r["answerable"]]
    print(f"\n同义改写 {len(para)} 条（缓存里有它的原问题），无答案原问题 {len(neg)} 条（缓存里没有）")
    print(f"最近一条就是自己原问题的比例（不设阈值）：{sum(r['correct'] for r in para) / len(para):.1%}")
    print("| τ | 命中率（改写命中缓存） | 命中里的正确率 | 召回（正确命中 / 全部改写） | 无答案题误命中 |")
    print("|---|---|---|---|---|")
    for tau in (0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.96):
        hit = [r for r in para if r["sim"] >= tau]
        ok = sum(r["correct"] for r in hit)
        print(f"| {tau} | {len(hit) / len(para):.1%} | {ok / len(hit) if hit else 0:.1%} | "
              f"{ok / len(para):.1%} | {sum(r['sim'] >= tau for r in neg)}/{len(neg)} |")

    if args.gate:
        gate(rows, by_id)


def gate(rows, by_id) -> None:
    """相似度 ≥ 0.80 的命中交给决策模型：缓存的回答能否回答当前问题。"""
    import httpx

    answers = {r["qid"]: r["answer"] for r in read_jsonl(ROOT / "eval/runs/qa_m4v4/routing.jsonl")
               if not r.get("refused") and r.get("answer")}
    client = httpx.Client(base_url=os.environ.get("DECIDER_BASE_URL", "http://127.0.0.1:8765"),
                          timeout=60, trust_env=False)
    question = {"type": "noul", "instructions": "缓存里的这条回答能直接、准确地回答当前问题（问的是同一件事、同一个平台和站点），不是只沾边"}
    judged = []
    for r in rows:
        if r["sim"] < 0.80 or r["hit_qid"] not in answers:
            continue
        cached_q = by_id[r["hit_qid"]]["query"]
        state = f"当前问题：{r['query']}\n\n缓存的问题：{cached_q}\n\n缓存的回答：{answers[r['hit_qid']][:1500]}"
        resp = client.post("/v1/systemone", json={"state": state, "questions": {"reuse": question}})
        resp.raise_for_status()
        p = float(resp.json()["answers"]["reuse"]["noul"])
        judged.append({**r, "p_reuse": p})
    print(f"\n决策模型把关（相似度 ≥ 0.80 的 {len(judged)} 条）：")
    print("| τ | 把关阈值 | 通过 | 通过里的正确率 | 召回（正确且通过 / 全部改写） | 拦下的正确命中 |")
    print("|---|---|---|---|---|---|")
    total = sum(1 for r in rows if r["answerable"])
    for tau in (0.80, 0.85, 0.88, 0.90):
        for th in (0.5, 0.7):
            cand = [j for j in judged if j["sim"] >= tau]
            passed = [j for j in cand if j["p_reuse"] >= th]
            ok = sum(j["correct"] for j in passed)
            blocked_ok = sum(j["correct"] for j in cand if j["p_reuse"] < th)
            print(f"| {tau} | {th} | {len(passed)} | {ok / len(passed) if passed else 0:.1%} | "
                  f"{ok / total:.1%} | {blocked_ok} |")
    (ROOT / "eval/runs/probe_cache_gate.jsonl").write_text(
        "".join(json.dumps(j, ensure_ascii=False) + "\n" for j in judged), encoding="utf-8")


if __name__ == "__main__":
    main()
