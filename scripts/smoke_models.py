"""M0 本地模型冒烟测试：BGE-M3（dense + 稀疏）和 bge-reranker-v2-m3 能否在 GPU 上跑通，各占多少显存。

用法：python scripts/smoke_models.py
"""

import os
import sys
import time
from pathlib import Path

import torch
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8")
# 系统环境变量里可能残留其他项目的同名变量，以 .env 为准
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

GIB = 1024**3


def used_gib() -> float:
    free, total = torch.cuda.mem_get_info()
    return (total - free) / GIB


def main() -> None:
    from FlagEmbedding import BGEM3FlagModel, FlagReranker

    print(f"GPU: {torch.cuda.get_device_name(0)}，加载前已用 {used_gib():.2f}GB")

    t = time.perf_counter()
    embedder = BGEM3FlagModel(os.environ["EMBED_MODEL"], use_fp16=True, devices="cuda:0")
    print(f"BGE-M3 加载 {time.perf_counter() - t:.1f}s，已用 {used_gib():.2f}GB")

    texts = [
        "德国站卖蓝牙耳机需要哪些认证？",
        "欧盟 RED 指令（2014/53/EU）适用于所有无线电设备，蓝牙耳机需要通过 CE-RED 认证。",
        "Temu 半托管模式下，卖家自行负责海外仓发货。",
    ]
    t = time.perf_counter()
    out = embedder.encode(texts, return_dense=True, return_sparse=True, max_length=1024)
    dense, sparse = out["dense_vecs"], out["lexical_weights"]
    print(f"编码 {len(texts)} 条 {1000 * (time.perf_counter() - t):.0f}ms，dense 维度 {dense.shape[1]}")
    sims = dense[0] @ dense[1:].T
    print(f"问题与两段文本的 dense 相似度：{sims[0]:.3f}（相关） / {sims[1]:.3f}（无关）")
    top = sorted(sparse[1].items(), key=lambda kv: -kv[1])[:5]
    tokens = embedder.tokenizer.convert_ids_to_tokens([int(k) for k, _ in top])
    print("第 2 段稀疏权重 Top5：", ", ".join(f"{tok}={w:.2f}" for tok, (_, w) in zip(tokens, top)))

    t = time.perf_counter()
    reranker = FlagReranker(os.environ["RERANK_MODEL"], use_fp16=True, devices="cuda:0")
    print(f"reranker 加载 {time.perf_counter() - t:.1f}s，已用 {used_gib():.2f}GB")
    scores = reranker.compute_score([[texts[0], texts[1]], [texts[0], texts[2]]], normalize=True)
    print(f"rerank 得分：{scores[0]:.3f}（相关） / {scores[1]:.3f}（无关）")

    print(f"两个模型都加载后已用 {used_gib():.2f}GB，峰值分配 {torch.cuda.max_memory_allocated() / GIB:.2f}GB")


if __name__ == "__main__":
    main()
