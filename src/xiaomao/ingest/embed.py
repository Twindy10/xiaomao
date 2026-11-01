"""BGE-M3 向量化：dense + 稀疏向量，带自动降批和显存统计。"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


def embed_input(chunk: dict) -> str:
    """送进 BGE-M3 的文本：FAQ 只算问题，其他算「标题路径 + 正文」。"""
    if chunk.get("collection") == "xm_faq":
        return str(chunk.get("question") or chunk.get("text") or "")
    title_path = str(chunk.get("title_path") or "")
    text = str(chunk.get("text") or "")
    return f"{title_path}\n{text}" if title_path else text


@dataclass
class Embedding:
    """一条片段的向量。"""

    dense: list[float]          # 1024 维，Python float
    sparse: dict[int, float]    # BGE-M3 稀疏向量，key 是 token id


class Embedder:
    """BGE-M3 封装；模型可注入，方便测试。"""

    def __init__(
        self,
        model_name: str | None = None,
        batch_size: int = 16,
        max_length: int = 1024,
        model: object | None = None,
    ) -> None:
        self.model_name = model_name or os.environ["EMBED_MODEL"]
        self.batch_size = batch_size
        self.max_length = max_length
        self.encode_seconds = 0.0
        self._model = model
        self._batch_size = batch_size

    @property
    def model(self):
        """第一次用到时才真正加载模型。"""
        if self._model is None:
            from FlagEmbedding import BGEM3FlagModel

            self._model = BGEM3FlagModel(self.model_name, use_fp16=True, devices="cuda:0")
        return self._model

    def encode(self, texts: list[str]) -> list[Embedding]:
        """按文本长度排序后分批编码，返回顺序与输入一致。"""
        if not texts:
            return []

        order = sorted(range(len(texts)), key=lambda index: len(texts[index]))
        results: list[Embedding | None] = [None] * len(texts)
        _ = self.model  # 先加载，免得把约 12 秒的加载时间算进编码耗时
        started = time.perf_counter()

        position = 0
        while position < len(order):
            indexes = order[position: position + self._batch_size]
            batch = [texts[index] for index in indexes]
            try:
                embeddings = self._encode_batch(batch)
            except Exception as exc:  # 显存不足时降批重试
                if not self._is_oom(exc):
                    raise
                self._release_cuda_memory()
                self._batch_size //= 2
                if self._batch_size < 4:
                    raise
                logger.warning("显存不足，batch_size 降到 %d", self._batch_size)
                continue
            for index, embedding in zip(indexes, embeddings):
                results[index] = embedding
            position += len(indexes)

        self.encode_seconds += time.perf_counter() - started
        return [item for item in results if item is not None]

    def peak_vram_mb(self) -> float | None:
        """Peak 显存占用（MB），没有 CUDA 返回 None。"""
        try:
            import torch

            if not torch.cuda.is_available():
                return None
            return torch.cuda.max_memory_allocated() / 2**20
        except Exception:  # torch 没装也不该影响主流程
            return None

    def _encode_batch(self, batch: list[str]) -> list[Embedding]:
        """编码一批文本，转成 Python 原生类型。"""
        output = self.model.encode(
            batch,
            batch_size=len(batch),
            max_length=self.max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        embeddings: list[Embedding] = []
        for dense, sparse in zip(output["dense_vecs"], output["lexical_weights"]):
            weights = {int(token): float(weight) for token, weight in sparse.items() if float(weight) > 0}
            if not weights:
                raise ValueError("稀疏向量为空，Milvus 不接受")
            embeddings.append(Embedding(dense=[float(value) for value in dense], sparse=weights))
        return embeddings

    @staticmethod
    def _is_oom(exc: Exception) -> bool:
        """判断是不是 CUDA 显存不足。"""
        try:
            import torch

            if isinstance(exc, torch.cuda.OutOfMemoryError):
                return True
        except Exception:
            pass
        return "out of memory" in str(exc).lower()

    @staticmethod
    def _release_cuda_memory() -> None:
        """清空 CUDA 缓存。"""
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
