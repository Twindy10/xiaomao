"""重排：API、本地、不重排三种实现，失败时保留原顺序。"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, replace
from typing import Protocol

import httpx

from xiaomao.retrieval.search import Candidate

API_PATH = "/api/v1/services/rerank/text-rerank/text-rerank"


@dataclass
class RerankResult:
    """一次重排的结果。"""

    candidates: list[Candidate]
    skipped: bool
    latency_ms: float
    tokens: int
    error: str | None = None


class Reranker(Protocol):
    """重排器的统一接口。"""

    def rerank(self, query: str, candidates: list[Candidate], top_n: int | None = None) -> RerankResult:
        """对候选重新排序。"""
        ...


def rerank_text(candidate: Candidate) -> str:
    """送进 reranker 的文本：Milvus 里的 text（已含标题路径）。"""
    return str(candidate.fields.get("text") or "")


def _renumber(candidates: list[Candidate]) -> list[Candidate]:
    """按当前顺序重编 rank。"""
    return [replace(candidate, rank=index) for index, candidate in enumerate(candidates, start=1)]


class NoopReranker:
    """不重排，原样返回（消融里的"不重排"）。"""

    def rerank(self, query: str, candidates: list[Candidate], top_n: int | None = None) -> RerankResult:
        """直接截取前 top_n。"""
        limit = top_n or len(candidates)
        return RerankResult(candidates=_renumber(candidates[:limit]), skipped=False, latency_ms=0.0, tokens=0)


class ApiReranker:
    """走阿里云百炼的 qwen3.7-text-rerank。"""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = 10.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = (base_url or os.environ.get("RERANK_API_BASE_URL") or "").rstrip("/")
        self.model = model or os.environ.get("RERANK_API_MODEL", "")
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "")
        if not self.base_url or not self.model:
            raise RuntimeError("RERANK_API_BASE_URL / RERANK_API_MODEL 未配置")
        self.client = client or httpx.Client(timeout=timeout, trust_env=False)

    def rerank(self, query: str, candidates: list[Candidate], top_n: int | None = None) -> RerankResult:
        """调 API 重排；出错不重试、不兜底，返回原顺序。"""
        limit = top_n or len(candidates)
        documents = [rerank_text(candidate) for candidate in candidates]
        body = {
            "model": self.model,
            "input": {"query": query, "documents": documents},
            "parameters": {"top_n": limit, "return_documents": False},
        }
        started = time.perf_counter()
        try:
            response = self.client.post(
                self.base_url + API_PATH,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=body,
            )
            if response.status_code != 200:
                raise RuntimeError(f"HTTP {response.status_code}")
            payload = response.json()
            results = payload["output"]["results"]
            reordered: list[Candidate] = []
            for position, item in enumerate(results[:limit], start=1):
                index = int(item["index"])
                if not 0 <= index < len(candidates):
                    raise ValueError(f"返回下标越界：{index}")
                reordered.append(
                    replace(candidates[index], score=float(item["relevance_score"]), rank=position)
                )
            tokens = int((payload.get("usage") or {}).get("total_tokens") or 0)
            return RerankResult(
                candidates=reordered,
                skipped=False,
                latency_ms=(time.perf_counter() - started) * 1000,
                tokens=tokens,
            )
        except Exception as exc:  # 超时、非 200、结构不对都在这里
            return RerankResult(
                candidates=_renumber(candidates[:limit]),
                skipped=True,
                latency_ms=(time.perf_counter() - started) * 1000,
                tokens=0,
                error=f"{type(exc).__name__}: {exc}"[:200],
            )


class LocalReranker:
    """本地 bge-reranker-v2-m3。"""

    def __init__(
        self,
        model_name: str | None = None,
        batch_size: int = 8,
        model: object | None = None,
    ) -> None:
        self.model_name = model_name or os.environ["RERANK_MODEL"]
        self.batch_size = batch_size
        self._model = model

    @property
    def model(self):
        """第一次用到时才加载模型。"""
        if self._model is None:
            from FlagEmbedding import FlagReranker

            self._model = FlagReranker(self.model_name, use_fp16=True, devices="cuda:0")
        return self._model

    def rerank(self, query: str, candidates: list[Candidate], top_n: int | None = None) -> RerankResult:
        """本地打分并重排。"""
        limit = top_n or len(candidates)
        started = time.perf_counter()
        pairs = [[query, rerank_text(candidate)] for candidate in candidates]
        scores = self.model.compute_score(
            pairs, batch_size=self.batch_size, max_length=1024, normalize=True
        )
        if not isinstance(scores, list):
            scores = [scores]
        order = sorted(range(len(candidates)), key=lambda index: -float(scores[index]))[:limit]
        reordered = [
            replace(candidates[index], score=float(scores[index]), rank=position)
            for position, index in enumerate(order, start=1)
        ]
        return RerankResult(
            candidates=reordered,
            skipped=False,
            latency_ms=(time.perf_counter() - started) * 1000,
            tokens=0,
        )

    def peak_vram_mb(self) -> float | None:
        """Peak 显存（MB）。"""
        try:
            import torch

            if not torch.cuda.is_available():
                return None
            return torch.cuda.max_memory_allocated() / 2**20
        except Exception:
            return None


def get_reranker(backend: str | None = None) -> Reranker:
    """按 RERANK_BACKEND 返回实现；none 表示不重排。"""
    name = (backend or os.environ.get("RERANK_BACKEND", "api")).lower()
    if name == "api":
        return ApiReranker()
    if name == "local":
        return LocalReranker()
    if name in ("none", "noop"):
        return NoopReranker()
    raise ValueError(f"未知的 rerank 后端：{name}")
