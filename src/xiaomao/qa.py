"""问答链路：改写 → 决策 → 缓存 → 检索 → 证据判断 → Backtrack → 流式生成。"""

from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Iterator

from xiaomao.cache.policy import should_cache
from xiaomao.decision.base import DeciderError
from xiaomao.decision.questions import ROUTING_QUESTIONS
from xiaomao.generation.answer import (
    CHITCHAT_REPLY,
    OUT_OF_SCOPE_REPLY,
    build_messages,
    parse_citations,
    refusal_text,
)
from xiaomao.router.evidence import judge_evidence
from xiaomao.router.retrieve import backtrack_steps, retrieve

NO_RETRIEVAL = {"chitchat", "out_of_scope"}
RERANK_HIGH = 0.7  # 和 judge_evidence 的 high 一致
DEFAULT_INTENT = "platform_policy"
DEFAULT_STRATEGY = "direct"


@dataclass
class QAOptions:
    """链路开关（评估时用来做消融）。"""

    strategy: str = "auto"
    use_rerank: bool = True
    profile: dict | None = None
    backtrack: bool = True
    translate: bool = True   # 加一路英文译文召回（需要 rerank 才生效）
    use_cache: bool = False  # 评估脚本默认不开，接口开
    history: list[dict] | None = None
    judge_evidence: bool = True  # 消融：关掉证据判断，Top-5 直接交给生成


@dataclass
class QAResult:
    """一次问答的结果。"""

    answer: str
    citations: list[dict]
    refused: bool
    intent: str
    strategy: str
    evidence: list
    trace: dict = field(default_factory=dict)


class QAPipeline:
    """把决策、检索、生成串起来。"""

    def __init__(
        self, *, decider, retriever, reranker, llm_json, generator,
        translator=None, cache=None, rewriter=None,
    ) -> None:
        self.decider = decider
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm_json          # 用于 SubQuery / HyDE 的 LLMClient
        self.generator = generator
        self.translator = translator  # router.translate.Translator；None 表示不翻译
        self.cache = cache            # cache.semantic.SemanticCache；None 表示不缓存
        self.rewriter = rewriter      # router.rewrite.Rewriter；None 表示不改写
        self._pool: ThreadPoolExecutor | None = None

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------

    def answer(self, query: str, options: QAOptions | None = None) -> QAResult:
        """非流式：消费流式事件，返回最后的 result。"""
        result: QAResult | None = None
        for event in self.answer_stream(query, options):
            if event["type"] == "done":
                result = event["result"]
        if result is None:  # 理论上不会发生
            raise RuntimeError("链路没有产出结果")
        return result

    def answer_stream(self, query: str, options: QAOptions | None = None) -> Iterator[dict]:
        """流式：meta → delta… → done。"""
        options = options or QAOptions()
        started = time.perf_counter()
        trace: dict = {
            "query_rewritten": query,
            "cache_hit": False,
            "backtracked": 0,
            "latency_ms": {},
            "llm_tokens": {
                "rewrite": 0, "decide": 0, "translate": 0,
                "retrieve": 0, "generate": 0, "cache_gate": 0,
            },
            "rerank_tokens": 0,
        }

        # 0) 多轮改写：之后所有步骤都用改写后的问题
        question = query
        if options.history and self.rewriter is not None:
            rewrite_started = time.perf_counter()
            question = self.rewriter(query, options.history)
            trace["query_rewritten"] = question
            trace["latency_ms"]["rewrite"] = (time.perf_counter() - rewrite_started) * 1000
            trace["llm_tokens"]["rewrite"] = int(getattr(self.rewriter, "last_tokens", 0) or 0)

        # 1) 翻译和决策同时发出：翻译不依赖决策结果，延迟被决策覆盖
        translation = None
        if self.translator is not None and options.translate and options.use_rerank:
            if self._pool is None:
                self._pool = ThreadPoolExecutor(max_workers=1)
            translation = self._pool.submit(self._translate, question)

        decide_started = time.perf_counter()
        decisions, degraded = self._decide(question)
        trace["latency_ms"]["decide"] = (time.perf_counter() - decide_started) * 1000
        trace["llm_tokens"]["decide"] += self._decider_tokens()
        trace["decisions"] = {
            name: {
                "value": answer.value,
                "confidence": answer.confidence,
                "backend": answer.backend,
                "fell_back": answer.fell_back,
            }
            for name, answer in decisions.items()
        }
        if degraded:
            trace["decision_error"] = True

        intent = str(decisions["intent"].value) if "intent" in decisions else DEFAULT_INTENT
        trace["intent"] = intent

        # 2) 闲聊 / 超出范围：先探一次检索，rerank 够高就当业务问题继续
        if intent in NO_RETRIEVAL and options.use_rerank:
            probe = retrieve(
                question, "direct", retriever=self.retriever, reranker=self.reranker, llm=self.llm,
                use_rerank=True, top_n=1,
            )
            trace["rerank_tokens"] += probe.rerank_tokens
            if not probe.rerank_skipped and probe.candidates and probe.candidates[0].score >= RERANK_HIGH:
                trace["intent_overridden"] = intent
                intent = DEFAULT_INTENT
                trace["intent"] = intent
        if intent in NO_RETRIEVAL:
            reply = CHITCHAT_REPLY if intent == "chitchat" else OUT_OF_SCOPE_REPLY
            trace["strategy"] = None
            trace["evidence_sufficient"] = False
            trace["evidence_by"] = "not_retrieved"
            trace["chunk_ids"] = []
            trace["latency_ms"]["total"] = (time.perf_counter() - started) * 1000
            result = QAResult(reply, [], False, intent, "none", [], trace)
            yield {"type": "meta", "intent": intent, "strategy": None,
                   "cache_hit": False, "query_rewritten": question}
            yield {"type": "delta", "text": reply}
            yield {"type": "done", "result": result}
            return

        # 3) 过滤条件：问题里先提，没提用画像补
        platform, platform_from = self._resolve(decisions, "platform", options.profile)
        site, site_from = self._resolve(decisions, "site", options.profile)
        trace["filters"] = {"platform": platform, "site": site,
                            "platform_from": platform_from, "site_from": site_from}

        # 4) 策略
        strategy = (
            str(decisions["strategy"].value)
            if options.strategy == "auto" and "strategy" in decisions
            else (options.strategy if options.strategy != "auto" else DEFAULT_STRATEGY)
        )
        trace["strategy"] = strategy

        # 5) 查缓存（用改写后的问题和它的向量）
        vectors = None
        if options.use_cache and self.cache is not None:
            encode_started = time.perf_counter()
            vectors = self._encode(question)
            trace["latency_ms"]["encode"] = (time.perf_counter() - encode_started) * 1000
        if options.use_cache and self.cache is not None and vectors is not None:
            cache_started = time.perf_counter()
            hit, detail = self.cache.lookup(question, vectors.dense, platform=platform, site=site)
            trace["latency_ms"]["cache"] = (time.perf_counter() - cache_started) * 1000
            trace["cache"] = detail
            trace["llm_tokens"]["cache_gate"] = int(detail.get("gate_tokens") or 0)
            if hit is not None:
                trace["cache_hit"] = True
                trace["evidence_sufficient"] = True
                trace["evidence_by"] = "cache"
                trace["chunk_ids"] = []
                trace["rerank_scores"] = []
                trace["latency_ms"]["total"] = (time.perf_counter() - started) * 1000
                result = QAResult(hit.answer, hit.citations, False, intent, strategy, [], trace)
                yield {"type": "meta", "intent": intent, "strategy": strategy,
                       "cache_hit": True, "query_rewritten": question}
                yield {"type": "delta", "text": hit.answer}
                yield {"type": "done", "result": result}
                return

        # 6) 检索 + 证据判断（含 Backtrack）
        extra_queries = self._collect_translation(translation, trace)
        retrieve_started = time.perf_counter()
        outcome = retrieve(
            question, strategy,
            retriever=self.retriever, reranker=self.reranker, llm=self.llm,
            platform=platform, site=site, use_rerank=options.use_rerank, top_n=5,
            extra_queries=extra_queries, vectors=vectors,
        )
        trace["latency_ms"]["retrieve"] = (time.perf_counter() - retrieve_started) * 1000
        trace["llm_tokens"]["retrieve"] += outcome.tokens
        trace["rerank_tokens"] += outcome.rerank_tokens
        trace["subqueries"] = outcome.subqueries
        trace["hypothetical"] = outcome.hypothetical
        trace["rerank_skipped"] = outcome.rerank_skipped

        evidence_started = time.perf_counter()
        if options.judge_evidence:
            verdict = judge_evidence(question, outcome, self.decider)
            trace["llm_tokens"]["decide"] += self._decider_tokens(verdict)
        else:
            verdict = None
        trace["latency_ms"]["evidence"] = (time.perf_counter() - evidence_started) * 1000

        backtracked = 0
        if verdict is not None and not verdict.sufficient and options.backtrack:
            for step in backtrack_steps(platform, site):
                outcome = retrieve(
                    question, strategy,
                    retriever=self.retriever, reranker=self.reranker, llm=self.llm,
                    platform=step["platform"], site=step["site"],
                    use_rerank=options.use_rerank, top_n=5,
                    extra_queries=extra_queries, vectors=vectors,
                )
                backtracked += 1
                trace["llm_tokens"]["retrieve"] += outcome.tokens
                trace["rerank_tokens"] += outcome.rerank_tokens
                verdict = judge_evidence(question, outcome, self.decider)
                trace["llm_tokens"]["decide"] += self._decider_tokens(verdict)
                if verdict.sufficient:
                    trace["filters"]["backtrack_to"] = step
                    break

        trace["backtracked"] = backtracked
        trace["evidence_sufficient"] = verdict.sufficient if verdict is not None else None
        trace["evidence_by"] = verdict.by if verdict is not None else "skipped"
        trace["chunk_ids"] = [candidate.chunk_id for candidate in outcome.candidates]
        trace["rerank_scores"] = [candidate.score for candidate in outcome.candidates]

        # 7) 不充分就拒答，不调生成模型
        if verdict is not None and not verdict.sufficient:
            reply = refusal_text(platform, site)
            trace["latency_ms"]["llm"] = 0.0
            trace["latency_ms"]["total"] = (time.perf_counter() - started) * 1000
            result = QAResult(reply, [], True, intent, strategy, outcome.candidates, trace)
            yield {"type": "meta", "intent": intent, "strategy": strategy,
                   "cache_hit": False, "query_rewritten": question}
            yield {"type": "delta", "text": reply}
            yield {"type": "done", "result": result}
            return

        # 8) 生成（流式）
        yield {"type": "meta", "intent": intent, "strategy": strategy,
               "cache_hit": False, "query_rewritten": question}
        generator_started = time.perf_counter()
        pieces: list[str] = []
        for piece in self._stream(question, outcome.candidates):
            pieces.append(piece)
            yield {"type": "delta", "text": piece}
        text = "".join(pieces)
        cleaned, citations = parse_citations(text, outcome.candidates)
        prompt_tokens, completion_tokens = self._generation_tokens()

        trace["latency_ms"]["generate"] = (time.perf_counter() - generator_started) * 1000
        trace["latency_ms"]["llm"] = trace["latency_ms"]["generate"]
        trace["llm_model"] = getattr(self.generator, "model", None)
        trace["prompt_tokens"] = prompt_tokens
        trace["completion_tokens"] = completion_tokens
        trace["llm_tokens"]["generate"] = prompt_tokens + completion_tokens
        trace["latency_ms"]["total"] = (time.perf_counter() - started) * 1000
        result = QAResult(cleaned, citations, False, intent, strategy, outcome.candidates, trace)

        # 9) 写缓存
        if options.use_cache and self.cache is not None:
            self._write_cache(result, question, vectors, platform, site, intent)

        yield {"type": "done", "result": result}

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _stream(self, question: str, candidates) -> Iterator[str]:
        """优先用生成模型的流式接口。"""
        messages = build_messages(question, candidates)
        if hasattr(self.generator, "stream"):
            yield from self.generator.stream(messages)
            return
        result = self.generator.complete(messages)
        yield result.text

    def _generation_tokens(self) -> tuple[int, int]:
        """生成这一步的 tokens。"""
        usage = getattr(self.generator, "last_usage", None) or {}
        return int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)

    def _write_cache(self, result: QAResult, question: str, vectors, platform, site, intent) -> None:
        """按规则写缓存；失败只记 trace，不影响回答。"""
        allowed, reason = should_cache(result)
        if not allowed:
            result.trace["cache_write"] = reason
            return
        doc_ids = {
            int(entry["doc_id"])
            for entry in result.citations
            if entry.get("doc_id") not in (None, "")
        }
        try:
            cache_id = self.cache.write(
                question=question,
                vector=vectors.dense,
                platform=platform,
                site=site,
                intent=intent,
                answer=result.answer,
                citations=result.citations,
                doc_ids=list(doc_ids),
            )
            result.trace["cache_write"] = "written"
            result.trace["cache_write_id"] = cache_id
        except Exception as exc:
            result.trace["cache_write_error"] = f"{type(exc).__name__}: {exc}"

    def _encode(self, question: str):
        """编码一次问题，缓存和检索共用。"""
        encoder = getattr(self.retriever, "encode", None)
        if encoder is not None:
            try:
                return encoder(question)
            except Exception:
                return None
        embedder = getattr(self.retriever, "embedder", None)
        if embedder is None:
            return None
        from xiaomao.retrieval.search import encode_queries

        try:
            return encode_queries(embedder, [question])[0]
        except Exception:
            return None

    def _translate(self, query: str) -> tuple[str, int, float]:
        """在后台线程里翻译，返回（译文, tokens, 耗时 ms）。"""
        started = time.perf_counter()
        text = self.translator(query)
        return text, int(getattr(self.translator, "last_tokens", 0) or 0), (time.perf_counter() - started) * 1000

    @staticmethod
    def _collect_translation(translation: Future | None, trace: dict) -> list[str]:
        """等翻译结果（通常早已完成），记进 trace；失败或没开就返回空列表。"""
        if translation is None:
            return []
        waited = time.perf_counter()
        try:
            text, tokens, elapsed = translation.result(timeout=10)
        except Exception:
            return []
        trace["query_en"] = text
        trace["llm_tokens"]["translate"] = tokens
        trace["latency_ms"]["translate"] = elapsed
        trace["latency_ms"]["translate_wait"] = (time.perf_counter() - waited) * 1000
        return [text] if text else []

    def _decider_tokens(self, verdict=None) -> int:
        """刚才那次决策调用里兜底 LLM 花的 tokens；证据判断没问决策模型时为 0。"""
        if verdict is not None and verdict.by not in {"decider", "decider_fallback"}:
            return 0
        return int(getattr(self.decider, "last_tokens", 0) or 0)

    def _decide(self, query: str):
        """跑决策层；整体失败时用默认值继续。"""
        try:
            return self.decider.decide(query, ROUTING_QUESTIONS), False
        except (DeciderError, Exception):
            return {}, True

    @staticmethod
    def _resolve(decisions: dict, name: str, profile: dict | None):
        """问题里提到的优先，没提到用画像补。"""
        answer = decisions.get(name)
        value = str(answer.value) if answer is not None else "none"
        if value and value != "none":
            return value, "query"
        if profile and profile.get(name):
            return str(profile[name]), "profile"
        return None, "none"
