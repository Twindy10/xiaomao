"""问答链路测试：策略、证据判断、Backtrack、引用解析，全部用假对象。"""

from __future__ import annotations

from dataclasses import dataclass

from xiaomao.decision.base import Answer, DeciderError
from xiaomao.generation.answer import GeneratedAnswer, parse_citations
from xiaomao.generation.llm import GenerationResult
from xiaomao.qa import QAPipeline, QAOptions
from xiaomao.retrieval.rerank import RerankResult
from xiaomao.retrieval.search import Candidate, QueryVectors
from xiaomao.router.evidence import judge_evidence
from xiaomao.router.retrieve import RetrievalOutcome, backtrack_steps, retrieve


def candidate(chunk_id: str, score: float = 0.9, rank: int = 1, text: str | None = None) -> Candidate:
    """构造候选。"""
    return Candidate(
        chunk_id=chunk_id,
        collection="xm_compliance",
        score=score,
        rank=rank,
        fields={
            "text": text or f"正文{chunk_id}",
            "title_path": "标题",
            "source_url": "https://example.com/1",
            "effective_date": None,
        },
    )


class FakeDecider:
    """路由答案和证据概率可调。"""

    def __init__(self, routing: dict | None = None, evidence_p: float = 0.9,
                 evidence_p_list: list[float] | None = None, fail: bool = False) -> None:
        self.routing = routing or {
            "intent": "compliance", "strategy": "direct", "platform": "none", "site": "none",
        }
        self.evidence_p = evidence_p
        self.evidence_p_list = list(evidence_p_list or [])
        self.fail = fail
        self.calls: list[list[str]] = []

    def decide(self, state, questions):
        """返回预设答案。"""
        names = [question.name for question in questions]
        self.calls.append(names)
        if self.fail:
            raise DeciderError("决策模型挂了")
        if names == ["evidence"]:
            probability = self.evidence_p_list.pop(0) if self.evidence_p_list else self.evidence_p
            return {"evidence": Answer(probability, {"yes": probability, "no": 1 - probability},
                                       abs(2 * probability - 1), "fake")}
        return {
            name: Answer(value, {str(value): 1.0}, 0.9, "fake")
            for name, value in self.routing.items()
            if name in names
        }


class FakeRetriever:
    """按问题返回预设候选，并记录调用参数。"""

    def __init__(self, mapping: dict[str, list[Candidate]] | None = None,
                 default: list[Candidate] | None = None) -> None:
        self.mapping = mapping or {}
        self.default = default or [candidate("c1", 0.9, 1)]
        self.calls: list[dict] = []

    def search(self, query, *, platform=None, site=None, top_k=20, dense_text=None, vectors=None):
        """返回候选。"""
        self.calls.append({"query": query, "platform": platform, "site": site,
                           "dense_text": dense_text, "vectors": vectors})
        return list(self.mapping.get(query, self.default))


class FakeReranker:
    """按原顺序返回，可标记 skipped。"""

    def __init__(self, skipped: bool = False, score: float | None = None,
                 tokens: int = 0) -> None:
        self.skipped = skipped
        self.score = score
        self.tokens = tokens
        self.calls: list[str] = []

    def rerank(self, query, candidates, top_n=None):
        """截断并可选改分数。"""
        self.calls.append(query)
        limit = top_n or len(candidates)
        ranked = [
            Candidate(item.chunk_id, item.collection,
                      self.score if self.score is not None else item.score,
                      index + 1, item.fields)
            for index, item in enumerate(candidates[:limit])
        ]
        return RerankResult(candidates=ranked, skipped=self.skipped,
                            latency_ms=1.0, tokens=self.tokens)


class FakeUsage:
    """假 usage。"""

    def __init__(self) -> None:
        self.total_tokens = 0


class FakeLLM:
    """用于 SubQuery / HyDE 的假 LLMClient。"""

    def __init__(self, payloads: list[dict] | None = None) -> None:
        self.payloads = list(payloads or [])
        self.usage = FakeUsage()
        self.prompts: list[str] = []

    def chat_json(self, prompt, system, max_tokens=3000, temperature=0.7):
        """返回下一个预设 JSON。"""
        self.prompts.append(prompt)
        self.usage.total_tokens += 50
        return self.payloads.pop(0)


class FakeGenerator:
    """返回固定文本。"""

    def __init__(self, text: str = "回答内容[1]") -> None:
        self.text = text
        self.calls: list[list[dict]] = []
        self.model = "fake-model"

    def complete(self, messages, max_tokens=1200, temperature=0.3) -> GenerationResult:
        """记录调用并返回文本。"""
        self.calls.append(messages)
        return GenerationResult(text=self.text, prompt_tokens=100, completion_tokens=20)


def build_pipeline(decider=None, retriever=None, reranker=None, llm=None, generator=None) -> QAPipeline:
    """组装链路。"""
    return QAPipeline(
        decider=decider or FakeDecider(),
        retriever=retriever or FakeRetriever(),
        reranker=reranker or FakeReranker(),
        llm_json=llm or FakeLLM(),
        generator=generator or FakeGenerator(),
    )


# ---------------------------------------------------------------------------
# 链路
# ---------------------------------------------------------------------------


def test_chitchat_skips_generation() -> None:
    """闲聊只查一次兜底（rerank 分数低），不生成。"""
    decider = FakeDecider(routing={"intent": "chitchat", "strategy": "direct",
                                   "platform": "none", "site": "none"})
    retriever = FakeRetriever()
    generator = FakeGenerator()
    pipeline = build_pipeline(decider=decider, retriever=retriever,
                              reranker=FakeReranker(score=0.1), generator=generator)
    result = pipeline.answer("你好")

    assert result.refused is False
    assert len(retriever.calls) == 1
    assert generator.calls == []
    assert result.intent == "chitchat"


def test_chitchat_misjudged_business_question_still_answered() -> None:
    """判成闲聊但 rerank 第 1 名够高：当业务问题回答。"""
    decider = FakeDecider(routing={"intent": "chitchat", "strategy": "direct",
                                   "platform": "none", "site": "none"})
    generator = FakeGenerator()
    pipeline = build_pipeline(decider=decider, reranker=FakeReranker(score=0.9), generator=generator)
    result = pipeline.answer("亚马逊企业购扩展到了哪些国家？")

    assert result.intent == "platform_policy"
    assert result.trace["intent_overridden"] == "chitchat"
    assert len(generator.calls) == 1


def test_profile_fills_filters() -> None:
    """问题没提平台站点时用画像补全，trace 记来源。"""
    pipeline = build_pipeline()
    result = pipeline.answer("耳机要什么认证", QAOptions(profile={"platform": "amazon", "site": "DE"}))

    assert result.trace["filters"]["platform"] == "amazon"
    assert result.trace["filters"]["platform_from"] == "profile"
    assert result.trace["filters"]["site_from"] == "profile"


def test_query_filters_take_priority_over_profile() -> None:
    """问题里提到的优先于画像。"""
    decider = FakeDecider(routing={"intent": "compliance", "strategy": "direct",
                                   "platform": "temu", "site": "US"})
    result = build_pipeline(decider=decider).answer(
        "temu 美国站怎么发货", QAOptions(profile={"platform": "amazon", "site": "DE"})
    )
    assert result.trace["filters"]["platform"] == "temu"
    assert result.trace["filters"]["platform_from"] == "query"


def test_subquery_round_robin() -> None:
    """不重排时：原问题和每个子问题轮流取，重复的跳过。"""
    retriever = FakeRetriever(mapping={
        "复合问题": [candidate("o1", 0.9, 1), candidate("a1", 0.8, 2)],
        "q1": [candidate("a1", 0.9, 1), candidate("a2", 0.8, 2)],
        "q2": [candidate("b1", 0.9, 1), candidate("b2", 0.8, 2)],
    })
    llm = FakeLLM([{"subqueries": ["q1", "q2"]}])
    outcome = retrieve("复合问题", "subquery", retriever=retriever, reranker=FakeReranker(),
                       llm=llm, top_n=5, use_rerank=False)

    assert outcome.subqueries == ["q1", "q2"]
    assert [item.chunk_id for item in outcome.candidates] == ["o1", "a1", "b1", "a2", "b2"]


def test_subquery_pool_reranked_once_with_original_query() -> None:
    """重排时：原问题、译文、子问题的候选合并去重，只用原问题 rerank 一次。"""
    retriever = FakeRetriever(mapping={
        "复合问题": [candidate("o1", 0.9, 1)],
        "english": [candidate("e1", 0.9, 1)],
        "q1": [candidate("o1", 0.9, 1), candidate("a1", 0.8, 2)],
        "q2": [candidate("b1", 0.9, 1)],
    })
    reranker = FakeReranker()
    llm = FakeLLM([{"subqueries": ["q1", "q2"]}])
    outcome = retrieve("复合问题", "subquery", retriever=retriever, reranker=reranker,
                       llm=llm, top_n=5, extra_queries=["english"])

    assert reranker.calls == ["复合问题"]
    assert {item.chunk_id for item in outcome.candidates} == {"o1", "e1", "a1", "b1"}


def test_subquery_falls_back_to_direct_when_one() -> None:
    """只拆出一个子问题时按 direct 处理。"""
    retriever = FakeRetriever(default=[candidate("c1")])
    llm = FakeLLM([{"subqueries": ["只有这一个"]}])
    outcome = retrieve("问题", "subquery", retriever=retriever, reranker=FakeReranker(), llm=llm)

    assert outcome.subqueries == []
    assert retriever.calls[0]["query"] == "问题"


def test_hyde_uses_hypothetical_for_dense_only() -> None:
    """HyDE 用「问题 + 假设答案」编码 dense，rerank 仍用原问题。"""
    retriever = FakeRetriever()
    reranker = FakeReranker()
    llm = FakeLLM([{"answer": "假设的答案内容"}])
    outcome = retrieve("模糊问题", "hyde", retriever=retriever, reranker=reranker, llm=llm)

    assert outcome.hypothetical == "假设的答案内容"
    assert retriever.calls[0]["dense_text"] == "模糊问题\n假设的答案内容"
    assert retriever.calls[0]["query"] == "模糊问题"
    assert reranker.calls == ["模糊问题"]


def test_no_rerank_keeps_fusion_order() -> None:
    """不重排时直接取融合顺序，并标记 skipped。"""
    retriever = FakeRetriever(default=[candidate("c1", 0.5, 1), candidate("c2", 0.4, 2)])
    outcome = retrieve("问题", "direct", retriever=retriever, reranker=FakeReranker(), llm=FakeLLM(),
                       use_rerank=False, top_n=1)
    assert [item.chunk_id for item in outcome.candidates] == ["c1"]
    assert outcome.rerank_skipped is True


# ---------------------------------------------------------------------------
# 证据判断
# ---------------------------------------------------------------------------


def make_outcome(score: float, skipped: bool = False, empty: bool = False) -> RetrievalOutcome:
    """构造检索结果。"""
    candidates = [] if empty else [candidate("c1", score, 1)]
    return RetrievalOutcome(candidates=candidates, rerank_skipped=skipped)


def test_evidence_high_score_is_sufficient_without_decider() -> None:
    """高分直接充分，不调决策模型。"""
    decider = FakeDecider(evidence_p=0.1)
    verdict = judge_evidence("问题", make_outcome(0.9), decider)
    assert verdict.sufficient is True and verdict.by == "rerank_high"
    assert decider.calls == []


def test_evidence_low_score_is_insufficient_without_decider() -> None:
    """低分直接不充分。"""
    decider = FakeDecider(evidence_p=0.9)
    verdict = judge_evidence("问题", make_outcome(0.1), decider)
    assert verdict.sufficient is False and verdict.by == "rerank_low"
    assert decider.calls == []


def test_evidence_middle_asks_decider() -> None:
    """中间段问决策模型。"""
    decider = FakeDecider(evidence_p=0.8)
    verdict = judge_evidence("问题", make_outcome(0.5), decider)
    assert verdict.sufficient is True and verdict.by == "decider"
    assert decider.calls == [["evidence"]]


def test_evidence_skipped_rerank_asks_decider() -> None:
    """rerank 被跳过时分数是融合分，直接问决策模型。"""
    decider = FakeDecider(evidence_p=0.2)
    verdict = judge_evidence("问题", make_outcome(0.99, skipped=True), decider)
    assert verdict.by == "decider" and verdict.sufficient is False


def test_evidence_empty_candidates() -> None:
    """没有候选直接不充分。"""
    verdict = judge_evidence("问题", make_outcome(0.0, empty=True), FakeDecider())
    assert verdict.sufficient is False


# ---------------------------------------------------------------------------
# Backtrack 与整条链路
# ---------------------------------------------------------------------------


def test_backtrack_steps() -> None:
    """放宽顺序：先去掉站点，再去掉平台。"""
    assert backtrack_steps("amazon", "DE") == [
        {"platform": "amazon", "site": None},
        {"platform": None, "site": None},
    ]
    assert backtrack_steps(None, "DE") == [{"platform": None, "site": None}]
    assert backtrack_steps(None, None) == []


def test_backtrack_stops_at_first_sufficient() -> None:
    """放宽一次就够时不再继续放宽。"""
    decider = FakeDecider(routing={"intent": "compliance", "strategy": "direct",
                                   "platform": "amazon", "site": "DE"},
                          evidence_p=0.1)
    # 第一次检索分数低 → 不充分；放宽后分数高 → 充分
    scores = iter([0.1, 0.9])
    retriever = FakeRetriever()

    class ScoreReranker(FakeReranker):
        def rerank(self, query, candidates, top_n=None):
            self.calls.append(query)
            score = next(scores)
            return RerankResult(
                candidates=[Candidate("c1", "xm_compliance", score, 1, candidates[0].fields)],
                skipped=False, latency_ms=1.0, tokens=0,
            )

    result = build_pipeline(decider=decider, retriever=retriever, reranker=ScoreReranker()).answer(
        "德国站亚马逊耳机认证"
    )
    assert result.refused is False
    assert result.trace["backtracked"] == 1
    assert result.trace["filters"]["backtrack_to"] == {"platform": "amazon", "site": None}


def test_refusal_when_never_sufficient() -> None:
    """一直不充分就拒答，不调生成模型。"""
    decider = FakeDecider(routing={"intent": "compliance", "strategy": "direct",
                                   "platform": "amazon", "site": "DE"},
                          evidence_p=0.0)
    generator = FakeGenerator()
    result = build_pipeline(decider=decider, reranker=FakeReranker(score=0.1), generator=generator).answer(
        "德国站亚马逊耳机认证"
    )

    assert result.refused is True
    assert generator.calls == []
    assert result.answer.startswith("暂未找到与该问题相关的资料")


def test_decision_failure_uses_defaults() -> None:
    """决策层整体失败时按默认值继续。"""
    decider = FakeDecider(fail=True)
    result = build_pipeline(decider=decider).answer("耳机认证")
    assert result.intent == "platform_policy"
    assert result.trace["decision_error"] is True


def test_citation_parsing_drops_out_of_range() -> None:
    """越界的 [n] 从正文删掉，只保留被引用的资料。"""
    candidates = [candidate("c1"), candidate("c2")]
    text, citations = parse_citations("结论一[1]，结论二[9]。", candidates)

    assert "[9]" not in text
    assert "[1]" in text
    assert [entry["n"] for entry in citations] == [1]
    assert citations[0]["chunk_id"] == "c1"


def test_trace_fields() -> None:
    """trace 字段齐全。"""
    result = build_pipeline().answer("耳机认证")
    trace = result.trace
    for field in ("query_rewritten", "intent", "strategy", "decisions", "cache_hit",
                  "chunk_ids", "rerank_scores", "evidence_sufficient", "evidence_by",
                  "backtracked", "filters", "latency_ms"):
        assert field in trace, field
    for stage in ("decide", "retrieve", "evidence", "llm", "total"):
        assert stage in trace["latency_ms"], stage


# ---------------------------------------------------------------------------
# 英文译文作为附加改写
# ---------------------------------------------------------------------------


class FakeTranslator:
    """返回固定译文，记录调用。"""

    def __init__(self, text: str = "english query") -> None:
        self.text = text
        self.calls: list[str] = []
        self.last_tokens = 0

    def __call__(self, query: str) -> str:
        self.calls.append(query)
        self.last_tokens = 120
        return self.text


def test_extra_query_candidates_reranked_with_original_query() -> None:
    """译文召回的候选并进原问题的候选池，去重后统一用原问题重排。"""
    retriever = FakeRetriever(mapping={
        "中文问题": [candidate("z1", 0.9, 1), candidate("x", 0.8, 2)],
        "english query": [candidate("x", 0.9, 1), candidate("e1", 0.8, 2)],
    })
    reranker = FakeReranker()
    outcome = retrieve("中文问题", "direct", retriever=retriever, reranker=reranker, llm=FakeLLM(),
                       extra_queries=["english query"])

    assert [item.chunk_id for item in outcome.candidates] == ["z1", "x", "e1"]
    assert reranker.calls == ["中文问题"]


def test_extra_query_ignored_without_rerank() -> None:
    """不重排时没法合并，译文直接忽略。"""
    retriever = FakeRetriever()
    retrieve("中文问题", "direct", retriever=retriever, reranker=FakeReranker(), llm=FakeLLM(),
             use_rerank=False, extra_queries=["english query"])

    assert [call["query"] for call in retriever.calls] == ["中文问题"]


def test_pipeline_translates_and_records_trace() -> None:
    """链路把译文交给检索，trace 里记下译文和 tokens。"""
    retriever = FakeRetriever()
    translator = FakeTranslator()
    pipeline = QAPipeline(
        decider=FakeDecider(), retriever=retriever, reranker=FakeReranker(),
        llm_json=FakeLLM(), generator=FakeGenerator(), translator=translator,
    )
    result = pipeline.answer("耳机认证")

    assert translator.calls == ["耳机认证"]
    assert result.trace["query_en"] == "english query"
    assert result.trace["llm_tokens"]["translate"] == 120
    assert "english query" in [call["query"] for call in retriever.calls]


def test_pipeline_skips_translation_without_rerank() -> None:
    """hybrid（不重排）配置不翻译，省一次调用。"""
    translator = FakeTranslator()
    pipeline = QAPipeline(
        decider=FakeDecider(), retriever=FakeRetriever(), reranker=FakeReranker(),
        llm_json=FakeLLM(), generator=FakeGenerator(), translator=translator,
    )
    pipeline.answer("耳机认证", QAOptions(use_rerank=False))

    assert translator.calls == []


# ---------------------------------------------------------------------------
# M4b：缓存、流式、向量复用
# ---------------------------------------------------------------------------


class FakeCache:
    """假语义缓存。"""

    def __init__(self, hit=None) -> None:
        self.hit = hit
        self.written: list[dict] = []

    def lookup(self, question, vector, *, platform, site):
        """返回预设命中。"""
        detail = {"similarity": 0.92, "candidate_id": "c1", "gate_p": 0.9,
                  "gate_tokens": 30, "lookup_ms": 3.0, "gate_ms": 700.0}
        return self.hit, detail

    def write(self, **kwargs):
        """记录写入。"""
        self.written.append(kwargs)
        return "new-id"


class Hit:
    """假的 CacheHit。"""

    cache_id = "c1"
    question = "缓存的问题"
    answer = "缓存里的回答"
    citations = [{"n": 1, "chunk_id": "cached-1"}]
    similarity = 0.92
    gate_p = 0.9
    gate_tokens = 30


class EncodingRetriever(FakeRetriever):
    """能编码的假检索器。"""

    def __init__(self, default=None) -> None:
        super().__init__(default=default or [candidate("c1", 0.9, 1, text="正文")])
        self.encoded: list[str] = []
        self.vectors_seen = []

    def encode(self, text):
        """返回带标记的向量。"""
        self.encoded.append(text)
        return QueryVectors(dense=[0.1] * 4, sparse={1: 0.5})

    def search(self, query, **kwargs):
        """记录传进来的 vectors。"""
        self.vectors_seen.append(kwargs.get("vectors"))
        return super().search(query, **kwargs)


def test_cache_hit_skips_retrieval_and_generation() -> None:
    """命中缓存时不检索、不生成。"""
    retriever = EncodingRetriever()
    generator = FakeGenerator()
    pipeline = build_pipeline(retriever=retriever, generator=generator)
    pipeline.cache = FakeCache(hit=Hit())

    result = pipeline.answer("德国站耳机要认证吗", QAOptions(use_cache=True))

    assert result.answer == "缓存里的回答"
    assert result.trace["cache_hit"] is True
    assert retriever.calls == []
    assert generator.calls == []


def test_miss_writes_cache_with_doc_ids() -> None:
    """未命中时回答完按规则写缓存，带上引用资料的 doc_id。"""
    cand = candidate("c1", 0.9, 1, text="正文")
    cand.fields["doc_id"] = 42
    pipeline = build_pipeline(retriever=EncodingRetriever(default=[cand]),
                              generator=FakeGenerator(text="回答[1]"))
    pipeline.cache = FakeCache()

    result = pipeline.answer("德国站耳机要认证吗", QAOptions(use_cache=True))

    assert result.trace["cache_hit"] is False
    written = pipeline.cache.written
    assert len(written) == 1
    assert written[0]["doc_ids"] == [42]
    assert written[0]["question"] == "德国站耳机要认证吗"


def test_refusal_does_not_write_cache() -> None:
    """拒答不写缓存。"""
    pipeline = build_pipeline(reranker=FakeReranker(score=0.1))
    pipeline.cache = FakeCache()

    pipeline.answer("德国站耳机要认证吗", QAOptions(use_cache=True))

    assert pipeline.cache.written == []


def test_stream_event_order_and_equivalence() -> None:
    """事件顺序 meta → delta → done，answer() 与流式结果一致。"""
    pipeline = build_pipeline()

    events = list(pipeline.answer_stream("问题"))
    assert events[0]["type"] == "meta"
    assert events[-1]["type"] == "done"
    assert all(event["type"] == "delta" for event in events[1:-1])

    streamed = events[-1]["result"]
    direct = pipeline.answer("问题")
    assert streamed.answer == direct.answer
    assert streamed.citations == direct.citations


def test_question_vector_encoded_once_and_reused() -> None:
    """问题向量只编码一次，缓存和检索共用。"""
    retriever = EncodingRetriever()
    pipeline = build_pipeline(retriever=retriever)
    pipeline.cache = FakeCache()

    pipeline.answer("德国站耳机要认证吗", QAOptions(use_cache=True))

    assert retriever.encoded == ["德国站耳机要认证吗"]
    assert retriever.vectors_seen and retriever.vectors_seen[0] is not None


# ---------------------------------------------------------------------------
# M6：消融开关和 rerank tokens
# ---------------------------------------------------------------------------


def test_judge_evidence_off_always_generates() -> None:
    """关掉证据判断：不问决策模型、不做 Backtrack、总会生成。"""
    decider = FakeDecider(routing={"intent": "compliance", "strategy": "direct",
                                   "platform": "amazon", "site": "DE"}, evidence_p=0.0)
    retriever = FakeRetriever()
    generator = FakeGenerator()
    result = build_pipeline(decider=decider, retriever=retriever,
                            reranker=FakeReranker(score=0.1), generator=generator).answer(
        "德国站耳机认证", QAOptions(judge_evidence=False)
    )

    assert result.refused is False
    assert result.answer == generator.text
    assert result.trace["evidence_by"] == "skipped"
    assert result.trace["evidence_sufficient"] is None
    assert result.trace["backtracked"] == 0
    assert len(retriever.calls) == 1  # 只检索一次，没有 Backtrack
    assert all("evidence" not in call for call in decider.calls)


def test_rerank_tokens_sum_across_backtrack() -> None:
    """rerank tokens 是所有 retrieve 调用（含 Backtrack）的和。"""
    decider = FakeDecider(routing={"intent": "compliance", "strategy": "direct",
                                   "platform": "amazon", "site": "DE"}, evidence_p=0.0)
    result = build_pipeline(decider=decider, reranker=FakeReranker(score=0.1, tokens=7)).answer(
        "德国站耳机认证"
    )

    assert result.refused is True
    assert result.trace["rerank_tokens"] == 21  # 首次 + 两次 Backtrack


def test_rerank_tokens_zero_without_rerank() -> None:
    """不重排时为 0。"""
    result = build_pipeline(reranker=FakeReranker(tokens=99)).answer(
        "问题", QAOptions(use_rerank=False)
    )
    assert result.trace["rerank_tokens"] == 0
