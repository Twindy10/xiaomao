"""检索层测试：过滤、融合、检索入口、rerank，全部用假对象。"""

from __future__ import annotations

import httpx
import pytest

from xiaomao.retrieval.filters import EU_MEMBERS, build_filter
from xiaomao.retrieval.hybrid import (
    FusionConfig,
    Retriever,
    fuse,
    merge_collections,
    parse_config,
    rrf_fuse,
    zscore_fuse,
)
from xiaomao.retrieval.rerank import ApiReranker, LocalReranker, NoopReranker
from xiaomao.retrieval.search import Candidate, QueryVectors


def make_candidate(chunk_id: str, score: float, rank: int, collection: str = "xm_compliance") -> Candidate:
    """构造一条候选。"""
    return Candidate(
        chunk_id=chunk_id,
        collection=collection,
        score=score,
        rank=rank,
        fields={"text": f"正文{chunk_id}"},
    )


# ---------------------------------------------------------------------------
# filters
# ---------------------------------------------------------------------------


def test_build_filter_only_dates() -> None:
    """不给平台站点时只有日期条件。"""
    from datetime import date

    expr = build_filter(today=date(2026, 10, 31))
    assert expr == "effective_date <= 20261031 and expire_date > 20261031"


def test_build_filter_platform() -> None:
    """平台包含"通用"空串。"""
    expr = build_filter(platform="amazon")
    assert 'platform in ["amazon", ""]' in expr


def test_build_filter_site_eu_member_adds_eu() -> None:
    """德国站要能搜到欧盟法规；英国不包含 EU。"""
    assert "DE" in EU_MEMBERS
    expr_de = build_filter(site="DE")
    assert 'site in ["DE", "", "EU"]' in expr_de

    expr_uk = build_filter(site="UK")
    assert '"EU"' not in expr_uk
    assert 'site in ["UK", ""]' in expr_uk


def test_build_filter_rejects_illegal_values() -> None:
    """非法取值要抛错，防止拼进表达式。"""
    with pytest.raises(ValueError):
        build_filter(platform="Amazon")
    with pytest.raises(ValueError):
        build_filter(platform='amazon"] or 1==1')
    with pytest.raises(ValueError):
        build_filter(site="de")
    with pytest.raises(ValueError):
        build_filter(site="DEU")


# ---------------------------------------------------------------------------
# hybrid
# ---------------------------------------------------------------------------


def test_parse_config_weights() -> None:
    """权重写法。"""
    config = parse_config("dense+bm25")
    assert config.weights == {"dense": 1.0, "bm25": 1.0}

    weighted = parse_config("dense:2+bm25")
    assert weighted.weights == {"dense": 2.0, "bm25": 1.0}
    assert weighted.name == "dense:2+bm25"


def test_parse_config_unknown_route() -> None:
    """未知路名抛错。"""
    with pytest.raises(ValueError):
        parse_config("dense+colbert")


def test_merge_collections_interleaves_by_score() -> None:
    """两个 collection 按分数合并并重编 rank。"""
    left = [make_candidate("a", 0.9, 1, "xm_platform_policy"), make_candidate("b", 0.5, 2, "xm_platform_policy")]
    right = [make_candidate("c", 0.8, 1, "xm_compliance"), make_candidate("d", 0.4, 2, "xm_compliance")]

    merged = merge_collections({"xm_platform_policy": left, "xm_compliance": right})

    assert [item.chunk_id for item in merged] == ["a", "c", "b", "d"]
    assert [item.rank for item in merged] == [1, 2, 3, 4]


def test_rrf_fuse_hand_computed() -> None:
    """手算三路/两路 RRF 的分数与顺序。"""
    route_lists = {
        "dense": [make_candidate("a", 0.9, 1), make_candidate("b", 0.8, 2), make_candidate("c", 0.7, 3)],
        "bm25": [make_candidate("b", 5.0, 1), make_candidate("c", 4.0, 2), make_candidate("d", 3.0, 3)],
    }
    fused = rrf_fuse(route_lists, parse_config("dense+bm25"), limit=20)

    scores = {item.chunk_id: item.score for item in fused}
    assert scores["a"] == pytest.approx(1 / 61)
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["c"] == pytest.approx(1 / 63 + 1 / 62)
    assert scores["d"] == pytest.approx(1 / 63)
    assert [item.chunk_id for item in fused] == ["b", "c", "a", "d"]
    assert [item.rank for item in fused] == [1, 2, 3, 4]


def test_rrf_fuse_weight_and_limit() -> None:
    """权重生效；limit 截断。"""
    route_lists = {
        "dense": [make_candidate("a", 0.9, 1), make_candidate("b", 0.8, 2)],
        "bm25": [make_candidate("b", 5.0, 1)],
    }
    weighted = rrf_fuse(route_lists, parse_config("dense:2+bm25"), limit=10)
    scores = {item.chunk_id: item.score for item in weighted}
    assert scores["a"] == pytest.approx(2 / 61)
    assert scores["b"] == pytest.approx(2 / 62 + 1 / 61)
    assert [item.chunk_id for item in weighted] == ["b", "a"]  # b 在两路都出现，仍然第一

    limited = rrf_fuse(route_lists, parse_config("dense+bm25"), limit=1)
    assert len(limited) == 1 and limited[0].rank == 1


def test_rrf_fuse_single_route_keeps_order() -> None:
    """单路时顺序不变，且重复 chunk 只出现一次。"""
    route_lists = {"dense": [make_candidate("a", 0.9, 1), make_candidate("b", 0.8, 2)]}
    fused = rrf_fuse(route_lists, parse_config("dense"))
    assert [item.chunk_id for item in fused] == ["a", "b"]


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


class FakeClient:
    """按 anns_field 返回固定结果，并记录调用。"""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def search(self, collection_name, data, anns_field, search_params, limit, filter, output_fields):
        """返回两条命中。"""
        self.calls.append(
            {
                "collection": collection_name,
                "anns_field": anns_field,
                "filter": filter,
                "limit": limit,
                "data": data,
            }
        )
        prefix = {"dense": "d", "sparse_bm25": "b", "sparse_m3": "s"}[anns_field]
        return [[
            {"chunk_id": f"{prefix}-{collection_name}-1", "distance": 0.9,
             "entity": {"text": "正文1", "title_path": "标题"}},
            {"chunk_id": f"{prefix}-{collection_name}-2", "distance": 0.8,
             "entity": {"text": "正文2", "title_path": "标题"}},
        ]]


class FakeEmbedder:
    """返回固定向量。"""

    def encode(self, texts):
        """忽略输入，返回常数向量。"""
        from xiaomao.ingest.embed import Embedding

        return [Embedding(dense=[0.1] * 4, sparse={3: 0.5}) for _ in texts]


def test_retriever_searches_each_route_and_collection() -> None:
    """每路每个 collection 都查一次，filter 传进去。"""
    client = FakeClient()
    retriever = Retriever(client, FakeEmbedder(), collections=("xm_platform_policy", "xm_compliance"),
                          config="dense+bm25")

    results = retriever.search("德国站耳机要认证吗", site="DE", top_k=5)

    # 2 路 × 2 collection
    assert len(client.calls) == 4
    assert all(call["filter"] == build_filter(site="DE") for call in client.calls)
    assert {call["anns_field"] for call in client.calls} == {"dense", "sparse_bm25"}
    assert results and results[0].rank == 1
    assert all(item.collection for item in results)


def test_retriever_passes_bm25_raw_query() -> None:
    """BM25 那一路直接传问题原文。"""
    client = FakeClient()
    Retriever(client, FakeEmbedder(), collections=("xm_faq",), config="bm25").search("问题")
    assert client.calls[0]["data"] == ["问题"]


# ---------------------------------------------------------------------------
# rerank
# ---------------------------------------------------------------------------


def _candidates() -> list[Candidate]:
    return [
        make_candidate("a", 0.9, 1),
        make_candidate("b", 0.8, 2),
        make_candidate("c", 0.7, 3),
    ]


def test_noop_reranker_keeps_order() -> None:
    """不重排时顺序不变。"""
    result = NoopReranker().rerank("问题", _candidates(), top_n=2)
    assert [item.chunk_id for item in result.candidates] == ["a", "b"]
    assert result.skipped is False


def test_api_reranker_reorders_by_index() -> None:
    """按返回的 index 重排，score 换成 relevance_score，tokens 取 usage。"""
    def handler(request: httpx.Request) -> httpx.Response:
        body = request.read().decode("utf-8")
        assert "api/v1/services/rerank/text-rerank/text-rerank" in str(request.url)
        assert "Bearer test-key" in request.headers["Authorization"]
        assert '"top_n":3' in body.replace(" ", "") or '"top_n": 3' in body
        return httpx.Response(
            200,
            json={
                "output": {"results": [
                    {"index": 2, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.5},
                    {"index": 1, "relevance_score": 0.1},
                ]},
                "usage": {"total_tokens": 123},
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler), trust_env=False)
    reranker = ApiReranker(base_url="https://example.com", model="qwen3.7-text-rerank",
                           api_key="test-key", client=client)

    result = reranker.rerank("问题", _candidates(), top_n=None)

    assert [item.chunk_id for item in result.candidates] == ["c", "a", "b"]
    assert result.candidates[0].score == 0.9
    assert result.tokens == 123
    assert result.skipped is False


def test_api_reranker_handles_http_error() -> None:
    """非 200 时 skipped，顺序不变。"""
    client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(500, json={})),
                          trust_env=False)
    reranker = ApiReranker(base_url="https://example.com", model="m", api_key="k", client=client)

    result = reranker.rerank("问题", _candidates(), top_n=None)

    assert result.skipped is True
    assert [item.chunk_id for item in result.candidates] == ["a", "b", "c"]
    assert result.error and "500" in result.error


def test_api_reranker_handles_timeout() -> None:
    """超时同样 skipped。"""
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timeout", request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler), trust_env=False)
    reranker = ApiReranker(base_url="https://example.com", model="m", api_key="k", client=client)

    result = reranker.rerank("问题", _candidates(), top_n=2)
    assert result.skipped is True
    assert len(result.candidates) == 2


class FakeRerankerModel:
    """假本地 reranker。"""

    def compute_score(self, pairs, batch_size, max_length, normalize):
        """按文本长度给分，验证排序即可。"""
        return [0.1, 0.9, 0.5][: len(pairs)]


def test_local_reranker_sorts_by_score() -> None:
    """按模型打分重排。"""
    reranker = LocalReranker(model_name="fake", model=FakeRerankerModel())
    result = reranker.rerank("问题", _candidates(), top_n=None)

    assert [item.chunk_id for item in result.candidates] == ["b", "c", "a"]
    assert result.candidates[0].score == 0.9
    assert result.skipped is False


def test_parse_config_fusion_method() -> None:
    """@z 选 z-score 加权融合，默认 rrf，未知后缀报错。"""
    config = parse_config("dense+bm25:0.2@z")
    assert config.method == "zscore" and config.weights == {"dense": 1.0, "bm25": 0.2}
    assert parse_config("dense+bm25").method == "rrf"
    with pytest.raises(ValueError):
        parse_config("dense+bm25@max")


def test_zscore_fuse_normalizes_scale_and_fills_missing() -> None:
    """BM25 分数大一两个量级也不会压倒 dense；某路没召回到的候选记这一路的最低分。"""
    dense = [make_candidate("a", 0.80, 1), make_candidate("b", 0.70, 2), make_candidate("c", 0.60, 3)]
    bm25 = [make_candidate("c", 30.0, 1), make_candidate("b", 20.0, 2), make_candidate("d", 10.0, 3)]
    config = parse_config("dense+bm25:0.2@z")
    fused = zscore_fuse({"dense": dense, "bm25": bm25}, config)

    z = 1.0 / (2 / 3) ** 0.5  # 三个等距值的 z-score 是 ±1.2247
    expected = {"a": z + 0.2 * -z, "b": 0.0, "c": -z + 0.2 * z, "d": -z + 0.2 * -z}
    assert [item.chunk_id for item in fused] == ["a", "b", "c", "d"]
    for item in fused:
        assert item.score == pytest.approx(expected[item.chunk_id])
    assert [item.rank for item in fused] == [1, 2, 3, 4]


def test_zscore_keeps_dense_winner_against_rank_one_noise() -> None:
    """RRF 会把稀疏路第 1 名的噪声顶上去；z-score 加权时 dense 第 1 名保持第 1。"""
    dense = [make_candidate(f"d{i}", 0.80 - 0.01 * i, i + 1) for i in range(10)]
    noise = [make_candidate("d9", 5.0, 1)] + [make_candidate(f"n{i}", 4.9 - 0.1 * i, i + 2) for i in range(9)]
    lists = {"dense": dense, "bm25": noise}
    assert rrf_fuse(lists, parse_config("dense+bm25"))[0].chunk_id == "d9"
    assert fuse(lists, parse_config("dense+bm25:0.2@z"))[0].chunk_id == "d0"
