"""chunk.py 的单元测试：用假 tokenizer，不加载真实模型。"""

from __future__ import annotations

from xiaomao.ingest.chunk import (
    MAX_CHARS,
    MAX_TOKENS,
    MIN_TOKENS,
    chunk_document,
    merge_short_sections,
    pack_paragraphs,
    parse_sections,
    split_paragraph,
)


def char_counter(text: str) -> int:
    """测试用的假 tokenizer：一个字符算一个 token。"""
    return len(text)


def make_document(text: str, **overrides: object) -> dict:
    """构造一篇最小的 documents 行。"""
    document = {
        "doc_key": "3f2a9c1d0b7e4a56",
        "collection": "xm_compliance",
        "source_url": "https://example.com/a",
        "title": "德国站合规指南",
        "platform": None,
        "site": "DE",
        "category": None,
        "doc_type": "guide",
        "lang": "zh",
        "effective_date": "2026-01-01",
        "expire_date": None,
        "content_hash": "x" * 64,
        "fetched_at": "2026-10-28T10:00:00+08:00",
        "text": text,
        "question": None,
        "answer": None,
    }
    document.update(overrides)
    return document


def test_title_path_follows_heading_levels() -> None:
    """多级标题的 title_path 正确，一级标题和文档标题重复时不叠加。"""
    text = "# 德国站合规指南\n\n引言段落。\n\n## 一、CE-RED 认证\n\n认证内容。\n\n### 1.1 报告要求\n\n报告内容。"
    sections = parse_sections(text, "德国站合规指南")

    paths = [path for path, _ in sections]
    assert paths[0] == ("德国站合规指南",)                              # 文档标题那一节（引言）
    assert paths[1] == ("德国站合规指南", "一、CE-RED 认证")
    assert paths[2] == ("德国站合规指南", "一、CE-RED 认证", "1.1 报告要求")


def test_clause_markers_are_headings() -> None:
    """Article 9、第九条、§ 3、Section 2 被识别成条款标题。"""
    text = "## 制造商义务\n\n正文。\n\nArticle 9 安全信息\n\n条款正文。\n\n第九条 责任\n\n条款正文。"
    paths = [path for path, _ in parse_sections(text, "GPSR")]

    assert any("Article 9 安全信息" in " > ".join(path) for path in paths)
    assert any("第九条 责任" in " > ".join(path) for path in paths)


def test_long_paragraph_is_split_within_limits() -> None:
    """超长段落被切开，每块都不超过上限。"""
    paragraph = "这是一句话。" * 300  # 1800 字符
    pieces = split_paragraph(paragraph, char_counter)

    assert len(pieces) > 1
    assert all(len(piece) <= MAX_CHARS for piece in pieces)
    assert all(char_counter(piece) <= MAX_TOKENS for piece in pieces)


def test_pack_paragraphs_stops_near_target() -> None:
    """贪心合并时到 TARGET 就断开，块不超上限。"""
    paragraphs = ["一段。" * 100, "二段。" * 100, "三段。" * 100]  # 每段 300 字符
    chunks = pack_paragraphs(paragraphs, char_counter)

    assert len(chunks) >= 2
    assert all(char_counter(chunk) <= MAX_TOKENS for chunk in chunks)


def test_short_section_merges_into_next_sibling() -> None:
    """很短的节并入下一个同级节，title_path 用共同上一级。"""
    long_body = "这是一段足够长的正文。" * 40  # 440 字符 > MIN_TOKENS
    sections = [
        (("指南", "第一章", "适用范围"), ["很短的一句。"]),
        (("指南", "第一章", "具体要求"), [long_body]),
    ]
    merged = merge_short_sections(sections, char_counter)

    assert len(merged) == 1
    path, lines = merged[0]
    assert path == ("指南", "第一章")
    assert "很短的一句。" in "\n".join(lines)


def test_short_section_merges_into_previous_when_no_next() -> None:
    """没有下一个同级节时并入上一块。"""
    long_body = "这是一段足够长的正文。" * 40
    sections = [
        (("指南", "第一章"), [long_body]),
        (("指南", "第二章"), ["很短的一句。"]),
    ]
    merged = merge_short_sections(sections, char_counter)

    assert len(merged) == 1
    assert "很短的一句。" in "\n".join(merged[0][1])


def test_short_section_threshold_is_80_tokens() -> None:
    """刚好超过阈值的节不合并。"""
    body = "字" * (MIN_TOKENS + 1)
    sections = [
        (("指南", "A"), [body]),
        (("指南", "B"), ["字" * 200]),
    ]
    assert len(merge_short_sections(sections, char_counter)) == 2


def test_table_is_not_broken_and_repeats_header() -> None:
    """表格不被切断；超长表格每块都带表头。"""
    header = "| 材料 | 费率 |\n|---|---|"
    rows = [f"| 材料{i} | {i}.00 |" for i in range(200)]
    table = "\n".join([header, *rows])

    pieces = split_paragraph(table, char_counter)

    assert len(pieces) > 1
    for piece in pieces:
        assert piece.startswith("| 材料 | 费率 |\n|---|---|")
        assert char_counter(piece) <= MAX_TOKENS


def test_chunk_document_faq_is_single_chunk() -> None:
    """FAQ 一篇一块，title_path 是问题本身。"""
    document = make_document(
        "德国站卖蓝牙耳机要什么认证？\n需要 CE-RED 认证。",
        collection="xm_faq",
        title="德国站卖蓝牙耳机要什么认证？",
        question="德国站卖蓝牙耳机要什么认证？",
        answer="需要 CE-RED 认证。",
    )
    chunks = chunk_document(document, char_counter)

    assert len(chunks) == 1
    assert chunks[0]["title_path"] == "德国站卖蓝牙耳机要什么认证？"
    assert chunks[0]["text"] == "德国站卖蓝牙耳机要什么认证？\n需要 CE-RED 认证。"
    assert chunks[0]["question"] == "德国站卖蓝牙耳机要什么认证？"
    assert chunks[0]["token_count"] == len(chunks[0]["text"])


def test_chunk_document_produces_ordered_metadata() -> None:
    """普通文档：多个块，元数据从文档复制，块内不重复标题路径。"""
    text = (
        "# 德国站合规指南\n\n## 一、认证\n\n" + "认证正文。" * 120
        + "\n\n## 二、包装法\n\n" + "包装正文。" * 120
    )
    document = make_document(text)
    chunks = chunk_document(document, char_counter)

    assert len(chunks) >= 2
    assert all(chunk["collection"] == "xm_compliance" for chunk in chunks)
    assert all(chunk["site"] == "DE" for chunk in chunks)
    assert all(
        chunk["title_path"] in ("德国站合规指南 > 一、认证", "德国站合规指南 > 二、包装法")
        for chunk in chunks
    )
    assert all("德国站合规指南 >" not in chunk["text"] for chunk in chunks)


def test_chunk_ids_are_continuous_per_document() -> None:
    """chunk_id 形如 doc_key-0000，且文档内连续。"""
    text = "# 指南\n\n" + "正文。" * 400
    document = make_document(text)
    chunks = chunk_document(document, char_counter)

    from xiaomao.ingest.chunk import main as _unused  # noqa: F401

    rows = [
        {"chunk_id": f"{chunk['doc_key']}-{index:04d}", "chunk_index": index, **chunk}
        for index, chunk in enumerate(chunks)
    ]
    assert [row["chunk_index"] for row in rows] == list(range(len(rows)))
    assert all(row["chunk_id"].startswith("3f2a9c1d0b7e4a56-") for row in rows)
    assert rows[0]["chunk_id"] == "3f2a9c1d0b7e4a56-0000"


def test_heading_only_section_is_dropped() -> None:
    """只有标题没有正文的节不产出片段。"""
    text = "# 指南\n\n## 只有标题\n\n## 有正文\n\n" + "正文。" * 100
    chunks = chunk_document(make_document(text), char_counter)

    assert chunks
    assert all("只有标题" not in chunk["text"] for chunk in chunks)


def test_deep_headings_enter_title_path() -> None:
    """4–6 级标题也要出现在 title_path 里。"""
    text = (
        "# 德国站合规指南\n\n## 一、总则\n\n" + "正文内容。" * 50
        + "\n\n#### 小节标题\n\n" + "小节正文内容。" * 50
        + "\n\n##### 小小节标题\n\n" + "小小节正文内容。" * 50
    )
    chunks = chunk_document(make_document(text), char_counter)
    paths = {chunk["title_path"] for chunk in chunks}

    assert "德国站合规指南 > 一、总则 > 小节标题" in paths
    assert any("小小节标题" in path for path in paths)


def test_long_section_line_is_not_a_heading() -> None:
    """以 Section 26 开头但超过 80 字符的长句是正文，不能进 title_path。"""
    long_line = (
        "Section 26 VerpackDG regulates the incentivised structure of participation fees "
        "and sets out how the system is financed by producers."
    )
    assert len(long_line) > 80
    text = f"## 一、费用\n\n{long_line}\n"
    paths = [" > ".join(path) for path, _ in parse_sections(text, "文档")]

    assert all("Section 26 VerpackDG" not in path for path in paths)


def test_tiny_chunk_merges_into_previous_and_renumbers() -> None:
    """最后一节只有按钮文字时，并进前一块，chunk_index 重新连续编号。"""
    text = "## 一、总则\n\n" + "正文内容。" * 120 + "\n\n领取福利"
    chunks = chunk_document(make_document(text), char_counter)

    assert len(chunks) == 1
    assert "领取福利" in chunks[0]["text"]
    assert chunks[0]["chunk_index"] == 0
    assert [chunk["chunk_index"] for chunk in chunks] == list(range(len(chunks)))


def test_second_level_path_equal_to_title_is_dropped() -> None:
    """第二节标题和文档标题相同（只差首尾空白）时，title_path 不重复。"""
    title = "亚马逊全球智能枢纽仓（GWD）"
    text = f"## {title}\n\n" + "正文内容。" * 100
    sections = parse_sections(text, f"  {title}  ")

    assert sections[0][0] == (title,)
    assert sections[0][0].count(title) == 1
