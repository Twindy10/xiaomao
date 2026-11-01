"""parse.py 的单元测试：全部用自造数据，不依赖网络、GPU 和 .env。"""

from __future__ import annotations

import json
from pathlib import Path

from xiaomao.ingest.parse import (
    build_document,
    clean_text,
    compute_content_hash,
    compute_doc_key,
    faq_item_to_document,
    html_to_markdown,
)

ARTICLE_HTML = """<!doctype html><html><head><meta charset="utf-8">
<title>德国站合规指南</title></head><body>
<nav><a href="/">首页</a><a href="/de">德国站</a></nav>
<article>
<h1>德国站合规指南</h1>
<p>本指南说明在德国站销售蓝牙耳机需要满足的合规要求，包括 CE 认证、包装法和 WEEE 注册，
卖家在入驻前应当逐项核对，避免因为材料缺失导致产品被下架。</p>
<h2>一、CE-RED 认证</h2>
<p>蓝牙耳机属于无线设备，需要符合无线电设备指令 2014/53/EU，也就是通常说的 CE-RED。
测试项目包括射频、电磁兼容和安全，测试报告需要由欧盟公告机构出具，报告上要有实验室签字。</p>
<h3>1.1 测试报告要求</h3>
<p>报告必须包含产品型号、测试标准、测试日期，有效期通常为三年，过期后需要重新送测。</p>
<h2>二、德国包装法</h2>
<p>卖家需要在 LUCID 注册包装，并按包装材料的重量缴纳回收费，注册号要填到平台后台，
否则商品会被平台判定为不合规。</p>
<table>
<tr><th>材料</th><th>费率</th></tr>
<tr><td>纸</td><td>0.15 欧元/公斤</td></tr>
<tr><td>塑料</td><td>0.90 欧元/公斤</td></tr>
</table>
</article>
<footer>版权所有 2026 某公司</footer></body></html>"""


def test_html_to_markdown_keeps_headings_and_tables_drops_chrome() -> None:
    """保留 #/## 标题和表格，去掉导航和页脚。"""
    markdown = html_to_markdown(ARTICLE_HTML.encode("utf-8"), "https://example.com/de/guide")

    assert markdown.startswith("# 德国站合规指南")
    assert "## 一、CE-RED 认证" in markdown
    assert "### 1.1 测试报告要求" in markdown
    assert "| 材料 | 费率 |" in markdown
    assert "0.90 欧元/公斤" in markdown
    assert "首页" not in markdown
    assert "版权所有" not in markdown


def test_clean_text_removes_zero_width_and_noise_lines() -> None:
    """去零宽字符、全角空格、行尾空白、多余空行和模板噪声行。"""
    raw = "标题\u200b\n\n\n\nWas this article helpful?\n正文\u3000 内容   \n\n\n结尾"
    cleaned = clean_text(raw)

    assert "\u200b" not in cleaned
    assert "\u3000" not in cleaned
    assert "Was this article helpful" not in cleaned
    assert "\n\n\n" not in cleaned
    assert cleaned.splitlines()[0] == "标题"
    assert "正文  内容" in cleaned


def test_clean_text_keeps_heading_levels() -> None:
    """标题层级不能被改动。"""
    cleaned = clean_text("# 一级\n## 二级\n### 三级")
    assert cleaned == "# 一级\n## 二级\n### 三级"


def test_compute_doc_key_is_stable_and_16_chars() -> None:
    """doc_key 稳定、16 位，且随 collection 变化。"""
    first = compute_doc_key("xm_compliance", "https://example.com/a")
    assert first == compute_doc_key("xm_compliance", "https://example.com/a")
    assert len(first) == 16
    assert first != compute_doc_key("xm_platform_policy", "https://example.com/a")
    assert first != compute_doc_key("xm_compliance", "https://example.com/b")


def test_compute_content_hash_is_64_hex() -> None:
    """content_hash 是 64 位十六进制，同文同 hash。"""
    assert len(compute_content_hash("正文")) == 64
    assert compute_content_hash("正文") == compute_content_hash("正文")
    assert compute_content_hash("正文") != compute_content_hash("正文 ")


def _write_raw(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _entry(raw_path: Path, **overrides: object) -> dict:
    entry = {
        "collection": "xm_compliance",
        "source_url": "https://example.com/de/guide",
        "title": "",
        "platform": None,
        "site": "DE",
        "doc_type": "guide",
        "lang": "zh",
        "category": None,
        "effective_date": "2026-01-01",
        "expire_date": None,
        "fetched_at": "2026-10-28T10:00:00+08:00",
        "raw_path": raw_path.name,
        "content_type": "text/html",
        "http_status": 200,
    }
    entry.update(overrides)
    return entry


def test_build_document_returns_none_when_text_too_short(tmp_path: Path) -> None:
    """正文太短时返回 None，并记录丢弃原因。"""
    raw = _write_raw(tmp_path, "short.html", "<html><body><h1>短</h1><p>太短了</p></body></html>")
    stats: dict[str, int] = {}

    assert build_document(_entry(raw), tmp_path, stats) is None
    assert stats.get("too_short") == 1


def test_build_document_returns_none_on_bad_status(tmp_path: Path) -> None:
    """http_status 不是 200 直接丢弃。"""
    raw = _write_raw(tmp_path, "ok.html", ARTICLE_HTML)
    stats: dict[str, int] = {}

    assert build_document(_entry(raw, http_status=404), tmp_path, stats) is None
    assert stats.get("http_not_200") == 1


def test_build_document_fills_fields(tmp_path: Path) -> None:
    """正常条目产出规定字段，标题回退到正文一级标题。"""
    raw = _write_raw(tmp_path, "ok.html", ARTICLE_HTML)
    document = build_document(_entry(raw), tmp_path)

    assert document is not None
    assert document["collection"] == "xm_compliance"
    assert document["title"] == "德国站合规指南"
    assert document["doc_key"] == compute_doc_key("xm_compliance", "https://example.com/de/guide")
    assert document["content_hash"] == compute_content_hash(document["text"])
    assert document["question"] is None and document["answer"] is None
    assert document["site"] == "DE"


def test_build_document_keeps_manifest_metadata(tmp_path: Path) -> None:
    """manifest 里是 null 的元数据保持 null，不去猜。"""
    raw = _write_raw(tmp_path, "ok.html", ARTICLE_HTML)
    document = build_document(_entry(raw, platform="amazon", site=None), tmp_path)

    assert document is not None
    assert document["platform"] == "amazon"
    assert document["site"] is None


def test_faq_item_to_document() -> None:
    """FAQ 转文档：text = question + 换行 + answer，source_url 以 faq://generated/ 开头。"""
    item = {
        "faq_id": "faq-0007",
        "question": "德国站卖蓝牙耳机要什么认证？",
        "answer": "需要 CE-RED 认证，另外要在 LUCID 注册包装。",
        "intent": "compliance",
        "platform": None,
        "site": "DE",
        "category": "3C",
        "lang": "zh",
        "effective_date": "2026-01-01",
        "evidence": [{"doc_key": "abc", "quote": "需要符合无线电设备指令"}],
        "source": "llm_generated",
        "reviewed": True,
        "reviewed_by": "claude",
        "note": None,
        "generated_at": "2026-10-28T12:00:00+08:00",
    }
    document = faq_item_to_document(item)

    assert document["text"] == f"{item['question']}\n{item['answer']}"
    assert document["source_url"].startswith("faq://generated/")
    assert document["source_url"].endswith("faq-0007")
    assert document["collection"] == "xm_faq"
    assert document["doc_type"] == "faq"
    assert document["title"] == item["question"]
    assert document["fetched_at"] == "2026-10-28T12:00:00+08:00"
    assert document["content_hash"] == compute_content_hash(document["text"])


def test_main_writes_documents_jsonl(tmp_path: Path) -> None:
    """端到端：manifest + FAQ -> documents.jsonl，重复 doc_key 保留最新。"""
    from xiaomao.ingest import parse

    raw = _write_raw(tmp_path, "ok.html", ARTICLE_HTML)
    manifest = tmp_path / "manifest.jsonl"
    entries = [
        _entry(raw, fetched_at="2026-10-28T09:00:00+08:00"),
        _entry(raw, fetched_at="2026-10-28T11:00:00+08:00"),  # 同 doc_key，应该覆盖上一条
    ]
    manifest.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in entries) + "\n", encoding="utf-8"
    )
    faq = tmp_path / "faq_items.jsonl"
    faq.write_text(
        json.dumps(
            {
                "faq_id": "faq-0001",
                "question": "Q",
                "answer": "A",
                "intent": "compliance",
                "platform": None,
                "site": None,
                "category": None,
                "lang": "zh",
                "effective_date": None,
                "evidence": [],
                "source": "llm_generated",
                "reviewed": True,
                "reviewed_by": "claude",
                "note": None,
            },
            ensure_ascii=False,
        )
        + "\n"
        + json.dumps({"faq_id": "faq-0002", "question": "未审核", "answer": "A", "reviewed": False},
                     ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "documents.jsonl"

    argv = sys_argv(manifest, faq, out, tmp_path)
    try:
        parse.main()
    finally:
        restore_argv(argv)

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2  # 一篇 HTML + 一条审核通过的 FAQ
    assert {row["collection"] for row in rows} == {"xm_compliance", "xm_faq"}
    html_row = next(row for row in rows if row["collection"] == "xm_compliance")
    assert html_row["fetched_at"] == "2026-10-28T11:00:00+08:00"


def sys_argv(manifest: Path, faq: Path, out: Path, project_root: Path) -> list[str]:
    """替换 sys.argv，返回原值。"""
    import sys

    original = sys.argv
    sys.argv = [
        "parse",
        "--manifest", str(manifest),
        "--faq", str(faq),
        "--out", str(out),
        "--project-root", str(project_root),
    ]
    return original


def restore_argv(original: list[str]) -> None:
    """还原 sys.argv。"""
    import sys

    sys.argv = original


# ---------------------------------------------------------------------------
# C3 反馈后的新增用例
# ---------------------------------------------------------------------------


def test_amazon_page_keeps_all_content_blocks() -> None:
    """正文分散在多个并列块里时，每一块都要保留；顶部菜单要删掉。"""
    filler = "这是页面正文的一段说明文字。" * 20
    html = (
        "<html><body><div id='sc-content-container'>"
        "<div id='GLOBALSELLING-NAV'>一站式跨境新手指南</div>"
        f"<div class='as-body'>第一块正文。{filler}</div>"
        f"<div class='as-body'>第二块正文。{filler}</div>"
        "</div></body></html>"
    )
    markdown = html_to_markdown(html.encode("utf-8"), "https://gs.amazon.cn/x")

    assert "第一块正文" in markdown
    assert "第二块正文" in markdown
    assert "一站式跨境新手指南" not in markdown


def test_orphan_li_is_wrapped_for_both_paths() -> None:
    """父节点不是 ul/ol 的 li 也要变成列表项：trafilatura 路径和站点规则路径都测。"""
    filler = "这是页面正文的一段说明文字。" * 20
    plain = (
        f"<html><body><div><p>例如：</p><li>条目一</li><li>条目二</li>"
        f"<p>{filler}</p></div></body></html>"
    )
    markdown = html_to_markdown(plain.encode("utf-8"), "https://example.org/page")
    assert "- 条目一" in markdown
    assert "- 条目二" in markdown

    amazon = (
        "<html><body><div id='sc-content-container'><div class='as-body'>"
        f"<p>例如：</p><li>条目甲</li><li>条目乙</li><p>{filler}</p>"
        "</div></div></body></html>"
    )
    markdown = html_to_markdown(amazon.encode("utf-8"), "https://gs.amazon.cn/y")
    assert "- 条目甲" in markdown
    assert "- 条目乙" in markdown


def test_content_popup_is_kept_but_megamenu_is_dropped() -> None:
    """CE 类别表在弹窗里，是正文要保留；注册弹窗要删掉。"""
    filler = "这是页面正文的一段说明文字。" * 20
    html = (
        "<html><body><div id='sc-content-container'>"
        "<div id='gppop2' class='popup-layover hide'>CE 类别表</div>"
        "<div id='as_cn_ags_megamenu_reg_pop'>扫码入群</div>"
        f"<div class='as-body'>{filler}</div>"
        "</div></body></html>"
    )
    markdown = html_to_markdown(html.encode("utf-8"), "https://gs.amazon.cn/z")

    assert "CE 类别表" in markdown
    assert "扫码入群" not in markdown


def test_element_to_markdown_covers_common_tags() -> None:
    """h1–h6、嵌套列表、表格（含竖线）、加粗、相邻重复段落。"""
    import lxml.html

    from xiaomao.ingest.parse import element_to_markdown

    html = (
        "<div>"
        "<h1>一级</h1><h2>二级</h2><h3>三级</h3><h4>四级</h4><h5>五级</h5><h6>六级</h6>"
        "<p>正文<strong>加粗</strong>结束</p>"
        "<ul><li>父项<ul><li>子项</li></ul></li></ul>"
        "<table><tr><th>列a</th><th>列|b</th></tr><tr><td>1</td><td>2</td></tr></table>"
        "<p>重复段落</p><p>重复段落</p>"
        "</div>"
    )
    markdown = element_to_markdown(lxml.html.fromstring(html))

    for level in range(1, 7):
        assert "#" * level + " " in markdown
    assert "**加粗**" in markdown
    assert "- 父项" in markdown
    assert "  - 子项" in markdown
    assert "| 列a | 列\\|b |" in markdown
    assert markdown.count("重复段落") == 1


def test_pdf_plain_text_joins_lines_and_drops_headers(tmp_path: Path) -> None:
    """PDF 纯文本路径：删页眉和页码，把断行拼回句子，项目符号转列表。"""
    import pymupdf

    from xiaomao.ingest.parse import _pdf_plain_text

    path = tmp_path / "slides.pdf"
    document = pymupdf.open()
    header = "亚马逊卖家行为准则"
    bodies = [
        ["亚马逊要求卖家在亚马逊商城遵循公平", "、诚实的行事原则", "，以确保安全的购买和销售体验", "。",
         "• 不得试图损害其他卖家及其商品", "/评分或者加以滥用；"],
        ["第二页的正文第一行", "第二页的正文第二行"],
        ["第三页的正文第一行", "第三页的正文第二行"],
    ]
    for body in bodies:
        page = document.new_page()
        page.insert_text((72, 40), header, fontname="china-s", fontsize=10)
        y = 80
        for line in body:
            page.insert_text((72, y), line, fontname="china-s", fontsize=10)
            y += 16
        page.insert_text((72, 780), "1", fontname="china-s", fontsize=9)
    document.save(str(path))
    document.close()

    reopened = pymupdf.open(str(path))
    text = _pdf_plain_text(reopened)
    reopened.close()

    assert header not in text
    assert "亚马逊要求卖家在亚马逊商城遵循公平、诚实的行事原则，以确保安全的购买和销售体验。" in text
    assert "- 不得试图损害其他卖家及其商品/评分或者加以滥用；" in text
    assert "\n1\n" not in text


def test_noise_prefix_and_html_comment_are_removed() -> None:
    """前缀噪声和 HTML 注释都要清掉。"""
    text = (
        "Beginning July 8, 2026, importers of most regulated consumer products "
        "will be required to electronically file certificate data.\n正文<!-- 这是注释 -->继续"
    )
    cleaned = clean_text(text)

    assert "Beginning July 8, 2026" not in cleaned
    assert "<!--" not in cleaned
    assert "正文继续" in cleaned


def test_pdf_keeps_lone_period_lines(tmp_path: Path) -> None:
    """每页都有一行单独的句号：不能当页眉删掉，要接在上一行末尾。"""
    import pymupdf

    from xiaomao.ingest.parse import _pdf_plain_text

    path = tmp_path / "periods.pdf"
    document = pymupdf.open()
    for page_index in range(3):
        page = document.new_page()
        page.insert_text((72, 40), "亚马逊卖家行为准则", fontname="china-s", fontsize=10)
        page.insert_text((72, 80), f"第{page_index + 1}页正文第一行", fontname="china-s", fontsize=10)
        page.insert_text((72, 96), "。", fontname="china-s", fontsize=10)
        page.insert_text((72, 120), f"第{page_index + 1}页正文第二行", fontname="china-s", fontsize=10)
    document.save(str(path))
    document.close()

    reopened = pymupdf.open(str(path))
    text = _pdf_plain_text(reopened)
    reopened.close()

    assert "亚马逊卖家行为准则" not in text
    for page_index in range(1, 4):
        assert f"第{page_index}页正文第一行。" in text
        # 句号之后另起一行
        assert f"第{page_index}页正文第一行。\n第{page_index}页正文第二行" in text
