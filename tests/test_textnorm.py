"""textnorm.normalize_for_match 的测试。"""

from __future__ import annotations

from xiaomao.textnorm import normalize_for_match


def test_full_width_punctuation_is_normalized() -> None:
    """全角标点转半角（NFKC）。"""
    assert normalize_for_match("（ ）！ ：") == "()!:"


def test_markdown_marks_are_removed() -> None:
    """加粗、列表、表格、转义符号都去掉。"""
    assert normalize_for_match("**加粗**") == "加粗"
    assert normalize_for_match("- 列表项") == "列表项"
    assert normalize_for_match("| 列a | 列b |") == "列a列b"
    assert normalize_for_match(r"\*转义") == "转义"
    assert normalize_for_match("> 引用") == "引用"


def test_whitespace_is_removed() -> None:
    """空白（含换行、不换行空格）全部去掉。"""
    assert normalize_for_match("德国站 卖耳机\n需要认证\t。") == "德国站卖耳机需要认证。"


def test_mixed_chinese_and_english() -> None:
    """中英文混排、带 Markdown 的原文和丢掉符号的引文能匹配上。"""
    source = "卖家需要在 **LUCID** 注册包装，并填写 `VerpackG` 的注册号。"
    quote = "卖家需要在 LUCID 注册包装，并填写 VerpackG 的注册号。"
    assert normalize_for_match(quote) in normalize_for_match(source)
