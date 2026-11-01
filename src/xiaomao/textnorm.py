"""文本归一化：判断引文是否原样来自原文。

`docs/data_format.md` 第 5 节规定：两边都做 NFKC 归一化，再去掉空白和 Markdown 符号，
然后做子串匹配。生成脚本和 `validate_data.py` 必须共用这一份实现。
"""

from __future__ import annotations

import re
import unicodedata

# Markdown 符号：模型抄引文时通常会把它们去掉
MARKDOWN_MARKS = "*_`#|>\\"
_SPACE_RE = re.compile(r"\s+")
# 列表标记：行首的 "- "、"* "、"1. " 也要去掉（`*` 在 MARKDOWN_MARKS 里已处理，但 `-` 和编号不在）
_LIST_MARK_RE = re.compile(r"(?m)^[ \t]*(?:[-+]|\d+[.)])[ \t]+")
_DROP_TABLE = {ord(char): None for char in MARKDOWN_MARKS}


def normalize_for_match(text: str) -> str:
    """NFKC 归一化（全角标点转半角），去掉行首列表标记、空白和 Markdown 符号。"""
    normalized = unicodedata.normalize("NFKC", text or "")
    normalized = _LIST_MARK_RE.sub("", normalized)
    return _SPACE_RE.sub("", normalized).translate(_DROP_TABLE)
