"""解析清洗：把抓下来的原文和审核通过的 FAQ 变成 data/processed/documents.jsonl。

用法：
    python -m xiaomao.ingest.parse --manifest data/raw/manifest.jsonl \
        --faq data/processed/faq_items.jsonl --out data/processed/documents.jsonl

正文抽取两条路：
1. 默认用 trafilatura；
2. 命中 SITE_RULES 的站点（正文分散在多个并列块里，trafilatura 只挑一块）改用 lxml 自己转 Markdown。
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import lxml.html
from lxml import etree

sys.stdout.reconfigure(encoding="utf-8")  # Windows 控制台默认 GBK

ROOT_DIR = Path(__file__).resolve().parents[3]  # src/xiaomao/ingest/parse.py -> 项目根

MIN_TEXT_CHARS = 200          # 正文清洗后少于这个长度就丢弃
FAQ_COLLECTION = "xm_faq"

ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff\u00ad"), None)
FULL_WIDTH_SPACE = str.maketrans({"\u3000": " "})
COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)

# 模板噪声行：整行命中就删掉。比较时忽略大小写、空白和标点
NOISE_LINES = {
    "这篇文章有帮助吗",
    "这篇文章对你有帮助吗",
    "本文是否有帮助",
    "内容是否有帮助",
    "was this article helpful",
    "was this helpful",
    "did this article help you",
    "share this page",
    "share this article",
    "print this page",
    "打印本页",
    "上一篇",
    "下一篇",
    "返回顶部",
    "回到顶部",
    "相关文章",
    "related articles",
    "ai新动能 开售即全球",
    "2026亚马逊全球开店跨境峰会",
    "专业销售账户的费用为 $39.99/月 + 销售费用。了解更多信息",
    "专业销售计划的费用为 $39.99/月 + 销售费用。了解更多信息",
    "for more information on the efiling requirement, please visit efiling – cpsc's modern approach for filing certificate data | cpsc.gov.",
    "在注册站点的同时，您可以同步扫码添加运营经理，入群获得注册指导和其他扶持资源！",
    "立刻加入亚马逊印度站官方答疑群，印度站开店有人帮！",
}

# 前缀噪声：太长的横幅文案整行匹配不好维护，按前缀删
NOISE_PREFIXES = {
    "Beginning July 8, 2026, importers of most regulated consumer products will be required to electronically file",
}

# 页面框架：所有站点都先删掉
GENERIC_DROP = [
    "//script", "//style", "//noscript", "//nav", "//header", "//footer",
    "//form", "//button", "//iframe", "//svg", "//img", "//video",
]

# Amazon 的弹窗不能按 class 一律删：CE 商品类别表就在 div#gppop2.popup-layover 里，是正文
AMAZON_DROP = [
    "//*[@id='GLOBALSELLING-NAV']",
    "//*[@id='GLOBALSELLING-FOOTER']",
    "//*[starts-with(@id,'as_cn_ags_megamenu_')]",
    "//*[contains(concat(' ',@class,' '),' nav-overlay-dropdown-content ')]",
    "//*[contains(concat(' ',@class,' '),' sidecar-widget ')]",
    "//*[starts-with(@id,'hmd-')]",
    "//*[contains(@id,'picker-popup')]",
]

SITE_RULES: dict[str, dict] = {
    "gs.amazon.cn": {"content": "//*[@id='sc-content-container']", "drop": AMAZON_DROP},
    "sell.amazon.com": {"content": "//*[@id='sc-content-container']", "drop": AMAZON_DROP},
    "environment.ec.europa.eu": {"content": "//*[@id='block-ewcms-theme-main-page-content']", "drop": []},
    "mp.weixin.qq.com": {"content": "//*[@id='js_content']", "drop": []},
}

HEADING_TAGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}
BLOCK_TAGS = {
    "h1", "h2", "h3", "h4", "h5", "h6", "p", "blockquote", "ul", "ol", "table",
    "div", "section", "article", "main", "aside", "figure", "pre", "dl", "dt", "dd",
}

SENTENCE_END = "。！？；："
BULLET_RE = re.compile(r"^[•·▪■]\s*")
NUMBERED_RE = re.compile(r"^\d+(\.\d+)+")
# 只有标点的行（幻灯片常把句号单独排一行）：不算页眉页脚，一律接到上一行
PUNCT_ONLY_RE = re.compile(r"^[\s。！？；：，、…—·.!?;:,]+$")
MIN_FOOTER_CHARS = 4      # 页面重复行至少要有这么多实义字符，才算页眉页脚


def _meaningful_chars(text: str) -> int:
    """去掉空白和标点后还剩几个字符。"""
    return len(re.sub(r"[\s\W_]+", "", text))


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    """当前时间，ISO 8601，带本地时区。"""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _decode_html(raw: bytes) -> str:
    """按 HTML 里的 charset 声明解码，失败再依次退到 UTF-8 和 GB18030。"""
    head = raw[:2048].decode("ascii", errors="ignore")
    match = re.search(r"""charset=["']?([\w-]+)""", head, re.IGNORECASE)
    candidates = [match.group(1)] if match else []
    for encoding in [*candidates, "utf-8", "gb18030"]:
        try:
            return raw.decode(encoding)
        except (LookupError, UnicodeDecodeError):
            continue
    return raw.decode("utf-8", errors="replace")


def _tag(node) -> str:
    """取标签名；注释和文本节点返回空串。"""
    return node.tag if isinstance(node.tag, str) else ""


def _collapse(text: str) -> str:
    """压缩行内空白（含 \xa0），去掉首尾空白。"""
    return re.sub(r"[ \t]+", " ", text.replace("\xa0", " ")).strip()


def _count_cjk(text: str) -> int:
    """统计中日韩统一表意文字个数。"""
    return sum(1 for char in text if "\u4e00" <= char <= "\u9fff")


# ---------------------------------------------------------------------------
# lxml -> Markdown
# ---------------------------------------------------------------------------


def _inline(node) -> str:
    """把行内内容摊平成文字，保留 strong/b 的加粗和 br 的换行。"""
    if isinstance(node, str):
        return node.replace("\xa0", " ")
    tag = _tag(node)
    if not tag:
        return ""
    if tag == "br":
        return "\n"
    parts = [node.text or ""]
    for child in node:
        parts.append(_inline(child))
        parts.append(child.tail or "")
    inner = "".join(parts)
    if tag in ("strong", "b") and inner.strip():
        return f"**{inner.strip()}**"
    return inner


def _inline_excluding(node, skip: set[str]) -> str:
    """摊平行内内容，但跳过指定的子标签（列表项里的嵌套列表）。"""
    parts = [node.text or ""]
    for child in node:
        if _tag(child) not in skip:
            parts.append(_inline(child))
        if child.tail:
            parts.append(child.tail)
    return _collapse("".join(parts))


def _render_table(el) -> str:
    """表格转 Markdown：第一行当表头，单元格里的换行压成空格，竖线转义。"""
    rows: list[list[str]] = []
    for row in el.iter("tr"):
        cells = []
        for cell in row:
            if _tag(cell) not in ("td", "th"):
                continue
            text = _collapse(_inline(cell)).replace("\n", " ")
            cells.append(text.replace("|", r"\|"))
        if cells:
            rows.append(cells)
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
    lines.extend("| " + " | ".join(row) + " |" for row in rows[1:])
    return "\n".join(lines)


def _render_list(el, depth: int = 0) -> list[str]:
    """列表转 Markdown，每层缩进 2 个空格。"""
    lines: list[str] = []
    ordered = _tag(el) == "ol"
    index = 0
    for item in el:
        if _tag(item) != "li":
            continue
        index += 1
        marker = f"{index}. " if ordered else "- "
        text = _inline_excluding(item, {"ul", "ol"})
        lines.append("  " * depth + marker + text)
        for child in item:
            if _tag(child) in ("ul", "ol"):
                lines.extend(_render_list(child, depth + 1))
    return lines


def _render_block(el) -> list[str]:
    """把一个块级元素渲染成若干段落。"""
    tag = _tag(el)
    if tag in HEADING_TAGS:
        text = _collapse(_inline(el))
        return [f"{'#' * HEADING_TAGS[tag]} {text}"] if text else []
    if tag == "table":
        table = _render_table(el)
        return [table] if table else []
    if tag in ("ul", "ol"):
        lines = _render_list(el)
        return ["\n".join(lines)] if lines else []
    blocks = _render_blocks(el)
    if blocks:
        return blocks
    text = _collapse(_inline(el))
    return [text] if text else []


def _render_blocks(node) -> list[str]:
    """按子节点顺序收集段落，行内内容聚成一段。lxml 的文本在 .text 和 .tail 上，不是子节点。"""
    blocks: list[str] = []
    inline_parts: list[str] = []

    def flush() -> None:
        if inline_parts:
            text = _collapse("".join(inline_parts))
            if text:
                blocks.append(text)
            inline_parts.clear()

    if node.text:
        inline_parts.append(node.text.replace("\xa0", " "))
    for child in node:
        if isinstance(child, str):  # 理论上不会出现，兜底
            inline_parts.append(child)
            continue
        tag = _tag(child)
        if not tag:  # 注释：只保留 tail
            if child.tail:
                inline_parts.append(child.tail.replace("\xa0", " "))
            continue
        if tag in BLOCK_TAGS:
            flush()
            blocks.extend(_render_block(child))
        else:
            inline_parts.append(_inline(child))
        if child.tail:
            inline_parts.append(child.tail.replace("\xa0", " "))
    flush()
    return blocks


def element_to_markdown(el) -> str:
    """把一个 DOM 子树转成 Markdown；相邻完全相同的段落只保留一段。"""
    blocks = _render_blocks(el)
    deduped: list[str] = []
    for block in blocks:
        text = block.strip()
        if not text:
            continue
        if deduped and deduped[-1] == text:
            continue
        deduped.append(text)
    return "\n\n".join(deduped)


def _wrap_orphan_li(tree) -> bool:
    """把父节点不是 ul/ol 的 li 包进新建的 ul，相邻的放进同一个 ul；返回是否改动过。"""
    changed = False
    parents = [
        parent for parent in tree.iter()
        if _tag(parent) not in ("ul", "ol") and any(_tag(child) == "li" for child in parent)
    ]
    for parent in parents:
        children = list(parent)
        index = 0
        while index < len(children):
            if _tag(children[index]) != "li":
                index += 1
                continue
            run = []
            while index < len(children) and _tag(children[index]) == "li":
                run.append(children[index])
                index += 1
            wrapper = etree.Element("ul")
            run[0].addprevious(wrapper)
            for item in run:
                wrapper.append(item)
            changed = True
    return changed


def html_to_markdown(raw: bytes, url: str) -> str:
    """HTML 转 Markdown：先做通用预处理，再按站点规则选择抽取器。"""
    import trafilatura

    try:
        tree = lxml.html.fromstring(_decode_html(raw))
    except Exception:
        return ""

    wrapped = _wrap_orphan_li(tree)
    rule = SITE_RULES.get(urlparse(url).netloc.lower())

    drop = GENERIC_DROP + (rule["drop"] if rule else [])
    for node in tree.xpath(" | ".join(drop)):
        node.drop_tree()

    if rule:
        found = tree.xpath(rule["content"])
        if found:
            markdown = element_to_markdown(found[0])
            if len(clean_text(markdown)) >= MIN_TEXT_CHARS:
                return markdown

    prepared = etree.tostring(tree, encoding="unicode", method="html")
    markdown = trafilatura.extract(
        prepared,
        url=url or None,
        output_format="markdown",
        include_tables=True,
        include_links=False,
        include_formatting=True,
    ) or ""

    # 修过孤立 li 的页面，trafilatura 仍会把这些条目当模板丢掉；两种结果取更全的那个
    if wrapped:
        container = tree.find("body")
        own = element_to_markdown(container if container is not None else tree)
        if len(clean_text(own)) > len(clean_text(markdown)):
            return own
    return markdown


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------


def _join_pdf_lines(lines: list[str]) -> str:
    """把 PDF 里被版面切断的行拼回去；只有标点的行一律接到上一行末尾。"""
    out: list[str] = []
    for line in lines:
        if out and PUNCT_ONLY_RE.fullmatch(line):
            out[-1] = out[-1] + line.strip()
            continue
        bullet = BULLET_RE.match(line)
        if bullet:
            out.append("- " + line[bullet.end():].strip())
            continue
        if NUMBERED_RE.match(line) or (out and out[-1].endswith(tuple(SENTENCE_END))):
            out.append(line)
            continue
        if not out:
            out.append(line)
            continue
        previous = out[-1]
        glue = ""
        if previous and previous[-1].isascii() and previous[-1].isalnum() and line[0].isascii() and line[0].isalnum():
            glue = " "
        out[-1] = previous + glue + line
    return "\n".join(out)


def _pdf_plain_text(doc) -> str:
    """PDF 纯文本兜底：去页眉页脚和页码，再把断行拼回句子。"""
    pages: list[list[str]] = []
    for page in doc:
        lines = [line.strip() for line in page.get_text().splitlines()]
        pages.append([line for line in lines if line])

    repeated: set[str] = set()
    if len(pages) >= 3:
        counts: dict[str, int] = {}
        for page_lines in pages:
            for line in set(page_lines):
                if _meaningful_chars(line) < MIN_FOOTER_CHARS:  # 只有标点、页码之类的行不算页眉页脚
                    continue
                counts[line] = counts.get(line, 0) + 1
        repeated = {line for line, count in counts.items() if count >= len(pages) * 0.5}

    blocks = []
    for page_lines in pages:
        kept = [line for line in page_lines if line not in repeated and not re.fullmatch(r"\d+", line)]
        joined = _join_pdf_lines(kept)
        if joined.strip():
            blocks.append(joined)
    return "\n\n".join(blocks)


def pdf_to_markdown(path: Path) -> str:
    """PDF 转 Markdown；中文大面积丢失时退回纯文本路径。"""
    import pymupdf
    from pymupdf import pymupdf4llm

    document = pymupdf.open(str(path))
    try:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            markdown = pymupdf4llm.to_markdown(str(path))
        total_cjk = sum(_count_cjk(page.get_text()) for page in document)
        if total_cjk >= 100 and _count_cjk(markdown or "") < total_cjk * 0.5:
            return _pdf_plain_text(document)
        return markdown or ""
    finally:
        document.close()


# ---------------------------------------------------------------------------
# 清洗
# ---------------------------------------------------------------------------


def _normalize_line(line: str) -> str:
    """比较噪声行用：去掉空白、标点和符号，转小写。"""
    return re.sub(r"[\s\W_]+", "", line).lower()


_NOISE_SET = {_normalize_line(item) for item in NOISE_LINES}
_NOISE_PREFIX_SET = {_normalize_line(item) for item in NOISE_PREFIXES}


def _is_noise_line(line: str) -> bool:
    """判断一整行是不是模板噪声（整行匹配或前缀匹配）。"""
    normalized = _normalize_line(line)
    if not normalized:
        return False
    if normalized in _NOISE_SET:
        return True
    return any(normalized.startswith(prefix) for prefix in _NOISE_PREFIX_SET)


def clean_text(md: str) -> str:
    """统一换行、去零宽字符和 HTML 注释、压空行、删模板噪声行；不改正文和标题层级。"""
    text = md.replace("\r\n", "\n").replace("\r", "\n")
    text = COMMENT_RE.sub("", text)
    text = text.translate(ZERO_WIDTH).translate(FULL_WIDTH_SPACE)

    kept: list[str] = []
    for line in text.split("\n"):
        line = line.rstrip()
        if line and _is_noise_line(line):
            continue
        kept.append(line)

    text = "\n".join(kept)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip("\n")


# ---------------------------------------------------------------------------
# 文档组装
# ---------------------------------------------------------------------------


def compute_doc_key(collection: str, source_url: str) -> str:
    """文档稳定标识：sha256(collection|source_url) 的前 16 位十六进制。"""
    digest = hashlib.sha256(f"{collection}|{source_url}".encode("utf-8")).hexdigest()
    return digest[:16]


def compute_content_hash(text: str) -> str:
    """对清洗后的正文做 sha256，返回完整 64 位十六进制。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _first_heading(text: str) -> str:
    """取正文第一个一级标题作为标题。"""
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def _title_from_url(source_url: str) -> str:
    """URL 最后一段作为兜底标题。"""
    tail = source_url.rstrip("/").rsplit("/", 1)[-1]
    return tail.split("?")[0].split("#")[0] or source_url


def build_document(entry: dict, project_root: Path, stats: dict[str, int] | None = None) -> dict | None:
    """把 manifest 的一行变成 documents 的一行；不适用的条目返回 None 并记原因。"""
    def drop(reason: str) -> None:
        if stats is not None:
            stats[reason] = stats.get(reason, 0) + 1

    if int(entry.get("http_status") or 0) != 200:
        drop("http_not_200")
        return None

    raw_path = project_root / str(entry.get("raw_path") or "")
    if not raw_path.is_file():
        drop("raw_missing")
        return None

    try:
        content_type = str(entry.get("content_type") or "").lower()
        if "pdf" in content_type:
            markdown = pdf_to_markdown(raw_path)
        else:
            markdown = html_to_markdown(raw_path.read_bytes(), str(entry.get("source_url") or ""))
    except Exception:  # 单篇解析失败不该中断整批
        drop("parse_error")
        return None

    text = clean_text(markdown)
    if not text:
        drop("empty_text")
        return None
    if len(text) < MIN_TEXT_CHARS:
        drop("too_short")
        return None

    source_url = str(entry.get("source_url") or "")
    collection = str(entry.get("collection") or "")
    title = str(entry.get("title") or "") or _first_heading(text) or _title_from_url(source_url)
    return {
        "doc_key": compute_doc_key(collection, source_url),
        "collection": collection,
        "source_url": source_url,
        "title": title,
        "platform": entry.get("platform"),
        "site": entry.get("site"),
        "doc_type": entry.get("doc_type"),
        "lang": entry.get("lang"),
        "category": entry.get("category"),
        "effective_date": entry.get("effective_date"),
        "expire_date": entry.get("expire_date"),
        "content_hash": compute_content_hash(text),
        "fetched_at": entry.get("fetched_at") or _now_iso(),
        "text": text,
        "question": None,
        "answer": None,
    }


def faq_item_to_document(item: dict) -> dict:
    """把一条审核通过的 FAQ 转成 documents 的一行（collection = xm_faq）。"""
    question = str(item.get("question") or "").strip()
    answer = str(item.get("answer") or "").strip()
    text = f"{question}\n{answer}"
    source_url = f"faq://generated/{item.get('faq_id')}"
    return {
        "doc_key": compute_doc_key(FAQ_COLLECTION, source_url),
        "collection": FAQ_COLLECTION,
        "source_url": source_url,
        "title": question,
        "platform": item.get("platform"),
        "site": item.get("site"),
        "doc_type": "faq",
        "lang": item.get("lang") or "zh",
        "category": item.get("category"),
        "effective_date": item.get("effective_date"),
        "expire_date": None,
        "content_hash": compute_content_hash(text),
        "fetched_at": item.get("generated_at") or _now_iso(),
        "text": text,
        "question": question,
        "answer": answer,
    }


# ---------------------------------------------------------------------------
# 命令行
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict]:
    """读 JSONL，跳过空行；坏行直接报错，避免把问题带进下游。"""
    items: list[dict] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line:
            items.append(json.loads(line))
    return items


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    """先写临时文件再改名，避免中途出错留下半截文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _keep_latest(selected: dict[str, dict], document: dict) -> bool:
    """同一个 doc_key 只保留 fetched_at 最新的一条。"""
    doc_key = document["doc_key"]
    current = selected.get(doc_key)
    if current is None or str(document["fetched_at"]) >= str(current["fetched_at"]):
        selected[doc_key] = document
        return True
    return False


def main() -> int:
    """读 manifest 和 FAQ，产出 documents.jsonl。"""
    parser = argparse.ArgumentParser(description="解析清洗：生成 documents.jsonl")
    parser.add_argument("--manifest", type=Path, default=Path("data/raw/manifest.jsonl"))
    parser.add_argument("--faq", type=Path, default=Path("data/processed/faq_items.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/documents.jsonl"))
    parser.add_argument("--project-root", type=Path, default=ROOT_DIR)
    args = parser.parse_args()

    stats: dict[str, int] = {}
    selected: dict[str, dict] = {}
    duplicates = 0

    if args.manifest.is_file():
        entries = _read_jsonl(args.manifest)
        print(f"读入 manifest {len(entries)} 条")
        for entry in entries:
            document = build_document(entry, args.project_root, stats)
            if document is None:
                continue
            if not _keep_latest(selected, document):
                duplicates += 1
    else:
        print(f"[WARN] manifest 不存在：{args.manifest}")

    if args.faq.is_file():
        items = [item for item in _read_jsonl(args.faq) if item.get("reviewed") is True]
        print(f"读入 FAQ {len(items)} 条（reviewed=true）")
        for item in items:
            document = faq_item_to_document(item)
            if not _keep_latest(selected, document):
                duplicates += 1
    else:
        print(f"[WARN] FAQ 文件不存在，跳过：{args.faq}")

    rows = sorted(selected.values(), key=lambda row: (row["collection"], row["doc_key"]))
    _write_jsonl(args.out, rows)

    dropped = {key: value for key, value in stats.items() if key != "duplicate"}
    print(f"\n输出 {len(rows)} 篇 -> {args.out}")
    for collection in sorted({row["collection"] for row in rows}):
        print(f"  {collection}: {sum(1 for row in rows if row['collection'] == collection)} 篇")
    if duplicates:
        print(f"  重复 doc_key 丢弃：{duplicates}")
    if dropped:
        print("丢弃原因：" + "，".join(f"{key}={value}" for key, value in sorted(dropped.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
