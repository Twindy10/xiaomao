"""分块：把 data/processed/documents.jsonl 切成 data/processed/chunks.jsonl。

用法：
    python -m xiaomao.ingest.chunk --docs data/processed/documents.jsonl --out data/processed/chunks.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Callable

sys.stdout.reconfigure(encoding="utf-8")  # Windows 控制台默认 GBK

ROOT_DIR = Path(__file__).resolve().parents[3]  # src/xiaomao/ingest/chunk.py -> 项目根

TARGET_TOKENS = 500       # 一块的目标 token 数，够了就断开
MAX_TOKENS = 800          # 硬上限
MIN_TOKENS = 80           # 低于这个长度的节要考虑合并
MIN_CHUNK_TOKENS = 20     # 低于这个 token 数的片段是按钮文字之类的碎块，并入相邻块
MAX_CHARS = 1500          # Milvus VARCHAR(8192) 按中文 3 字节算的安全上限
PATH_SEP = " > "
MAX_CLAUSE_CHARS = 80     # 超过这个长度的「Article 9 …」是正文句子，不是条款标题

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
CLAUSE_RES = (
    re.compile(r"^(?:Article|Art\.)\s+\d+", re.IGNORECASE),
    re.compile(r"^第[一二三四五六七八九十百零〇0-9]+条"),
    re.compile(r"^§\s*\d+"),
    re.compile(r"^Section\s+\d+", re.IGNORECASE),
)
SENTENCE_RE = re.compile(r"[^。！？.!?]*[。！？.!?]|[^。！？.!?]+$")
TABLE_ROW_RE = re.compile(r"^\s*\|")

Tokenizer = Callable[[str], int]
_TOKENIZER = None


def _default_tokenizer(text: str) -> int:
    """用 EMBED_MODEL 的 tokenizer 数 token；模型只加载一次。"""
    global _TOKENIZER
    if _TOKENIZER is None:
        from dotenv import dotenv_values
        from transformers import AutoTokenizer

        config = dotenv_values(ROOT_DIR / ".env")
        path = config.get("EMBED_MODEL") or "BAAI/bge-m3"
        _TOKENIZER = AutoTokenizer.from_pretrained(path, local_files_only=True)
    return len(_TOKENIZER.encode(text, add_special_tokens=False))


def _is_clause_heading(line: str) -> bool:
    """判断一行是不是条款标记（Article 9、第九条、§ 3、Section 2）；超过 80 字符的算正文。"""
    stripped = line.strip()
    if not stripped or len(stripped) > MAX_CLAUSE_CHARS:
        return False
    return any(pattern.match(stripped) for pattern in CLAUSE_RES)


def parse_sections(text: str, title: str) -> list[tuple[tuple[str, ...], list[str]]]:
    """按 Markdown 标题和条款标记把正文切成「节」，每节带完整标题路径。"""
    sections: list[tuple[tuple[str, ...], list[str]]] = []
    stack: list[tuple[int, str]] = []
    body: list[str] = []
    path: tuple[str, ...] = ()
    deepest_md = 0

    def flush() -> None:
        nonlocal body
        if path or body:
            sections.append((path, body))
        body = []

    for line in text.split("\n"):
        heading = HEADING_RE.match(line)
        level: int | None = None
        heading_text = ""
        if heading:
            level = len(heading.group(1))
            heading_text = heading.group(2).strip()
            deepest_md = max(deepest_md, level)
        elif _is_clause_heading(line):
            level = deepest_md + 1  # 条款比当前最深的 Markdown 标题再低一级
            heading_text = line.strip()

        if level is not None and heading_text:
            flush()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading_text))
            path = tuple(item for _, item in stack)
        else:
            body.append(line)

    flush()

    # 标题路径的第一层固定是文档标题；相邻两层相同就去掉后一层（含第二层等于标题的情况）
    doc_title = title.strip()
    normalized: list[tuple[tuple[str, ...], list[str]]] = []
    for section_path, lines in sections:
        path = list(section_path) or [doc_title]
        if path[0].strip() != doc_title:
            path = [doc_title, *path]
        else:
            path[0] = doc_title
        deduped = [path[0]]
        for segment in path[1:]:
            if segment.strip() == deduped[-1].strip():
                continue
            deduped.append(segment)
        normalized.append((tuple(deduped), lines))
    return normalized


def _split_paragraphs(lines: list[str]) -> list[str]:
    """按空行切段落；连续的表格行算一个整体段落。"""
    paragraphs: list[str] = []
    current: list[str] = []
    in_table = False

    def flush() -> None:
        nonlocal current
        text = "\n".join(current).strip("\n")
        if text.strip():
            paragraphs.append(text)
        current = []

    for line in lines:
        is_table_row = bool(TABLE_ROW_RE.match(line))
        if not line.strip():
            if in_table:  # 表格中间的空行忽略，保持表格完整
                continue
            flush()
            continue
        if in_table and not is_table_row:
            flush()
        current.append(line)
        in_table = is_table_row
    flush()
    return paragraphs


def _split_by_chars(text: str, count: Tokenizer) -> list[str]:
    """单句仍然超长时按字符硬切。"""
    pieces: list[str] = []
    remaining = text
    while remaining:
        cut = min(MAX_CHARS, len(remaining))
        while cut > 1 and count(remaining[:cut]) > MAX_TOKENS:
            cut = int(cut * 0.8)
        cut = max(cut, 1)
        pieces.append(remaining[:cut])
        remaining = remaining[cut:]
    return pieces


def _split_sentences(text: str) -> list[str]:
    """按中英文句末标点切成句子。"""
    return [match.group(0) for match in SENTENCE_RE.finditer(text) if match.group(0).strip()]


def _split_table(paragraph: str, count: Tokenizer) -> list[str]:
    """超长表格按行切，每块都重复表头两行。"""
    rows = [line for line in paragraph.split("\n") if line.strip()]
    if len(rows) <= 2:
        return _split_by_chars(paragraph, count)
    header, body = rows[:2], rows[2:]
    pieces: list[str] = []
    current: list[str] = []
    for row in body:
        candidate = "\n".join([*header, *current, row])
        if current and (count(candidate) > MAX_TOKENS or len(candidate) > MAX_CHARS):
            pieces.append("\n".join([*header, *current]))
            current = [row]
        else:
            current.append(row)
    if current:
        pieces.append("\n".join([*header, *current]))
    return pieces or [paragraph]


def split_paragraph(paragraph: str, count: Tokenizer) -> list[str]:
    """把超长段落切成不超过上限的若干块；普通段落原样返回。"""
    if count(paragraph) <= MAX_TOKENS and len(paragraph) <= MAX_CHARS:
        return [paragraph]
    if TABLE_ROW_RE.match(paragraph):
        return _split_table(paragraph, count)
    sentences = [part for part in _split_sentences(paragraph) if part.strip()]
    if not sentences:
        return _split_by_chars(paragraph, count)

    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = current + sentence
        if current and (count(candidate) > MAX_TOKENS or len(candidate) > MAX_CHARS):
            pieces.append(current)
            current = sentence
        else:
            current = candidate
        if count(current) > MAX_TOKENS or len(current) > MAX_CHARS:
            pieces.extend(_split_by_chars(current, count))
            current = ""
    if current.strip():
        pieces.append(current)
    return pieces


def pack_paragraphs(paragraphs: list[str], count: Tokenizer) -> list[str]:
    """贪心合并段落：接近 TARGET 就断开，最多不超过 MAX。"""
    units: list[str] = []
    for paragraph in paragraphs:
        units.extend(split_paragraph(paragraph, count))

    chunks: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current}\n\n{unit}" if current else unit
        if current and (count(candidate) > MAX_TOKENS or len(candidate) > MAX_CHARS):
            chunks.append(current)
            current = unit
        else:
            current = candidate
        if count(current) >= TARGET_TOKENS:
            chunks.append(current)
            current = ""
    if current.strip():
        chunks.append(current)
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def _common_prefix(left: tuple[str, ...], right: tuple[str, ...]) -> tuple[str, ...]:
    """两段标题路径的公共前缀。"""
    size = 0
    for index in range(min(len(left), len(right))):
        if left[index] != right[index]:
            break
        size = index + 1
    return left[:size]


def merge_short_sections(
    sections: list[tuple[tuple[str, ...], list[str]]], count: Tokenizer
) -> list[tuple[tuple[str, ...], list[str]]]:
    """把内容太少的节并入下一个同级节；没有下一个就并入上一块。"""
    merged: list[tuple[tuple[str, ...], list[str]]] = []
    pending: list[tuple[tuple[str, ...], list[str]]] = list(sections)

    index = 0
    while index < len(pending):
        path, lines = pending[index]
        paragraphs = _split_paragraphs(lines)
        tokens = count("\n\n".join(paragraphs)) if paragraphs else 0
        if paragraphs and tokens < MIN_TOKENS:
            if index + 1 < len(pending):
                next_path, next_lines = pending[index + 1]
                pending[index + 1] = (_common_prefix(path, next_path), [*lines, "", *next_lines])
                index += 1
                continue
            if merged:
                merged[-1] = (merged[-1][0], [*merged[-1][1], ""] + lines)
                index += 1
                continue
        merged.append((path, lines))
        index += 1
    return merged


def _merge_tiny_chunks(
    chunks: list[dict], count: Tokenizer, stats: dict[str, int] | None = None
) -> list[dict]:
    """把过短的片段并入前一块（没有前一块就并入后一块）；并进去会超上限就丢弃。"""
    merged: list[dict] = []
    dropped = 0
    index = 0
    while index < len(chunks):
        chunk = chunks[index]
        if chunk["token_count"] >= MIN_CHUNK_TOKENS:
            merged.append(chunk)
            index += 1
            continue

        if merged:
            target = merged[-1]
        elif index + 1 < len(chunks):
            target = chunks[index + 1]
        else:
            target = None

        if target is None:
            dropped += 1
        else:
            combined = f"{target['text']}\n\n{chunk['text']}"
            if count(combined) <= MAX_TOKENS and len(combined) <= MAX_CHARS:
                target["text"] = combined
                target["token_count"] = count(combined)
            else:
                dropped += 1
        index += 1

    for position, chunk in enumerate(merged):
        chunk["chunk_index"] = position
    if stats is not None and dropped:
        stats["tiny_dropped"] = stats.get("tiny_dropped", 0) + dropped
    return merged


def chunk_document(
    document: dict, count: Tokenizer, stats: dict[str, int] | None = None
) -> list[dict]:
    """把一篇文档切成若干片段（不含 chunk_id，由调用方编号）。"""
    base = {
        "doc_key": document["doc_key"],
        "collection": document["collection"],
        "platform": document.get("platform"),
        "site": document.get("site"),
        "category": document.get("category"),
        "doc_type": document.get("doc_type"),
        "lang": document.get("lang"),
        "effective_date": document.get("effective_date"),
        "expire_date": document.get("expire_date"),
        "source_url": document["source_url"],
    }

    if document["collection"] == "xm_faq":
        text = document["text"].strip()
        return [
            {
                **base,
                "title_path": document.get("question") or document["title"],
                "text": text,
                "question": document.get("question"),
                "answer": document.get("answer"),
                "token_count": count(text),
                "chunk_index": 0,
            }
        ]

    sections = parse_sections(document["text"], document["title"])
    chunks: list[dict] = []
    for path, lines in merge_short_sections(sections, count):
        paragraphs = _split_paragraphs(lines)
        for text in pack_paragraphs(paragraphs, count):
            chunks.append(
                {
                    **base,
                    "title_path": PATH_SEP.join(path),
                    "text": text,
                    "question": None,
                    "answer": None,
                    "token_count": count(text),
                }
            )
    return _merge_tiny_chunks(chunks, count, stats)


def _read_jsonl(path: Path) -> list[dict]:
    """读 JSONL，跳过空行。"""
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    """先写临时文件再改名。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _percentile(values: list[int], ratio: float) -> float:
    """线性插值分位数。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * ratio
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize(chunks: list[dict]) -> str:
    """汇总：块数、分布和超长块占比。"""
    by_collection: dict[str, int] = {}
    for chunk in chunks:
        by_collection[chunk["collection"]] = by_collection.get(chunk["collection"], 0) + 1
    tokens = [chunk["token_count"] for chunk in chunks]
    over = sum(1 for value in tokens if value > TARGET_TOKENS)
    lines = [f"总块数 {len(chunks)}"]
    for collection in sorted(by_collection):
        lines.append(f"  {collection}: {by_collection[collection]} 块")
    if tokens:
        lines.append(
            "token_count 最小 {min} / 中位数 {median:.0f} / P95 {p95:.0f} / 最大 {max}".format(
                min=min(tokens),
                median=_percentile(tokens, 0.5),
                p95=_percentile(tokens, 0.95),
                max=max(tokens),
            )
        )
        lines.append(f"超过 {TARGET_TOKENS} token 的块 {over} 个（{over / len(tokens):.1%}）")
    return "\n".join(lines)


def main() -> int:
    """读 documents.jsonl，产出 chunks.jsonl。"""
    parser = argparse.ArgumentParser(description="分块：生成 chunks.jsonl")
    parser.add_argument("--docs", type=Path, default=Path("data/processed/documents.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/chunks.jsonl"))
    args = parser.parse_args()

    if not args.docs.is_file():
        print(f"[ERROR] 找不到 {args.docs}，先跑 python -m xiaomao.ingest.parse")
        return 1

    count = _default_tokenizer
    documents = _read_jsonl(args.docs)
    stats: dict[str, int] = {}
    rows: list[dict] = []
    for document in documents:
        chunks = chunk_document(document, count, stats)
        for chunk in chunks:
            rows.append({"chunk_id": f"{chunk['doc_key']}-{chunk['chunk_index']:04d}", **chunk})

    _write_jsonl(args.out, rows)
    print(f"读入文档 {len(documents)} 篇")
    print(summarize(rows))
    if stats.get("tiny_dropped"):
        print(f"过短且并不进去、已丢弃的片段：{stats['tiny_dropped']} 个")
    print(f"输出 -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
