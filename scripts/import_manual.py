"""把用户手动保存的网页和 PDF 登记进 data/raw/manifest.jsonl。

用法：
  1. 把页面另存为 HTML（"网页，仅 HTML"或"网页，全部"都行，文件名随意），放进 data/raw/manual/
  2. python scripts/import_manual.py

怎么对上元数据：
  - HTML：从文件里浏览器写下的原网址（"saved from url" 注释、canonical、og:url）匹配
    data/manual_sources.jsonl；也认 m01.html 这种按编号命名的文件
  - 清单以外的文件（包括 PDF）：在 data/manual_extra.jsonl 里按文件名写一行元数据

已经登记过的会跳过，可以分批保存、反复运行。
"""

import argparse
import contextlib
import hashlib
import io
import json
import re
import shutil
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import trafilatura

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
CN_TZ = timezone(timedelta(hours=8))
MIN_TEXT_CHARS = 200
META_FIELDS = ["collection", "source_url", "title", "platform", "site", "category",
               "doc_type", "lang", "effective_date", "expire_date"]
# gs.amazon.cn 和 globalselling.amazon.com 是同一个站，浏览器保存时记下的可能是后者
URL_ALIASES = {"https://globalselling.amazon.com": "https://gs.amazon.cn"}


def doc_key(collection: str, source_url: str) -> str:
    return hashlib.sha256(f"{collection}|{source_url}".encode("utf-8")).hexdigest()[:16]


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def normalize_url(url: str) -> str:
    url = url.split("?")[0].split("#")[0].rstrip("/")
    for old, new in URL_ALIASES.items():
        if url.startswith(old):
            url = new + url[len(old):]
    return url


def original_url(html: str) -> str:
    """取浏览器另存为时记下的原网址。"""
    for pat in (r"saved from url=\(\d+\)(\S+)",
                r'<link[^>]+rel="canonical"[^>]+href="([^"]+)"',
                r'<link[^>]+href="([^"]+)"[^>]+rel="canonical"',
                r'<meta[^>]+property="og:url"[^>]+content="([^"]+)"'):
        m = re.search(pat, html)
        if m:
            return normalize_url(m.group(1))
    return ""


def extract(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pymupdf import pymupdf4llm
        with contextlib.redirect_stdout(io.StringIO()):  # 它会往 stdout 打印解析提示
            return pymupdf4llm.to_markdown(str(path))
    html = path.read_bytes().decode("utf-8", errors="replace")
    return trafilatura.extract(html, output_format="markdown", include_tables=True) or ""


def main() -> None:
    ap = argparse.ArgumentParser(description="登记手动保存的网页和 PDF")
    ap.add_argument("--list", type=Path, default=ROOT / "data" / "manual_sources.jsonl")
    ap.add_argument("--extra", type=Path, default=ROOT / "data" / "manual_extra.jsonl")
    ap.add_argument("--manual-dir", type=Path, default=ROOT / "data" / "raw" / "manual")
    ap.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    args = ap.parse_args()

    by_url = {normalize_url(r["source_url"]): r for r in read_jsonl(args.list)}
    by_id = {r["manual_id"]: r for r in read_jsonl(args.list)}
    by_name = {r["filename"]: r for r in read_jsonl(args.extra)}

    manifest = args.raw_dir / "manifest.jsonl"
    existing = read_jsonl(manifest)
    done = {r["doc_key"] for r in existing}
    seen_text = {r["text_sha"]: r["source_url"] for r in existing if r.get("text_sha")}

    stats = {k: 0 for k in ("imported", "already_done", "no_metadata", "thin_content", "duplicate_content")}
    matched_ids = set()
    files = sorted(p for p in args.manual_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in (".html", ".htm", ".pdf"))
    with manifest.open("a", encoding="utf-8") as out:
        for path in files:
            item = by_name.get(path.name) or by_id.get(path.stem)
            if item is None and path.suffix.lower() != ".pdf":
                head = path.read_bytes()[:500_000].decode("utf-8", errors="replace")
                item = by_url.get(original_url(head))
            if item is None:
                print(f"找不到元数据，跳过：{path.name}（清单以外的文件请写进 {args.extra.name}）")
                stats["no_metadata"] += 1
                continue
            if "manual_id" in item:
                matched_ids.add(item["manual_id"])
            key = doc_key(item["collection"], item["source_url"])
            if key in done:
                stats["already_done"] += 1
                continue
            text = extract(path)
            if len(text) < MIN_TEXT_CHARS:
                print(f"{path.name} 正文只有 {len(text)} 字，可能保存的不是完整网页，请重新保存")
                stats["thin_content"] += 1
                continue
            text_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
            if text_sha in seen_text:
                print(f"{path.name} 正文与 {seen_text[text_sha]} 相同，跳过")
                stats["duplicate_content"] += 1
                continue
            is_pdf = path.suffix.lower() == ".pdf"
            dest = args.raw_dir / f"{key}.{'pdf' if is_pdf else 'html'}"
            shutil.copyfile(path, dest)
            saved_at = datetime.fromtimestamp(path.stat().st_mtime, CN_TZ).isoformat(timespec="seconds")
            row = {"doc_key": key, **{f: item[f] for f in META_FIELDS},
                   "raw_path": (dest.relative_to(ROOT) if dest.is_relative_to(ROOT) else dest).as_posix(),
                   "content_type": "application/pdf" if is_pdf else "text/html",
                   "http_status": 200, "fetched_at": saved_at, "text_sha": "" if is_pdf else text_sha,
                   "acquired_by": "manual"}
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            done.add(key)
            if not is_pdf:
                seen_text[text_sha] = item["source_url"]
            stats["imported"] += 1
            label = item.get("manual_id", "额外")
            print(f"{label:<4} 已登记（正文 {len(text)} 字）：{item['title'][:30]}")

    print("\n汇总：" + "，".join(f"{k} {v}" for k, v in stats.items()))
    skipped = sorted(set(by_id) - matched_ids)
    if skipped:
        print(f"清单里没有保存的 {len(skipped)} 个：" + " ".join(skipped))


if __name__ == "__main__":
    main()
