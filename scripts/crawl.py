"""按 data/sources.jsonl 抓取公开页面，原文存到 data/raw/，登记到 data/raw/manifest.jsonl。

抓取规则（学习项目，小规模）：
- 每个 URL 抓取前检查 robots.txt，不允许就跳过
- 同一站点两次请求间隔至少 --delay 秒（robots.txt 声明的 Crawl-delay 更大时以它为准）
- User-Agent 如实标明是学习项目的爬虫
- 被重定向到首页、需要登录或返回反爬挑战页的，直接跳过，不做任何绕过

用法：
  python scripts/crawl.py                         # 抓取 sources.jsonl 里还没抓过的
  python scripts/crawl.py --probe urls.txt        # 只探测：每个 URL 能否直接拿到正文，不落盘
"""

import argparse
import hashlib
import json
import sys
import time
import urllib.robotparser
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import trafilatura

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
UA = "xiaomao-study-bot/0.1 (personal RAG learning project; low-rate crawl)"
CN_TZ = timezone(timedelta(hours=8))
MIN_TEXT_CHARS = 200
# 抓取清单里每行必须有的元数据（见 docs/data_format.md 第 2a 节）
META_FIELDS = ["collection", "source_url", "title", "platform", "site", "category",
               "doc_type", "lang", "effective_date", "expire_date"]


def doc_key(collection: str, source_url: str) -> str:
    return hashlib.sha256(f"{collection}|{source_url}".encode("utf-8")).hexdigest()[:16]


class PoliteClient:
    """带 robots.txt 检查和按站点限速的 HTTP 客户端。"""

    def __init__(self, delay: float) -> None:
        # 只接受 gzip/deflate：httpx 未装 brotli 时，br 压缩的响应会解码失败
        self.http = httpx.Client(
            headers={"User-Agent": UA, "Accept-Encoding": "gzip, deflate",
                     "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"},
            timeout=30, follow_redirects=True)
        self.delay = delay
        self.robots: dict[str, urllib.robotparser.RobotFileParser] = {}
        self.last_hit: dict[str, float] = {}

    def _robots(self, host: str) -> urllib.robotparser.RobotFileParser:
        if host not in self.robots:
            rp = urllib.robotparser.RobotFileParser()
            try:
                r = self.http.get(f"https://{host}/robots.txt")
                # 没有 robots.txt（404）视为全部允许；其他异常状态视为不允许
                if r.status_code == 200:
                    rp.parse(r.text.splitlines())
                elif r.status_code == 404:
                    rp.parse([])
                else:
                    rp.parse(["User-agent: *", "Disallow: /"])
            except httpx.HTTPError:
                rp.parse(["User-agent: *", "Disallow: /"])
            self.robots[host] = rp
        return self.robots[host]

    def fetch(self, url: str) -> tuple[httpx.Response | None, str]:
        """返回 (响应, 跳过原因)。跳过原因为空表示可用。"""
        host = urlsplit(url).netloc
        rp = self._robots(host)
        if not rp.can_fetch(UA, url):
            return None, "robots_disallow"
        wait = max(self.delay, rp.crawl_delay(UA) or 0)
        elapsed = time.monotonic() - self.last_hit.get(host, 0)
        if elapsed < wait:
            time.sleep(wait - elapsed)
        try:
            resp = self.http.get(url)
        except httpx.HTTPError as exc:
            return None, f"http_error:{type(exc).__name__}"
        finally:
            self.last_hit[host] = time.monotonic()
        if resp.status_code != 200:
            return resp, f"http_{resp.status_code}"
        final = urlsplit(str(resp.url))
        if final.path in ("", "/") and urlsplit(url).path not in ("", "/"):
            return resp, "redirected_to_home"
        if any(k in final.path.lower() for k in ("login", "signin", "sign-in", "passport")):
            return resp, "login_required"
        return resp, ""


def extract_text(resp: httpx.Response) -> str:
    ctype = resp.headers.get("content-type", "")
    if "pdf" in ctype:
        return "PDF"  # PDF 留给 parse.py 处理，探测时只确认能下载
    return trafilatura.extract(resp.text, output_format="markdown", include_tables=True) or ""


def probe(client: PoliteClient, urls: list[str]) -> None:
    for url in urls:
        resp, reason = client.fetch(url)
        if reason:
            print(f"SKIP {reason:<20} {url}")
            continue
        text = extract_text(resp)
        meta = trafilatura.extract_metadata(resp.text) if text != "PDF" else None
        # 不截断：这里打印的标题会被抄进 sources.jsonl，截断过的标题曾导致 title_path 重复一层
        title = (meta.title if meta and meta.title else "").replace("\t", " ")
        status = "OK  " if text == "PDF" or len(text) >= MIN_TEXT_CHARS else "THIN"
        print(f"{status} {len(text):>6}  {url}\t{title}")


def crawl(client: PoliteClient, sources: Path, raw_dir: Path) -> None:
    manifest = raw_dir / "manifest.jsonl"
    done = set()
    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["doc_key"])
    entries = [json.loads(l) for l in sources.read_text(encoding="utf-8").splitlines() if l.strip()]
    stats: dict[str, int] = {}
    # 同一页面的不同标签页（如 /japan/faq 和 /japan/pricing）常常渲染出完全相同的正文，按正文去重
    seen_text: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines() if manifest.exists() else []:
        if line.strip():
            row = json.loads(line)
            if row.get("text_sha"):
                seen_text[row["text_sha"]] = row["source_url"]
    with manifest.open("a", encoding="utf-8") as out:
        for i, src in enumerate(entries, 1):
            missing = [f for f in META_FIELDS if f not in src]
            if missing:
                print(f"[{i}/{len(entries)}] 缺字段 {missing}：{src.get('source_url')}")
                stats["bad_source"] = stats.get("bad_source", 0) + 1
                continue
            key = doc_key(src["collection"], src["source_url"])
            if key in done:
                stats["already_done"] = stats.get("already_done", 0) + 1
                continue
            resp, reason = client.fetch(src["source_url"])
            text_sha = ""
            if not reason:
                text = extract_text(resp)
                if text != "PDF" and len(text) < MIN_TEXT_CHARS:
                    reason = "thin_content"
                elif text != "PDF":
                    text_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
                    if text_sha in seen_text:
                        reason = "duplicate_content"
                        print(f"[{i}/{len(entries)}] 正文与 {seen_text[text_sha]} 相同")
            if reason:
                print(f"[{i}/{len(entries)}] 跳过 {reason}：{src['source_url']}")
                stats[reason] = stats.get(reason, 0) + 1
                continue
            is_pdf = "pdf" in resp.headers.get("content-type", "")
            raw_path = raw_dir / f"{key}.{'pdf' if is_pdf else 'html'}"
            raw_path.write_bytes(resp.content)
            row = {"doc_key": key, **{f: src[f] for f in META_FIELDS},
                   "raw_path": raw_path.relative_to(ROOT).as_posix(),
                   "content_type": "application/pdf" if is_pdf else "text/html",
                   "http_status": resp.status_code,
                   "fetched_at": datetime.now(CN_TZ).isoformat(timespec="seconds"),
                   "text_sha": text_sha,
                   "acquired_by": "crawler"}
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            done.add(key)
            if text_sha:
                seen_text[text_sha] = src["source_url"]
            stats["saved"] = stats.get("saved", 0) + 1
            print(f"[{i}/{len(entries)}] 已保存 {raw_path.name}  {src['title'][:40]}")
    print("\n汇总：" + "，".join(f"{k} {v}" for k, v in sorted(stats.items())))


def main() -> None:
    ap = argparse.ArgumentParser(description="小贸同学：低频率抓取公开政策页面")
    ap.add_argument("--sources", type=Path, default=ROOT / "data" / "sources.jsonl")
    ap.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    ap.add_argument("--delay", type=float, default=2.0, help="同一站点两次请求的最小间隔（秒）")
    ap.add_argument("--probe", type=Path, help="只探测这个文件里的 URL（一行一个），不保存")
    args = ap.parse_args()
    client = PoliteClient(args.delay)
    if args.probe:
        urls = [u.strip() for u in args.probe.read_text(encoding="utf-8").splitlines() if u.strip() and not u.startswith("#")]
        probe(client, urls)
        return
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    crawl(client, args.sources, args.raw_dir)


if __name__ == "__main__":
    main()
