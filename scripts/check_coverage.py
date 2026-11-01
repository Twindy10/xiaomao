"""抽取覆盖率：对比 documents.jsonl 的正文和原始 HTML 主内容的可见文字，找出被解析丢掉正文的站点。

覆盖率 = 抽出的字数 / 页面主内容可见字数（都去掉空白和 Markdown 符号）。
只是粗估：分母里还残留一些菜单和侧栏，所以 0.85 以上基本正常，远低于这个值就说明丢了正文；
明显大于 1 说明混进了页面以外的文字（比如菜单）。

用法：
    python scripts/check_coverage.py [--show 15]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

from lxml import html as LH

sys.stdout.reconfigure(encoding="utf-8")  # Windows 控制台默认 GBK

ROOT_DIR = Path(__file__).resolve().parents[1]
CHROME_XPATH = (
    "//nav|//header|//footer|//script|//style|//noscript|//aside"
    "|//*[contains(@class,'sidecar')]|//*[contains(@class,'breadcrumb')]"
    # Amazon 的菜单、登录/注册弹窗、反馈问卷、语言选择（和 parse.py 的 AMAZON_DROP 对应）
    "|//*[starts-with(@id,'GLOBALSELLING-')]|//*[starts-with(@id,'as_cn_ags_megamenu_')]"
    "|//*[starts-with(@id,'hmd-')]|//*[contains(@id,'picker-popup')]"
)


def _norm(text: str) -> str:
    """去掉空白和 Markdown 符号，只比较文字本身。"""
    return re.sub(r"[\s*\\#|\-]+", "", text)


def _visible_chars(raw: bytes) -> int:
    """原始 HTML 里主内容的可见字数（去掉导航、页眉页脚、侧栏）。"""
    tree = LH.fromstring(raw)
    for node in tree.xpath(CHROME_XPATH):
        node.drop_tree()
    main = tree.xpath("//main") or [tree]
    return len(_norm(main[0].text_content()))


def main() -> int:
    """按站点汇总覆盖率，并列出最低的若干篇。"""
    parser = argparse.ArgumentParser(description="统计解析覆盖率")
    parser.add_argument("--manifest", type=Path, default=ROOT_DIR / "data/raw/manifest.jsonl")
    parser.add_argument("--docs", type=Path, default=ROOT_DIR / "data/processed/documents.jsonl")
    parser.add_argument("--show", type=int, default=15, help="列出覆盖率最低的 N 篇")
    args = parser.parse_args()

    manifest = {
        row["doc_key"]: row
        for row in map(json.loads, args.manifest.read_text(encoding="utf-8").splitlines())
    }
    docs = [json.loads(line) for line in args.docs.read_text(encoding="utf-8").splitlines() if line.strip()]

    rows: list[tuple[float, int, int, str, str]] = []
    for doc in docs:
        entry = manifest[doc["doc_key"]]
        if "pdf" in entry["content_type"]:
            continue
        visible = _visible_chars((ROOT_DIR / entry["raw_path"]).read_bytes()) or 1
        got = len(_norm(doc["text"]))
        rows.append((got / visible, got, visible, urlparse(doc["source_url"]).netloc, doc["title"][:45]))

    by_host: dict[str, list[float]] = defaultdict(list)
    for ratio, _, _, host, _ in rows:
        by_host[host].append(ratio)
    print(f"{'站点':34s} 篇数  中位数  最低  最高")
    for host, ratios in sorted(by_host.items()):
        ratios.sort()
        print(f"{host:36s} {len(ratios):3d}  {ratios[len(ratios) // 2]:.2f}  {ratios[0]:.2f}  {ratios[-1]:.2f}")

    print(f"\n覆盖率最低的 {args.show} 篇：")
    for ratio, got, visible, host, title in sorted(rows)[: args.show]:
        print(f"  {ratio:.2f}  抽出 {got:6d} / 可见 {visible:6d}  {host:24s} {title}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
