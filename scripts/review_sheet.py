"""C5 人工审核表：把评估集和 FAQ 抽检导出成 Excel，审完再导回。

用法：
    uv run python scripts/review_sheet.py export [--out data/review/审核表.xlsx] [--faq-ratio 0.1]
    uv run python scripts/review_sheet.py import data/review/审核表.xlsx [--dry-run]

- 只导出还没人审的行（评估集 reviewed_by != human；FAQ 是还没抽检过的），可以分几次审完。
- 每行在"结论"列选"通过"或"删除"；要改内容直接改白底的单元格，灰底的列改了也不会导入。
- 结论留空的行导入时原样跳过。导入前先整体校验，有任何错误就一行都不写。
- 删除的评估题移到 data/eval/rejected.jsonl；删除的 FAQ 标成 reviewed=false，不进知识库。
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

from xiaomao.textnorm import normalize_for_match

sys.stdout.reconfigure(encoding="utf-8")  # Windows 控制台默认 GBK

ROOT_DIR = Path(__file__).resolve().parents[1]

INTENTS = ("platform_policy", "compliance", "logistics", "tax", "ops", "account", "chitchat", "out_of_scope")
STRATEGIES = ("direct", "subquery", "hyde")
PLATFORMS = ("amazon", "temu", "tiktok_shop", "shopee")
NO_STRATEGY_INTENTS = ("chitchat", "out_of_scope")
VERDICTS = ("通过", "删除")
FAQ_CHECKED = "[人工抽检通过]"

READONLY_FILL = PatternFill("solid", fgColor="EEEEEE")
HEADER_FILL = PatternFill("solid", fgColor="DDEBF7")
VERDICT_FILL = PatternFill("solid", fgColor="FFF2CC")


@dataclass(frozen=True)
class Column:
    """审核表的一列。key 是数据字段名；editable=False 的列导入时忽略。"""

    header: str
    key: str
    width: int
    editable: bool = False
    choices: tuple[str, ...] = ()


INTENT_COLUMNS = (
    Column("编号", "id", 10),
    Column("问题", "query", 50, editable=True),
    Column("意图", "intent", 16, editable=True, choices=INTENTS),
    Column("策略", "strategy", 11, editable=True, choices=STRATEGIES),
    Column("平台", "platform", 12, editable=True, choices=PLATFORMS),
    Column("站点", "site", 8, editable=True),
    Column("生成备注", "note", 24),
    Column("结论", "_verdict", 8, editable=True, choices=VERDICTS),
    Column("审核备注", "_comment", 24, editable=True),
)

QA_COLUMNS = (
    Column("编号", "id", 9),
    Column("问题", "query", 36, editable=True),
    Column("可回答", "answerable", 8),
    Column("参考答案", "reference_answer", 45, editable=True),
    Column("同义改写（一行一条）", "paraphrases", 36, editable=True),
    Column("意图", "intent", 15, editable=True, choices=INTENTS),
    Column("平台", "platform", 11, editable=True, choices=PLATFORMS),
    Column("站点", "site", 7, editable=True),
    Column("证据引文", "_evidence", 45),
    Column("证据所在片段（核对参考答案用）", "_context", 70),
    Column("来源", "_source", 30),
    Column("生成备注", "note", 20),
    Column("结论", "_verdict", 8, editable=True, choices=VERDICTS),
    Column("审核备注", "_comment", 24, editable=True),
)

FAQ_COLUMNS = (
    Column("编号", "faq_id", 10),
    Column("问题", "question", 36, editable=True),
    Column("答案", "answer", 50, editable=True),
    Column("证据引文", "_evidence", 40),
    Column("证据所在片段（核对答案用）", "_context", 70),
    Column("来源", "_source", 30),
    Column("初审备注", "note", 20),
    Column("结论", "_verdict", 8, editable=True, choices=VERDICTS),
    Column("审核备注", "_comment", 24, editable=True),
)

SHEETS = {"意图集": INTENT_COLUMNS, "检索集": QA_COLUMNS, "FAQ抽检": FAQ_COLUMNS}


class Paths:
    """数据文件位置；测试时指向临时目录。"""

    def __init__(self, data_dir: Path) -> None:
        self.intent = data_dir / "eval" / "intent.jsonl"
        self.qa = data_dir / "eval" / "retrieval_qa.jsonl"
        self.rejected = data_dir / "eval" / "rejected.jsonl"
        self.faq = data_dir / "processed" / "faq_items.jsonl"
        self.documents = data_dir / "processed" / "documents.jsonl"
        self.chunks = data_dir / "processed" / "chunks.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    """读 JSONL，跳过空行；文件不存在返回空列表。"""
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """先写临时文件再改名。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


# ---------------------------------------------------------------- 导出


class EvidenceLookup:
    """把 evidence 展开成 引文 / 所在片段 / 来源 三段文字。"""

    def __init__(self, paths: Paths) -> None:
        self.docs = {row["doc_key"]: row for row in read_jsonl(paths.documents)}
        self.chunks: dict[str, list[dict]] = {}
        for chunk in read_jsonl(paths.chunks):
            self.chunks.setdefault(chunk["doc_key"], []).append(chunk)

    def describe(self, evidence: list[dict]) -> tuple[str, str, str]:
        quotes, contexts, sources = [], [], []
        for index, entry in enumerate(evidence, 1):
            doc = self.docs.get(entry["doc_key"], {})
            mark = f"[{index}] " if len(evidence) > 1 else ""
            quotes.append(f"{mark}{entry['quote']}")
            contexts.append(f"{mark}{self._chunk_text(entry)}")
            sources.append(f"{mark}{doc.get('title', '')}\n{doc.get('source_url', '')}")
        return "\n\n".join(quotes), "\n\n".join(contexts), "\n\n".join(sources)

    def _chunk_text(self, entry: dict) -> str:
        quote = normalize_for_match(entry["quote"])
        for chunk in self.chunks.get(entry["doc_key"], []):
            if quote in normalize_for_match(chunk["text"]):
                return f"（{chunk['title_path']}）\n{chunk['text']}"
        return "（引文不在单个片段里）"


def _cell_value(row: dict, key: str) -> object:
    value = row.get(key)
    if key == "paraphrases":
        return "\n".join(value or [])
    if key == "answerable":
        return "是" if value else "否"
    return value


def _fill_sheet(sheet, columns: tuple[Column, ...], rows: list[dict]) -> None:
    """写表头、数据、下拉框和样式。"""
    sheet.append([column.header for column in columns])
    for index, column in enumerate(columns, 1):
        cell = sheet.cell(row=1, column=index)
        cell.font = Font(bold=True)
        cell.fill = VERDICT_FILL if column.key == "_verdict" else HEADER_FILL
        sheet.column_dimensions[cell.column_letter].width = column.width
    for row in rows:
        sheet.append([_cell_value(row, column.key) for column in columns])

    last = sheet.max_row
    for index, column in enumerate(columns, 1):
        letter = sheet.cell(row=1, column=index).column_letter
        for row_index in range(2, last + 1):
            cell = sheet.cell(row=row_index, column=index)
            cell.alignment = Alignment(wrap_text=True, vertical="top")
            if not column.editable:
                cell.fill = READONLY_FILL
            elif column.key == "_verdict":
                cell.fill = VERDICT_FILL
        if column.choices and last >= 2:
            validation = DataValidation(type="list", formula1='"' + ",".join(column.choices) + '"', allow_blank=True)
            validation.add(f"{letter}2:{letter}{last}")
            sheet.add_data_validation(validation)
    sheet.freeze_panes = "C2"
    sheet.auto_filter.ref = sheet.dimensions


def _coverage_text(docs: dict[str, dict]) -> str:
    """知识库覆盖范围，审"不可回答"题时对照。"""
    counter = Counter(
        (row["collection"], row.get("platform") or "-", row.get("site") or "不限")
        for row in docs.values()
        if row["collection"] != "xm_faq"
    )
    lines = [f"  {collection}  平台 {platform}  站点 {site}：{count} 篇" for (collection, platform, site), count in sorted(counter.items())]
    return "\n".join(lines)


def _instructions(counts: dict[str, int], coverage: str) -> list[str]:
    return [
        "人工审核说明",
        "",
        f"本表共 {sum(counts.values())} 行：" + "，".join(f"{name} {count}" for name, count in counts.items()),
        "预计用时：意图集约 10 秒一条，检索集约 1 分钟一条（要核对证据），FAQ 抽检约 30 秒一条。可以分几次审，每次导入一部分。",
        "",
        "怎么填",
        "1. 每行在黄色的“结论”列选“通过”或“删除”。留空的行这次不导入，下次导出还会出现。",
        "2. 内容有小问题就直接改白底的单元格再选“通过”；问题大到改不动就选“删除”，在“审核备注”写原因。",
        "3. 灰底的列只供参考，改了也不会导入。",
        "",
        "意图集看什么",
        "- 意图是否标对；策略：direct 只问一个点，subquery 一句话问了 2–3 个点，hyde 先描述具体情况再求助；闲聊和超范围的策略留空。",
        "- 平台、站点只标问题里明确提到的，没提就留空，不要按常识推测。平台：amazon / temu / tiktok_shop / shopee；站点用大写国家代码，如 US、DE、UK、JP，欧盟整体写 EU。",
        "- 问题要像卖家真会问的话；太假、太像、和别的题重复的删掉。",
        "",
        "检索集看什么",
        "- 可回答的题：参考答案的每一句都要能在“证据所在片段”里找到依据；有超出片段的说法，删掉那句或者整题删除。",
        "- 同义改写至少 2 条，意思要和问题一样。",
        "- 不可回答的题：确认知识库里确实没有答案（对照下面的覆盖范围）。知识库其实能答的，删除。",
        "- 表里已经由 Claude 审过的题是抽检：选“通过”只会记一笔人工抽检，不会改成人审；发现问题照样可以改或删。",
        "",
        "FAQ 抽检看什么",
        "- 答案是否忠于“证据所在片段”、有没有答非所问。这些已经经过模型筛查和 Claude 初审，抽检是为了确认初审靠得住。",
        "",
        "改完保存，然后运行：uv run python scripts/review_sheet.py import <这个文件>",
        "",
        "知识库覆盖范围",
        coverage,
    ]


def cmd_export(args: argparse.Namespace) -> int:
    """导出审核表。"""
    paths = Paths(args.data_dir)
    lookup = EvidenceLookup(paths)

    intents = [row for row in read_jsonl(paths.intent) if row.get("reviewed_by") != "human"]
    all_qa = read_jsonl(paths.qa)
    # 还没审的题全部导出；Claude 审过的题再按比例抽一部分给用户抽检
    qa_rows = [row for row in all_qa if not row.get("reviewed")]
    by_claude = [row for row in all_qa if row.get("reviewed_by") == "claude"]
    qa_checked = sum(1 for row in by_claude if FAQ_CHECKED in (row.get("note") or ""))
    qa_need = max(0, math.ceil(len(by_claude) * args.qa_spot_ratio) - qa_checked)
    qa_pool = [row for row in by_claude if FAQ_CHECKED not in (row.get("note") or "")]
    qa_rows += random.Random(args.seed).sample(qa_pool, min(qa_need, len(qa_pool)))
    qa_rows.sort(key=lambda row: row["id"])
    for row in qa_rows:
        row["_evidence"], row["_context"], row["_source"] = lookup.describe(row.get("evidence") or [])

    faqs = read_jsonl(paths.faq)
    reviewed = [row for row in faqs if row.get("reviewed") is True]
    checked = sum(1 for row in reviewed if FAQ_CHECKED in (row.get("note") or ""))
    target = args.faq_count if args.faq_count is not None else math.ceil(len(reviewed) * args.faq_ratio)
    need = max(0, target - checked)
    pool = [row for row in reviewed if FAQ_CHECKED not in (row.get("note") or "")]
    sample = sorted(random.Random(args.seed).sample(pool, min(need, len(pool))), key=lambda row: row["faq_id"])
    for row in sample:
        row["_evidence"], row["_context"], row["_source"] = lookup.describe(row["evidence"])

    workbook = Workbook()
    guide = workbook.active
    guide.title = "说明"
    counts = {"意图集": len(intents), "检索集": len(qa_rows), "FAQ抽检": len(sample)}
    for line in _instructions(counts, _coverage_text(lookup.docs)):
        guide.append([line])
    guide.column_dimensions["A"].width = 120
    guide["A1"].font = Font(bold=True, size=14)
    for name, rows in (("意图集", intents), ("检索集", qa_rows), ("FAQ抽检", sample)):
        _fill_sheet(workbook.create_sheet(name), SHEETS[name], rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(args.out)
    print(f"导出 {args.out}")
    for name, count in counts.items():
        print(f"  {name}：{count} 行")
    print(f"  （检索集里 Claude 审过 {len(by_claude)} 条，已抽检 {qa_checked} 条；FAQ 已抽检 {checked} 条，目标 {target} 条）")
    return 0


# ---------------------------------------------------------------- 导入


def _clean(value: object) -> str | None:
    """单元格值转成字符串；空白当作 None。"""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _sheet_rows(sheet, columns: tuple[Column, ...]) -> list[tuple[int, dict]]:
    """按表头读出每一行；表头和导出时不一致就报错。"""
    headers = list(next(sheet.iter_rows(min_row=1, max_row=1, values_only=True), ()))
    expected = [column.header for column in columns]
    if headers[: len(expected)] != expected:
        raise ValueError(f"「{sheet.title}」的表头和导出时不一致，不要增删或移动列")
    rows = []
    for row_index, values in enumerate(sheet.iter_rows(min_row=2, values_only=True), 2):
        if all(value is None for value in values):
            continue
        rows.append((row_index, {column.key: _clean(value) for column, value in zip(columns, values)}))
    return rows


def _check_common(cells: dict, errors: list[str], where: str) -> None:
    """意图、平台、站点的取值检查。"""
    if cells.get("intent") not in INTENTS:
        errors.append(f"{where}：意图必须是 {' / '.join(INTENTS)} 之一，现在是 {cells.get('intent')!r}")
    if cells.get("platform") is not None and cells["platform"] not in PLATFORMS:
        errors.append(f"{where}：平台必须是 {' / '.join(PLATFORMS)} 或留空")
    site = cells.get("site")
    if site is not None and not (site.isascii() and site.isalpha() and site.isupper() and len(site) == 2):
        errors.append(f"{where}：站点要写两位大写代码（如 US、DE、EU）或留空，现在是 {site!r}")


def _apply_intent(row: dict, cells: dict, errors: list[str], where: str) -> None:
    _check_common(cells, errors, where)
    strategy = cells.get("strategy")
    if cells.get("intent") in NO_STRATEGY_INTENTS:
        if strategy is not None:
            errors.append(f"{where}：闲聊和超范围的题策略要留空")
    elif strategy not in STRATEGIES:
        errors.append(f"{where}：策略必须是 {' / '.join(STRATEGIES)} 之一")
    if not cells.get("query"):
        errors.append(f"{where}：问题不能为空")
    for key in ("query", "intent", "strategy", "platform", "site"):
        row[key] = cells.get(key)


def _apply_qa(row: dict, cells: dict, errors: list[str], where: str) -> None:
    _check_common(cells, errors, where)
    paraphrases = [line.strip() for line in (cells.get("paraphrases") or "").splitlines() if line.strip()]
    if len(paraphrases) < 2:
        errors.append(f"{where}：同义改写至少 2 条（一行一条）")
    if not cells.get("query"):
        errors.append(f"{where}：问题不能为空")
    answer = cells.get("reference_answer")
    if row.get("answerable") and not answer:
        errors.append(f"{where}：可回答的题必须有参考答案")
    if not row.get("answerable") and answer:
        errors.append(f"{where}：不可回答的题参考答案要留空")
    for key in ("query", "intent", "platform", "site"):
        row[key] = cells.get(key)
    row["reference_answer"] = answer
    row["paraphrases"] = paraphrases


def _apply_faq(row: dict, cells: dict, errors: list[str], where: str) -> None:
    if not cells.get("question") or not cells.get("answer"):
        errors.append(f"{where}：问题和答案都不能为空")
    elif not 20 <= len(cells["answer"]) <= 400:
        errors.append(f"{where}：答案要 20–400 字，现在 {len(cells['answer'])} 字")
    row["question"], row["answer"] = cells.get("question"), cells.get("answer")


def _merge_note(old: str | None, extra: str | None) -> str | None:
    parts = [part for part in (old, extra) if part]
    return "；".join(parts) or None


def import_workbook(path: Path, paths: Paths, dry_run: bool = False) -> tuple[Counter, list[str]]:
    """读入审完的表，校验通过才写回。返回 (统计, 错误列表)。"""
    workbook = load_workbook(path, read_only=True)
    stats: Counter[str] = Counter()
    errors: list[str] = []

    data = {"意图集": read_jsonl(paths.intent), "检索集": read_jsonl(paths.qa), "FAQ抽检": read_jsonl(paths.faq)}
    id_key = {"意图集": "id", "检索集": "id", "FAQ抽检": "faq_id"}
    appliers = {"意图集": _apply_intent, "检索集": _apply_qa, "FAQ抽检": _apply_faq}
    removed: dict[str, set[str]] = {"意图集": set(), "检索集": set()}
    rejected = read_jsonl(paths.rejected)

    for name, columns in SHEETS.items():
        if name not in workbook.sheetnames:
            continue
        index = {row[id_key[name]]: row for row in data[name]}
        for row_index, cells in _sheet_rows(workbook[name], columns):
            where = f"「{name}」第 {row_index} 行"
            verdict = cells.get("_verdict")
            if verdict is None:
                stats[f"{name} 未审"] += 1
                continue
            if verdict not in VERDICTS:
                errors.append(f"{where}：结论只能选“通过”或“删除”")
                continue
            row = index.get(cells.get(id_key[name]))
            if row is None:
                errors.append(f"{where}：编号 {cells.get(id_key[name])!r} 在数据文件里找不到")
                continue
            comment = cells.get("_comment")
            if name == "FAQ抽检":
                if verdict == "通过":
                    candidate = dict(row)
                    appliers[name](candidate, cells, errors, where)
                    row.update(candidate)
                    row["note"] = _merge_note(row.get("note"), FAQ_CHECKED + (comment or ""))
                else:
                    row["reviewed"], row["reviewed_by"] = False, None
                    row["note"] = _merge_note(row.get("note"), f"人工抽检删除：{comment or '未注明'}")
            elif verdict == "通过":
                candidate = dict(row)
                appliers[name](candidate, cells, errors, where)
                row.update(candidate)
                if row.get("reviewed_by") == "claude":
                    # Claude 审过的题是在抽检：保持 reviewed_by=claude，只留下人工抽检的记录
                    row["note"] = _merge_note(row.get("note"), FAQ_CHECKED + (comment or ""))
                else:
                    row["reviewed"], row["reviewed_by"] = True, "human"
                    row["note"] = _merge_note(row.get("note"), comment)
            else:
                removed[name].add(row[id_key[name]])
                rejected.append({"set": name, "reason": comment or "未注明", "row": row})
            stats[f"{name} {verdict}"] += 1

    if errors or dry_run:
        return stats, errors

    write_jsonl(paths.intent, [row for row in data["意图集"] if row["id"] not in removed["意图集"]])
    write_jsonl(paths.qa, [row for row in data["检索集"] if row["id"] not in removed["检索集"]])
    write_jsonl(paths.faq, data["FAQ抽检"])
    if removed["意图集"] or removed["检索集"]:
        write_jsonl(paths.rejected, rejected)
    return stats, errors


def cmd_import(args: argparse.Namespace) -> int:
    """导入审核表。"""
    paths = Paths(args.data_dir)
    stats, errors = import_workbook(args.workbook, paths, dry_run=args.dry_run)
    for key, value in sorted(stats.items()):
        print(f"  {key}：{value}")
    if errors:
        print(f"\n有 {len(errors)} 处要改，这次一行都没有写入：")
        for message in errors[:50]:
            print(f"  {message}")
        return 1
    if args.dry_run:
        print("\n[dry-run] 校验通过，没有写入")
        return 0
    intent_human = sum(1 for row in read_jsonl(paths.intent) if row.get("reviewed_by") == "human")
    qa_rows = read_jsonl(paths.qa)
    qa_reviewed = [row for row in qa_rows if row.get("reviewed") is True]
    by_claude = [row for row in qa_reviewed if row.get("reviewed_by") == "claude"]
    qa_checked = sum(1 for row in by_claude if FAQ_CHECKED in (row.get("note") or ""))
    qa_need = math.ceil(len(by_claude) * 0.1)
    faq_checked = sum(1 for row in read_jsonl(paths.faq) if FAQ_CHECKED in (row.get("note") or ""))
    # 口径和 validate_data.py 一致：意图集要人审；检索集可由 Claude 审，但其中 10% 要人工抽检
    print("\n写入完成。")
    print(f"  意图集：人审通过 {intent_human} 条（要求 ≥300）")
    print(f"  检索集：审核通过 {len(qa_reviewed)} 条（要求 ≥150），其中 Claude 审 {len(by_claude)} 条、"
          f"已人工抽检 {qa_checked} 条（要求 ≥{qa_need}）")
    print(f"  FAQ：已人工抽检 {faq_checked} 条")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="C5 人工审核表：导出 / 导入")
    parser.add_argument("--data-dir", type=Path, default=ROOT_DIR / "data")
    sub = parser.add_subparsers(dest="command", required=True)

    export = sub.add_parser("export", help="导出还没人审的行")
    export.add_argument("--out", type=Path, default=ROOT_DIR / "data" / "review" / "审核表.xlsx")
    export.add_argument("--faq-ratio", type=float, default=0.1, help="FAQ 抽检比例")
    export.add_argument("--faq-count", type=int, default=None, help="FAQ 抽检条数，给了就不按比例算")
    export.add_argument("--qa-spot-ratio", type=float, default=0.1, help="检索集里 Claude 审过的题抽检比例")
    export.add_argument("--seed", type=int, default=42)
    export.set_defaults(func=cmd_export)

    imp = sub.add_parser("import", help="导入审完的表")
    imp.add_argument("workbook", type=Path)
    imp.add_argument("--dry-run", action="store_true", help="只校验不写入")
    imp.set_defaults(func=cmd_import)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
