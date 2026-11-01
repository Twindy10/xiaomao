"""scripts/review_sheet.py 的往返测试：导出 → 在表里审 → 导入。"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest
from openpyxl import load_workbook

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "review_sheet.py"


def load_module() -> ModuleType:
    """按路径加载 scripts/review_sheet.py。"""
    spec = importlib.util.spec_from_file_location("review_sheet", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclass 解析字符串注解时要能在 sys.modules 里找到模块
    spec.loader.exec_module(module)
    return module


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.fixture()
def data_dir(tmp_path: Path) -> Path:
    """造一套最小数据：1 篇文档、2 个片段、3 条意图题、2 条检索题、10 条 FAQ。"""
    write_jsonl(tmp_path / "processed" / "documents.jsonl", [
        {"doc_key": "d1", "collection": "xm_compliance", "platform": None, "site": "UK",
         "title": "Batteries", "source_url": "https://example.gov.uk/b", "text": "..."},
    ])
    write_jsonl(tmp_path / "processed" / "chunks.jsonl", [
        {"doc_key": "d1", "title_path": "Batteries > Small", "text": "You must register on the **NPWD**."},
        {"doc_key": "d1", "title_path": "Batteries > Large", "text": "Join a BCS by 15 October."},
    ])
    base = {"platform": None, "site": None, "source": "llm_generated", "reviewed": False, "reviewed_by": None, "note": None}
    write_jsonl(tmp_path / "eval" / "intent.jsonl", [
        {"id": "int-0001", "query": "英国卖电池要注册吗", "intent": "tax", "strategy": "direct", **base},
        {"id": "int-0002", "query": "随便聊聊", "intent": "chitchat", "strategy": None, **base},
        {"id": "int-0003", "query": "FBA 怎么收费", "intent": "logistics", "strategy": "direct", **base},
    ])
    write_jsonl(tmp_path / "eval" / "retrieval_qa.jsonl", [
        {"id": "qa-0001", "query": "小电池生产者在哪注册", "intent": "compliance", "answerable": True,
         "evidence": [{"doc_key": "d1", "quote": "You must register on the NPWD."}],
         "reference_answer": "在 NPWD 注册。", "paraphrases": ["a", "b"], **base},
        {"id": "qa-0002", "query": "Temu 怎么入驻", "intent": "account", "answerable": False,
         "evidence": [], "reference_answer": None, "paraphrases": ["c", "d"], **base},
    ])
    write_jsonl(tmp_path / "processed" / "faq_items.jsonl", [
        {"faq_id": f"faq-{n:04d}", "question": f"问题{n}", "answer": "这是一个足够长的答案，满足二十个字的下限要求。",
         "evidence": [{"doc_key": "d1", "quote": "Join a BCS by 15 October."}],
         "reviewed": True, "reviewed_by": "claude", "note": None}
        for n in range(1, 11)
    ])
    return tmp_path


def export(module: ModuleType, data_dir: Path) -> Path:
    out = data_dir / "review" / "sheet.xlsx"
    assert module.main(["--data-dir", str(data_dir), "export", "--out", str(out)]) == 0
    return out


def set_cells(path: Path, sheet_name: str, row_id: str, values: dict[str, object]) -> None:
    """按表头名改某一行（模拟用户在 Excel 里审）。"""
    workbook = load_workbook(path)
    sheet = workbook[sheet_name]
    headers = [cell.value for cell in sheet[1]]
    for row in sheet.iter_rows(min_row=2):
        if row[0].value == row_id:
            for header, value in values.items():
                row[headers.index(header)].value = value
    workbook.save(path)


def test_export_contents(data_dir: Path) -> None:
    module = load_module()
    workbook = load_workbook(export(module, data_dir))
    assert workbook.sheetnames == ["说明", "意图集", "检索集", "FAQ抽检"]
    assert workbook["意图集"].max_row == 4
    qa = workbook["检索集"]
    headers = [cell.value for cell in qa[1]]
    context = qa.cell(row=2, column=headers.index("证据所在片段（核对参考答案用）") + 1).value
    assert "Batteries > Small" in context and "NPWD" in context  # 引文去掉 ** 后也能找到片段
    assert qa.cell(row=2, column=headers.index("同义改写（一行一条）") + 1).value == "a\nb"
    assert workbook["FAQ抽检"].max_row == 2  # 10 条的 10% = 1 条


def test_round_trip(data_dir: Path) -> None:
    module = load_module()
    path = export(module, data_dir)
    set_cells(path, "意图集", "int-0001", {"意图": "compliance", "站点": "UK", "结论": "通过", "审核备注": "改了意图"})
    set_cells(path, "意图集", "int-0002", {"结论": "删除", "审核备注": "太假"})
    set_cells(path, "检索集", "qa-0001", {"同义改写（一行一条）": "改写一\n改写二\n改写三", "结论": "通过"})
    sample_id = load_workbook(path)["FAQ抽检"].cell(row=2, column=1).value
    set_cells(path, "FAQ抽检", sample_id, {"结论": "通过"})

    stats, errors = module.import_workbook(path, module.Paths(data_dir))
    assert errors == []
    assert stats["意图集 通过"] == 1 and stats["意图集 删除"] == 1 and stats["意图集 未审"] == 1

    intents = {row["id"]: row for row in read_jsonl(data_dir / "eval" / "intent.jsonl")}
    assert set(intents) == {"int-0001", "int-0003"}  # 删除的移走了
    assert intents["int-0001"]["intent"] == "compliance" and intents["int-0001"]["site"] == "UK"
    assert intents["int-0001"]["reviewed_by"] == "human" and intents["int-0001"]["note"] == "改了意图"
    assert intents["int-0003"]["reviewed_by"] is None  # 结论留空的不动
    rejected = read_jsonl(data_dir / "eval" / "rejected.jsonl")
    assert rejected[0]["row"]["id"] == "int-0002" and rejected[0]["reason"] == "太假"

    qa = {row["id"]: row for row in read_jsonl(data_dir / "eval" / "retrieval_qa.jsonl")}
    assert qa["qa-0001"]["paraphrases"] == ["改写一", "改写二", "改写三"]
    assert qa["qa-0001"]["evidence"][0]["doc_key"] == "d1"  # 灰底列不会被导入
    faqs = {row["faq_id"]: row for row in read_jsonl(data_dir / "processed" / "faq_items.jsonl")}
    assert module.FAQ_CHECKED in faqs[sample_id]["note"]

    # 再导出时，审过的不再出现，FAQ 抽检也已够数
    workbook = load_workbook(export(module, data_dir))
    assert [row[0].value for row in workbook["意图集"].iter_rows(min_row=2)] == ["int-0003"]
    assert workbook["FAQ抽检"].max_row == 1


def test_invalid_rows_block_the_whole_import(data_dir: Path) -> None:
    module = load_module()
    path = export(module, data_dir)
    set_cells(path, "意图集", "int-0001", {"结论": "通过"})
    set_cells(path, "意图集", "int-0003", {"意图": "物流", "结论": "通过"})
    set_cells(path, "检索集", "qa-0001", {"同义改写（一行一条）": "只有一条", "结论": "通过"})
    set_cells(path, "检索集", "qa-0002", {"参考答案": "不该有答案", "结论": "通过"})
    before = (data_dir / "eval" / "intent.jsonl").read_text(encoding="utf-8")

    stats, errors = module.import_workbook(path, module.Paths(data_dir))
    assert len(errors) == 3
    assert any("意图必须是" in message for message in errors)
    assert any("至少 2 条" in message for message in errors)
    assert any("参考答案要留空" in message for message in errors)
    assert (data_dir / "eval" / "intent.jsonl").read_text(encoding="utf-8") == before  # 一行都没写


def test_strategy_rules(data_dir: Path) -> None:
    module = load_module()
    path = export(module, data_dir)
    set_cells(path, "意图集", "int-0002", {"策略": "direct", "结论": "通过"})
    set_cells(path, "意图集", "int-0003", {"策略": None, "结论": "通过"})
    _, errors = module.import_workbook(path, module.Paths(data_dir))
    assert any("策略要留空" in message for message in errors)
    assert any("策略必须是" in message for message in errors)


def test_spot_check_of_claude_reviewed_qa(data_dir: Path) -> None:
    """Claude 审过的检索题按比例进抽检；选通过后保持 reviewed_by=claude，只加抽检标记。"""
    module = load_module()
    qa_path = data_dir / "eval" / "retrieval_qa.jsonl"
    rows = read_jsonl(qa_path)
    for row in rows:
        row["reviewed"], row["reviewed_by"] = True, "claude"
    write_jsonl(qa_path, rows)

    path = export(module, data_dir)
    sheet = load_workbook(path)["检索集"]
    exported = [row[0].value for row in sheet.iter_rows(min_row=2)]
    assert len(exported) == 1  # 2 条的 10% 向上取整 = 1 条
    set_cells(path, "检索集", exported[0], {"结论": "通过"})

    _, errors = module.import_workbook(path, module.Paths(data_dir))
    assert errors == []
    checked = {row["id"]: row for row in read_jsonl(qa_path)}[exported[0]]
    assert checked["reviewed_by"] == "claude"
    assert module.FAQ_CHECKED in checked["note"]
    assert load_workbook(export(module, data_dir))["检索集"].max_row == 1  # 抽检够数后不再导出


def test_header_change_is_rejected(data_dir: Path) -> None:
    module = load_module()
    path = export(module, data_dir)
    workbook = load_workbook(path)
    workbook["意图集"].insert_cols(2)
    workbook.save(path)
    with pytest.raises(ValueError, match="表头"):
        module.import_workbook(path, module.Paths(data_dir))
