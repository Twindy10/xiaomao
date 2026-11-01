"""流量回放测试：计划和节奏控制（不连网）。"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "replay_traffic.py"


def load_script() -> ModuleType:
    """按路径加载脚本。"""
    spec = importlib.util.spec_from_file_location("replay_script", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sample_items() -> list[dict]:
    """三题有答案、一题无答案。"""
    return [
        {"id": "a", "query": "问题 A", "answerable": True, "site": "DE",
         "paraphrases": ["换个说法的 A", "A 的另一种问法"]},
        {"id": "b", "query": "问题 B", "answerable": True, "site": "US",
         "paraphrases": ["换个说法的 B"]},
        {"id": "c", "query": "无答案 C", "answerable": False, "site": None, "paraphrases": []},
    ]


def sample_intents() -> list[dict]:
    """闲聊和业务各一条。"""
    return [
        {"id": "i1", "query": "你好", "intent": "chitchat"},
        {"id": "i2", "query": "德国站耳机要认证吗", "intent": "compliance"},
    ]


def test_plan_is_stable_with_seed() -> None:
    """同一个 seed 计划完全一样。"""
    script = load_script()
    first = script.build_plan(sample_items(), sample_intents(), total=50, seed=7)
    second = script.build_plan(sample_items(), sample_intents(), total=50, seed=7)
    assert first == second
    assert len(first) == 50


def test_plan_has_all_categories_and_ratios() -> None:
    """比例大致对得上（大样本）。"""
    script = load_script()
    plan = script.build_plan(sample_items(), sample_intents(), total=400, seed=1)
    counts: dict[str, int] = {}
    for task in plan:
        counts[task["category"]] = counts.get(task["category"], 0) + 1

    assert set(counts) == set(script.RATIOS)
    for category, ratio in script.RATIOS.items():
        assert abs(counts[category] / len(plan) - ratio) < 0.06, category


def test_multi_turn_pairs_are_adjacent() -> None:
    """两轮对话两条相邻，第二条是追问。"""
    script = load_script()
    plan = script.build_plan(sample_items(), sample_intents(), total=200, seed=3)
    for index, task in enumerate(plan):
        if task["follow_up"]:
            assert plan[index - 1]["category"] == "multi_turn"
            assert plan[index - 1]["follow_up"] is False
            assert not task["open_new"]


def test_group_units_pairs_multi_turn() -> None:
    """单元分组：两轮对话绑一起，其他各自成组。"""
    script = load_script()
    plan = [
        {"index": 0, "category": "original", "follow_up": False},
        {"index": 1, "category": "multi_turn", "follow_up": False},
        {"index": 2, "category": "multi_turn", "follow_up": True},
        {"index": 3, "category": "casual", "follow_up": False},
    ]
    units = script.group_units(plan)
    assert [len(unit) for unit in units] == [1, 2, 1]
    assert units[1][1]["follow_up"] is True


class FakeClock:
    """可推进的时钟。"""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        """当前时间。"""
        return self.now

    def sleep(self, seconds: float) -> None:
        """推进时间。"""
        self.sleeps.append(seconds)
        self.now += seconds


def test_pacer_allows_per_minute_then_waits() -> None:
    """9 个请求直接放行，第 10 个要等到窗口滑动。"""
    script = load_script()
    fake = FakeClock()
    pacer = script.Pacer(per_minute=9, clock=fake.clock, sleep=fake.sleep)

    for _ in range(9):
        assert pacer.wait() == 0.0
    waited = pacer.wait()

    assert waited > 0
    assert fake.sleeps and fake.sleeps[0] == waited
    assert fake.now >= 60


def test_pacer_window_slides() -> None:
    """过了 60 秒后窗口内没有旧请求，不再等待。"""
    script = load_script()
    fake = FakeClock()
    pacer = script.Pacer(per_minute=2, clock=fake.clock, sleep=fake.sleep)

    pacer.wait()
    pacer.wait()
    assert pacer.wait() > 0
    fake.now += 120
    assert pacer.wait() == 0.0


def test_follow_up_uses_same_account_as_first_turn() -> None:
    """两轮对话的追问必须和首问同一个账号，否则会话不属于他，接口返回 404（M6 回放时出过）。"""
    script = load_script()
    unit = [{"index": 5, "category": "multi_turn"}, {"index": 6, "category": "multi_turn"}]
    assert script.seat_for(unit, 8) == 5
    assert script.seat_for(unit[1:], 8) == 6  # 只看单元第一条，不按每条请求自己的序号
