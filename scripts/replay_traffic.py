"""流量回放：通过真实接口打一批请求，产生 qa_trace 数据。

注意：**只用于本机开发库**——脚本会把测试账号的 plan_id 直接改成 2（专业版）。

    uv run python scripts/replay_traffic.py --run m6 --total 300 --concurrency 4
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

for _name in ("NO_PROXY", "no_proxy"):
    os.environ[_name] = ",".join(filter(None, [os.environ.get(_name), "127.0.0.1,localhost"]))

from dotenv import load_dotenv  # noqa: E402

RATIOS = {"original": 0.4, "paraphrase": 0.3, "unanswerable": 0.1, "casual": 0.1, "multi_turn": 0.1}
FOLLOW_UPS = ["那{site}呢？", "这个要多久？", "需要准备哪些材料？", "流程复杂吗？"]
SITE_NAMES = {"US": "美国站", "UK": "英国站", "DE": "德国站", "JP": "日本站", "FR": "法国站"}
PER_MINUTE = 9
WINDOW = 60.0


def read_jsonl(path: Path) -> list[dict]:
    """读 JSONL。"""
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, row: dict) -> None:
    """追加一行。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def percentile(values: list[float], ratio: float) -> float:
    """线性插值分位数。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * ratio
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def build_plan(items: list[dict], intents: list[dict], total: int = 300, seed: int = 42) -> list[dict]:
    """按比例生成请求计划；两轮对话一次加两条。"""
    rng = random.Random(seed)
    answerable = [item for item in items if item.get("answerable") is True]
    unanswerable = [item for item in items if item.get("answerable") is False]
    casual = [row for row in intents if row.get("intent") in ("chitchat", "out_of_scope")]
    paraphrased = [(item, text) for item in answerable for text in (item.get("paraphrases") or [])]

    categories = list(RATIOS)
    # 两轮对话一次加两条，所以抽中的概率减半，最终请求占比才是 10%
    weights = [RATIOS[name] / 2 if name == "multi_turn" else RATIOS[name] for name in categories]
    plan: list[dict] = []

    while len(plan) < total:
        category = rng.choices(categories, weights=weights, k=1)[0]
        index = len(plan)
        if category == "original" and answerable:
            item = rng.choice(answerable)
            plan.append({"index": index, "category": "original", "question": item["query"],
                         "open_new": True, "follow_up": False})
        elif category == "paraphrase" and paraphrased:
            item, text = rng.choice(paraphrased)
            plan.append({"index": index, "category": "paraphrase", "question": text,
                         "open_new": True, "follow_up": False})
        elif category == "unanswerable" and unanswerable:
            item = rng.choice(unanswerable)
            plan.append({"index": index, "category": "unanswerable", "question": item["query"],
                         "open_new": True, "follow_up": False})
        elif category == "casual" and casual:
            row = rng.choice(casual)
            plan.append({"index": index, "category": "casual", "question": row["query"],
                         "open_new": True, "follow_up": False})
        elif category == "multi_turn" and answerable:
            item = rng.choice(answerable)
            site = rng.choice(list(SITE_NAMES)) if item.get("site") else "DE"
            plan.append({"index": index, "category": "multi_turn", "question": item["query"],
                         "open_new": True, "follow_up": False})
            if len(plan) < total:
                plan.append({"index": len(plan), "category": "multi_turn",
                             "question": rng.choice(FOLLOW_UPS).format(site=SITE_NAMES.get(site, "德国站")),
                             "open_new": False, "follow_up": True})
        else:
            item = rng.choice(answerable) if answerable else {"query": "德国站耳机要认证吗"}
            plan.append({"index": index, "category": "original", "question": item["query"],
                         "open_new": True, "follow_up": False})
    return plan[:total]


def seat_for(unit: list[dict], n_users: int) -> int:
    """一个单元用哪个账号：按单元第一条的序号分配，两轮对话的首问和追问必须是同一个账号。"""
    return unit[0]["index"] % n_users


class Pacer:
    """每个账号每分钟最多 N 个请求；时间函数和 sleep 可注入。"""

    def __init__(self, per_minute: int = PER_MINUTE, clock=time.monotonic, sleep=time.sleep) -> None:
        self.per_minute = per_minute
        self.clock = clock
        self.sleep = sleep
        self.stamps: list[float] = []
        self.lock = threading.Lock()

    def wait(self) -> float:
        """必要时 sleep，返回等了多久。"""
        with self.lock:
            now = self.clock()
            self.stamps = [stamp for stamp in self.stamps if now - stamp < WINDOW]
            wait_for = 0.0
            if len(self.stamps) >= self.per_minute:
                wait_for = WINDOW - (now - self.stamps[0]) + 0.01
            else:
                self.stamps.append(now)
                return 0.0
            if wait_for > 0:
                self.sleep(wait_for)
                now = self.clock()
                self.stamps = [stamp for stamp in self.stamps if now - stamp < WINDOW]
            self.stamps.append(self.clock())
            return wait_for


def parse_sse(lines) -> list[dict]:
    """把 SSE 行解析成事件列表。"""
    events: list[dict] = []
    for line in lines:
        if not line.startswith("data: "):
            continue
        try:
            events.append(json.loads(line[6:]))
        except json.JSONDecodeError:
            continue
    return events


def send_chat(client, token: str, question: str, conversation_id: int | None) -> dict:
    """发一次请求，返回客户端侧的结果。"""
    body = {"question": question}
    if conversation_id is not None:
        body["conversation_id"] = conversation_id
    started = time.perf_counter()
    first_ms = None
    events: list[dict] = []
    status = 0
    error = None
    try:
        with client.stream("POST", "/api/chat", headers={"Authorization": f"Bearer {token}"},
                           json=body) as response:
            status = response.status_code
            for line in response.iter_lines():
                if line.startswith("data: ") and first_ms is None:
                    first_ms = (time.perf_counter() - started) * 1000
                events.extend(parse_sse([line]))
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    total_ms = (time.perf_counter() - started) * 1000

    meta = next((event for event in events if "conversation_id" in event), {})
    done = next((event for event in events if "message_id" in event), {})
    failed = next((event for event in events if "request_id" in event and "error" in event), None)
    return {
        "status": status,
        "first_ms": first_ms,
        "total_ms": total_ms,
        "conversation_id": meta.get("conversation_id"),
        "cache_hit": bool(meta.get("cache_hit")),
        "message_id": done.get("message_id"),
        "refused": bool(done.get("refused")),
        "tokens": int(done.get("tokens") or 0),
        "error": error or (f"event:{failed['request_id']}" if failed else None),
    }


def group_units(plan: list[dict]) -> list[list[dict]]:
    """两轮对话的两条绑成一个单元，单元内按顺序执行。"""
    units: list[list[dict]] = []
    index = 0
    while index < len(plan):
        task = plan[index]
        paired = (
            task["category"] == "multi_turn" and not task["follow_up"]
            and index + 1 < len(plan) and plan[index + 1].get("follow_up")
        )
        if paired:
            units.append([task, plan[index + 1]])
            index += 2
        else:
            units.append([task])
            index += 1
    return units


def summarize(rows: list[dict], wall_seconds: float | None = None) -> str:
    """Markdown 汇总。"""
    ok = [row for row in rows if row.get("status") == 200 and not row.get("error")]
    lines = ["# 流量回放", ""]
    lines.append(f"- 请求 {len(rows)}，成功 {len(ok)}（{len(ok) / len(rows):.1%}）" if rows else "- 没有请求")
    if wall_seconds:
        lines.append(f"- 吞吐：{len(rows) / max(wall_seconds, 1e-6):.2f} 请求/秒")
    lines.append("")
    lines.append("| 类别 | 请求 | 成功率 | 首字 P50 | 首字 P95 | 总耗时 P50 | 总耗时 P95 |")
    lines.append("|---|---|---|---|---|---|---|")
    for category in dict.fromkeys(row.get("category") for row in rows):
        group = [row for row in rows if row.get("category") == category]
        good = [row for row in group if row.get("status") == 200 and not row.get("error")]
        firsts = [float(row["first_ms"]) for row in good if row.get("first_ms") is not None]
        totals = [float(row["total_ms"]) for row in good]
        lines.append(
            f"| {category} | {len(group)} | {len(good) / len(group):.1%} | "
            f"{percentile(firsts, 0.5):.0f}ms | {percentile(firsts, 0.95):.0f}ms | "
            f"{percentile(totals, 0.5):.0f}ms | {percentile(totals, 0.95):.0f}ms |"
        )
    lines.append("")
    cache_hits = sum(1 for row in ok if row.get("cache_hit"))
    if ok:
        lines.append(f"- 缓存命中 {cache_hits}/{len(ok)}（{cache_hits / len(ok):.1%}）")
    return "\n".join(lines)


def ensure_users(client, count: int, users_path: Path) -> list[dict]:
    """注册并登录测试账号，返回 [{user_id, token, phone_last4}]。"""
    import httpx

    users: list[dict] = []
    for _ in range(count):
        phone = "199" + "".join(str(random.randint(0, 9)) for _ in range(8))
        password = "replay1234"
        register = client.post("/api/auth/register", json={"phone": phone, "password": password})
        if register.status_code not in (201, 409):
            raise SystemExit(f"注册失败：{register.status_code} {register.text}")
        login = client.post("/api/auth/login", json={"phone": phone, "password": password})
        if login.status_code != 200:
            raise SystemExit(f"登录失败：{login.status_code} {login.text}")
        body = login.json()
        users.append({"user_id": body["user"]["id"], "token": body["access_token"],
                      "phone_last4": phone[-4:]})

    # 开发库限定：把测试账号升到专业版（每天 200 问），否则 300 个请求会被配额拦住
    from xiaomao.runtime import build_pool

    pool = build_pool()
    try:
        with pool.connection() as conn, conn.cursor() as cursor:
            for user in users:
                cursor.execute("UPDATE `user` SET plan_id=2 WHERE id=%s", (user["user_id"],))
    finally:
        pool.close()

    # 续跑时会注册一批新账号（密码不落盘，旧账号登不回来）：追加进 users.json，
    # 不能覆盖，否则 trace_stats --users 只统计得到续跑那一小段
    users_path.parent.mkdir(parents=True, exist_ok=True)
    previous = json.loads(users_path.read_text(encoding="utf-8")) if users_path.is_file() else []
    known = {row["user_id"] for row in previous}
    merged = previous + [{"user_id": user["user_id"], "phone_last4": user["phone_last4"]}
                         for user in users if user["user_id"] not in known]
    users_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    return users


def main(argv: list[str] | None = None) -> int:
    """回放流量。"""
    load_dotenv(ROOT_DIR / ".env", override=True)
    import httpx

    parser = argparse.ArgumentParser(description="流量回放（只用于本机开发库）")
    parser.add_argument("--run", default="m6")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--total", type=int, default=300)
    parser.add_argument("--users", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)

    out_dir = ROOT_DIR / "eval" / "runs" / f"replay_{args.run}"
    out_dir.mkdir(parents=True, exist_ok=True)
    requests_path = out_dir / "requests.jsonl"
    users_path = out_dir / "users.json"

    items = read_jsonl(ROOT_DIR / "data" / "eval" / "retrieval_qa.jsonl")
    intents = read_jsonl(ROOT_DIR / "data" / "eval" / "intent.jsonl")
    plan = build_plan(items, intents, args.total, args.seed)

    done = {row["index"] for row in read_jsonl(requests_path)}
    pending = [item for item in plan if item["index"] not in done]
    if not pending:
        print("没有待回放的请求")
        return 0

    client = httpx.Client(base_url=args.base_url, timeout=180, trust_env=False)
    users = ensure_users(client, args.users, users_path)
    pacers = [Pacer() for _ in users]
    conversations: dict[int, int | None] = {}
    lock = threading.Lock()

    def run_task(task: dict, seat: int, conversation_id: int | None) -> dict:
        """用指定账号控节奏、发请求。"""
        user = users[seat]
        pacers[seat].wait()
        return send_chat(client, user["token"], task["question"], conversation_id)

    def worker(unit: list[dict]) -> None:
        """一个单元里的任务按顺序跑，两轮对话共用会话。
        账号按单元的第一条分配：追问必须和首问是同一个账号，否则会话不属于他，接口返回 404。"""
        conversation_id = None
        seat = seat_for(unit, len(users))
        for task in unit:
            if task["index"] in done:
                continue
            result = run_task(task, seat, conversation_id)
            if task["category"] == "multi_turn" and result.get("conversation_id"):
                conversation_id = result["conversation_id"]
            with lock:
                append_jsonl(requests_path, {
                    "index": task["index"],
                    "category": task["category"],
                    "question": task["question"],
                    "user_id": users[seat]["user_id"],
                    **result,
                })

    units = [unit for unit in group_units(plan) if any(task["index"] not in done for task in unit)]
    started_at = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for offset in range(0, len(units), args.concurrency):
            batch = units[offset:offset + args.concurrency]
            for future in [pool.submit(worker, unit) for unit in batch]:
                future.result()
            print(f"  已完成 {min(offset + args.concurrency, len(units))}/{len(units)} 个单元")
    wall = time.perf_counter() - started_at

    rows = read_jsonl(requests_path)
    report = summarize(rows, wall)
    (out_dir / "summary.md").write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"\n明细 -> {requests_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
