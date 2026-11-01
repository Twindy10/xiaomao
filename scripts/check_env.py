#!/usr/bin/env python3
"""小贸同学 M0 环境检查：配置、Milvus、MySQL、Redis、GPU、生成模型、决策模型。用法见 --help。"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Callable
from urllib.parse import urlsplit, urlunsplit

sys.stdout.reconfigure(encoding="utf-8")  # Windows 控制台默认 GBK，先切成 UTF-8

ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT_DIR / ".env"

OK, WARN, FAIL = "OK", "WARN", "FAIL"
Result = tuple[str, str, str]

EXPECTED_TABLES = "plan user seller_profile conversation message qa_trace feedback usage_daily kb_document".split()
REQUIRED_KEYS = (
    "LLM_BASE_URL LLM_API_KEY LLM_MODEL DECIDER_BACKEND MILVUS_URI "
    "MYSQL_HOST MYSQL_PORT MYSQL_USER MYSQL_PASSWORD MYSQL_DATABASE REDIS_URL"
).split()

CONFIG: dict[str, str] = {}

def env(key: str, default: str = "") -> str:
    """取配置值，未配置时返回默认值。"""
    return CONFIG.get(key) or default

def load_env() -> dict[str, str]:
    """读取项目根目录的 .env，合并进进程环境变量（.env 优先）。"""
    values: dict[str, str] = {}
    try:
        from dotenv import dotenv_values, load_dotenv
    except ImportError:
        print("[WARN] 未安装 python-dotenv，改为手工解析 .env")
    else:
        load_dotenv(ENV_FILE)
        values = {k: (v or "") for k, v in dotenv_values(ENV_FILE).items()}

    if not values and ENV_FILE.exists():
        for raw in ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                values[key.strip()] = value.strip().strip('"').strip("'")

    # .env 优先：宿主机上可能残留别的项目的同名变量（例如 MYSQL_USER）
    for key in set(REQUIRED_KEYS) | {"MILVUS_TOKEN", "DECIDER_BASE_URL"}:
        if key not in values and os.environ.get(key):
            values[key] = os.environ[key]
    return values

def mask_url(url: str) -> str:
    """把 URL 里的密码段打码，其余部分保留。"""
    parts = urlsplit(url)
    if parts.password is None:
        return url
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    userinfo = f"{parts.username}:***@" if parts.username else "***@"
    return urlunsplit((parts.scheme, userinfo + host, parts.path, parts.query, parts.fragment))

def check_config() -> Result:
    """检查必需的环境变量是否齐全（输出里不出现明文密钥）。"""
    missing = [key for key in REQUIRED_KEYS if not env(key)]
    if env("DECIDER_BACKEND") not in ("", "llm_json") and not env("DECIDER_BASE_URL"):
        missing.append("DECIDER_BASE_URL")
    if missing:
        return "Config", FAIL, "缺少必需变量：" + ", ".join(missing)
    return "Config", OK, f"{len(REQUIRED_KEYS)} 个必需变量齐全，REDIS_URL={mask_url(env('REDIS_URL'))}"

def check_milvus() -> Result:
    """检查 Milvus 连接、服务端版本和 xm_ 开头的集合。"""
    from pymilvus import MilvusClient

    client = MilvusClient(uri=env("MILVUS_URI"), token=env("MILVUS_TOKEN"))
    try:
        version = client.get_server_version()
    except Exception:  # 老版本客户端没有这个方法，不影响连接结论
        version = "未知"
    xm_names = sorted(str(n) for n in client.list_collections() if str(n).startswith("xm_"))
    suffix = f"xm_ 集合：{', '.join(xm_names)}" if xm_names else "尚无 xm_ 集合"
    return "Milvus", OK, f"v{version}，{suffix}"

def check_mysql() -> Result:
    """检查 MySQL 版本、默认字符集和 9 张业务表是否齐全。"""
    import pymysql

    conn = pymysql.connect(
        host=env("MYSQL_HOST", "127.0.0.1"),
        port=int(env("MYSQL_PORT", "3306")),
        user=env("MYSQL_USER"),
        password=env("MYSQL_PASSWORD"),
        database=env("MYSQL_DATABASE"),
        charset="utf8mb4",
        connect_timeout=5,
    )
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT VERSION(), @@character_set_database")
            version, charset = (str(v) for v in cur.fetchone())
            cur.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = %s",
                (env("MYSQL_DATABASE"),),
            )
            existing = {str(row[0]) for row in cur.fetchall()}
    finally:
        conn.close()

    notes: list[str] = []
    if not version.startswith("8."):
        notes.append(f"版本 {version}，需要 8.x")
    if charset != "utf8mb4":
        notes.append(f"默认字符集 {charset}，建议 utf8mb4")
    missing = [table for table in EXPECTED_TABLES if table not in existing]
    if missing:
        notes.append("缺少表：" + ", ".join(missing) + "（M0 阶段可能还没建表）")
    if not notes:
        return "MySQL", OK, f"{version}，字符集 utf8mb4，9 张表齐全"
    return "MySQL", FAIL if not version.startswith("8.") else WARN, f"{version}；" + "；".join(notes)

def check_redis() -> Result:
    """检查 Redis 连接、版本 ≥ 8、查询引擎、淘汰策略和持久化配置。"""
    import redis
    from redis.exceptions import ResponseError

    client = redis.Redis.from_url(env("REDIS_URL"), socket_timeout=5, socket_connect_timeout=5)
    client.ping()
    version = str(client.info("server").get("redis_version", "未知"))
    major = int(version.split(".")[0]) if version.split(".")[0].isdigit() else 0

    fatal: list[str] = []
    warnings: list[str] = []
    if major < 8:
        fatal.append(f"redis_version={version}，需要 ≥ 8（语义缓存要用内置向量检索）")
    try:
        client.execute_command("FT._LIST")
    except Exception as exc:
        fatal.append(f"FT._LIST 失败，查询引擎不可用：{type(exc).__name__}: {exc}")
    try:
        policy = str(client.config_get("maxmemory-policy").get("maxmemory-policy", ""))
        if policy != "volatile-lru":
            warnings.append(f"maxmemory-policy={policy}，建议 volatile-lru")
        appendonly = str(client.config_get("appendonly").get("appendonly", ""))
        if appendonly.lower() != "yes":
            warnings.append(f"appendonly={appendonly}，建议 yes")
    except ResponseError:
        warnings.append("CONFIG 命令被禁用，未检查内存策略与持久化")

    if fatal:
        return "Redis", FAIL, f"{version}；" + "；".join(fatal + warnings)
    if warnings:
        return "Redis", WARN, f"{version}；" + "；".join(warnings)
    return "Redis", OK, f"{version}，FT 可用，volatile-lru，appendonly=yes"

def check_gpu() -> Result:
    """检查 CUDA 是否可用，以及可用显存是否够跑完整链路。"""
    try:
        import torch
    except ImportError:
        return "GPU", WARN, "torch 未安装，跳过"
    if not torch.cuda.is_available():
        return "GPU", WARN, "CUDA 不可用"
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    gib = 1024.0**3
    free_gib = free_bytes / gib
    detail = f"{torch.cuda.get_device_name(0)}，总显存 {total_bytes / gib:.1f}GB，可用 {free_gib:.1f}GB"
    if free_gib < 4.5:
        return "GPU", WARN, f"{detail}；BGE-M3 + reranker + 决策模型约需 5.2GB"
    return "GPU", OK, detail

def check_llm() -> Result:
    """调用生成模型 API 发一条 ping，确认 Key 与网络可用。"""
    from openai import OpenAI

    client = OpenAI(base_url=env("LLM_BASE_URL"), api_key=env("LLM_API_KEY"), timeout=20)
    started = time.perf_counter()
    response = client.chat.completions.create(
        model=env("LLM_MODEL"),
        messages=[{"role": "user", "content": "ping"}],
        max_tokens=5,
        # 千问 3.x 默认先输出思考过程，关掉才能测出真实的首字延迟
        extra_body={"enable_thinking": env("LLM_ENABLE_THINKING", "false").lower() == "true"},
    )
    cost_ms = (time.perf_counter() - started) * 1000
    model = getattr(response, "model", None) or env("LLM_MODEL")
    return "LLM", OK, f"{model}，{cost_ms:.0f}ms"

def check_decider() -> Result:
    """检查决策模型服务是否可达（llm_json 后端不需要额外服务）。"""
    if env("DECIDER_BACKEND") == "llm_json":
        return "Decider", OK, "使用 LLM 兜底，无需额外服务"

    import httpx

    base_url = env("DECIDER_BASE_URL")
    started = time.perf_counter()
    # 本地服务必须绕过系统代理（Clash 等会把 127.0.0.1 的请求也劫持成 502）；健康检查是 /healthz
    try:
        response = httpx.get(base_url.rstrip("/") + "/healthz", timeout=3, trust_env=False)
    except Exception as exc:
        return "Decider", WARN, f"{base_url} 连不上（{type(exc).__name__}），M0 阶段可能还没部署"
    cost_ms = (time.perf_counter() - started) * 1000
    if response.status_code != 200:
        return "Decider", WARN, f"/healthz 返回 HTTP {response.status_code}，{cost_ms:.0f}ms"
    return "Decider", OK, f"HTTP {response.status_code}，{cost_ms:.0f}ms"

CHECKS: list[tuple[str, str, Callable[[], Result]]] = [
    ("config", "Config", check_config), ("milvus", "Milvus", check_milvus), ("mysql", "MySQL", check_mysql),
    ("redis", "Redis", check_redis), ("gpu", "GPU", check_gpu),
    ("llm", "LLM", check_llm), ("decider", "Decider", check_decider),
]

def run_check(label: str, func: Callable[[], Result]) -> str:
    """执行单个检查，异常统一转成 FAIL，保证不影响其他检查。"""
    try:
        _, status, detail = func()
    except Exception as exc:
        status, detail = FAIL, f"{type(exc).__name__}: {exc}"
    print(f"[{status:<4}] {label:<12}{detail}")
    return status

def main() -> int:
    """解析参数、依次执行检查并输出汇总，返回进程退出码。"""
    parser = argparse.ArgumentParser(description="小贸同学 M0 环境检查")
    parser.add_argument("--skip-llm", action="store_true", help="跳过生成模型 API 检查（不消耗额度）")
    parser.add_argument("--skip-gpu", action="store_true", help="跳过 GPU 检查")
    parser.add_argument(
        "--only", action="append", metavar="NAME", choices=[key for key, _, _ in CHECKS],
        help="只运行指定检查，可重复传多次",
    )
    args = parser.parse_args()

    global CONFIG
    CONFIG = load_env()

    skipped = {key for key, flag in (("llm", args.skip_llm), ("gpu", args.skip_gpu)) if flag}
    selected = [c for c in CHECKS if (not args.only or c[0] in args.only) and c[0] not in skipped]

    counts = {OK: 0, WARN: 0, FAIL: 0}
    for _, label, func in selected:
        counts[run_check(label, func)] += 1
    print(f"\n共 {len(selected)} 项：OK {counts[OK]}，WARN {counts[WARN]}，FAIL {counts[FAIL]}")
    return 1 if counts[FAIL] else 0

if __name__ == "__main__":
    sys.exit(main())
