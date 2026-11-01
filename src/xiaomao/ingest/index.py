"""Milvus 存储层：建集合、写片段、删片段、对账。"""

from __future__ import annotations

import re
from datetime import date

from pymilvus import CollectionSchema, DataType, Function, FunctionType, MilvusClient
from pymilvus.milvus_client.index import IndexParams

from xiaomao.ingest.embed import Embedding

COLLECTIONS: tuple[str, ...] = ("xm_platform_policy", "xm_compliance", "xm_faq")
NO_EXPIRE = 99991231

# 分词：内置 chinese 会把"半托管"切成"半/托管"、也不转小写，所以换成 jieba + 业务词典
JIEBA_USER_WORDS = [
    "半托管", "全托管", "CE-RED", "亚马逊物流", "品牌备案", "欧代", "英代",
    "授权代表", "生产者责任延伸", "卖家中心", "账户健康", "购买按钮",
]
ANALYZER_PARAMS = {
    "tokenizer": {
        "type": "jieba",
        "dict": ["_default_", *JIEBA_USER_WORDS],
        "mode": "exact",
        "hmm": True,
    },
    "filter": ["lowercase", "cnalphanumonly"],
}

# VARCHAR 字段的字节上限，超了直接报错而不是悄悄截断
VARCHAR_LIMITS = {
    "chunk_id": 64,
    "text": 8192,
    "platform": 32,
    "site": 32,
    "doc_type": 32,
    "lang": 32,
    "title_path": 512,
    "source_url": 1024,
}

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def build_schema() -> CollectionSchema:
    """三个 collection 共用的 schema。"""
    schema = MilvusClient.create_schema(auto_id=False)
    schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=64)
    schema.add_field("doc_id", DataType.INT64)
    schema.add_field(
        "text",
        DataType.VARCHAR,
        max_length=VARCHAR_LIMITS["text"],
        enable_analyzer=True,
        analyzer_params=ANALYZER_PARAMS,
    )
    schema.add_field("dense", DataType.FLOAT_VECTOR, dim=1024)
    schema.add_field("sparse_m3", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field("sparse_bm25", DataType.SPARSE_FLOAT_VECTOR)
    for field in ("platform", "site", "doc_type", "lang"):
        schema.add_field(field, DataType.VARCHAR, max_length=VARCHAR_LIMITS[field])
    schema.add_field("effective_date", DataType.INT64)
    schema.add_field("expire_date", DataType.INT64)
    schema.add_field("title_path", DataType.VARCHAR, max_length=VARCHAR_LIMITS["title_path"])
    schema.add_field("source_url", DataType.VARCHAR, max_length=VARCHAR_LIMITS["source_url"])

    schema.add_function(
        Function(
            name="bm25",
            function_type=FunctionType.BM25,
            input_field_names=["text"],
            output_field_names=["sparse_bm25"],
        )
    )
    return schema


def build_index_params() -> IndexParams:
    """向量索引 + 标量索引。"""
    index = MilvusClient.prepare_index_params()
    index.add_index("dense", index_type="HNSW", metric_type="COSINE", params={"M": 16, "efConstruction": 200})
    index.add_index("sparse_m3", index_type="SPARSE_INVERTED_INDEX", metric_type="IP")
    index.add_index("sparse_bm25", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
    for field in ("platform", "site", "effective_date", "expire_date", "doc_id"):
        index.add_index(field, index_type="INVERTED")
    return index


def ensure_collection(client: MilvusClient, name: str, drop: bool = False) -> bool:
    """不存在就建（含索引）并 load；返回是否新建。"""
    exists = client.has_collection(name)
    if exists and not drop:
        return False
    if exists:
        client.drop_collection(name)
    client.create_collection(collection_name=name, schema=build_schema(), index_params=build_index_params())
    client.load_collection(name)
    return True


def _date_to_int(value: object) -> int:
    """'YYYY-MM-DD' -> 20261028；None 保持 0，由调用方决定 expire 的默认值。"""
    if value is None or value == "":
        return 0
    if isinstance(value, date):
        return int(value.strftime("%Y%m%d"))
    text = str(value)
    if not DATE_RE.match(text):
        raise ValueError(f"日期格式不对：{text!r}")
    return int(text.replace("-", ""))


def _check_length(row: dict, chunk_id: str) -> None:
    """按 UTF-8 字节数检查 VARCHAR 字段。"""
    for field, limit in VARCHAR_LIMITS.items():
        value = row.get(field)
        if value is None:
            continue
        size = len(str(value).encode("utf-8"))
        if size > limit:
            raise ValueError(f"{chunk_id} 的 {field} 有 {size} 字节，超过上限 {limit}")


def bm25_text(chunk: dict) -> str:
    """写进 Milvus text 字段的文本，BM25 从它生成。
    政策和法规带上标题路径（2026-10-31 实测：BM25 Hit@5 从 70/122 升到 81/122）；FAQ 的 text 已经是"问题\n答案"。"""
    text = str(chunk["text"])
    title_path = str(chunk.get("title_path") or "")
    if chunk.get("collection") == "xm_faq" or not title_path:
        return text
    return f"{title_path}\n{text}"


def chunk_to_row(chunk: dict, doc_id: int, embedding: Embedding) -> dict:
    """把 chunks.jsonl 的一行转成 Milvus 的一行。"""
    expire = _date_to_int(chunk.get("expire_date"))
    row = {
        "chunk_id": str(chunk["chunk_id"]),
        "doc_id": int(doc_id),
        "text": bm25_text(chunk),
        "dense": embedding.dense,
        "sparse_m3": embedding.sparse,
        "platform": chunk.get("platform") or "",
        "site": chunk.get("site") or "",
        "doc_type": chunk.get("doc_type") or "",
        "lang": chunk.get("lang") or "",
        "effective_date": _date_to_int(chunk.get("effective_date")),
        "expire_date": expire or NO_EXPIRE,
        "title_path": str(chunk.get("title_path") or ""),
        "source_url": str(chunk.get("source_url") or ""),
    }
    _check_length(row, row["chunk_id"])
    return row


def upsert_rows(client: MilvusClient, name: str, rows: list[dict], batch: int = 500) -> int:
    """分批 upsert，返回写入条数。"""
    written = 0
    for start in range(0, len(rows), batch):
        piece = rows[start: start + batch]
        client.upsert(collection_name=name, data=piece)
        written += len(piece)
    return written


def delete_doc(client: MilvusClient, name: str, doc_id: int) -> None:
    """删掉某篇文档的全部片段。"""
    client.delete(collection_name=name, filter=f"doc_id == {int(doc_id)}")


def count_rows(client: MilvusClient, name: str) -> int:
    """集合里的片段条数（强一致读数）。"""
    result = client.query(
        collection_name=name,
        filter="",
        output_fields=["count(*)"],
        consistency_level="Strong",
    )
    return int(result[0]["count(*)"]) if result else 0


def finish(client: MilvusClient, name: str) -> None:
    """写完 flush 并重新 load。"""
    client.flush(name)
    client.load_collection(name)
