# 数据格式约定（M1 起生效）

所有中间数据都是 **UTF-8、无 BOM 的 JSONL**：一行一个 JSON 对象，写入时用 `ensure_ascii=False`。字段名和取值以本文为准；要改格式，先改本文，再改代码。

```
data/
├── sources.jsonl                 # 要抓取的原文清单（人工整理）
├── raw/                          # 原始抓取（HTML/PDF 原样保存），不进版本库
│   └── manifest.jsonl            # 抓取结果登记：每个原文对应哪个文件、元数据是什么
├── processed/
│   ├── faq_items.jsonl           # LLM 生成的 FAQ（审核用的中间文件）
│   ├── documents.jsonl           # 第 1 步：解析清洗后的整篇文档，一行一篇（含审核通过的 FAQ）
│   └── chunks.jsonl              # 第 2 步：分块结果，一行一个片段
└── eval/
    ├── intent.jsonl              # 意图标注集（约 300 条）
    └── retrieval_qa.jsonl        # 检索/问答评估集（约 150–200 条）
```

## 1. 公共取值

| 字段 | 取值 |
|---|---|
| `collection` | `xm_platform_policy` / `xm_compliance` / `xm_faq` |
| `platform` | `amazon` / `temu` / `tiktok_shop` / `shopee` / `null`（法规类文档不属于任何平台） |
| `site` | ISO 国家代码大写：`US` `UK` `DE` `FR` `IT` `ES` `JP` …；欧盟整体适用写 `EU`；不限站点写 `null` |
| `doc_type` | `policy`（平台政策）/ `guide`（操作指南）/ `regulation`（法规条文）/ `announcement`（政策公告）/ `faq` |
| `lang` | `zh` / `en` / `de` / `ja` / `fr` / `es` / `it` |
| 日期 | 字符串 `YYYY-MM-DD`，未知写 `null`。入库时由代码转成整数 `yyyymmdd`；`expire_date` 为 `null` 时存 `99991231` |
| `intent` | `platform_policy` / `compliance` / `logistics` / `tax` / `ops` / `account` / `chitchat` / `out_of_scope` |
| `strategy` | `direct` / `subquery` / `hyde` |

## 2. `documents.jsonl`（整篇文档）

| 字段 | 类型 | 说明 |
|---|---|---|
| `doc_key` | str | 文档的稳定标识：`sha256(collection + "\|" + source_url)` 的前 16 位十六进制。同一篇原文重新抓取后 `doc_key` 不变 |
| `collection` | str | 见公共取值 |
| `source_url` | str | 原文地址；FAQ 这类没有网址的写 `faq://<来源>/<序号>` |
| `title` | str | 文档标题 |
| `platform` / `site` / `doc_type` / `lang` | str \| null | 见公共取值 |
| `category` | str \| null | 适用品类，如 `3C`、`家居`；不限品类写 `null` |
| `effective_date` / `expire_date` | str \| null | 见公共取值 |
| `content_hash` | str | 对清洗后 `text` 做 `sha256` 的完整十六进制（64 位）。对应 MySQL `kb_document.content_hash` |
| `fetched_at` | str | 抓取时间，ISO 8601，带时区，如 `2026-10-28T19:30:00+08:00` |
| `text` | str | 清洗后的全文，Markdown 格式，保留标题层级（`#`、`##`、`###`）和条款编号；去掉导航、页眉页脚、广告 |
| `question` / `answer` | str \| null | 只有 `xm_faq` 的文档有；此时 `text = question + "\n" + answer`，其他文档写 `null` |

说明：
- `doc_id`（MySQL 自增主键）**不出现在这两个文件里**。它在入库时由 `kb_document` 分配，再写进 Milvus。
- 同一个 `doc_key` 在 `documents.jsonl` 里只能出现一次，保留最新抓取的版本。
- FAQ 文档来自 `faq_items.jsonl` 里 `reviewed=true` 的条目，由解析步骤一并写进来：`collection=xm_faq`、`doc_type=faq`、`source_url=faq://generated/<faq_id>`、`title=question`、`fetched_at` 写生成时间。

## 2a. `raw/manifest.jsonl`（抓取登记）

抓取脚本每成功保存一个原文就写一行，字段和 `documents.jsonl` 的元数据一致，另加：

| 字段 | 类型 | 说明 |
|---|---|---|
| `raw_path` | str | 原文文件路径，相对项目根目录，如 `data/raw/3f2a9c1d0b7e4a56.html` |
| `content_type` | str | `text/html` 或 `application/pdf` |
| `http_status` | int | 抓取时的 HTTP 状态码 |
| `text_sha` | str | 抓取时用 trafilatura 抽出的正文做 sha256，取前 16 位，用来跳过正文完全相同的页面（同一页面的不同标签页常常这样）；PDF 为空字符串 |
| `acquired_by` | str | `crawler`（`scripts/crawl.py` 自动抓取）/ `manual`（用户在自己的浏览器里手动另存为，再用 `scripts/import_manual.py` 登记） |

**数据来源说明**：平台类资料（Amazon）的网站条款禁止使用爬虫等自动工具，所以 2026-10-28 发现这一点后就停止了自动抓取。之前自动抓下的 49 页，经用户决定保留，仅用于个人非商业学习；其余平台页面由用户手动保存。政府合规资料的许可都允许抓取：GOV.UK 是 OGL 开放许可，欧盟委员会允许注明出处后再利用，CPSC 是美国联邦政府作品，ZSVR 允许私人非商业下载。`data/raw/` 和 `data/processed/` 不进版本库，也不对外分发。

`sources.jsonl` 的字段是 manifest 去掉 `doc_key`、`raw_path`、`content_type`、`http_status`、`fetched_at` 之后剩下的部分（`source_url` 和元数据），由人工整理。

## 2b. `processed/faq_items.jsonl`（生成的 FAQ，待审核）

| 字段 | 类型 | 说明 |
|---|---|---|
| `faq_id` | str | `faq-0001` 起递增 |
| `question` / `answer` | str | 问题口语化，答案 2–5 句话，只能用证据里有的信息 |
| `intent` / `platform` / `site` / `category` / `lang` | — | 见公共取值；`lang` 是问答本身的语言，一般为 `zh` |
| `effective_date` | str \| null | 取证据文档的生效日期 |
| `evidence` | list | `[{"doc_key": "...", "quote": "原文里原样复制的一句话"}]`，至少 1 条 |
| `source` / `reviewed` / `reviewed_by` / `note` | — | 同评估集（第 4 节） |

只有 `reviewed=true` 的条目会进入知识库。

## 3. `chunks.jsonl`（片段）

| 字段 | 类型 | 说明 |
|---|---|---|
| `chunk_id` | str | `f"{doc_key}-{chunk_index:04d}"`，例如 `3f2a9c1d0b7e4a56-0003`。长度 21，不超过 Milvus 主键上限 64 |
| `doc_key` | str | 所属文档 |
| `chunk_index` | int | 在文档内的序号，从 0 开始 |
| `collection` / `platform` / `site` / `category` / `doc_type` / `lang` / `effective_date` / `expire_date` / `source_url` | — | 从所属文档原样复制，方便入库时不用再关联 |
| `title_path` | str | 标题路径，用 `" > "` 连接，如 `欧盟 GPSR > 第 9 条 制造商义务 > 第 3 款`；FAQ 写问题本身 |
| `text` | str | 片段正文，**不含**标题路径。送进 BGE-M3 的文本由入库代码拼成 `title_path + "\n" + text` |
| `question` / `answer` | str \| null | 只有 `xm_faq` 有；此时 `text = question + "\n" + answer` |
| `token_count` | int | 用 BGE-M3 的 tokenizer 数出来的 token 数（不含标题路径） |

分块规则（M1 先按这个实现，M3 再调）：
- **政策和法规**：按 `#` 到 `######` 标题和条款编号切；一块目标 300–500 token，**上限 800 token**，超过就按段落再切。不做重叠，上下文靠 `title_path` 补。（2026-10-29 从 1–3 级扩到 1–6 级：Amazon 中文页大量用 4–6 级标题，只认 1–3 级会让 `title_path` 丢掉这些小节名。）
- 条款编号行（`Article 9`、`第九条`、`§ 3`、`Section 2`）只有在不超过 80 个字符时才算标题，更长的是以条款号开头的正文句子。
- **FAQ**：一问一答一块，不切。
- `text` 不超过 1500 个字符，保证 UTF-8 编码后不超过 Milvus `VARCHAR(8192)` 的字节上限（中文每字 3 字节）。
- 空片段、只有标题没有正文的片段直接丢弃。

## 4. `eval/intent.jsonl`（意图标注集）

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | str | `int-0001` 起递增 |
| `query` | str | 卖家原话，口语化，可以带错别字 |
| `intent` | str | 8 类之一 |
| `strategy` | str \| null | `chitchat`、`out_of_scope` 写 `null` |
| `platform` | str \| null | 问题里**明确提到**的平台，没提写 `null`（不要根据常识推测） |
| `site` | str \| null | 同上 |
| `source` | str | `handwritten`（人写）/ `llm_generated`（LLM 生成） |
| `reviewed` | bool | 是否已审核；只有 `true` 的条目进入正式评估 |
| `reviewed_by` | str \| null | `human`（用户本人审过）/ `claude`（只经过 Claude 初审）；未审核写 `null` |
| `note` | str \| null | 标注时的疑问或边界情况说明 |

**审核要求**（2026-10-30 调整：检索集工作量太大，用户委托 Claude 审）：
- `intent.jsonl`：必须 `reviewed_by=human`（用户本人审过）。
- `retrieval_qa.jsonl`：`reviewed_by=human` 或 `claude` 都可以。Claude 审的部分，用户至少抽检 10%，抽检通过的条目在 `note` 里带 `[人工抽检通过]`。审核方法：先由比生成模型更强的模型对照证据筛查，Claude 再逐条复核标出的问题（`scripts/review_qa.py`）。
- FAQ 进知识库允许 `reviewed_by=claude`。Claude 在初审之外，又对照原文逐条复查了 10% 的抽样（`note` 带 `[Claude 复查…]`）；用户抽检的条目带 `[人工抽检通过]`。

对外的说法要和实际一致：意图集是"人工标注审核"，检索集和 FAQ 是"LLM 生成 + 模型筛查 + Claude 复核 + 人工抽检"，不能说成全部人工校对。

数量要求：8 类意图每类至少 25 条；`chitchat`、`out_of_scope` 各不少于 20 条；`subquery` 和 `hyde` 各不少于 40 条。

## 5. `eval/retrieval_qa.jsonl`（检索/问答评估集）

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | str | `qa-0001` 起递增 |
| `query` | str | 问题 |
| `intent` / `platform` / `site` | — | 同意图标注集 |
| `answerable` | bool | 知识库里是否有答案。`false` 的条目用来评估"知识库未覆盖"的判断，约占 15% |
| `evidence` | list | 支撑答案的原文片段：`[{"doc_key": "...", "quote": "原文里能精确找到的一句话"}]`；`answerable=false` 时为 `[]` |
| `reference_answer` | str \| null | 标准答案，2–5 句话；`answerable=false` 时写 `null` |
| `paraphrases` | list[str] | 2–3 个同义改写，用来画语义缓存阈值 τ 的 P-R 曲线 |
| `source` / `reviewed` / `reviewed_by` / `note` | — | 同意图标注集 |

**为什么证据用原文引用 `quote`，而不是 `chunk_id`**：分块策略在 M3 会调整，片段编号会跟着变。用原文引用标注，换了分块方式标注依然有效。计算 Hit@K 时，前 K 个片段里只要有一个包含某条 `quote`，就算命中。所以 `quote` 必须是从 `documents.jsonl` 的 `text` 里原样复制的，不能改写。

**引文匹配规则**（生成脚本、`validate_data.py`、Hit@K 统一使用 `xiaomao.textnorm.normalize_for_match`）：两边都先做 NFKC 归一化（全角标点转半角），再去掉所有空白和 Markdown 符号 `*` `_` `` ` `` `#` `>` `|` `\`，然后做子串匹配。之所以要去掉 Markdown 符号，是因为 129 篇文档里有 93 篇带 `**` 加粗，模型抄引文时通常会把它们去掉。
- `quote` 长度 10–200 个字符，不能是标题行。
- `retrieval_qa` 的每条 `quote` 还必须完整落在**当前**某一个片段的 `text` 里，否则这条证据在 Hit@K 里永远不可能命中。M3 换分块方式后要重新检查一遍，被切断的证据由人工修正。

## 6. 校验

M1 要附带一个校验脚本 `scripts/validate_data.py`，检查以上所有约定：字段齐全、取值合法、`chunk_id` 格式和唯一性、`token_count` 不超过上限、`quote` 能在对应文档里找到、各类数量达标、评估集满足上面第 4 节末尾的审核要求。校验不通过不进入 M2。

**防止评估集泄题**：`retrieval_qa.jsonl` 的问题不能和 `faq_items.jsonl` 的问题太像，否则 FAQ 库会"背出"评估题。生成评估集时用 BGE-M3 算相似度，和任何一条 FAQ 问题的 dense 余弦相似度 > 0.90 的直接丢弃。
