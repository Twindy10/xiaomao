// 知识库收录情况，从 data/processed/documents.jsonl 统计（2026-10-29 抓取，不含 FAQ）。
// 入库数据变了要重新统计：按 collection、platform、site 计数
export const KB_UPDATED = "2026-10-29";
export const KB_PASSAGES = 1447;
export const KB_FAQ = 339;

export interface CoverageRow {
  label: string;
  docs: number;
}

export const AMAZON_POLICY: CoverageRow[] = [
  { label: "各站通用", docs: 48 },
  { label: "美国站", docs: 15 },
  { label: "欧洲站", docs: 5 },
  { label: "日本站", docs: 3 },
  { label: "澳大利亚、阿联酋", docs: 4 },
];

export const COMPLIANCE: CoverageRow[] = [
  { label: "英国", docs: 25 },
  { label: "欧盟", docs: 14 },
  { label: "德国", docs: 10 },
  { label: "美国", docs: 5 },
  { label: "日本及其他地区", docs: 0 },
];

const sum = (rows: CoverageRow[]) => rows.reduce((total, row) => total + row.docs, 0);
export const AMAZON_TOTAL = sum(AMAZON_POLICY);
export const COMPLIANCE_TOTAL = sum(COMPLIANCE);
export const KB_DOCS = AMAZON_TOTAL + COMPLIANCE_TOTAL;

console.assert(AMAZON_TOTAL === 75, "亚马逊政策应为 75 篇", AMAZON_TOTAL);
console.assert(COMPLIANCE_TOTAL === 54, "合规法规应为 54 篇", COMPLIANCE_TOTAL);
console.assert(KB_DOCS === 129, "知识库应为 129 篇", KB_DOCS);
