// 界面上的中文标签。后端 options 里的说明是给决策模型看的提示词，不适合直接展示

export const PLATFORM_LABELS: Record<string, string> = {
  amazon: "亚马逊",
  temu: "Temu",
  tiktok_shop: "TikTok Shop",
  shopee: "Shopee",
};

export const SITE_LABELS: Record<string, string> = {
  US: "美国",
  UK: "英国",
  DE: "德国",
  JP: "日本",
  FR: "法国",
  IT: "意大利",
  ES: "西班牙",
  EU: "欧盟",
};

export const STAGE_LABELS: Record<string, { title: string; hint: string }> = {
  newbie: { title: "刚起步", hint: "还在入驻、上新阶段" },
  growing: { title: "稳定经营", hint: "已经有稳定的出单" },
};

export const REASON_LABELS = {
  wrong: "答错了",
  outdated: "政策过时",
  off_topic: "答非所问",
  other: "其他",
} as const;

export const PLAN_LABELS: Record<string, string> = { free: "免费版", pro: "专业版" };

// 首页建议的问题都要是知识库答得上来的（评估集里答对过的题），否则演示时一点就拒答
export const SUGGESTIONS = [
  { tag: "合规认证", text: "德国站卖蓝牙耳机需要哪些认证？" },
  { tag: "合规认证", text: "在欧盟给产品贴 CE 标志需要走哪几步？" },
  { tag: "物流费用", text: "亚马逊美国站 FBA 的仓储费怎么计算？" },
  { tag: "关税", text: "怎么查英国进出口商品的关税编码？" },
];

export function hostOf(url?: string | null): string {
  if (!url) return "";
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return "";
  }
}
