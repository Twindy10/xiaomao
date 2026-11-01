import type { Conversation } from "../api/types";

/** 后端返回的 MySQL DATETIME 可能是 "2026-10-31 12:00:00.123"，没有时区，按本地时间解析 */
export function parseTime(value: string): Date {
  return new Date(value.includes("T") ? value : value.replace(" ", "T"));
}

const startOfDay = (date: Date) => new Date(date.getFullYear(), date.getMonth(), date.getDate()).getTime();

export function groupConversations(list: Conversation[]): { label: string; items: Conversation[] }[] {
  const today = startOfDay(new Date());
  const day = 86_400_000;
  const groups: Record<string, Conversation[]> = { 今天: [], 昨天: [], "近 7 天": [], 更早: [] };
  for (const item of list) {
    const t = startOfDay(parseTime(item.updated_at));
    if (t >= today) groups["今天"].push(item);
    else if (t >= today - day) groups["昨天"].push(item);
    else if (t >= today - 7 * day) groups["近 7 天"].push(item);
    else groups["更早"].push(item);
  }
  return Object.entries(groups)
    .filter(([, items]) => items.length)
    .map(([label, items]) => ({ label, items }));
}

export function formatDate(value?: string | null): string {
  if (!value) return "";
  const date = parseTime(String(value));
  if (Number.isNaN(date.getTime())) return String(value);
  return `${date.getFullYear()}.${String(date.getMonth() + 1).padStart(2, "0")}.${String(date.getDate()).padStart(2, "0")}`;
}

export function greeting(): string {
  const hour = new Date().getHours();
  if (hour < 6) return "夜深了";
  if (hour < 11) return "早上好";
  if (hour < 14) return "中午好";
  if (hour < 18) return "下午好";
  return "晚上好";
}
