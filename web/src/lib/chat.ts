import type { Citation, Message } from "../api/types";

export type Phase = "understanding" | "retrieving" | "writing";

/** 界面上的一条消息：比后端的 Message 多了流式状态 */
export interface UIMessage {
  key: string;
  id: number | null;
  role: "user" | "assistant";
  content: string;
  citations: Citation[];
  rating: 1 | -1 | null;
  status: "streaming" | "done" | "error";
  phase?: Phase;
  cacheHit?: boolean;
  intent?: string | null;
  refused?: boolean;
  animate?: boolean; // 本次会话里流式生成的才做打字动画，历史消息直接显示
  error?: string;
  requestId?: string;
  question?: string; // 出错时用来重试
  stopped?: boolean; // 用户点了停止
}

let seq = 0;
export const newKey = () => `m${Date.now().toString(36)}${(seq++).toString(36)}`;

// 历史消息没有 refused 字段，按拒答文案的开头认（generation/answer.py 的 REFUSAL_TEMPLATE）；
// 2026-11-01 改过一次文案，旧开头留着，老对话照样识别
const REFUSAL_HINT = /^(暂未找到与该问题相关的资料|这个问题知识库里暂时没有可靠的资料覆盖)/;

export function fromServer(message: Message): UIMessage {
  return {
    key: `s${message.id}`,
    id: message.id,
    role: message.role,
    content: message.content,
    citations: message.citations ?? [],
    rating: message.rating ?? null,
    status: "done",
    refused: message.role === "assistant" && !(message.citations ?? []).length && REFUSAL_HINT.test(message.content),
  };
}

export const INTENT_LABELS: Record<string, string> = {
  platform_policy: "平台规则",
  compliance: "合规认证",
  logistics: "物流",
  tax: "税务",
  ops: "运营技巧",
  account: "账号问题",
};
