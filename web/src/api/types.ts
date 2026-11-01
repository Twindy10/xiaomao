// 和后端约定的数据结构（docs/deepseek/M5_tasks.md、M4b 的 /api/chat）

export interface User {
  id: number;
  nickname: string | null;
  plan: string;
}

export interface Me extends User {
  phone: string;
  daily_quota: number;
}

export interface Citation {
  n: number;
  chunk_id: string;
  collection?: string;
  source_url?: string | null;
  title_path?: string | null;
  effective_date?: string | null;
  rank?: number;
  doc_id?: number | null;
}

export interface Conversation {
  id: number;
  title: string;
  updated_at: string;
}

export interface Message {
  id: number;
  role: "user" | "assistant";
  content: string;
  citations: Citation[] | null;
  created_at: string;
  rating?: 1 | -1 | null;
}

export interface Profile {
  platforms: string[];
  sites: string[];
  categories: string[];
  stage: "newbie" | "growing" | null;
}

export interface ProfileResponse extends Profile {
  options: { platforms: Record<string, string>; sites: Record<string, string> };
}

export interface Usage {
  day: string;
  used: number;
  daily_quota: number;
  tokens: number;
  plan: string;
}

export interface ChatMeta {
  conversation_id: number;
  intent: string | null;
  strategy: string | null;
  cache_hit: boolean;
}

export interface ChatDone {
  message_id: number | null;
  citations: Citation[];
  refused: boolean;
  latency_ms: Record<string, number>;
  answer?: string; // 模拟接口会带上最终版回答；真实接口没有时用流式拼出来的全文
}

export interface ChatHandlers {
  onMeta: (meta: ChatMeta) => void;
  onDelta: (text: string) => void;
  onDone: (done: ChatDone) => void;
}

export type FeedbackReason = "wrong" | "outdated" | "off_topic" | "other";

export interface Api {
  readonly mock: boolean;
  hasSession(): boolean;
  register(phone: string, password: string, nickname?: string): Promise<void>;
  login(phone: string, password: string): Promise<User>;
  logout(): Promise<void>;
  me(): Promise<Me>;
  getProfile(): Promise<ProfileResponse>;
  saveProfile(profile: Profile): Promise<ProfileResponse>;
  listConversations(): Promise<Conversation[]>;
  deleteConversation(id: number): Promise<void>;
  listMessages(conversationId: number): Promise<Message[]>;
  chat(question: string, conversationId: number | null, handlers: ChatHandlers, signal: AbortSignal): Promise<void>;
  feedback(messageId: number, rating: 1 | -1, reason?: FeedbackReason, comment?: string): Promise<void>;
  usage(): Promise<Usage>;
}

/** 接口错误：status 是 HTTP 状态码，code 是后端的 error 字段 */
export class ApiError extends Error {
  status: number;
  code: string;
  requestId?: string;

  constructor(status: number, code: string, message: string, requestId?: string) {
    super(message);
    this.status = status;
    this.code = code;
    this.requestId = requestId;
  }
}
