// 真实后端：JWT + 自动续期，SSE 用 fetch 读流（EventSource 不支持 POST）
import {
  ApiError,
  type Api,
  type ChatHandlers,
  type Citation,
  type Conversation,
  type FeedbackReason,
  type Me,
  type Message,
  type Profile,
  type ProfileResponse,
  type Usage,
  type User,
} from "./types";

const ACCESS_KEY = "xm.access";
const REFRESH_KEY = "xm.refresh";

/** 会话失效（refresh 也失败）时通知 App 回到登录页 */
export const sessionEvents = new EventTarget();

function readToken(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

function saveTokens(access: string | null, refresh: string | null) {
  try {
    if (access) localStorage.setItem(ACCESS_KEY, access);
    else localStorage.removeItem(ACCESS_KEY);
    if (refresh) localStorage.setItem(REFRESH_KEY, refresh);
    else localStorage.removeItem(REFRESH_KEY);
  } catch {
    /* 隐私模式下存不了，就只在本次会话里有效 */
  }
}

async function toError(response: Response): Promise<ApiError> {
  let code = `http_${response.status}`;
  let message = "请求失败，请稍后再试";
  try {
    const body = await response.json();
    if (typeof body?.error === "string") code = body.error;
    if (typeof body?.message === "string") message = body.message;
    else if (typeof body?.error === "string") message = body.error;
  } catch {
    /* 不是 JSON */
  }
  return new ApiError(response.status, code, message, response.headers.get("X-Request-Id") ?? undefined);
}

// 并发的请求同时遇到 401 时只换一次 token
let refreshing: Promise<boolean> | null = null;

async function refreshTokens(): Promise<boolean> {
  const refreshToken = readToken(REFRESH_KEY);
  if (!refreshToken) return false;
  refreshing ??= (async () => {
    try {
      const response = await fetch("/api/auth/refresh", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ refresh_token: refreshToken }),
      });
      if (!response.ok) return false;
      const body = await response.json();
      saveTokens(body.access_token, body.refresh_token);
      return true;
    } catch {
      return false;
    } finally {
      queueMicrotask(() => (refreshing = null));
    }
  })();
  return refreshing;
}

async function request(path: string, init: RequestInit = {}, retry = true): Promise<Response> {
  const headers = new Headers(init.headers);
  const access = readToken(ACCESS_KEY);
  if (access) headers.set("Authorization", `Bearer ${access}`);
  if (init.body && !headers.has("Content-Type")) headers.set("Content-Type", "application/json");

  const response = await fetch(path, { ...init, headers });
  if (response.status === 401 && retry && !path.startsWith("/api/auth/")) {
    if (await refreshTokens()) return request(path, init, false);
    saveTokens(null, null);
    sessionEvents.dispatchEvent(new Event("expired"));
  }
  if (!response.ok) throw await toError(response);
  return response;
}

async function json<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await request(path, init);
  return (response.status === 204 ? undefined : await response.json()) as T;
}

/** MySQL 的 JSON 列经 pymysql 取出来是字符串，这里统一解析 */
function parseCitations(value: unknown): Citation[] {
  if (Array.isArray(value)) return value as Citation[];
  if (typeof value === "string" && value) {
    try {
      const parsed = JSON.parse(value);
      return Array.isArray(parsed) ? parsed : [];
    } catch {
      return [];
    }
  }
  return [];
}

/** 按 SSE 规范切事件：事件之间空一行，每行 `event:` 或 `data:` */
async function readSse(response: Response, onEvent: (event: string, data: string) => void) {
  const reader = response.body!.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value.replace(/\r\n/g, "\n");
    let boundary: number;
    while ((boundary = buffer.indexOf("\n\n")) >= 0) {
      const block = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      let event = "message";
      const data: string[] = [];
      for (const line of block.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data.push(line.slice(5).replace(/^ /, ""));
      }
      if (data.length) onEvent(event, data.join("\n"));
    }
  }
}

export const realApi: Api = {
  mock: false,

  hasSession: () => Boolean(readToken(ACCESS_KEY) || readToken(REFRESH_KEY)),

  async register(phone, password, nickname) {
    await json("/api/auth/register", {
      method: "POST",
      body: JSON.stringify({ phone, password, nickname: nickname || null }),
    });
  },

  async login(phone, password) {
    const body = await json<{ access_token: string; refresh_token: string; user: User }>("/api/auth/login", {
      method: "POST",
      body: JSON.stringify({ phone, password }),
    });
    saveTokens(body.access_token, body.refresh_token);
    return body.user;
  },

  async logout() {
    try {
      await request("/api/auth/logout", {
        method: "POST",
        body: JSON.stringify({ refresh_token: readToken(REFRESH_KEY) }),
      }, false);
    } catch {
      /* 本地照样退出 */
    }
    saveTokens(null, null);
  },

  me: () => json<Me>("/api/me"),
  getProfile: () => json<ProfileResponse>("/api/profile"),
  saveProfile: (profile: Profile) =>
    json<ProfileResponse>("/api/profile", { method: "PUT", body: JSON.stringify(profile) }),
  listConversations: () => json<Conversation[]>("/api/conversations?limit=50"),
  async deleteConversation(id) {
    await json(`/api/conversations/${id}`, { method: "DELETE" });
  },

  async listMessages(conversationId) {
    const rows = await json<Message[]>(`/api/conversations/${conversationId}/messages`);
    return rows.map((row) => ({ ...row, citations: parseCitations(row.citations) }));
  },

  async chat(question, conversationId, handlers: ChatHandlers, signal) {
    const response = await request("/api/chat", {
      method: "POST",
      body: JSON.stringify({ question, conversation_id: conversationId }),
      signal,
    });
    let finished = false;
    await readSse(response, (event, data) => {
      const payload = JSON.parse(data);
      if (event === "meta") handlers.onMeta(payload);
      else if (event === "delta") handlers.onDelta(payload.text ?? "");
      else if (event === "done") {
        finished = true;
        handlers.onDone({ ...payload, citations: parseCitations(payload.citations) });
      } else if (event === "error") {
        finished = true;
        throw new ApiError(500, "stream_error", payload.error ?? "内部错误", payload.request_id);
      }
    });
    if (!finished) throw new ApiError(0, "stream_cut", "连接中断了，请重试");
  },

  async feedback(messageId, rating, reason?: FeedbackReason, comment?: string) {
    await json(`/api/messages/${messageId}/feedback`, {
      method: "POST",
      body: JSON.stringify({ rating, reason: rating === -1 ? reason ?? null : null, comment: comment || null }),
    });
  },

  usage: () => json<Usage>("/api/usage"),
};
