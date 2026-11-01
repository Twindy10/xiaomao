// 模拟接口：后端 M5 交付前用来开发和演示界面（npm run dev:mock）
import {
  ApiError,
  type Api,
  type ChatHandlers,
  type Citation,
  type Conversation,
  type Message,
  type Profile,
  type ProfileResponse,
} from "./types";

const OPTIONS = {
  platforms: { amazon: "Amazon", temu: "Temu", tiktok_shop: "TikTok Shop", shopee: "Shopee" },
  sites: {
    US: "美国站", UK: "英国站", DE: "德国站", JP: "日本站", FR: "法国站",
    IT: "意大利站", ES: "西班牙站", EU: "欧盟整体",
  },
};

const sleep = (ms: number, signal?: AbortSignal) =>
  new Promise<void>((resolve, reject) => {
    const timer = setTimeout(resolve, ms);
    signal?.addEventListener("abort", () => {
      clearTimeout(timer);
      reject(new DOMException("aborted", "AbortError"));
    });
  });

const hoursAgo = (h: number) => new Date(Date.now() - h * 3600_000).toISOString();

interface Template {
  match: RegExp;
  answer: string;
  citations: Citation[];
}

const TEMPLATES: Template[] = [
  {
    match: /认证|CE|合规|耳机|FCC|GPSR/i,
    answer:
      "在德国站销售蓝牙耳机，主要需要满足以下几项要求：\n\n" +
      "1. **CE 标志（RED 指令）**：蓝牙耳机属于无线电设备，需要按无线电设备指令 2014/53/EU 做符合性评估，出具欧盟符合性声明（DoC）并加贴 CE 标志 [1]。\n" +
      "2. **RoHS**：限制电子电气产品中铅、汞、镉等有害物质的使用 [2]。\n" +
      "3. **WEEE 注册**：在德国销售电子电气设备，需要在 stiftung ear 注册并取得 WEEE 注册号，listing 里要填写这个编号 [3]。\n" +
      "4. **电池法规**：耳机内置锂电池时，还要履行电池生产者登记义务 [3]。\n\n" +
      "另外，亚马逊德国站会在上架或抽查时要求上传 DoC 和检测报告，建议提前准备好欧盟境内的负责人信息。",
    citations: [
      { n: 1, chunk_id: "c-red-01", collection: "xm_compliance", source_url: "https://single-market-economy.ec.europa.eu/sectors/electrical-and-electronic-engineering-industries-eei/radio-equipment-directive-red_en", title_path: "无线电设备指令 RED > 适用范围与符合性评估", effective_date: "2025-08-01" },
      { n: 2, chunk_id: "c-rohs-02", collection: "xm_compliance", source_url: "https://environment.ec.europa.eu/topics/waste-and-recycling/rohs-directive_en", title_path: "RoHS 指令 > 限用物质清单", effective_date: "2024-11-15" },
      { n: 3, chunk_id: "c-weee-03", collection: "xm_compliance", source_url: "https://www.stiftung-ear.de/en/", title_path: "德国 WEEE 与电池法 > 生产者注册义务", effective_date: "2026-01-01" },
    ],
  },
  {
    match: /FBA|物流|仓储|配送|发货|头程/i,
    answer:
      "亚马逊美国站 FBA 的费用主要分两块：\n\n" +
      "- **配送费**：按商品尺寸分段和发货重量计算，低价商品（售价低于 10 美元）可以享受更低的配送费率 [1]。\n" +
      "- **月度仓储费**：按日均占用体积（立方英尺）收取，10–12 月是旺季，费率明显更高 [2]。\n\n" +
      "库龄超过 181 天的库存还会额外收取**库龄附加费**，建议定期清理滞销库存 [2]。",
    citations: [
      { n: 1, chunk_id: "p-fba-11", collection: "xm_platform_policy", source_url: "https://sellercentral.amazon.com/help/hub/reference/external/GPDC3KPYAGDTVDJP", title_path: "FBA 费用 > 配送费 > 尺寸分段", effective_date: "2026-01-15" },
      { n: 2, chunk_id: "p-fba-12", collection: "xm_platform_policy", source_url: "https://sellercentral.amazon.com/help/hub/reference/external/G3EDYEF6KUCFQTNM", title_path: "FBA 费用 > 仓储费与库龄附加费", effective_date: "2026-01-15" },
    ],
  },
  {
    match: /.*/,
    answer:
      "Temu 的两种模式主要区别在于**谁负责履约**：\n\n" +
      "- **全托管**：卖家只负责供货，把货送到 Temu 国内仓，定价、运营、物流和售后都由平台负责 [1]。\n" +
      "- **半托管**：卖家自己负责海外仓发货和本地履约，对定价和库存有更多自主权，适合已经有海外仓的卖家 [2]。\n\n" +
      "入驻时两种模式都需要企业资质，半托管还需要提供海外仓或本地发货能力的证明。",
    citations: [
      { n: 1, chunk_id: "p-temu-01", collection: "xm_platform_policy", source_url: "https://seller.temu.com/", title_path: "Temu 招商 > 全托管模式说明", effective_date: "2026-03-01" },
      { n: 2, chunk_id: "p-temu-02", collection: "xm_platform_policy", source_url: "https://seller.temu.com/", title_path: "Temu 招商 > 半托管模式说明", effective_date: "2026-03-01" },
    ],
  },
];

interface MockState {
  conversations: Conversation[];
  messages: Record<number, Message[]>;
  profile: Profile;
  used: number;
  answered: Set<string>;
  nextId: number;
}

function seed(): MockState {
  const t = TEMPLATES[0];
  return {
    conversations: [
      { id: 3, title: "德国站卖蓝牙耳机需要哪些认证？", updated_at: hoursAgo(2) },
      { id: 2, title: "FBA 仓储费怎么计算", updated_at: hoursAgo(26) },
      { id: 1, title: "Temu 半托管和全托管有什么区别", updated_at: hoursAgo(24 * 5) },
    ],
    messages: {
      3: [
        { id: 31, role: "user", content: "德国站卖蓝牙耳机需要哪些认证？", citations: null, created_at: hoursAgo(2) },
        { id: 32, role: "assistant", content: t.answer, citations: t.citations, created_at: hoursAgo(2), rating: 1 },
      ],
      2: [
        { id: 21, role: "user", content: "FBA 仓储费怎么计算", citations: null, created_at: hoursAgo(26) },
        { id: 22, role: "assistant", content: TEMPLATES[1].answer, citations: TEMPLATES[1].citations, created_at: hoursAgo(26), rating: null },
      ],
      1: [
        { id: 11, role: "user", content: "Temu 半托管和全托管有什么区别", citations: null, created_at: hoursAgo(120) },
        { id: 12, role: "assistant", content: TEMPLATES[2].answer, citations: TEMPLATES[2].citations, created_at: hoursAgo(120), rating: null },
      ],
    },
    profile: { platforms: ["amazon"], sites: ["DE"], categories: ["3C 数码"], stage: "growing" },
    used: 3,
    answered: new Set(["德国站卖蓝牙耳机需要哪些认证？"]),
    nextId: 100,
  };
}

const state = seed();
const DAILY_QUOTA = 20;
let loggedIn = (() => {
  try {
    return localStorage.getItem("xm.mock.session") === "1";
  } catch {
    return false;
  }
})();
let nickname = "林小满";

function setSession(value: boolean) {
  loggedIn = value;
  try {
    if (value) localStorage.setItem("xm.mock.session", "1");
    else localStorage.removeItem("xm.mock.session");
  } catch {
    /* 忽略 */
  }
}

function requireLogin() {
  if (!loggedIn) throw new ApiError(401, "unauthorized", "请先登录");
}

const profileResponse = (): ProfileResponse => ({ ...state.profile, options: OPTIONS });

export const mockApi: Api = {
  mock: true,
  hasSession: () => loggedIn,

  async register(phone, password, name) {
    await sleep(500);
    if (!/^1[3-9]\d{9}$/.test(phone)) throw new ApiError(422, "invalid", "手机号格式不对");
    if (password.length < 8 || !/[a-zA-Z]/.test(password) || !/\d/.test(password))
      throw new ApiError(422, "invalid", "密码至少 8 位，需要同时包含字母和数字");
    if (name) nickname = name;
  },

  async login(phone, password) {
    await sleep(600);
    if (!/^1[3-9]\d{9}$/.test(phone) || password.length < 8)
      throw new ApiError(401, "invalid_credentials", "手机号或密码错误");
    setSession(true);
    return { id: 1, nickname, plan: "free" };
  },

  async logout() {
    setSession(false);
  },

  async me() {
    requireLogin();
    return { id: 1, nickname, plan: "free", phone: "138****1234", daily_quota: DAILY_QUOTA };
  },

  async getProfile() {
    requireLogin();
    await sleep(200);
    return profileResponse();
  },

  async saveProfile(profile) {
    requireLogin();
    await sleep(450);
    state.profile = profile;
    return profileResponse();
  },

  async listConversations() {
    requireLogin();
    await sleep(250);
    return [...state.conversations].sort((a, b) => b.updated_at.localeCompare(a.updated_at));
  },

  async deleteConversation(id) {
    await sleep(200);
    state.conversations = state.conversations.filter((c) => c.id !== id);
    delete state.messages[id];
  },

  async listMessages(id) {
    requireLogin();
    await sleep(220);
    if (!state.messages[id]) throw new ApiError(404, "not_found", "会话不存在");
    return state.messages[id].map((m) => ({ ...m }));
  },

  async chat(question, conversationId, handlers: ChatHandlers, signal) {
    requireLogin();
    if (state.used >= DAILY_QUOTA) throw new ApiError(429, "quota_exceeded", "今日额度已用完，明天再来吧");
    state.used += 1;

    let cid = conversationId;
    if (cid == null) {
      cid = state.nextId++;
      state.conversations.unshift({ id: cid, title: question.slice(0, 30), updated_at: new Date().toISOString() });
      state.messages[cid] = [];
    }
    const now = new Date().toISOString();
    state.messages[cid].push({ id: state.nextId++, role: "user", content: question, citations: null, created_at: now });

    const chitchat = /^(你好|您好|谢谢|hi|hello)/i.test(question.trim());
    const cacheHit = !chitchat && state.answered.has(question.trim());
    const template = TEMPLATES.find((t) => t.match.test(question))!;
    const answer = chitchat
      ? "你好！我是小贸同学，可以帮你查平台规则、产品合规认证、物流和税务这些问题。直接问我就好。"
      : template.answer;
    const citations = chitchat ? [] : template.citations;

    await sleep(cacheHit ? 650 : 1100, signal);
    handlers.onMeta({ conversation_id: cid, intent: chitchat ? "chitchat" : "compliance", strategy: chitchat ? null : "direct", cache_hit: cacheHit });

    if (cacheHit || chitchat) {
      await sleep(120, signal);
      handlers.onDelta(answer);
    } else {
      await sleep(900, signal);
      // 真实模型是一段一段吐的，长短不一
      for (let i = 0; i < answer.length; ) {
        const size = 2 + Math.floor(Math.random() * 7);
        handlers.onDelta(answer.slice(i, i + size));
        i += size;
        await sleep(25 + Math.random() * 55, signal);
      }
    }

    const messageId = state.nextId++;
    state.messages[cid].push({ id: messageId, role: "assistant", content: answer, citations, created_at: new Date().toISOString(), rating: null });
    const conversation = state.conversations.find((c) => c.id === cid);
    if (conversation) conversation.updated_at = new Date().toISOString();
    if (!chitchat) state.answered.add(question.trim());
    handlers.onDone({ message_id: messageId, citations, refused: false, latency_ms: { total: cacheHit ? 800 : 4200 }, answer });
  },

  async feedback(messageId, rating) {
    await sleep(300);
    for (const list of Object.values(state.messages)) {
      const message = list.find((m) => m.id === messageId);
      if (message) message.rating = rating;
    }
  },

  async usage() {
    requireLogin();
    return { day: new Date().toISOString().slice(0, 10), used: state.used, daily_quota: DAILY_QUOTA, tokens: state.used * 2700, plan: "free" };
  },
};
