import { ArrowDown } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, ApiError, type FeedbackReason, type Profile } from "../api";
import { fromServer, newKey, type UIMessage } from "../lib/chat";
import Composer, { type ComposerHandle } from "./Composer";
import { scopeOf } from "./scope";
import Turn from "./Turn";
import { useToast } from "./ui";

interface Props {
  conversationId: number | null;
  initialQuestion?: string;
  title: string | null;
  profile: Profile | null;
  onCreated: (id: number, title: string) => void;
  onAnswered: () => void;
  onOpenSettings: () => void;
}

export default function Chat({ conversationId, initialQuestion, title, profile, onCreated, onAnswered, onOpenSettings }: Props) {
  const toast = useToast();
  const [messages, setMessages] = useState<UIMessage[]>([]);
  const [loading, setLoading] = useState(conversationId != null);
  const [busy, setBusy] = useState(false);
  const [atBottom, setAtBottom] = useState(true);
  const scroller = useRef<HTMLDivElement>(null);
  const content = useRef<HTMLDivElement>(null);
  const stick = useRef(true);
  const lastTop = useRef(0);
  const abort = useRef<AbortController | null>(null);
  const composer = useRef<ComposerHandle>(null);
  const conversation = useRef(conversationId);

  // 打开已有对话：只在挂载时加载一次。这里新建的对话只改地址、不重新挂载，回答还在流式生成
  const [openedId] = useState(conversationId);
  useEffect(() => {
    if (openedId == null) return;
    let cancelled = false;
    api
      .listMessages(openedId)
      .then((rows) => !cancelled && setMessages(rows.map(fromServer)))
      .catch(() => !cancelled && toast("对话加载失败", "error"))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [openedId, toast]);

  useEffect(() => () => abort.current?.abort(), []);

  // 内容长高时，用户本来在底部就跟着滚
  useEffect(() => {
    const box = scroller.current;
    const inner = content.current;
    if (!box || !inner) return;
    const observer = new ResizeObserver(() => {
      if (stick.current) box.scrollTop = box.scrollHeight;
    });
    observer.observe(inner);
    return () => observer.disconnect();
  }, []);

  // 只有往上滚才停止跟随；平滑滚动的中间过程不算
  const onScroll = useCallback(() => {
    const box = scroller.current;
    if (!box) return;
    const near = box.scrollHeight - box.scrollTop - box.clientHeight < 80;
    if (near) stick.current = true;
    else if (box.scrollTop < lastTop.current - 2) stick.current = false;
    lastTop.current = box.scrollTop;
    setAtBottom(near);
  }, []);

  const scrollToBottom = useCallback((smooth = true) => {
    const box = scroller.current;
    if (!box) return;
    stick.current = true;
    box.scrollTo({ top: box.scrollHeight, behavior: smooth ? "smooth" : "auto" });
  }, []);

  const patch = useCallback((key: string, update: Partial<UIMessage> | ((m: UIMessage) => Partial<UIMessage>)) => {
    setMessages((list) => list.map((m) => (m.key === key ? { ...m, ...(typeof update === "function" ? update(m) : update) } : m)));
  }, []);

  const send = useCallback(
    async (question: string, replaceKey?: string) => {
      if (abort.current) return;
      const key = newKey();
      const placeholder: UIMessage = {
        key,
        id: null,
        role: "assistant",
        content: "",
        citations: [],
        rating: null,
        status: "streaming",
        phase: "understanding",
        animate: true,
        question,
      };
      setMessages((list) =>
        replaceKey
          ? [...list.filter((m) => m.key !== replaceKey), placeholder]
          : [...list, { key: newKey(), id: null, role: "user", content: question, citations: [], rating: null, status: "done" }, placeholder],
      );
      requestAnimationFrame(() => scrollToBottom(false));

      const controller = new AbortController();
      abort.current = controller;
      setBusy(true);
      try {
        await api.chat(
          question,
          conversation.current,
          {
            onMeta: (meta) => {
              if (conversation.current == null) {
                conversation.current = meta.conversation_id;
                onCreated(meta.conversation_id, question.slice(0, 30));
              }
              patch(key, { cacheHit: meta.cache_hit, intent: meta.intent, phase: meta.cache_hit ? "writing" : "retrieving" });
            },
            onDelta: (text) => patch(key, (m) => ({ content: m.content + text, phase: "writing" })),
            onDone: (done) =>
              patch(key, (m) => ({
                id: done.message_id,
                content: done.answer ?? m.content,
                citations: done.citations ?? [],
                refused: done.refused,
                status: "done",
              })),
          },
          controller.signal,
        );
      } catch (error) {
        if (error instanceof DOMException && error.name === "AbortError") {
          patch(key, (m) => ({ status: "done", stopped: true, content: m.content || "（已停止）" }));
        } else {
          const apiError = error instanceof ApiError ? error : null;
          const message = apiError?.status === 429 || apiError?.code === "stream_cut" ? apiError.message : "生成失败，请重试";
          patch(key, { status: "error", error: message, requestId: apiError?.requestId });
        }
      } finally {
        abort.current = null;
        setBusy(false);
        onAnswered();
      }
    },
    [onAnswered, onCreated, patch, scrollToBottom],
  );

  // 从首页带过来的问题：挂载后发一次（StrictMode 下 effect 会跑两遍，用 ref 挡住）
  const asked = useRef(false);
  useEffect(() => {
    if (initialQuestion && openedId == null && !asked.current) {
      asked.current = true;
      void send(initialQuestion);
    } else if (openedId == null) composer.current?.focus();
  }, [initialQuestion, openedId, send]);

  const rate = useCallback(
    async (message: UIMessage, rating: 1 | -1, reason?: FeedbackReason, comment?: string) => {
      if (message.id == null) return false;
      const previous = message.rating;
      patch(message.key, { rating });
      try {
        await api.feedback(message.id, rating, reason, comment);
        toast(rating === 1 ? "感谢反馈" : "反馈已提交");
        return true;
      } catch {
        patch(message.key, { rating: previous });
        toast("提交失败，请稍后重试", "error");
        return false;
      }
    },
    [patch, toast],
  );

  // 一问一答成一轮；重试时回答会被替换，问题保留
  const turns = useMemo(() => {
    const out: { question: UIMessage | null; answer: UIMessage | null }[] = [];
    for (const message of messages) {
      if (message.role === "user") out.push({ question: message, answer: null });
      else if (out.length && !out[out.length - 1].answer) out[out.length - 1].answer = message;
      else out.push({ question: null, answer: message });
    }
    return out;
  }, [messages]);

  const scope = scopeOf(profile);
  // 页头一行对话标题（会话标题只存了前 30 个字，有完整的第一个问题时用完整的）
  const firstQuestion = turns[0]?.question?.content ?? turns[0]?.answer?.question;
  const heading = firstQuestion ?? initialQuestion ?? title ?? "新对话";

  return (
    <div className="chat">
      <div className="chat-scroll" ref={scroller} onScroll={onScroll}>
        <div className="thread" ref={content}>
          <header className="chat-head">
            <h1>{heading}</h1>
          </header>

          {loading && <TurnSkeleton />}

          {turns.map((turn, index) => (
            <Turn
              key={turn.answer?.key ?? turn.question?.key ?? index}
              question={turn.question?.content ?? turn.answer?.question ?? ""}
              answer={turn.answer}
              onRate={(rating, reason, comment) => (turn.answer ? rate(turn.answer, rating, reason, comment) : Promise.resolve(false))}
              onRetry={() => turn.answer?.question && send(turn.answer.question, turn.answer.key)}
            />
          ))}
          <div style={{ height: 32 }} />
        </div>
      </div>

      <div className="chat-foot">
        <div className="thread">
          {!atBottom && turns.length > 0 && (
            <button className="to-bottom" onClick={() => scrollToBottom()} aria-label="回到底部">
              <ArrowDown size={16} strokeWidth={1.8} />
            </button>
          )}
          <Composer
            ref={composer}
            floating
            busy={busy}
            scope={scope}
            placeholder="继续提问"
            onSend={(text) => send(text)}
            onStop={() => abort.current?.abort()}
            onEditScope={onOpenSettings}
          />
          <p className="disclaimer">内容由 AI 生成，仅供参考，请以平台和官方发布的原文为准</p>
        </div>
      </div>
    </div>
  );
}

function TurnSkeleton() {
  return (
    <div className="turn" aria-hidden="true">
      <div className="msg-user">
        <div className="skeleton" style={{ height: 44, width: "42%", borderRadius: 18 }} />
      </div>
      <div className="msg-bot">
        <div className="skeleton" style={{ height: 26, width: 26, borderRadius: 8 }} />
        <div className="bot-body" style={{ display: "grid", gap: 10 }}>
          <div className="skeleton" style={{ height: 16, width: "92%" }} />
          <div className="skeleton" style={{ height: 16, width: "80%" }} />
          <div className="skeleton" style={{ height: 16, width: "60%" }} />
        </div>
      </div>
    </div>
  );
}
