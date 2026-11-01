import { Check, ChevronDown, Copy, RotateCw, ThumbsDown, ThumbsUp } from "lucide-react";
import { useCallback, useMemo, useState } from "react";
import type { Citation, FeedbackReason } from "../api";
import type { UIMessage } from "../lib/chat";
import { hostOf, REASON_LABELS } from "../lib/labels";
import { formatDate } from "../lib/time";
import { useSmoothText } from "../lib/useSmoothText";
import { Mark, Markdown } from "./ui";

const PHASE_TEXT = { understanding: "正在理解问题", retrieving: "正在检索资料", writing: "正在组织回答" };
const COLLAPSED = 3;

interface Props {
  question: string;
  answer: UIMessage | null;
  onRate: (rating: 1 | -1, reason?: FeedbackReason, comment?: string) => Promise<boolean>;
  onRetry: () => void;
}

/** 一问一答：问题是右侧气泡，回答在左侧，出处跟在回答下面 */
export default function Turn({ question, answer, onRate, onRetry }: Props) {
  return (
    <section className="turn">
      <div className="msg-user">
        <p className="bubble">{question}</p>
      </div>
      {answer && <Answer message={answer} onRate={onRate} onRetry={onRetry} />}
    </section>
  );
}

function Answer({ message, onRate, onRetry }: { message: UIMessage; onRate: Props["onRate"]; onRetry: () => void }) {
  const { text, caughtUp } = useSmoothText(message.content, Boolean(message.animate));
  const done = message.status === "done" && caughtUp;
  const waiting = message.status === "streaming" && !text;
  const prefix = `src-${message.key}`;
  const valid = useMemo(() => new Set(message.citations.map((c) => c.n)), [message.citations]);
  const [expanded, setExpanded] = useState(false);

  // 点正文里的编号：出处被折叠时先展开，再滚过去高亮
  const jump = useCallback(
    (n: number) => {
      const index = message.citations.findIndex((c) => c.n === n);
      if (index >= COLLAPSED) setExpanded(true);
      requestAnimationFrame(() => {
        const el = document.getElementById(`${prefix}-${n}`);
        if (!el) return;
        el.scrollIntoView({ behavior: "smooth", block: "nearest" });
        el.classList.remove("flash");
        void el.offsetWidth;
        el.classList.add("flash");
      });
    },
    [message.citations, prefix],
  );

  return (
    <div className="msg-bot">
      <span className="bot-avatar">
        <Mark size={26} />
      </span>
      <div className="bot-body">
        {message.status === "error" ? (
          <div className="error-box">
            {message.error || "生成失败，请重试"}
            <div style={{ marginTop: 8 }}>
              {message.question && (
                <button className="text-action" onClick={onRetry} style={{ marginLeft: -10 }}>
                  <RotateCw size={14} strokeWidth={1.6} /> 重试
                </button>
              )}
              {message.requestId && <span className="num" style={{ fontSize: 12 }}> 请求编号 {message.requestId}</span>}
            </div>
          </div>
        ) : waiting ? (
          <p className="thinking">{PHASE_TEXT[message.phase ?? "understanding"]}…</p>
        ) : message.refused ? (
          <div className="refusal">{text}</div>
        ) : (
          <Markdown text={text} valid={valid} onCite={jump} typing={!done} />
        )}
        {message.stopped && <p className="chat-sub">已停止生成</p>}
        {done && !message.refused && message.citations.length > 0 && (
          <Sources items={message.citations} prefix={prefix} expanded={expanded} onToggle={() => setExpanded((v) => !v)} />
        )}
        {done && !message.stopped && <Tools message={message} onRate={onRate} />}
      </div>
    </div>
  );
}

function Sources({
  items,
  prefix,
  expanded,
  onToggle,
}: {
  items: Citation[];
  prefix: string;
  expanded: boolean;
  onToggle: () => void;
}) {
  const visible = expanded ? items : items.slice(0, COLLAPSED);
  return (
    <section className="sources-block" aria-label="出处">
      <div className="sources-head">
        出处<span className="num"> · {items.length}</span>
      </div>
      <ol className="source-list">
        {visible.map((item) => (
          <li key={item.n}>
            <Source item={item} id={`${prefix}-${item.n}`} />
          </li>
        ))}
      </ol>
      {items.length > COLLAPSED && (
        <button className="text-action more" onClick={onToggle} aria-expanded={expanded}>
          {expanded ? "收起" : `展开其余 ${items.length - COLLAPSED} 条`}
          <ChevronDown size={14} strokeWidth={1.6} className={expanded ? "flip" : undefined} />
        </button>
      )}
    </section>
  );
}

function Source({ item, id }: { item: Citation; id: string }) {
  const segments = (item.title_path || "").split(/\s*>\s*/).filter(Boolean);
  const title = segments.at(-1) || hostOf(item.source_url) || "资料原文";
  const host = hostOf(item.source_url);
  const date = formatDate(item.effective_date);
  // 同一篇文档的几段，末级标题常常一样，补上上一级章节名才分得开
  const meta = [segments.at(-2), host, date && `生效 ${date}`].filter(Boolean).join(" · ");
  const body = (
    <>
      <span className="source-n">{item.n}</span>
      <span className="source-title" title={segments.join(" › ")}>
        {title}
      </span>
      {meta && <span className="source-meta num">{meta}</span>}
    </>
  );
  return item.source_url ? (
    <a id={id} className="source" href={item.source_url} target="_blank" rel="noopener noreferrer">
      {body}
    </a>
  ) : (
    <div id={id} className="source">
      {body}
    </div>
  );
}

function Tools({ message, onRate }: { message: UIMessage; onRate: Props["onRate"] }) {
  const [copied, setCopied] = useState(false);
  const [asking, setAsking] = useState(false);
  const [reason, setReason] = useState<FeedbackReason>("wrong");
  const [comment, setComment] = useState("");
  const [busy, setBusy] = useState(false);
  const disabled = message.id == null || busy;

  async function copy() {
    try {
      await navigator.clipboard.writeText(message.content);
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    } catch {
      /* 浏览器不允许写剪贴板时忽略 */
    }
  }

  async function rate(value: 1 | -1, withReason?: FeedbackReason) {
    setBusy(true);
    const ok = await onRate(value, withReason, withReason ? comment.trim() || undefined : undefined);
    setBusy(false);
    if (ok) setAsking(false);
  }

  return (
    <>
      <div className="answer-tools">
        <button className="text-action" onClick={copy}>
          {copied ? <Check size={14} strokeWidth={1.8} /> : <Copy size={14} strokeWidth={1.6} />}
          {copied ? "已复制" : "复制"}
        </button>
        <button className="text-action" aria-pressed={message.rating === 1} disabled={disabled} onClick={() => rate(1)}>
          <ThumbsUp size={14} strokeWidth={1.6} /> 有帮助
        </button>
        <button
          className="text-action"
          aria-pressed={message.rating === -1 || asking}
          aria-expanded={asking}
          disabled={disabled}
          onClick={() => setAsking((v) => !v)}
        >
          <ThumbsDown size={14} strokeWidth={1.6} /> 反馈问题
        </button>
      </div>
      {asking && (
        <div className="reasons">
          <p>问题类型</p>
          <div className="choice-row" role="group" aria-label="问题类型">
            {(Object.keys(REASON_LABELS) as FeedbackReason[]).map((key) => (
              <button key={key} className="choice" aria-pressed={reason === key} onClick={() => setReason(key)}>
                {REASON_LABELS[key]}
              </button>
            ))}
          </div>
          <label className="sr-only" htmlFor={`comment-${message.key}`}>
            补充说明
          </label>
          <textarea
            id={`comment-${message.key}`}
            rows={2}
            maxLength={500}
            placeholder="补充说明（选填）"
            value={comment}
            onChange={(e) => setComment(e.target.value)}
          />
          <div className="reasons-actions">
            <button className="btn btn-secondary" onClick={() => setAsking(false)}>
              取消
            </button>
            <button className="btn btn-primary" disabled={busy} onClick={() => rate(-1, reason)}>
              提交
            </button>
          </div>
        </div>
      )}
    </>
  );
}
