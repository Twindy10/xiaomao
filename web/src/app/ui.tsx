import { Component, createContext, memo, useCallback, useContext, useRef, useState, type ReactNode } from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";

/** 产品标志：强调色圆角方块里一道"跨过去"的弧线和一个点，和 favicon 同形 */
export function Mark({ size = 28 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden="true">
      <rect width="32" height="32" rx="9" fill="var(--accent)" />
      <path d="M9 21.5c2.2-6.6 4.3-10 7-10s4.8 3.4 7 10" fill="none" stroke="#fff" strokeWidth="2.4" strokeLinecap="round" />
      <circle cx="16" cy="11.5" r="2.2" fill="#fff" />
    </svg>
  );
}

/** 某个组件渲染出错时不至于整页空白 */
export class ErrorBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false };

  static getDerivedStateFromError() {
    return { failed: true };
  }

  render() {
    if (!this.state.failed) return this.props.children;
    return (
      <div className="boot" role="alert">
        <div className="crashed">
          <Mark size={32} />
          <h1>页面出错了</h1>
          <p>刷新后可继续使用，对话记录不受影响。</p>
          <button className="btn btn-primary" onClick={() => location.reload()}>
            刷新页面
          </button>
        </div>
      </div>
    );
  }
}

type ToastKind = "info" | "error";
const ToastContext = createContext<(text: string, kind?: ToastKind) => void>(() => {});
export const useToast = () => useContext(ToastContext);

export function ToastProvider({ children }: { children: ReactNode }) {
  const [items, setItems] = useState<{ id: number; text: string; kind: ToastKind }[]>([]);
  const seq = useRef(0);
  const show = useCallback((text: string, kind: ToastKind = "info") => {
    const id = ++seq.current;
    setItems((list) => [...list.slice(-2), { id, text, kind }]);
    setTimeout(() => setItems((list) => list.filter((item) => item.id !== id)), 2800);
  }, []);
  return (
    <ToastContext.Provider value={show}>
      {children}
      <div className="toasts" role="status" aria-live="polite">
        {items.map((item) => (
          <div key={item.id} className="toast" data-kind={item.kind}>
            {item.text}
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
}

// 正文里的 [n] 转成指向 #cite-n 的链接，再渲染成引用编号。不开 rehype-raw：模型输出的 HTML 按文本显示
const CITE_RE = /\[(\d{1,2})\](?!\()/g;

export const Markdown = memo(function Markdown({
  text,
  valid,
  onCite,
  typing,
}: {
  text: string;
  valid: Set<number>;
  onCite: (n: number) => void;
  typing?: boolean;
}) {
  const components: Components = {
    a({ href, children }) {
      const match = href?.match(/^#cite-(\d+)$/);
      if (match) {
        const n = Number(match[1]);
        const ok = valid.has(n);
        return (
          <button type="button" className="cite" disabled={!ok} onClick={() => onCite(n)} aria-label={`出处 ${n}`}>
            {n}
          </button>
        );
      }
      return (
        <a href={href} target="_blank" rel="noopener noreferrer">
          {children}
        </a>
      );
    },
  };
  return (
    <div className={`prose${typing ? " typing" : ""}`}>
      <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
        {text.replace(CITE_RE, "[$1](#cite-$1)")}
      </ReactMarkdown>
    </div>
  );
});
