import { ArrowUp, Crosshair, Square } from "lucide-react";
import { forwardRef, useImperativeHandle, useLayoutEffect, useRef, useState, type KeyboardEvent } from "react";

const MAX = 1000;

export interface ComposerHandle {
  focus: () => void;
}

interface Props {
  busy?: boolean;
  floating?: boolean;
  scope: string | null;
  placeholder?: string;
  onSend: (text: string) => void;
  onStop?: () => void;
  onEditScope: () => void;
}

const Composer = forwardRef<ComposerHandle, Props>(function Composer(
  { busy = false, floating = false, scope, placeholder, onSend, onStop, onEditScope },
  ref,
) {
  const [value, setValue] = useState("");
  const area = useRef<HTMLTextAreaElement>(null);
  useImperativeHandle(ref, () => ({ focus: () => area.current?.focus() }));

  // 随内容长高，最多约 7 行
  useLayoutEffect(() => {
    const el = area.current;
    if (!el) return;
    el.style.height = "0px";
    el.style.height = `${Math.min(el.scrollHeight, 188)}px`;
  }, [value]);

  const text = value.trim();
  const canSend = !busy && text.length > 0 && text.length <= MAX;

  function send() {
    if (!canSend) return;
    onSend(text);
    setValue("");
  }

  function onKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      send();
    }
  }

  return (
    <div className={`composer${floating ? " floating" : ""}`}>
      <label className="sr-only" htmlFor="question">
        问题
      </label>
      <textarea
        id="question"
        ref={area}
        rows={1}
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={onKeyDown}
        placeholder={placeholder ?? "输入问题，例如：德国站卖蓝牙耳机需要哪些认证？"}
      />
      <div className="composer-bar">
        <button type="button" className="scope-button" onClick={onEditScope} title="问题未指明平台和站点时使用">
          <Crosshair size={14} strokeWidth={1.6} />
          <span>{scope ? `检索范围：${scope}` : "设置检索范围"}</span>
        </button>
        {value.length > MAX * 0.8 && (
          <span className="composer-count num" data-over={value.length > MAX || undefined}>
            {value.length} / {MAX}
          </span>
        )}
        <button
          type="button"
          className="send"
          data-stop={busy || undefined}
          onClick={busy ? onStop : send}
          disabled={!busy && !canSend}
          aria-label={busy ? "停止生成" : "发送"}
        >
          {busy ? <Square size={12} fill="currentColor" /> : <ArrowUp size={18} strokeWidth={2} />}
        </button>
      </div>
    </div>
  );
});

export default Composer;
