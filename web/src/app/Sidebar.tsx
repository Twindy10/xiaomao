import { AnimatePresence, motion } from "motion/react";
import { Check, ChevronUp, LogOut, SlidersHorizontal, SquarePen, Trash2 } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import type { Conversation, Me, Profile, Usage } from "../api";
import { PLAN_LABELS } from "../lib/labels";
import { groupConversations } from "../lib/time";
import type { Route } from "./App";
import { scopeOf } from "./scope";
import { Mark } from "./ui";

interface Props {
  me: Me;
  usage: Usage | null;
  profile: Profile | null;
  conversations: Conversation[] | null;
  route: Route;
  onGo: (route: Route) => void;
  onDelete: (id: number) => void;
  onLogout: () => void;
}

export default function Sidebar({ me, usage, profile, conversations, route, onGo, onDelete, onLogout }: Props) {
  const groups = useMemo(() => groupConversations(conversations ?? []), [conversations]);
  const scope = scopeOf(profile);
  const activeId = route.view === "chat" ? route.id : null;

  return (
    <aside className="side" aria-label="导航">
      <div className="brand">
        <Mark size={28} />
        <div>
          <div className="brand-name">小贸同学</div>
          <button className="brand-scope" onClick={() => onGo({ view: "settings" })}>
            {scope ? `检索范围：${scope}` : "未设置检索范围"}
          </button>
        </div>
      </div>

      <nav className="side-actions">
        <button className="nav-item" aria-current={route.view === "home" ? "page" : undefined} onClick={() => onGo({ view: "home" })}>
          <SquarePen size={16} strokeWidth={1.6} />
          新对话
        </button>
        <button
          className="nav-item"
          aria-current={route.view === "settings" ? "page" : undefined}
          onClick={() => onGo({ view: "settings" })}
        >
          <SlidersHorizontal size={16} strokeWidth={1.6} />
          经营画像
        </button>
      </nav>

      <div className="side-scroll">
        {conversations === null ? null : conversations.length === 0 ? (
          <p className="side-empty">暂无对话</p>
        ) : (
          groups.map((group) => (
            <section key={group.label} aria-label={group.label}>
              <h2 className="group-label">{group.label}</h2>
              {group.items.map((item) => (
                <ConversationRow
                  key={item.id}
                  item={item}
                  active={item.id === activeId}
                  onOpen={() => onGo({ view: "chat", id: item.id })}
                  onDelete={() => onDelete(item.id)}
                />
              ))}
            </section>
          ))
        )}
      </div>

      <div className="side-foot">
        {usage && <Quota usage={usage} />}
        <UserMenu me={me} onSettings={() => onGo({ view: "settings" })} onLogout={onLogout} />
      </div>
    </aside>
  );
}

function ConversationRow({
  item,
  active,
  onOpen,
  onDelete,
}: {
  item: Conversation;
  active: boolean;
  onOpen: () => void;
  onDelete: () => void;
}) {
  const [confirm, setConfirm] = useState(false);
  useEffect(() => {
    if (!confirm) return;
    const timer = setTimeout(() => setConfirm(false), 2600);
    return () => clearTimeout(timer);
  }, [confirm]);

  return (
    <div className="conv">
      <button className="nav-item" aria-current={active ? "page" : undefined} onClick={onOpen} title={item.title}>
        <span>{item.title || "新对话"}</span>
      </button>
      <button
        className="conv-delete"
        data-confirm={confirm || undefined}
        onClick={() => (confirm ? onDelete() : setConfirm(true))}
        aria-label={confirm ? `确认删除"${item.title}"` : `删除"${item.title}"`}
      >
        {confirm ? (
          <>
            <Check size={13} /> 删除
          </>
        ) : (
          <Trash2 size={14} strokeWidth={1.6} />
        )}
      </button>
    </div>
  );
}

function Quota({ usage }: { usage: Usage }) {
  const left = Math.max(usage.daily_quota - usage.used, 0);
  const ratio = usage.daily_quota ? Math.min(usage.used / usage.daily_quota, 1) : 0;
  return (
    <div className="quota">
      <div className="quota-line">
        <span>
          今日剩余 <b className="num">{left}</b> 次
        </span>
        <span className="num">
          {usage.used} / {usage.daily_quota}
        </span>
      </div>
      <div className="track" data-low={left <= 3 || undefined} aria-hidden="true">
        <i style={{ width: `${ratio * 100}%` }} />
      </div>
    </div>
  );
}

function UserMenu({ me, onSettings, onLogout }: { me: Me; onSettings: () => void; onLogout: () => void }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const away = (event: MouseEvent) => {
      if (!ref.current?.contains(event.target as Node)) setOpen(false);
    };
    const key = (event: KeyboardEvent) => event.key === "Escape" && setOpen(false);
    window.addEventListener("mousedown", away);
    window.addEventListener("keydown", key);
    return () => {
      window.removeEventListener("mousedown", away);
      window.removeEventListener("keydown", key);
    };
  }, [open]);

  const name = me.nickname || "卖家朋友";
  return (
    <div className="user" ref={ref}>
      <AnimatePresence>
        {open && (
          <motion.div
            className="menu"
            role="menu"
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0, transition: { duration: 0.22, ease: [0.22, 1, 0.36, 1] } }}
            exit={{ opacity: 0, y: 4, transition: { duration: 0.14 } }}
          >
            <button
              role="menuitem"
              onClick={() => {
                setOpen(false);
                onSettings();
              }}
            >
              <SlidersHorizontal size={15} strokeWidth={1.6} />
              经营画像
            </button>
            <hr />
            <button role="menuitem" className="danger" onClick={onLogout}>
              <LogOut size={15} strokeWidth={1.6} />
              退出登录
            </button>
          </motion.div>
        )}
      </AnimatePresence>
      <button className="user-row" aria-expanded={open} aria-haspopup="menu" onClick={() => setOpen((v) => !v)}>
        <span className="avatar">{name.slice(0, 1)}</span>
        <span className="user-meta">
          <b>{name}</b>
          <span>
            {PLAN_LABELS[me.plan] ?? me.plan} · {me.phone}
          </span>
        </span>
        <ChevronUp className="chev" size={16} strokeWidth={1.6} />
      </button>
    </div>
  );
}
