import { useCallback, useEffect, useState } from "react";
import { api, sessionEvents, type Conversation, type Me, type ProfileResponse, type Usage } from "../api";
import Auth from "./Auth";
import Chat from "./Chat";
import Home from "./Home";
import Settings from "./Settings";
import Sidebar from "./Sidebar";
import { ErrorBoundary, Mark, ToastProvider, useToast } from "./ui";
import { Menu, X } from "lucide-react";

/** 三个视图，各有自己的地址：#/ 首页，#/c/3 一次对话，#/settings 经营画像 */
export type Route = { view: "home" } | { view: "chat"; id: number | null; ask?: string } | { view: "settings" };

function parse(hash: string): Route {
  const chat = hash.match(/^#\/c\/(\d+)$/);
  if (chat) return { view: "chat", id: Number(chat[1]) };
  if (hash === "#/settings") return { view: "settings" };
  return { view: "home" };
}

function hashOf(route: Route): string {
  if (route.view === "settings") return "#/settings";
  if (route.view === "chat" && route.id != null) return `#/c/${route.id}`;
  return "#/";
}

export default function App() {
  return (
    <ErrorBoundary>
      <ToastProvider>
        <Root />
      </ToastProvider>
    </ErrorBoundary>
  );
}

function Root() {
  const [phase, setPhase] = useState<"boot" | "auth" | "app">(() => (api.hasSession() ? "boot" : "auth"));
  const [me, setMe] = useState<Me | null>(null);

  const enter = useCallback(async () => {
    try {
      setMe(await api.me());
      setPhase("app");
    } catch {
      setPhase("auth");
    }
  }, []);

  useEffect(() => {
    if (api.hasSession()) void enter();
    const expired = () => setPhase("auth");
    sessionEvents.addEventListener("expired", expired);
    return () => sessionEvents.removeEventListener("expired", expired);
  }, [enter]);

  if (phase === "boot") {
    return (
      <div className="boot">
        <Mark size={32} />
      </div>
    );
  }
  if (phase === "auth" || !me) return <Auth onAuthed={() => void enter()} />;
  return (
    <Workspace
      me={me}
      onLogout={async () => {
        await api.logout();
        setMe(null);
        history.replaceState(null, "", "#/");
        setPhase("auth");
      }}
    />
  );
}

function Workspace({ me, onLogout }: { me: Me; onLogout: () => void }) {
  const toast = useToast();
  const [route, setRoute] = useState<Route>(() => parse(location.hash));
  // 用户切换视图时换一个新组件；对话里新建会话只改地址，不重新挂载（回答还在流式生成）
  const [viewKey, setViewKey] = useState(0);
  const [conversations, setConversations] = useState<Conversation[] | null>(null);
  const [usage, setUsage] = useState<Usage | null>(null);
  const [profile, setProfile] = useState<ProfileResponse | null>(null);
  const [sheet, setSheet] = useState(false);

  const refreshList = useCallback(() => {
    api.listConversations().then(setConversations).catch(() => setConversations((list) => list ?? []));
  }, []);
  const refreshUsage = useCallback(() => {
    api.usage().then(setUsage).catch(() => {});
  }, []);

  useEffect(() => {
    refreshList();
    refreshUsage();
    api.getProfile().then(setProfile).catch(() => {});
  }, [refreshList, refreshUsage]);

  useEffect(() => {
    const onHash = () => {
      setRoute(parse(location.hash));
      setViewKey((k) => k + 1);
    };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const go = useCallback((next: Route) => {
    setSheet(false);
    const hash = hashOf(next);
    if (location.hash !== hash) history.pushState(null, "", hash);
    setRoute(next);
    setViewKey((k) => k + 1);
  }, []);

  const created = useCallback((id: number, title: string) => {
    history.replaceState(null, "", `#/c/${id}`);
    setRoute({ view: "chat", id });
    setConversations((list) => [{ id, title, updated_at: new Date().toISOString() }, ...(list ?? []).filter((c) => c.id !== id)]);
  }, []);

  const answered = useCallback(() => {
    refreshUsage();
    refreshList();
  }, [refreshList, refreshUsage]);

  const remove = useCallback(
    async (id: number) => {
      const previous = conversations;
      setConversations((list) => list?.filter((c) => c.id !== id) ?? null);
      if (route.view === "chat" && route.id === id) go({ view: "home" });
      try {
        await api.deleteConversation(id);
      } catch {
        setConversations(previous);
        toast("删除失败，请稍后再试", "error");
      }
    },
    [conversations, go, route, toast],
  );

  const activeId = route.view === "chat" ? route.id : null;
  const title =
    route.view === "settings"
      ? "经营画像"
      : route.view === "chat"
        ? (conversations?.find((c) => c.id === activeId)?.title ?? route.ask ?? "对话")
        : "小贸同学";

  const sidebar = (
    <Sidebar
      me={me}
      usage={usage}
      profile={profile}
      conversations={conversations}
      route={route}
      onGo={go}
      onDelete={remove}
      onLogout={onLogout}
    />
  );

  return (
    <div className="shell">
      {sidebar}
      <header className="topbar">
        <button className="icon-button" onClick={() => setSheet(true)} aria-label="打开菜单">
          <Menu size={20} strokeWidth={1.6} />
        </button>
        <span className="topbar-title">{title}</span>
      </header>
      {sheet && (
        <div className="sheet" role="dialog" aria-label="菜单">
          <div className="sheet-head">
            <button className="icon-button" onClick={() => setSheet(false)} aria-label="关闭菜单">
              <X size={20} strokeWidth={1.6} />
            </button>
          </div>
          {sidebar}
        </div>
      )}

      <main className="canvas" key={viewKey}>
        {route.view === "home" && (
          <Home
            me={me}
            usage={usage}
            profile={profile}
            conversations={conversations}
            onAsk={(question) => go({ view: "chat", id: null, ask: question })}
            onOpen={(id) => go({ view: "chat", id })}
            onOpenSettings={() => go({ view: "settings" })}
          />
        )}
        {route.view === "chat" && (
          <Chat
            conversationId={route.id}
            initialQuestion={route.ask}
            title={route.id != null ? (conversations?.find((c) => c.id === route.id)?.title ?? null) : null}
            profile={profile}
            onCreated={created}
            onAnswered={answered}
            onOpenSettings={() => go({ view: "settings" })}
          />
        )}
        {route.view === "settings" && (
          <Settings me={me} usage={usage} profile={profile} onSaved={setProfile} onLogout={onLogout} />
        )}
      </main>
    </div>
  );
}
