import { Eye, EyeOff } from "lucide-react";
import { useState, type FormEvent } from "react";
import { api, ApiError } from "../api";
import { KB_DOCS, KB_FAQ, KB_PASSAGES } from "./coverage";
import { Mark } from "./ui";

export default function Auth({ onAuthed }: { onAuthed: () => void }) {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [phone, setPhone] = useState("");
  const [password, setPassword] = useState("");
  const [nickname, setNickname] = useState("");
  const [show, setShow] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const register = mode === "register";

  async function submit(event: FormEvent) {
    event.preventDefault();
    setError(null);
    setBusy(true);
    try {
      if (register) await api.register(phone.trim(), password, nickname.trim() || undefined);
      await api.login(phone.trim(), password);
      onAuthed();
    } catch (err) {
      if (err instanceof ApiError && err.status === 409) {
        setMode("login");
        setError("该手机号已注册，请直接登录");
      } else setError(err instanceof ApiError ? err.message : "网络异常，请稍后重试");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="auth">
      <section className="auth-about" aria-label="产品介绍">
        <div className="brand" style={{ padding: 0 }}>
          <Mark size={32} />
          <span className="brand-name">小贸同学</span>
        </div>
        <h2 className="greet">跨境电商规则与合规问答</h2>
        <p className="lede">
          收录亚马逊卖家政策，以及美国、英国、德国、欧盟的产品合规法规。
        </p>
        <div className="auth-facts">
          <div>
            <b>{KB_DOCS}</b>
            <span>篇政策和法规</span>
          </div>
          <div>
            <b>{KB_PASSAGES.toLocaleString("en-US")}</b>
            <span>段原文</span>
          </div>
          <div>
            <b>{KB_FAQ}</b>
            <span>条常见问题</span>
          </div>
        </div>
      </section>

      <div className="auth-form-wrap">
        <div className="auth-form">
          <h1>{register ? "注册" : "登录"}</h1>
          <p className="switch">
            {register ? "已有账号？" : "还没有账号？"}
            <button
              type="button"
              onClick={() => {
                setMode(register ? "login" : "register");
                setError(null);
              }}
            >
              {register ? "登录" : "注册"}
            </button>
          </p>
          <form onSubmit={submit}>
            <div className="field">
              <label className="field-label" htmlFor="phone">
                手机号
              </label>
              <input
                id="phone"
                className="input num"
                inputMode="numeric"
                autoComplete="tel"
                maxLength={11}
                value={phone}
                onChange={(e) => setPhone(e.target.value.replace(/\D/g, ""))}
                required
              />
            </div>
            {register && (
              <div className="field">
                <label className="field-label" htmlFor="nickname">
                  昵称<span style={{ color: "var(--muted)", fontWeight: 400 }}>（选填）</span>
                </label>
                <input id="nickname" className="input" maxLength={20} value={nickname} onChange={(e) => setNickname(e.target.value)} />
              </div>
            )}
            <div className="field">
              <label className="field-label" htmlFor="password">
                密码
              </label>
              <div className="password">
                <input
                  id="password"
                  className="input"
                  type={show ? "text" : "password"}
                  autoComplete={register ? "new-password" : "current-password"}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  aria-describedby={register ? "password-rule" : undefined}
                  required
                />
                <button type="button" onClick={() => setShow((v) => !v)} aria-label={show ? "隐藏密码" : "显示密码"}>
                  {show ? <EyeOff size={16} strokeWidth={1.6} /> : <Eye size={16} strokeWidth={1.6} />}
                </button>
              </div>
              {register && (
                <p className="field-hint" id="password-rule">
                  至少 8 位，包含字母和数字
                </p>
              )}
            </div>
            {error && (
              <p className="form-error" role="alert">
                {error}
              </p>
            )}
            <button className="btn btn-primary" disabled={busy || phone.length !== 11 || !password}>
              {busy ? "请稍候…" : register ? "注册并登录" : "登录"}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
