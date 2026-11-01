import { Check, X } from "lucide-react";
import { useEffect, useRef, useState, type KeyboardEvent } from "react";
import { api, ApiError, type Me, type Profile, type ProfileResponse, type Usage } from "../api";
import { PLAN_LABELS, PLATFORM_LABELS, SITE_LABELS, STAGE_LABELS } from "../lib/labels";
import { scopeOf } from "./scope";
import { useToast } from "./ui";

interface Props {
  me: Me;
  usage: Usage | null;
  profile: ProfileResponse | null;
  onSaved: (profile: ProfileResponse) => void;
  onLogout: () => void;
}

const EMPTY: Profile = { platforms: [], sites: [], categories: [], stage: null };

/** 每个控件改了就保存（一个视图一种保存方式），标题旁显示保存时间 */
export default function Settings({ me, usage, profile, onSaved, onLogout }: Props) {
  const toast = useToast();
  const [draft, setDraft] = useState<Profile>(() => (profile ? pick(profile) : EMPTY));
  const [savedAt, setSavedAt] = useState<string | null>(null);
  const [category, setCategory] = useState("");
  const loaded = useRef(profile != null);

  useEffect(() => {
    if (profile && !loaded.current) {
      loaded.current = true;
      setDraft(pick(profile));
    }
  }, [profile]);

  async function save(next: Profile) {
    setDraft(next);
    try {
      const saved = await api.saveProfile(next);
      onSaved(saved);
      setSavedAt(new Date().toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" }));
    } catch (error) {
      toast(error instanceof ApiError ? error.message : "保存失败，请稍后重试", "error");
    }
  }

  const toggle = (field: "platforms" | "sites", key: string) =>
    save({ ...draft, [field]: draft[field].includes(key) ? draft[field].filter((k) => k !== key) : [...draft[field], key] });

  function addCategory() {
    const value = category.trim().replace(/[,，]$/, "").slice(0, 20);
    setCategory("");
    if (value && !draft.categories.includes(value) && draft.categories.length < 10) {
      void save({ ...draft, categories: [...draft.categories, value] });
    }
  }

  function onCategoryKey(event: KeyboardEvent<HTMLInputElement>) {
    if ((event.key === "Enter" || event.key === "," || event.key === "，") && !event.nativeEvent.isComposing) {
      event.preventDefault();
      addCategory();
    }
  }

  const platformKeys = Object.keys(profile?.options.platforms ?? PLATFORM_LABELS);
  const siteKeys = Object.keys(profile?.options.sites ?? SITE_LABELS);
  const scope = scopeOf(draft);
  const used = usage?.used ?? 0;
  const quota = usage?.daily_quota ?? me.daily_quota;

  return (
    <div className="settings column">
      <h1>经营画像</h1>
      <p className="settings-sub">
        {savedAt ? (
          <span className="saved">
            <Check size={14} strokeWidth={2} /> 已保存 {savedAt}
          </span>
        ) : (
          "修改后自动保存。"
        )}
      </p>

      <section className="setting" aria-labelledby="s-platform">
        <div className="setting-label">
          <h2 id="s-platform">主营平台</h2>
          <p>仅选择一个平台时，未指明平台的问题按该平台检索。</p>
        </div>
        <div className="setting-panel">
          <div className="toggle-list" role="group" aria-labelledby="s-platform">
            {platformKeys.map((key) => (
              <button key={key} className="choice" aria-pressed={draft.platforms.includes(key)} onClick={() => toggle("platforms", key)}>
                {PLATFORM_LABELS[key] ?? key}
              </button>
            ))}
          </div>
          <p className="field-hint">平台政策目前仅收录亚马逊。</p>
        </div>
      </section>

      <section className="setting" aria-labelledby="s-site">
        <div className="setting-label">
          <h2 id="s-site">主营站点</h2>
          <p>仅选择一个站点时，未指明站点的问题按该站点检索。合规法规按站点所在国家或地区匹配。</p>
        </div>
        <div className="setting-panel">
          <div className="toggle-list" role="group" aria-labelledby="s-site">
            {siteKeys.map((key) => (
              <button key={key} className="choice" aria-pressed={draft.sites.includes(key)} onClick={() => toggle("sites", key)}>
                {SITE_LABELS[key] ?? key}
              </button>
            ))}
          </div>
          <div className="preview" aria-live="polite">
            <div className="preview-q">示例：卖蓝牙耳机需要哪些认证？</div>
            <div className="preview-a">
              {scope ? `检索范围：${scope}` : "检索范围：全部平台和站点"}
            </div>
          </div>
        </div>
      </section>

      <section className="setting" aria-labelledby="s-shop">
        <div className="setting-label">
          <h2 id="s-shop">店铺情况</h2>
          <p>用于理解问题，不影响检索范围。</p>
          <p className="setting-summary">
            当前：{draft.categories.length ? draft.categories.join("、") : "未填写品类"}
            {" · "}
            {draft.stage ? STAGE_LABELS[draft.stage].title : "未选择阶段"}
          </p>
        </div>
        <div className="setting-panel">
          <div className="field">
            <label className="field-label" htmlFor="category">
              主营品类
            </label>
            <input
              id="category"
              className="input"
              value={category}
              maxLength={20}
              onChange={(e) => setCategory(e.target.value)}
              onKeyDown={onCategoryKey}
              onBlur={addCategory}
              placeholder="例如 3C 数码，按回车添加"
              aria-describedby="category-hint"
            />
            <p className="field-hint" id="category-hint">
              最多 10 个，每个不超过 20 个字
            </p>
            {draft.categories.length > 0 && (
              <div className="tags">
                {draft.categories.map((item) => (
                  <span key={item} className="tag">
                    {item}
                    <button
                      onClick={() => save({ ...draft, categories: draft.categories.filter((c) => c !== item) })}
                      aria-label={`删除${item}`}
                    >
                      <X size={12} strokeWidth={2} />
                    </button>
                  </span>
                ))}
              </div>
            )}
          </div>
          <div className="field">
            <span className="field-label" id="stage">
              经营阶段
            </span>
            <div className="toggle-list" role="group" aria-labelledby="stage">
              {(Object.keys(STAGE_LABELS) as ("newbie" | "growing")[]).map((key) => (
                <button
                  key={key}
                  className="choice"
                  aria-pressed={draft.stage === key}
                  onClick={() => save({ ...draft, stage: draft.stage === key ? null : key })}
                  title={STAGE_LABELS[key].hint}
                >
                  {STAGE_LABELS[key].title}
                </button>
              ))}
            </div>
          </div>
        </div>
      </section>

      <section className="setting" aria-labelledby="s-usage">
        <div className="setting-label">
          <h2 id="s-usage">今日用量</h2>
          <p>每日 0 点重置，生成失败的提问不计入。</p>
        </div>
        <div className="setting-panel">
          <div className="usage-figure num">
            {used}
            <small>/ {quota} 次</small>
          </div>
          <div className="track" aria-hidden="true">
            <i style={{ width: `${quota ? Math.min(used / quota, 1) * 100 : 0}%` }} />
          </div>
          <div className="track-ends">
            <span>0</span>
            <span>{quota}</span>
          </div>
        </div>
      </section>

      <section className="setting" aria-labelledby="s-account">
        <div className="setting-label">
          <h2 id="s-account">账号</h2>
        </div>
        <div className="setting-panel">
          <div className="kv">
            <span>手机号</span>
            <b className="num">{me.phone}</b>
          </div>
          <div className="kv">
            <span>套餐</span>
            <b>
              {PLAN_LABELS[me.plan] ?? me.plan}，每日 {me.daily_quota} 次
            </b>
          </div>
          <div style={{ marginTop: 16 }}>
            <button className="btn btn-secondary" onClick={onLogout}>
              退出登录
            </button>
          </div>
        </div>
      </section>
    </div>
  );
}

function pick(profile: Profile): Profile {
  return { platforms: profile.platforms, sites: profile.sites, categories: profile.categories, stage: profile.stage };
}
