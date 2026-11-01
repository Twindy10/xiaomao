import { ArrowRight } from "lucide-react";
import type { Conversation, Me, Profile, Usage } from "../api";
import { PLAN_LABELS, SUGGESTIONS } from "../lib/labels";
import { greeting, parseTime } from "../lib/time";
import Composer from "./Composer";
import {
  AMAZON_POLICY,
  AMAZON_TOTAL,
  COMPLIANCE,
  COMPLIANCE_TOTAL,
  KB_DOCS,
  KB_PASSAGES,
  KB_UPDATED,
  type CoverageRow,
} from "./coverage";
import { scopeOf } from "./scope";

interface Props {
  me: Me;
  usage: Usage | null;
  profile: Profile | null;
  conversations: Conversation[] | null;
  onAsk: (question: string) => void;
  onOpen: (id: number) => void;
  onOpenSettings: () => void;
}

export function ago(value: string): string {
  const minutes = Math.round((Date.now() - parseTime(value).getTime()) / 60_000);
  if (minutes < 1) return "刚刚";
  if (minutes < 60) return `${minutes} 分钟前`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours} 小时前`;
  const days = Math.round(hours / 24);
  return days === 1 ? "昨天" : `${days} 天前`;
}

const monthDay = (iso: string) => {
  const [, m, d] = iso.split("-").map(Number);
  return `${m} 月 ${d} 日`;
};

export default function Home({ me, usage, profile, conversations, onAsk, onOpen, onOpenSettings }: Props) {
  const scope = scopeOf(profile);
  const left = usage ? Math.max(usage.daily_quota - usage.used, 0) : null;
  const latest = conversations?.[0];
  const name = me.nickname || "卖家朋友";

  return (
    <div className="home column">
      <h1 className="greet">
        {greeting()}，{name}。
        {left != null && <span className="greet-fact">{left > 0 ? `今日还可提问 ${left} 次。` : "今日提问次数已用完。"}</span>}
      </h1>

      <div className="home-composer">
        <Composer scope={scope} onSend={onAsk} onEditScope={onOpenSettings} />
      </div>

      <div className="grid tiles">
        <div className="tile span-3">
          <span className="tile-label">今日剩余</span>
          <span className="tile-value">
            {left ?? "—"}
            <small>次</small>
          </span>
          <span className="tile-caption">
            {usage ? `${PLAN_LABELS[usage.plan] ?? usage.plan}每日 ${usage.daily_quota} 次，0 点重置` : "加载中"}
          </span>
        </div>
        <button className="tile span-3" onClick={() => latest && onOpen(latest.id)} disabled={!latest}>
          <span className="tile-label">对话</span>
          <span className="tile-value">
            {conversations ? conversations.length : "—"}
            <small>个</small>
          </span>
          <span className="tile-caption">{latest ? `最近：${ago(latest.updated_at)}` : "暂无对话"}</span>
        </button>
        <button className="tile span-3" onClick={onOpenSettings}>
          <span className="tile-label">默认检索范围</span>
          <span className="tile-value text">{scope ?? "未设置"}</span>
          <span className="tile-caption">{scope ? "问题未指明平台和站点时使用" : "在经营画像中设置"}</span>
        </button>
        <div className="tile span-3">
          <span className="tile-label">知识库收录</span>
          <span className="tile-value">
            {KB_DOCS}
            <small>篇资料</small>
          </span>
          <span className="tile-caption num">
            {KB_PASSAGES.toLocaleString("en-US")} 段原文 · 更新于 {monthDay(KB_UPDATED)}
          </span>
        </div>
      </div>

      <div className="grid rows">
        <section className="panel span-12" aria-labelledby="try">
          <div className="panel-head">
            <h2 id="try">推荐问题</h2>
            <span className="count">{SUGGESTIONS.length}</span>
          </div>
          <ul className="row-cards two">
            {SUGGESTIONS.map((item) => (
              <li key={item.text}>
                <button className="row-card" onClick={() => onAsk(item.text)}>
                  <span className="body">
                    <span className="title">{item.text}</span>
                    <span className="sub">{item.tag}</span>
                  </span>
                  <ArrowRight className="go" size={16} strokeWidth={1.6} />
                </button>
              </li>
            ))}
          </ul>
        </section>
      </div>

      <div className="grid rows">
        <Ledger
          id="kb-amazon"
          title="亚马逊平台政策"
          total={AMAZON_TOTAL}
          rows={AMAZON_POLICY}
          note="按适用站点统计，通用政策适用于所有站点。"
        />
        <Ledger
          id="kb-compliance"
          title="产品合规法规"
          total={COMPLIANCE_TOTAL}
          rows={COMPLIANCE}
          note="其他国家和地区暂未收录。"
        />
      </div>
    </div>
  );
}

function Ledger({ id, title, total, rows, note }: { id: string; title: string; total: number; rows: CoverageRow[]; note: string }) {
  const max = Math.max(...rows.map((row) => row.docs));
  return (
    <section className="panel span-6" aria-labelledby={id}>
      <div className="panel-head">
        <h2 id={id}>{title}</h2>
        <span className="count">{total} 篇</span>
      </div>
      <ul className="ledger">
        {rows.map((row) => (
          <li key={row.label}>
            <span className="ledger-label">{row.label}</span>
            <span className={`ledger-docs${row.docs === 0 ? " zero" : ""}`}>{row.docs}</span>
            {/* 0 篇也留着条的位置（宽度 0），两个账本的行高才一致 */}
            <i className="ledger-bar" style={{ width: `${(row.docs / max) * 100}%` }} aria-hidden="true" />
          </li>
        ))}
      </ul>
      <p className="ledger-note">{note}</p>
    </section>
  );
}
