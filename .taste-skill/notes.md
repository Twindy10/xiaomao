# 小贸同学 v2 — design notes (taste-skill, app-screen run)

Style: editorial-greeting (seed 458233343). Chosen over achromatic-ledger: one seller's own tool,
mostly reading long cited answers; a centred column fits a transcript; the home already greets.
Light only (the style is light); v1 keeps dark mode.

## 2.1 Data and jobs
- Objects: conversation (title, updated_at); turn = question + answer (markdown with [n]),
  citations {n, title_path "a > b > c", source_url, effective_date}; refused flag; cache_hit.
- Usage: used / daily_quota (free 20, pro 200), resets daily. Profile: platforms, sites, categories, stage.
- Magnitudes: a seller asks 1-20 questions a day; an answer runs 300-900 characters with 2-7 sources;
  latency 1.5 s (cache) to 5 s (full).
- Knowledge base (data/processed, fetched 2026-10-29): 129 documents, 1447 passages, 339 reviewed FAQ.
  Amazon policy 75 (general 48, US 15, EU 5, JP 3, AU 2, AE 2); compliance 54 (UK 25, EU 14, DE 10, US 5).
- People: a small cross-border seller, quick look several times a day, desktop at work, phone in between.
- Jobs: ask (most frequent), read the source behind a claim, follow up, rate an answer, set default scope.
- Product's own things: the name 小贸同学 and its mark; the scope (亚马逊 · 德国站); the sources' sites
  (gs.amazon.cn, sellercentral, gov.uk, eur-lex); the refusal when the base has nothing.

## 2.2 Questions
- home: "What can I ask today, and will the knowledge base cover it?"
  Finding: the questions left today, and where the base is thin (a JP compliance question will be refused).
- detail (a conversation): "What do the sources say, and can I trust this answer?"
  Finding per answer: "依据 5 篇资料" / "知识库里没有可靠资料，未作答" / "与之前的相同问题一致（缓存）".
  Evidence beside the answer: each answer's sources sit in a rail at its right, [n] jumps to them.
- settings (经营画像): "What will be searched when I don't name a platform or site?"
  Preview of what the setting produces: an example question and the scope it would be searched in.

## 2.3 Face and signature
- Face: the greeting names the seller and today's one fact, set from data:
  "下午好，Q。今天还能问 17 个问题。" (serif greeting, one fact).
- Signature element: the coverage ledger — the knowledge base as two ranked lists (Amazon policy by site,
  compliance by region) with the thin places shown honestly ("其他地区 0"), so a seller knows before asking.
- The one hue (#1b6e66) on: the coverage bars (main data on home), the citation numbers, the send action.

## Grid
Centred 1100 column: 12 × 77 + 11 × 16 = 924 + 176 = 1100.
Spans: 3/3/3/3 (4 × 263 + 3 × 16 = 1100); 9/3 (821 + 16 + 263 = 1100); 6/6 (542 + 16 + 542 = 1100);
detail: 8/4 (728 + 16 + 356 = 1100).
Sidebar 240 on the page ground, one 1px hairline.

## Critique, last round (2026-11-01, inspect on the mock build at 1440 / 768 / 390)
Gate: home 0 FAIL 1 WARN (app-title-size: the 48px serif greeting is the dealt style's move, kept on purpose);
chat 0 FAIL 0 WARN; settings 0 FAIL 0 WARN; lint 0 FAIL 0 WARN (four floating shadows marked taste-skill-ignore).
1. Question answered by the biggest thing? home: yes, the greeting carries today's questions left; chat: the answer
   and its sources take the first screen; settings: each section's control sits beside its explanation.
2. Shared edges? Fixed rounds: recent-conversations panel left a 71px void, then a ragged row; removed the panel
   (the sidebar already lists conversations), suggestions span 12 in two columns, ledgers 6/6 with equal rows.
3. Colour only on data? The teal is on the coverage bars, the citation numbers, the status dot and the send button.
4. Numbers add up? Coverage totals are summed from the rows with console.assert (75 + 54 = 129).
5. Zeros: "日本及其他地区 0" printed as a zero with an empty bar.
6. Fixed: the first question repeated under the h1; the docked composer shadow was too heavy (now 6% / 8px);
   the settings label column beside the tall shop panel now carries the current values.
7. Phone: sidebar folds to a 48px bar with a full-screen sheet; tiles 2 × 2; sources move under the answer.
Still short: the knowledge-base counts are a static snapshot (coverage.ts) until the API exposes them;
the fonts come from Google Fonts and fall back to system faces where that is blocked.

## Changes after review with the user (2026-11-01)
- Copy: plain product wording only; no self-description ("每句都有出处""不会编"). Backend refusal / chitchat replies rewritten too.
- Ground: pure white was too bright → neutral #f2f2f2 ground, #fafafa panels, white composer and row cards (a warm grey tripped palette-cream).
- Chat: the editorial Q-as-heading + 8/4 sources rail read as a document; changed to a conversation — question in a right-hand
  bubble, answer on the left with the mark, sources (3, then "展开其余") under the answer, no status line or intent pills,
  760px reading column. Inspect: chat 0 FAIL 0 WARN.
