"""Self-contained HTML report written into every episodes results directory."""

import html
from collections import defaultdict
from typing import Dict, List

from uaug.schema import Episode, JudgeResult, TaskInstance

_CSS = """
:root { --bg:#FAFAF8; --surface:#FFF; --ink:#20241F; --ink2:#5A615A; --ink3:#8A918A;
  --line:#E3E6E0; --accent:#0F766E; --soft:#E4F0EE; --bad:#A8552F; --redact:#23261F; --redink:#F5F4EF; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#171B19; --surface:#1F2422; --ink:#E7EBE7; --ink2:#A6ADA6; --ink3:#757C75;
    --line:#333A35; --accent:#53BCAE; --soft:#24352F; --bad:#D98B63; --redact:#0C0E0C; --redink:#D6DAD3; } }
* { box-sizing:border-box; }
body { background:var(--bg); color:var(--ink); font-family:system-ui,"Segoe UI",sans-serif;
  line-height:1.55; margin:0; padding:40px 20px 80px; }
.wrap { max-width:980px; margin:0 auto; }
.mono { font-family:ui-monospace,Consolas,monospace; }
.eyebrow { font-family:ui-monospace,Consolas,monospace; font-size:12px; letter-spacing:.14em;
  text-transform:uppercase; color:var(--accent); }
h1 { font-size:28px; margin:6px 0 4px; }
.sub { color:var(--ink2); margin:0 0 30px; max-width:70ch; }
.tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; margin-bottom:14px; }
.tile { background:var(--surface); border:1px solid var(--line); border-radius:6px; padding:14px 16px 12px; }
.tile .v { font-size:24px; font-weight:650; font-variant-numeric:tabular-nums; }
.tile .k { font-size:12.5px; color:var(--ink2); margin-top:2px; }
.panel { background:var(--surface); border:1px solid var(--line); border-radius:6px; padding:16px 18px; margin-bottom:38px; }
.panel h2 { font-size:13px; margin:0 0 14px; color:var(--ink2); font-weight:600; letter-spacing:.04em; text-transform:uppercase; }
.bars { display:grid; grid-template-columns:max-content 1fr max-content; gap:7px 10px; align-items:center; font-size:13.5px; }
.bars .name { color:var(--ink2); font-family:ui-monospace,Consolas,monospace; font-size:12.5px; }
.bars .track { background:var(--line); border-radius:0 4px 4px 0; }
.bars .fill { height:14px; background:var(--accent); border-radius:0 4px 4px 0; min-width:2px; }
.bars .val { font-variant-numeric:tabular-nums; }
h2.big { font-size:19px; margin:0 0 6px; }
.ssub { color:var(--ink2); font-size:14px; margin:0 0 18px; max-width:72ch; }
.card { background:var(--surface); border:1px solid var(--line); border-radius:6px; padding:20px 22px; margin-bottom:18px; }
.ctop { display:flex; align-items:baseline; gap:10px; flex-wrap:wrap; margin-bottom:10px; }
.ctop .id { font-family:ui-monospace,Consolas,monospace; font-size:13px; color:var(--ink2); }
.chip { font-size:11px; font-family:ui-monospace,Consolas,monospace; letter-spacing:.06em; text-transform:uppercase;
  padding:2px 8px; border-radius:10px; background:var(--soft); color:var(--accent); }
.prompt { border-left:3px solid var(--accent); padding:2px 0 2px 14px; margin:0 0 14px; font-size:14.5px; max-width:72ch; }
.prompt .lbl, .lbl { display:block; font-size:11px; letter-spacing:.1em; text-transform:uppercase; color:var(--accent);
  font-family:ui-monospace,Consolas,monospace; margin-bottom:4px; }
.qa { margin:0 0 14px; display:grid; gap:6px; font-size:14px; max-width:72ch; }
.qa .q { color:var(--ink); } .qa .q::before { content:"Q  "; color:var(--accent); font-family:ui-monospace,Consolas,monospace; }
.qa .a { color:var(--ink2); } .qa .a::before { content:"A  "; color:var(--ink3); font-family:ui-monospace,Consolas,monospace; }
.verdicts { list-style:none; margin:0 0 14px; padding:0; display:grid; gap:8px; }
.verdicts li { display:grid; grid-template-columns:auto auto 1fr; gap:10px; align-items:start; font-size:14px; }
.mark { background:var(--redact); color:var(--redink); font-family:ui-monospace,Consolas,monospace; font-size:11px;
  padding:3px 7px; border-radius:3px; white-space:nowrap; margin-top:1px; }
.v-ok { color:var(--accent); font-weight:650; font-family:ui-monospace,Consolas,monospace; font-size:12px; white-space:nowrap; }
.v-miss { color:var(--bad); font-weight:650; font-family:ui-monospace,Consolas,monospace; font-size:12px; white-space:nowrap; }
details { margin-top:4px; }
summary { cursor:pointer; font-size:13px; color:var(--ink2); font-family:ui-monospace,Consolas,monospace; }
summary:hover { color:var(--accent); }
.final { font-size:13.5px; color:var(--ink2); background:var(--bg); border:1px solid var(--line); border-radius:4px;
  padding:12px 14px; margin-top:8px; max-width:76ch; }
.foot { color:var(--ink3); font-size:12.5px; border-top:1px solid var(--line); margin-top:40px; padding-top:14px; }
"""


def _verdict(flag: bool, label: str) -> str:
    css = "v-ok" if flag else "v-miss"
    symbol = "✓" if flag else "✗"
    return f'<span class="{css}">{symbol} {label}</span>'


def _example_card(instance: TaskInstance, episode: Episode, judgment: JudgeResult) -> str:
    qa_rows = []
    for turn in episode.turns:
        if turn.role == "agent_question":
            qa_rows.append(f'<div class="q">{html.escape(turn.text)}</div>')
        elif turn.role == "user_answer":
            qa_rows.append(f'<div class="a">{html.escape(turn.text)}</div>')
    qa_html = f'<div class="qa"><span class="lbl">clarification dialogue</span>{"".join(qa_rows)}</div>' if qa_rows else ""

    verdict_rows = []
    for detail in instance.redacted:
        verdict = judgment.per_detail.get(detail.id, {})
        verdict_rows.append(
            f'<li><span class="mark">{html.escape(detail.category)}</span>'
            f'<span>{_verdict(bool(verdict.get("elicited")), "elicited")}<br>{_verdict(bool(verdict.get("delivered")), "delivered")}</span>'
            f'<div>{html.escape(detail.text)}</div></li>'
        )

    return f"""
<div class="card">
  <div class="ctop"><span class="chip">u_rec {judgment.u_rec:.2f}</span>
    <span class="chip">{episode.n_questions} questions</span>
    <span class="id">{html.escape(instance.instance_id)}</span></div>
  <blockquote class="prompt"><span class="lbl">user request</span>{html.escape(instance.eroded_prompt)}</blockquote>
  {qa_html}
  <ul class="verdicts">{"".join(verdict_rows)}</ul>
  <details><summary>final output</summary><div class="final">{html.escape(episode.final_output)}</div></details>
</div>"""


def write_episodes_report(
    path: str,
    instances: List[TaskInstance],
    episodes: List[Episode],
    judgments: List[JudgeResult],
    report: dict,
    n_examples: int = 4,
):
    """Renders one agent's episode run into a standalone HTML file."""
    by_id: Dict[str, TaskInstance] = {i.instance_id: i for i in instances}
    judgment_key = {(j.instance_id, j.seed): j for j in judgments}

    agent = report.get("agent", "?")
    tiles = [
        (f"{report.get('u_rec_mean', 0) or 0:.3f}", "utility recovered (importance-weighted)"),
        (f"{report.get('eps_c_hat_mean', 0) or 0:.3f}", "ε̂C — weight never elicited"),
        (f"{report.get('eps_e_hat_mean', 0) or 0:.3f}", "ε̂E — elicited but not delivered"),
        (f"{report.get('preserved_mean', 0) or 0:.2f}", "kept details preserved in output"),
        (f"{report.get('n_questions_mean', 0) or 0:g}", "questions used per episode"),
    ]
    tiles_html = "".join(f'<div class="tile"><div class="v">{v}</div><div class="k">{k}</div></div>' for v, k in tiles)

    # Recovery rate by redacted-detail category
    delivered = defaultdict(int)
    totals = defaultdict(int)
    for judgment in judgments:
        instance = by_id.get(judgment.instance_id)
        if instance is None:
            continue
        categories = {d.id: d.category for d in instance.redacted}
        for detail_id, verdict in judgment.per_detail.items():
            category = categories.get(detail_id)
            if category:
                totals[category] += 1
                delivered[category] += bool(verdict.get("delivered"))
    bars_html = "".join(
        f'<span class="name">{html.escape(cat)}</span>'
        f'<div class="track"><div class="fill" style="width:{100 * delivered[cat] / totals[cat]:.1f}%"></div></div>'
        f'<span class="val">{delivered[cat]}/{totals[cat]}</span>'
        for cat, _ in sorted(totals.items(), key=lambda kv: -kv[1])
    )

    # Examples: most interesting episodes first (questions asked, mixed outcomes)
    scored = sorted(episodes, key=lambda e: -(e.n_questions + judgment_key[(e.instance_id, e.seed)].u_rec))
    examples = []
    for episode in scored[:n_examples]:
        instance = by_id.get(episode.instance_id)
        judgment = judgment_key.get((episode.instance_id, episode.seed))
        if instance is not None and judgment is not None:
            examples.append(_example_card(instance, episode, judgment))

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>UAUG episodes report — {html.escape(agent)}</title><style>{_CSS}</style></head><body>
<div class="wrap">
  <div class="eyebrow">uaug &middot; stage 2 &middot; episodes</div>
  <h1>Agent “{html.escape(agent)}” on {report.get("n_episodes", 0)} episodes ({html.escape(str(report.get("split", "")))})</h1>
  <p class="sub">Per redacted detail the judge scores <b>elicited</b> (revealed in the user's answers) and <b>delivered</b> (satisfied by the final output). ε̂C is the importance weight never elicited; ε̂E is elicited-but-fumbled.</p>
  <div class="tiles">{tiles_html}</div>
  <div class="panel"><h2>Delivered redacted details by category</h2><div class="bars">{bars_html}</div></div>
  <div class="eyebrow">example episodes &middot; {len(examples)} of {len(episodes)}</div>
  <h2 class="big">Transcripts and verdicts</h2>
  <p class="ssub">Sorted toward episodes with questions and recovery — the interesting ones.</p>
  {"".join(examples)}
  <div class="foot">Generated by <span class="mono">uaug.episodes.report</span></div>
</div></body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)
