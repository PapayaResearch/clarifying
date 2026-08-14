"""HTML report written into every episodes results directory."""

import html
from collections import defaultdict
from typing import Dict, List

from uaug.report.common import render_page, tiles_html
from uaug.schema import Episode, JudgeResult, TaskInstance

_CSS = """
.panel { margin-bottom:38px; }
.qa { margin:0 0 14px; display:grid; gap:6px; font-size:14px; max-width:72ch; }
.qa .q { color:var(--ink); } .qa .q::before { content:"Q  "; color:var(--accent); font-family:ui-monospace,Consolas,monospace; }
.qa .a { color:var(--ink2); } .qa .a::before { content:"A  "; color:var(--ink3); font-family:ui-monospace,Consolas,monospace; }
.verdicts { list-style:none; margin:0 0 14px; padding:0; display:grid; gap:8px; }
.verdicts li { display:grid; grid-template-columns:auto auto 1fr; gap:10px; align-items:start; font-size:14px; }
.v-ok { color:var(--accent); font-weight:650; font-family:ui-monospace,Consolas,monospace; font-size:12px; white-space:nowrap; }
.v-miss { color:var(--bad); font-weight:650; font-family:ui-monospace,Consolas,monospace; font-size:12px; white-space:nowrap; }
.final { font-size:13.5px; color:var(--ink2); background:var(--bg); border:1px solid var(--line); border-radius:4px;
  padding:12px 14px; margin-top:8px; max-width:76ch; }
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

    body = f"""{tiles_html(tiles)}
  <div class="panel"><h2>Delivered redacted details by category</h2><div class="bars">{bars_html}</div></div>
  <div class="eyebrow">example episodes &middot; {len(examples)} of {len(episodes)}</div>
  <h2 class="big">Transcripts and verdicts</h2>
  <p class="ssub">Sorted toward episodes with questions and recovery &mdash; the interesting ones.</p>
  {"".join(examples)}"""

    render_page(
        path,
        eyebrow="uaug &middot; stage 2 &middot; episodes",
        tab_title=f"UAUG episodes report — {agent}",
        heading=f"Agent “{html.escape(agent)}” on {report.get('n_episodes', 0)} episodes ({html.escape(str(report.get('split', '')))})",
        sub="Per redacted detail the judge scores <b>elicited</b> (revealed in the user's answers) and <b>delivered</b> (satisfied by the final output). ε̂C is the importance weight never elicited; ε̂E is elicited-but-fumbled.",
        body=body,
        module="uaug.report.episodes",
        extra_css=_CSS,
    )
