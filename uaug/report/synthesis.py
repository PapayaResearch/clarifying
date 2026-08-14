"""HTML report written into every synthesis results directory."""

import html
import statistics
from collections import Counter
from typing import List

from uaug.report.common import render_page, tiles_html
from uaug.schema import TaskInstance

_CSS = """
.band { display:grid; grid-template-columns:1fr 1.2fr; gap:12px; margin-bottom:38px; }
@media (max-width:720px) { .band { grid-template-columns:1fr; } }
.yield { display:flex; height:22px; border-radius:4px; overflow:hidden; gap:2px; margin:4px 0 10px; }
.legend { display:flex; gap:18px; font-size:13px; color:var(--ink2); flex-wrap:wrap; }
.legend .sw { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:6px; }
.kv { display:grid; grid-template-columns:auto 1fr; gap:4px 14px; font-size:13.5px; margin-top:16px; }
.kv dt { color:var(--ink2); } .kv dd { margin:0; font-variant-numeric:tabular-nums; }
.red { list-style:none; margin:0 0 4px; padding:0; display:grid; gap:10px; }
.red li { display:grid; grid-template-columns:auto 1fr; gap:10px; align-items:start; }
.red .t { font-size:14.5px; }
.red .check { display:block; font-size:12.5px; color:var(--ink3); margin-top:1px; }
.spec { font-size:13.5px; color:var(--ink2); background:var(--bg); border:1px solid var(--line); border-radius:4px;
  padding:12px 14px; margin-top:8px; max-width:76ch; }
.kept { font-size:13px; color:var(--ink2); margin-top:8px; }
.kept .cat { color:var(--ink3); font-family:ui-monospace,Consolas,monospace; font-size:11.5px; }
"""


def _example_card(instance: TaskInstance) -> str:
    redacted_rows = "".join(
        f'<li><span class="mark">{html.escape(d.category)} &middot; {d.importance:g}</span>'
        f'<div class="t">{html.escape(d.text)}'
        f'<span class="check"><b>judge asks:</b> {html.escape(d.check)}</span></div></li>'
        for d in instance.redacted
    )
    kept_rows = "".join(
        f'<li><span class="cat">[{html.escape(d.category)}]</span> {html.escape(d.text)}</li>'
        for d in instance.kept[:6]
    )
    return f"""
<div class="card">
  <div class="ctop"><span class="chip">{html.escape(instance.split)}</span>
    <span class="id">{html.escape(instance.instance_id)}</span>
    <span class="chip n">{len(instance.details)} details</span></div>
  <blockquote class="prompt"><span class="lbl">user request (eroded)</span>{html.escape(instance.eroded_prompt)}</blockquote>
  <ul class="red">{redacted_rows}</ul>
  <details><summary>kept details (sample) + full spec</summary>
    <ul class="kept">{kept_rows}</ul>
    <div class="spec">{html.escape(instance.full_spec)}</div>
  </details>
</div>"""


def write_synthesis_report(path: str, instances: List[TaskInstance], report: dict, n_examples: int = 4):
    """Renders instances + the synthesis report dict into a standalone HTML file."""
    n = len(instances)
    n_sources = report.get("n_sources", 0)
    dropped = n_sources - n

    if instances:
        detail_counts = [len(i.details) for i in instances]
        prompt_words = [len(i.eroded_prompt.split()) for i in instances]
        spec_words = [len(i.full_spec.split()) for i in instances]
        importances = [d.importance for i in instances for d in i.redacted]
        tiles = [
            (f"{n} <small>/ {n_sources}</small>", f"instances from sources ({n / max(n_sources, 1):.0%} yield)"),
            (f"{statistics.mean(detail_counts):.1f}", f"details per instance ({min(detail_counts)}&ndash;{max(detail_counts)})"),
            (f"{statistics.mean(prompt_words):.0f} <small>&larr; {statistics.mean(spec_words):.0f} words</small>", "eroded prompt vs full spec"),
            (f"{statistics.mean(importances):.2f}", "mean importance of redacted details"),
        ]
    else:
        tiles = [(f"0 <small>/ {n_sources}</small>", "instances from sources")]

    categories = Counter(d.category for i in instances for d in i.redacted)
    max_count = max(categories.values(), default=1)
    bars_html = "".join(
        f'<span class="name">{html.escape(cat)}</span>'
        f'<div><div class="fill" style="width:{100 * count / max_count:.1f}%"></div></div>'
        f'<span class="val">{count}</span>'
        for cat, count in categories.most_common()
    )

    splits = ", ".join(f"{s}: {c}" for s, c in report.get("per_split", {}).items())
    outcomes = ", ".join(f"{k}: {v}" for k, v in report.get("outcomes", {}).items())

    # Examples spread across the instance list for variety
    step = max(len(instances) // max(n_examples, 1), 1)
    examples = instances[::step][:n_examples]
    examples_html = "".join(_example_card(i) for i in examples)

    body = f"""{tiles_html(tiles)}
  <div class="band">
    <div class="panel">
      <h2>Yield &amp; splits</h2>
      <div class="yield"><div style="background:var(--accent);flex:{max(n, 1)}"></div><div style="background:var(--drop);flex:{max(dropped, 1)}"></div></div>
      <div class="legend"><span><span class="sw" style="background:var(--accent)"></span>accepted &middot; {n}</span>
        <span><span class="sw" style="background:var(--drop)"></span>dropped &middot; {dropped}</span></div>
      <dl class="kv"><dt>splits</dt><dd>{html.escape(splits)}</dd><dt>outcomes</dt><dd>{html.escape(outcomes)}</dd></dl>
    </div>
    <div class="panel">
      <h2>Redacted details by category &middot; {sum(categories.values())} total</h2>
      <div class="bars">{bars_html}</div>
    </div>
  </div>
  <div class="eyebrow">examples &middot; {len(examples)} of {n}</div>
  <h2 class="big">What the agent sees vs. what it must recover</h2>
  <p class="ssub">The agent gets only the <b>user request</b>. Redacted details are what the judge scores recovery on &mdash; each with the yes/no question the judge asks.</p>
  {examples_html}"""

    render_page(
        path,
        eyebrow="uaug &middot; stage 1 &middot; data synthesis",
        tab_title="UAUG synthesis report",
        heading=f"Synthesis report: {n} instances from {n_sources} sources",
        sub="Pipeline: decompose &rarr; redact &rarr; erode.",
        body=body,
        module="uaug.report.synthesis",
        extra_css=_CSS,
    )
