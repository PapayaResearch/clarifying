"""Self-contained HTML report written into every synthesis results directory."""

import html
import statistics
from collections import Counter
from typing import List

from uaug.schema import TaskInstance

_CSS = """
:root { --bg:#FAFAF8; --surface:#FFF; --ink:#20241F; --ink2:#5A615A; --ink3:#8A918A;
  --line:#E3E6E0; --accent:#0F766E; --soft:#E4F0EE; --drop:#B9BFB8; --redact:#23261F; --redink:#F5F4EF; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#171B19; --surface:#1F2422; --ink:#E7EBE7; --ink2:#A6ADA6; --ink3:#757C75;
    --line:#333A35; --accent:#53BCAE; --soft:#24352F; --redact:#0C0E0C; --redink:#D6DAD3; } }
* { box-sizing:border-box; }
body { background:var(--bg); color:var(--ink); font-family:system-ui,"Segoe UI",sans-serif;
  line-height:1.55; margin:0; padding:40px 20px 80px; }
.wrap { max-width:980px; margin:0 auto; }
.mono { font-family:ui-monospace,Consolas,monospace; }
.eyebrow { font-family:ui-monospace,Consolas,monospace; font-size:12px; letter-spacing:.14em;
  text-transform:uppercase; color:var(--accent); }
h1 { font-size:28px; margin:6px 0 4px; }
.sub { color:var(--ink2); margin:0 0 30px; max-width:70ch; }
.tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:12px; margin-bottom:14px; }
.tile { background:var(--surface); border:1px solid var(--line); border-radius:6px; padding:14px 16px 12px; }
.tile .v { font-size:24px; font-weight:650; font-variant-numeric:tabular-nums; }
.tile .v small { font-size:13px; font-weight:500; color:var(--ink2); }
.tile .k { font-size:12.5px; color:var(--ink2); margin-top:2px; }
.band { display:grid; grid-template-columns:1fr 1.2fr; gap:12px; margin-bottom:38px; }
@media (max-width:720px) { .band { grid-template-columns:1fr; } }
.panel { background:var(--surface); border:1px solid var(--line); border-radius:6px; padding:16px 18px; }
.panel h2 { font-size:13px; margin:0 0 14px; color:var(--ink2); font-weight:600; letter-spacing:.04em; text-transform:uppercase; }
.yield { display:flex; height:22px; border-radius:4px; overflow:hidden; gap:2px; margin:4px 0 10px; }
.legend { display:flex; gap:18px; font-size:13px; color:var(--ink2); flex-wrap:wrap; }
.legend .sw { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:6px; }
.kv { display:grid; grid-template-columns:auto 1fr; gap:4px 14px; font-size:13.5px; margin-top:16px; }
.kv dt { color:var(--ink2); } .kv dd { margin:0; font-variant-numeric:tabular-nums; }
.bars { display:grid; grid-template-columns:max-content 1fr max-content; gap:7px 10px; align-items:center; font-size:13.5px; }
.bars .name { color:var(--ink2); font-family:ui-monospace,Consolas,monospace; font-size:12.5px; }
.bars .fill { height:14px; background:var(--accent); border-radius:0 4px 4px 0; min-width:3px; }
.bars .val { font-variant-numeric:tabular-nums; }
h2.big { font-size:19px; margin:0 0 6px; }
.ssub { color:var(--ink2); font-size:14px; margin:0 0 18px; max-width:72ch; }
.card { background:var(--surface); border:1px solid var(--line); border-radius:6px; padding:20px 22px; margin-bottom:18px; }
.ctop { display:flex; align-items:baseline; gap:10px; flex-wrap:wrap; margin-bottom:10px; }
.ctop .id { font-family:ui-monospace,Consolas,monospace; font-size:13px; color:var(--ink2); }
.chip { font-size:11px; font-family:ui-monospace,Consolas,monospace; letter-spacing:.06em; text-transform:uppercase;
  padding:2px 8px; border-radius:10px; background:var(--soft); color:var(--accent); }
.chip.n { background:transparent; border:1px solid var(--line); color:var(--ink2); }
.prompt { border-left:3px solid var(--accent); padding:2px 0 2px 14px; margin:0 0 16px; font-size:15px; max-width:72ch; }
.prompt .lbl { display:block; font-size:11px; letter-spacing:.1em; text-transform:uppercase; color:var(--accent);
  font-family:ui-monospace,Consolas,monospace; margin-bottom:4px; }
.red { list-style:none; margin:0 0 4px; padding:0; display:grid; gap:10px; }
.red li { display:grid; grid-template-columns:auto 1fr; gap:10px; align-items:start; }
.mark { background:var(--redact); color:var(--redink); font-family:ui-monospace,Consolas,monospace; font-size:11px;
  padding:3px 7px; border-radius:3px; white-space:nowrap; margin-top:1px; }
.red .t { font-size:14.5px; }
.red .check { display:block; font-size:12.5px; color:var(--ink3); margin-top:1px; }
details { margin-top:12px; }
summary { cursor:pointer; font-size:13px; color:var(--ink2); font-family:ui-monospace,Consolas,monospace; }
summary:hover { color:var(--accent); }
.spec { font-size:13.5px; color:var(--ink2); background:var(--bg); border:1px solid var(--line); border-radius:4px;
  padding:12px 14px; margin-top:8px; max-width:76ch; }
.kept { font-size:13px; color:var(--ink2); margin-top:8px; }
.kept .cat { color:var(--ink3); font-family:ui-monospace,Consolas,monospace; font-size:11.5px; }
.foot { color:var(--ink3); font-size:12.5px; border-top:1px solid var(--line); margin-top:40px; padding-top:14px; }
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


def write_html_report(path: str, instances: List[TaskInstance], report: dict, n_examples: int = 4):
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
    tiles_html = "".join(f'<div class="tile"><div class="v">{v}</div><div class="k">{k}</div></div>' for v, k in tiles)

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

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>UAUG synthesis report</title><style>{_CSS}</style></head><body>
<div class="wrap">
  <div class="eyebrow">uaug &middot; stage 1 &middot; data synthesis</div>
  <h1>Synthesis report: {n} instances from {n_sources} sources</h1>
  <p class="sub">Pipeline: decompose &rarr; redact &rarr; erode.</p>
  <div class="tiles">{tiles_html}</div>
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
  {examples_html}
  <div class="foot">Generated by <span class="mono">uaug.synthesis.report</span></div>
</div></body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)
