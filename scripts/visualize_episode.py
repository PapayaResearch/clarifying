"""Render paper-ready per-instance visuals for an evaluation run.

Usage:
    python scripts/visualize_episode.py <eval_dir> [-n 5] [--seed 42] [--instances <id> ...] [--out <dir>]

Each figure shows the full pipeline for one instance: the original caption,
the redacted details (each marked recovered or not by the final output),
the interaction as a chat, and the utility recovered under both weightings
(details weighed equally / by LLM-assigned importance) on one line.
Instances are sampled at random unless --instances names them.
Output goes to <eval_dir>/figures/<instance_id>.png, or to --out.
"""
import argparse
import json
import os
import random
import textwrap

import matplotlib
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch

INK, MUTED, LINE = "#3c4043", "#80868b", "#dadce0"
CAPTION_BG, USER_BG, AGENT_BG = "#f8f9fa", "#f1f3f4", "#e8f0fe"
DELIVERED, MISSED = "#188038", "#c5221f"
PAGE_W = 100.0
FIG_W = 8.5
LH = 2.6
PAD = 2.0
NOTE_SIZE = 6.2  # the recovered / not recovered verdict inside each redacted-detail box
# inches per page unit: the axes of plt.subplots span the figure width minus the default subplot margins
UNIT = FIG_W * (plt.rcParams["figure.subplot.right"] - plt.rcParams["figure.subplot.left"]) / PAGE_W
MODE_TITLES = {"no-asking": "No-Asking Mode", "fixed-asking": "Fixed-Asking Mode", "open-asking": "Open-Asking Mode"}

_probe = plt.figure(figsize=(1, 1))
_renderer = _probe.canvas.get_renderer()


def width(text, size, weight="normal"):
    """Width of one line of text in page units."""
    w, _, _ = _renderer.get_text_width_height_descent(text, FontProperties(size=size, weight=weight), False)
    return w / _probe.dpi / UNIT


def wrap(text, chars):
    lines = []
    for para in text.split("\n"):
        lines.extend(textwrap.wrap(para, chars) or [""])
    return lines


class Page:
    """Top-down block layout in a 0..100 wide coordinate system."""

    def __init__(self):
        self.items = []
        self.y = 6.5

    def space(self, h):
        self.y += h

    def label(self, text, x=3.0):
        self.items.append(("label", x, self.y, text))
        self.y += LH

    def box(self, text, x, w, bg, edge=LINE, chars=None, tag=None, tag_color=None,
            tag_side="right", tail=None, left_tag=None, note=None, note_color=None, note_col=0.0):
        """note: short text inside the box, vertically centered and left-aligned in a column of note_col units that
        ends at the box's right padding, the wrapped text keeps clear of that column."""
        chars = chars or int((w - (note_col + 4.5 if note else 0.0)) * 1.24)
        lines = wrap(text, chars)
        h = len(lines) * LH + 2 * PAD
        self.items.append(("box", x, self.y, w, h, lines, bg, edge, tag, tag_color, tag_side, tail, left_tag,
                           note, note_color, note_col))
        self.y += h
        return h

    def runs_box(self, runs, x, w):
        """A box holding a single line of text runs [(text, size, weight, color, gap_after)]: the first run is the
        label at the box's left padding, the rest are centered between the label and the right padding."""
        h = LH + 2 * PAD
        self.items.append(("runs", x, self.y, w, h, runs))
        self.y += h

    def arrow(self, note=""):
        self.items.append(("arrow", 50.0, self.y, note))
        self.y += 6.0

    def render(self, path, title):
        height_units = self.y + 4.0
        fig_h = height_units * 0.055
        fig, ax = plt.subplots(figsize=(8.5, fig_h))
        ax.set_xlim(0, PAGE_W)
        ax.set_ylim(0, height_units)
        ax.invert_yaxis()
        ax.axis("off")
        ax.text(3.0, 1.5, title, fontsize=9.5, color=INK, fontweight="bold", va="top")
        for item in self.items:
            if item[0] == "label":
                _, x, y, text = item
                ax.text(x, y + 0.4, text.upper(), fontsize=6.2, color=MUTED, va="top",
                        fontweight="bold")
            elif item[0] == "arrow":
                _, x, y, note = item
                ax.annotate("", xy=(x, y + 4.6), xytext=(x, y + 0.6),
                            arrowprops=dict(arrowstyle="-|>", color=MUTED, linewidth=1.2))
                if note:
                    ax.text(x + 2.0, y + 2.6, note, fontsize=6.5, color=MUTED, va="center")
            elif item[0] == "runs":
                _, x, y, w, h, runs = item
                ax.add_patch(FancyBboxPatch(
                    (x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                    facecolor=CAPTION_BG, edgecolor=LINE, linewidth=1.0, mutation_aspect=0.5))
                (label, size, weight, color, gap), rest = runs[0], runs[1:]
                ax.text(x + PAD, y + h / 2, label, fontsize=size, color=color, va="center", fontweight=weight)
                left, right = x + PAD + width(label, size, weight) + gap, x + w - PAD
                total = sum(width(t, s, wt) + g for t, s, wt, _, g in rest) - rest[-1][4]
                cursor = left + (right - left - total) / 2
                for text, size, weight, color, gap in rest:
                    ax.text(cursor, y + h / 2, text, fontsize=size, color=color, va="center", fontweight=weight)
                    cursor += width(text, size, weight) + gap
            elif item[0] == "box":
                _, x, y, w, h, lines, bg, edge, tag, tag_color, tag_side, tail, left_tag, note, note_color, \
                    note_col = item
                ax.add_patch(FancyBboxPatch(
                    (x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                    facecolor=bg, edgecolor=edge, linewidth=1.0, mutation_aspect=0.5))
                if tail:
                    # WhatsApp-style wedge flush with the box top, squaring off that corner
                    sign = -1 if tail == "left" else 1
                    bx = x if tail == "left" else x + w
                    inner = bx - sign * 1.5
                    apex = (bx + sign * 1.8, y)
                    cover = plt.Polygon(
                        [(inner, y), (inner, y + 1.8), (bx, y + 1.8), apex],
                        closed=True, facecolor=bg, edgecolor="none", zorder=2)
                    ax.add_patch(cover)
                    ax.plot([inner, apex[0], bx], [y, apex[1], y + 1.8],
                            color=edge, linewidth=1.0, zorder=2, solid_capstyle="round")
                for i, line in enumerate(lines):
                    ax.text(x + PAD, y + PAD + (i + 0.5) * LH, line, fontsize=7.2,
                            color=INK, va="center")
                if note:  # vertically centered in the box
                    ax.text(x + w - PAD - note_col, y + h / 2, note, fontsize=NOTE_SIZE,
                            color=note_color or MUTED, va="center")
                if tag:
                    tag_x, align = (x + w, "right") if tag_side == "right" else (x, "left")
                    ax.text(tag_x, y - 0.2, tag, fontsize=6.0, color=tag_color or MUTED,
                            ha=align, va="bottom")
                if left_tag:
                    ax.text(x, y - 0.2, left_tag, fontsize=6.0, color=tag_color or MUTED,
                            ha="left", va="bottom")
        fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def visualize(instance, episode, judgment, out_path):
    page = Page()

    page.label("ground-truth intent")
    page.box(instance["full_spec"], 3.0, 94.0, CAPTION_BG)
    page.space(1.5)

    page.label(f"redacted details ({len(instance['redacted_ids'])})")
    details = {d["id"]: d for d in instance["details"]}
    notes = {True: "✓ Detail recovered", False: "✗ Detail not recovered"}
    verdicts = {i: bool(judgment["per_detail"].get(str(i), {}).get("delivered")) for i in instance["redacted_ids"]}
    # the verdicts start at one x in every box, the widest one present on this instance ending as far from the right
    # edge as the text is from the left (an all-recovered instance is not padded for the absent longer label)
    note_col = max(width(notes[v], NOTE_SIZE) for v in set(verdicts.values()))
    for detail_id in instance["redacted_ids"]:
        detail = details[detail_id]
        verdict = verdicts[detail_id]
        color = DELIVERED if verdict else MISSED
        page.space(3.4)
        page.box(detail["text"], 6.0, 88.0, "white", edge=color,
                    tag=f"Importance = {detail['importance']:.2f}", left_tag=detail["category"].title(),
                    note=notes[verdict], note_color=color, note_col=note_col)

    page.arrow("Erode: remaining details rewritten as the user request")

    page.label("user request and interaction (user left, agent right)")
    page.space(0.5)
    page.box(instance["eroded_prompt"], 4.5, 72.0, USER_BG, tail="left")
    page.space(1.8)
    q_num = 0
    for turn in episode["turns"]:
        if turn["role"] == "agent_question":
            q_num += 1
            page.box(turn["text"], 23.5, 72.0, AGENT_BG, tag=f"Agent Question {q_num}",
                        tail="right")
            page.space(1.8)
        elif turn["role"] == "user_answer":
            page.box(turn["text"], 4.5, 72.0, USER_BG, tag="User Answer",
                        tag_side="left", tail="left")
            page.space(1.8)

    page.arrow("Compose: the agent commits its final output")
    page.label("agent final output")
    page.box(episode["final_output"], 3.0, 94.0, AGENT_BG)

    page.arrow("Judge: one check per redacted detail against the final output")

    u_rec = judgment["u_rec"]
    page.label("episode metrics")
    page.space(0.5)
    page.runs_box([  # (text, size, weight, color, gap after)
        ("Utility Recovered", 7.2, "normal", INK, 2.5),
        (f"{u_rec['equal']:.3f}", 7.2, "bold", INK, 0.8),
        ("(details weighed equally)", 6.4, "normal", MUTED, 2.0),
        ("/", 7.2, "normal", MUTED, 2.0),
        (f"{u_rec['llm']:.3f}", 7.2, "bold", INK, 0.8),
        ("(details weighed by LLM-assigned importance)", 6.4, "normal", MUTED, 0.0),
    ], 6.0, 88.0)  # the same box as the redacted details

    agent = episode["agent_name"]
    title = MODE_TITLES.get(agent, f"{agent.replace('_', '-').title()} Mode")
    page.render(out_path, title)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("eval_dir", help="evaluation run directory")
    parser.add_argument("-n", type=int, default=5, help="number of random instances (ignored with --instances)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--instances", nargs="+", metavar="ID", help="instance ids to render instead of a random sample")
    parser.add_argument("--out", help="output directory (default <eval_dir>/figures)")
    args = parser.parse_args()

    with open(os.path.join(args.eval_dir, "config.yaml"), encoding="utf-8") as f:
        config = yaml.safe_load(f)
    instances_dir = config["evaluation"]["instances_dir"]
    split = config["evaluation"]["split"]
    if not os.path.isabs(instances_dir):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        instances_dir = os.path.join(repo_root, instances_dir)

    instances = {}
    with open(os.path.join(instances_dir, f"instances.{split}.jsonl"), encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            instances[row["instance_id"]] = row
    episodes = {}
    with open(os.path.join(args.eval_dir, "episodes.jsonl"), encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            episodes[row["instance_id"]] = row
    judgments = {}
    with open(os.path.join(args.eval_dir, "judgments.jsonl"), encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            judgments[row["instance_id"]] = row

    out_dir = args.out or os.path.join(args.eval_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)
    if args.instances:
        missing = [i for i in args.instances if i not in episodes]
        if missing:
            raise SystemExit(f"not in this evaluation run: {', '.join(missing)}")
        chosen = args.instances
    else:
        chosen = random.Random(args.seed).sample(sorted(episodes), min(args.n, len(episodes)))
    for instance_id in chosen:
        out_path = os.path.join(out_dir, f"{instance_id}.png")
        visualize(instances[instance_id], episodes[instance_id], judgments[instance_id], out_path)
        print("saved", out_path)


if __name__ == "__main__":
    main()
