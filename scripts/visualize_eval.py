"""Render paper-ready per-instance visuals for an evaluation run.

Usage:
    python scripts/visualize_eval.py <eval_dir> -n 5 [--seed 42] [--importance llm|equal]

Each figure shows the full pipeline for one instance: the original caption,
the redacted details (colored by whether the final output delivered them),
the interaction as a chat, and the episode metrics under the chosen weighting.
Output goes to <eval_dir>/figures/<importance>/<instance_id>.png.
"""
import argparse
import json
import os
import random
import textwrap

import matplotlib
import yaml

from uaug.utils.misc import clairvoyant_u_rec

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

INK, MUTED, LINE = "#3c4043", "#80868b", "#dadce0"
CAPTION_BG, USER_BG, AGENT_BG = "#f8f9fa", "#f1f3f4", "#e8f0fe"
DELIVERED, MISSED = "#188038", "#c5221f"
PAGE_W = 100.0
LH = 2.6
PAD = 2.0
MODE_TITLES = {"no-asking": "No-Asking Mode", "fixed-asking": "Fixed-Asking Mode", "open-asking": "Open-Asking Mode"}


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
            tag_side="right", tail=None, left_tag=None):
        chars = chars or int(w * 1.24)
        lines = wrap(text, chars)
        h = len(lines) * LH + 2 * PAD
        self.items.append(("box", x, self.y, w, h, lines, bg, edge, tag, tag_color, tag_side, tail, left_tag))
        self.y += h
        return h

    def metric_row(self, texts, gap=1.5):
        """One box per metric, the row as a whole centered on the page."""
        h = LH + 2 * PAD
        widths = [len(text) * 0.78 + 2 * PAD for text in texts]
        x = (PAGE_W - sum(widths) - gap * (len(texts) - 1)) / 2
        for text, w in zip(texts, widths):
            self.items.append(("box", x, self.y, w, h, [text], CAPTION_BG, LINE, None, None, "right", None, None))
            x += w + gap
        self.y += h

    def arrow(self, note=""):
        self.items.append(("arrow", 50.0, self.y, note))
        self.y += 6.0

    def render(self, path, title, legend=None):
        """legend: [(color, label)] drawn as small swatches in the top-right corner."""
        height_units = self.y + 4.0
        fig_h = height_units * 0.055
        fig, ax = plt.subplots(figsize=(8.5, fig_h))
        ax.set_xlim(0, PAGE_W)
        ax.set_ylim(0, height_units)
        ax.invert_yaxis()
        ax.axis("off")
        ax.text(3.0, 1.5, title, fontsize=9.5, color=INK, fontweight="bold", va="top")
        x_right = PAGE_W - 3.0
        renderer = fig.canvas.get_renderer()
        for color, label in reversed(legend or []):  # laid out right to left, swatch left of its label
            text = ax.text(x_right, 2.6, label, fontsize=6.2, color=MUTED, ha="right", va="center")
            bbox = text.get_window_extent(renderer).transformed(ax.transData.inverted())
            swatch_x = bbox.x0 - 0.8 - 1.8
            ax.add_patch(Rectangle((swatch_x, 1.8), 1.8, 1.6, facecolor="white", edgecolor=color, linewidth=1.0))
            x_right = swatch_x - 2.4
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
            elif item[0] == "box":
                _, x, y, w, h, lines, bg, edge, tag, tag_color, tag_side, tail, left_tag = item
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
                if tag:
                    tag_x, align = (x + w, "right") if tag_side == "right" else (x, "left")
                    ax.text(tag_x, y - 0.2, tag, fontsize=6.0, color=tag_color or MUTED,
                            ha=align, va="bottom")
                if left_tag:
                    ax.text(x, y - 0.2, left_tag, fontsize=6.0, color=tag_color or MUTED,
                            ha="left", va="bottom")
        fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def visualize(instance, episode, judgment, out_path, importance="llm"):
    page = Page()

    page.label("ground-truth intent")
    page.box(instance["full_spec"], 3.0, 94.0, CAPTION_BG)
    page.space(1.5)

    page.label(f"redacted details ({len(instance['redacted_ids'])})")
    details = {d["id"]: d for d in instance["details"]}
    for detail_id in instance["redacted_ids"]:
        detail = details[detail_id]
        verdict = judgment["per_detail"].get(str(detail_id), {}).get("delivered")
        color = DELIVERED if verdict else MISSED
        page.space(3.4)
        page.box(detail["text"], 6.0, 88.0, "white", edge=color,
                    tag=f"Importance = {detail['importance']:.2f}" if importance == "llm" else None,
                    tag_color=color, left_tag=detail["category"].title())

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

    n_delivered = sum(1 for d in instance["redacted_ids"]
                        if judgment["per_detail"].get(str(d), {}).get("delivered"))
    redacted_importances = [details[i]["importance"] for i in instance["redacted_ids"]]
    page.label("episode metrics")
    page.space(0.5)
    page.metric_row([
        f"Utility Recovered = {judgment['u_rec'][importance]:.3f}",
        f"Clairvoyant Solution = {clairvoyant_u_rec(redacted_importances, episode['n_questions'], importance):.3f}",
        f"Details Delivered = {n_delivered}/{len(instance['redacted_ids'])}",
        f"Questions Asked = {episode['n_questions']}",
    ])

    agent = episode["agent_name"]
    title = MODE_TITLES.get(agent, f"{agent.replace('_', '-').title()} Mode")
    page.render(out_path, title, legend=[(DELIVERED, "Detail delivered"), (MISSED, "Detail not delivered")])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("eval_dir", help="evaluation run directory")
    parser.add_argument("-n", type=int, default=5, help="number of random instances")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--importance", choices=("llm", "equal"), default="llm",
                        help="which weighting the metrics show (llm also labels each redacted detail's importance)")
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

    out_dir = os.path.join(args.eval_dir, "figures", args.importance)
    os.makedirs(out_dir, exist_ok=True)
    chosen = random.Random(args.seed).sample(sorted(episodes), min(args.n, len(episodes)))
    for instance_id in chosen:
        out_path = os.path.join(out_dir, f"{instance_id}.png")
        visualize(instances[instance_id], episodes[instance_id], judgments[instance_id], out_path, args.importance)
        print("saved", out_path)


if __name__ == "__main__":
    main()
