"""Render the paper's overview figure for one instance: the ground truth, the user request, the redacted details,
the clarification dialogue, the agent's output and the judge's verdict, with the images that
scripts/visualize_images.py made for the instance (run it first, --images is its output directory).

Usage:
    python scripts/visualize_instance.py <instance_id> <eval_dir> [--images visuals_images] [--max-words 150]
        [--out <dir>]

Output goes to scripts/figures/<experiment>/<instance_id>_visual.png and .pdf, the experiment named after the
evaluation run (k<k>_<agent>-<step or base>, e.g. k3_open-asking-369), or to --out. A final description longer
than --max-words is cut with an ellipsis, the judge saw the whole text.
"""
import argparse
import json
import os

import matplotlib
import yaml
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Polygon  # noqa: E402

INK, MUTED, LINE, CARD = "#3c4043", "#80868b", "#dadce0", "#f1f3f4"
CAPTION_BG, USER_BG, AGENT_BG = "#f8f9fa", "#f1f3f4", "#e8f0fe"
DELIVERED, MISSED = "#188038", "#c5221f"

# layout, everything in inches
FIG_W = 10.9
MARGIN = 0.4
GAP = 0.36  # between panels along the pipeline, the arrows span it
PANEL_PAD = 0.22  # inside a panel
CARD_SIDE = 2.1  # image cards
FONT, SMALL, TINY = 7.4, 6.5, 5.8
PAD = 0.1  # inside text boxes
RADIUS = 0.08


def line_height(size):
    return 1.38 * size / 72


class Canvas:
    """Drawing in inches with y running down from the top, rendered once the total height is known."""

    def __init__(self, width, probe=None):
        self.W = width
        self.ops = []
        self._probe = probe or plt.figure(figsize=(1, 1))
        self._renderer = self._probe.canvas.get_renderer()

    def scratch(self):
        """A canvas that measures like this one but whose drawing is thrown away, for dry-run layouts."""
        return Canvas(self.W, self._probe)

    def measure(self, text, size, weight="normal"):
        """Width of one line of text in inches."""
        w, _, _ = self._renderer.get_text_width_height_descent(text, FontProperties(size=size, weight=weight), False)
        return w / self._probe.dpi

    def wrap(self, text, width, size):
        lines = []
        for para in text.split("\n"):
            words = para.split()
            if not words:
                lines.append("")
                continue
            line = words[0]
            for word in words[1:]:
                if self.measure(f"{line} {word}", size) <= width:
                    line = f"{line} {word}"
                else:
                    lines.append(line)
                    line = word
            lines.append(line)
        return lines

    def text(self, x, y, s, size=FONT, color=INK, ha="left", va="center", weight="normal"):
        self.ops.append(lambda fig, ax, H: ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, fontweight=weight))

    def box(self, x, y, w, h, bg="white", edge=LINE, radius=RADIUS, dashed=False, tail=None, zorder=1):
        def op(fig, ax, H):
            ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}", facecolor=bg,
                                        edgecolor=edge, linewidth=1.0, linestyle=(0, (3, 2)) if dashed else "-",
                                        zorder=zorder))
            if tail:  # WhatsApp-style wedge flush with the box top, squaring off that corner
                t = 0.13
                sign = -1 if tail == "left" else 1
                bx = x if tail == "left" else x + w
                inner = bx - sign * 0.11
                apex = (bx + sign * t, y)
                ax.add_patch(Polygon([(inner, y), (inner, y + t), (bx, y + t), apex], closed=True, facecolor=bg,
                                     edgecolor="none", zorder=2))
                ax.plot([inner, apex[0], bx], [y, apex[1], y + t], color=edge, linewidth=1.0, zorder=2,
                        solid_capstyle="round")
        self.ops.append(op)

    def circle(self, x, y, r, color):
        self.ops.append(lambda fig, ax, H: ax.add_patch(Circle((x, y), r, facecolor=color, edgecolor="none")))

    def arrow(self, x0, y0, x1, y1):
        self.ops.append(lambda fig, ax, H: ax.annotate(
            "", xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(arrowstyle="-|>", color=MUTED, linewidth=1.1, mutation_scale=9, shrinkA=0, shrinkB=0)))

    def image(self, x, y, side, image):
        def op(fig, ax, H):
            draw_card(fig.add_axes([x / self.W, 1 - (y + side) / H, side / self.W, side / H]), image)
        self.ops.append(op)

    def render(self, out_base, height):
        fig = plt.figure(figsize=(self.W, height))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, self.W)
        ax.set_ylim(height, 0)
        ax.axis("off")
        for op in self.ops:
            op(fig, ax, height)
        for ext in ("png", "pdf"):
            fig.savefig(f"{out_base}.{ext}", dpi=200, facecolor="white")
        plt.close(fig)
        plt.close(self._probe)


def squared(image):
    """The image letterboxed on white to a square, so every card has the same shape."""
    w, h = image.size
    if w == h:
        return image
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), "white")
    canvas.paste(image, ((side - w) // 2, (side - h) // 2))
    return canvas


def draw_card(ax, image):
    """The image inside a rounded card with a soft grey frame."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.035,rounding_size=0.09", facecolor=CARD,
                                edgecolor=LINE, linewidth=1.0, clip_on=False, zorder=1))
    if image is None:
        ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=8, color=MUTED, zorder=3)
        return
    shown = ax.imshow(squared(image), extent=(0, 1, 0, 1), zorder=2)
    shown.set_clip_path(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.07", transform=ax.transData))


# Blocks: each draws at (x, y) within width w and returns the height it used


def paragraph(cv, x, y, w, text, size=FONT, color=INK, bg=None, edge=None, tail=None, tag=None, tag_side="right",
              pad=PAD):
    """Wrapped text, inside a rounded box when a background or edge is given, an optional small tag above it."""
    lines = cv.wrap(text, w - 2 * pad, size)
    step = line_height(size)
    h = len(lines) * step + 2 * pad
    if bg is not None or edge is not None:
        cv.box(x, y, w, h, bg or "white", edge or LINE, tail=tail)
    for i, line in enumerate(lines):
        cv.text(x + pad, y + pad + (i + 0.5) * step, line, size, color)
    if tag:
        tx, ha = (x + w, "right") if tag_side == "right" else (x, "left")
        cv.text(tx, y - 0.03, tag, SMALL, MUTED, ha=ha, va="bottom")
    return h


def header(cv, x, y, number, title):
    """Panel title, numbered when a number is given."""
    r = 0.095
    tx = x
    if number is not None:
        cv.circle(x + r, y + r, r, INK)
        cv.text(x + r, y + r + 0.005, str(number), 6.4, "white", ha="center", weight="bold")
        tx = x + 2 * r + 0.1
    cv.text(tx, y + r, title.upper(), 7.0, INK, weight="bold")
    return 2 * r


def detail_card(cv, x, y, w, detail, height=None):
    """A dashed card with the detail's category on top and its text wrapped below, drawn at the given height when
    set so that a row of cards lines up."""
    label_h = line_height(TINY) + 0.05
    lines = cv.wrap(detail["text"], w - 2 * PAD, FONT)
    step = line_height(FONT)
    h = height or PAD + label_h + len(lines) * step + PAD
    cv.box(x, y, w, h, "white", LINE, dashed=True)
    cv.text(x + PAD, y + PAD + 0.5 * line_height(TINY), detail["category"].upper(), TINY, MUTED, weight="bold")
    for i, line in enumerate(lines):
        cv.text(x + PAD, y + PAD + label_h + (i + 0.5) * step, line, FONT, INK)
    return h


def check_row(cv, x, y, w, text, delivered):
    """The judge's check question after a tick or a cross, the box edged in the verdict's color."""
    color, glyph = (DELIVERED, "✓") if delivered else (MISSED, "✗")
    lead = PAD + cv.measure(glyph, 8.5, "bold") + 0.07
    lines = cv.wrap(text, w - lead - PAD, FONT)
    step = line_height(FONT)
    h = len(lines) * step + 2 * PAD
    cv.box(x, y, w, h, "white", color)
    cv.text(x + PAD, y + PAD + 0.5 * step, glyph, 8.5, color, weight="bold")
    for i, line in enumerate(lines):
        cv.text(x + lead, y + PAD + (i + 0.5) * step, line, FONT, INK)
    return h


def runs_width(cv, runs):
    """runs: [(text, size, weight, color, gap_after)], the trailing gap not counted."""
    return sum(cv.measure(text, size, weight) + gap for text, size, weight, _, gap in runs) - runs[-1][4]


def draw_runs(cv, x, y, runs):
    for text, size, weight, color, gap in runs:
        cv.text(x, y, text, size, color, weight=weight)
        x += cv.measure(text, size, weight) + gap


UTILITY_LABEL = "Utility recovered"
UTILITY_INSET = PAD + 0.04


def utility_groups(u_rec):
    """The two (value, weighting note) run groups of the utility box."""
    return [[(f"{u_rec['equal']:.2f}", 9.5, "bold", INK, 0.06),
             ("(details weighed equally)", TINY, "normal", MUTED, 0.12)],
            [(f"{u_rec['llm']:.2f}", 9.5, "bold", INK, 0.06),
             ("(details weighed by LLM-assigned importance)", TINY, "normal", MUTED, 0.0)]]


def utility_min_width(cv, u_rec):
    """The narrowest utility box whose narrowest layout (label above the stacked values) still fits."""
    return 2 * UTILITY_INSET + max(runs_width(cv, group) for group in utility_groups(u_rec))


def utility_box(cv, x, y, w, u_rec):
    """Utility recovered under both weightings in one soft box. The label sits at the left with the two values and
    their weighting notes centered beside it, on one line when they fit and stacked otherwise, and when even that
    is too wide the label moves above the stacked values."""
    groups = utility_groups(u_rec)
    label_w = cv.measure(UTILITY_LABEL, SMALL)
    left, right = x + UTILITY_INSET + label_w + 0.3, x + w - UTILITY_INSET
    line = groups[0] + [("/", SMALL, "normal", MUTED, 0.12)] + groups[1]
    block = max(runs_width(cv, group) for group in groups)
    if runs_width(cv, line) <= right - left:  # one line
        h = 0.4
        cv.box(x, y, w, h, CAPTION_BG, LINE)
        cv.text(x + UTILITY_INSET, y + h / 2, UTILITY_LABEL, SMALL, MUTED)
        draw_runs(cv, left + (right - left - runs_width(cv, line)) / 2, y + h / 2, line)
        return h
    if block <= right - left:  # label beside the two stacked lines
        h = 0.58
        cv.box(x, y, w, h, CAPTION_BG, LINE)
        cv.text(x + UTILITY_INSET, y + h / 2, UTILITY_LABEL, SMALL, MUTED)
        for i, group in enumerate(groups):
            draw_runs(cv, left + (right - left - block) / 2, y + h / 2 + (i - 0.5) * 0.22, group)
        return h
    h = 0.8  # label above the two stacked lines
    cv.box(x, y, w, h, CAPTION_BG, LINE)
    cv.text(x + UTILITY_INSET, y + 0.17, UTILITY_LABEL, SMALL, MUTED)
    for i, group in enumerate(groups):
        draw_runs(cv, x + (w - block) / 2, y + 0.42 + i * 0.22, group)
    return h


# Panels: each lays out its content from the panel's top-left corner and returns the panel height


TITLE_PAD = 0.16  # above a panel title
HEADER_GAP = 0.1  # under a panel title


def image_text_panel(cv, x, y, w, number, title, image, text, bg, tail=None, height=None):
    """A title, then an image card beside a text box, each centered vertically in the space under the title. That
    space fills the panel when its height is given."""
    px, py, pw = x + PANEL_PAD, y + TITLE_PAD, w - 2 * PANEL_PAD
    hh = header(cv, px, py, number, title) + HEADER_GAP
    top = py + hh
    frame = 0.035 * CARD_SIDE  # the card frame reaches this far outside the card
    image_h = CARD_SIDE + 2 * frame
    tx, tw = px + CARD_SIDE + 0.3, pw - CARD_SIDE - 0.3
    text_h = len(cv.wrap(text, tw - 2 * PAD, FONT)) * line_height(FONT) + 2 * PAD
    area = max(image_h, text_h)
    if height:
        area = max(area, height - TITLE_PAD - hh - PANEL_PAD)
    cv.image(px, top + (area - image_h) / 2 + frame, CARD_SIDE, image)
    paragraph(cv, tx, top + (area - text_h) / 2, tw, text, bg=bg, tail=tail)
    return TITLE_PAD + hh + area + PANEL_PAD


def intent_panel(cv, x, y, w, instance, image, height=None):
    """The reference image beside the full description."""
    return image_text_panel(cv, x, y, w, 1, "Ground truth", image, instance["full_spec"], CAPTION_BG, height=height)


def request_panel(cv, x, y, w, instance, image, height=None):
    """The request image beside the eroded request."""
    return image_text_panel(cv, x, y, w, 2, "User request", image, instance["eroded_prompt"], USER_BG,
                            height=height)


def redacted_panel(cv, x, y, w, redacted):
    """The redacted details as a row of dashed cards of one height."""
    px, py, pw = x + PANEL_PAD, y + TITLE_PAD, w - 2 * PANEL_PAD
    n = len(redacted)
    cy = py + header(cv, px, py, None, "Redacted details") + HEADER_GAP
    gap = 0.2
    cw = (pw - gap * (n - 1)) / n
    h = max(detail_card(cv.scratch(), px, cy, cw, detail) for detail in redacted)
    for i, detail in enumerate(redacted):
        detail_card(cv, px + i * (cw + gap), cy, cw, detail, height=h)
    return cy + h + PANEL_PAD - y


def dialogue_panel(cv, x, y, w, episode):
    """The chat in two columns, agent questions on the left and simulated user answers on the right, each bubble
    starting at the vertical midpoint of the one before it (or just under the last bubble of its own column)."""
    px, py, pw = x + PANEL_PAD, y + TITLE_PAD, w - 2 * PANEL_PAD
    cy = py + header(cv, px, py, 3, "Clarification") + 0.06
    bw = 0.49 * pw
    columns = {"agent_question": (px, AGENT_BG, "left", "left"), "user_answer": (px + pw - bw, USER_BG, "right", "right")}
    bottoms = {role: cy for role in columns}  # the last bubble's bottom per column
    prev_top, prev_h, end, q = cy, 0.0, cy, 0
    for turn in episode["turns"]:
        role = turn["role"]
        if role not in columns:
            continue
        if role == "agent_question":
            q += 1
        bx, bg, tail, tag_side = columns[role]
        top = max(prev_top + prev_h / 2, bottoms[role] + 0.1) + 0.17  # 0.17 for the tag above the bubble
        tag = f"Agent question {q}" if role == "agent_question" else f"Simulated user answer {q}"
        h = paragraph(cv, bx, top, bw, turn["text"], bg=bg, tail=tail, tag=tag, tag_side=tag_side)
        bottoms[role] = top + h
        prev_top, prev_h, end = top, h, max(end, top + h)
    if q == 0:
        cv.text(px, cy + 0.25, "The agent asks no questions.", FONT, MUTED)
        end = cy + 0.5
    return end + PANEL_PAD - y


def truncated(text, max_words):
    """The text cut after max_words words, at the last clause break past the halfway point, with an ellipsis."""
    words = text.split()
    if not max_words or len(words) <= max_words:
        return text
    cut = " ".join(words[:max_words])
    for sep in (". ", ", ", "; "):
        at = cut.rfind(sep)
        if at > len(cut) / 2:
            cut = cut[:at]
            break
    return cut.rstrip(",;.") + " …"


def output_panel(cv, x, y, w, episode, image, max_words, height=None):
    """The output image beside the final description."""
    text = truncated(episode["final_output"], max_words)
    return image_text_panel(cv, x, y, w, 4, "Agent output", image, text, AGENT_BG, tail="right", height=height)


def judge_panel(cv, x, y, w, redacted, judgment, height=None):
    """The check rows bunched together, then the utility box, the spare room split equally between the three gaps
    (under the title, between the checks and the utility box, and under it) when the panel's height is given."""
    px, py, pw = x + PANEL_PAD, y + TITLE_PAD, w - 2 * PANEL_PAD
    hh = header(cv, px, py, 5, "Judge")
    verdicts = [(d["check"], bool(judgment["per_detail"].get(str(d["id"]), {}).get("delivered"))) for d in redacted]
    checks_h = sum(check_row(cv.scratch(), px, 0, pw, text, ok) for text, ok in verdicts) + 0.08 * (len(verdicts) - 1)
    box_h = utility_box(cv.scratch(), px, 0, pw, judgment["u_rec"])
    fixed = TITLE_PAD + hh + checks_h + box_h
    gap = max(0.16, (height - fixed) / 3) if height else 0.16
    cy = py + hh + gap
    for text, ok in verdicts:
        cy += check_row(cv, px, cy, pw, text, ok) + 0.08
    cy += gap - 0.08
    cy += utility_box(cv, px, cy, pw, judgment["u_rec"])
    return cy + gap - y


def bisect(predicate, lo, hi, tol=0.01):
    """The smallest value in [lo, hi] where the monotone predicate turns true (hi when it never does)."""
    if predicate(lo):
        return lo
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if predicate(mid):
            hi = mid
        else:
            lo = mid
    return hi


def balance(pair, lo, hi):
    """The split in [lo, hi] at which the two panel heights pair(split) come closest, the first minus the second
    decreasing in the split (dry runs on scratch canvases)."""
    split = bisect(lambda s: pair(s)[0] <= pair(s)[1], lo, hi)
    return min((split, max(lo, split - 0.02)), key=lambda s: abs(pair(s)[0] - pair(s)[1]))


# Data


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_eval(eval_dir):
    """(instances by id, episodes by id, judgments by id, config) of one evaluation run."""
    with open(os.path.join(eval_dir, "config.yaml"), encoding="utf-8") as f:
        config = yaml.safe_load(f)
    instances_dir = config["evaluation"]["instances_dir"]
    if not os.path.isabs(instances_dir):
        instances_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), instances_dir)
    split = config["evaluation"]["split"]
    instances = {r["instance_id"]: r for r in read_jsonl(os.path.join(instances_dir, f"instances.{split}.jsonl"))}
    episodes = {r["instance_id"]: r for r in read_jsonl(os.path.join(eval_dir, "episodes.jsonl"))}
    judgments = {r["instance_id"]: r for r in read_jsonl(os.path.join(eval_dir, "judgments.jsonl"))}
    return instances, episodes, judgments, config


def load_image(path):
    return Image.open(path).convert("RGB") if os.path.exists(path) else None


def experiment_name(config):
    """k<k>_<agent>-<step or base> of an evaluation run, e.g. k3_open-asking-369, the figure's subfolder."""
    checkpoint = config["model"].get("checkpoint")
    step = checkpoint.rsplit("step-", 1)[-1].lstrip("0") if checkpoint else "base"
    return f"k{config['general']['k']}_{config['agent']['name']}-{step}"


def describe(config):
    """The image column key of the evaluated agent, named as in visualize_images.py: the untrained open asker is
    naive-asking, otherwise the agent's mode."""
    fixed = config["agent"]["mode"] == "fixed"
    if not fixed and not config["model"].get("checkpoint"):
        return "naive-asking"
    return "fixed-asking" if fixed else "open-asking"


def visualize(instance, episode, judgment, config, images, out_base, max_words):
    details = {d["id"]: d for d in instance["details"]}
    redacted = [details[i] for i in instance["redacted_ids"]]
    key = describe(config)

    cv = Canvas(FIG_W)
    x0, inner_w = MARGIN, FIG_W - 2 * MARGIN
    y = MARGIN

    # the instance as one panel: intent and request side by side (the split searched so that the two come out the
    # same height), the redacted details under both, the pipeline arrows start after it
    def top(w2):
        w1 = inner_w - w2
        return (request_panel(cv.scratch(), x0 + w1, y, w2, instance, images["request"]),
                intent_panel(cv.scratch(), x0, y, w1, instance, images["reference"]))

    w2 = balance(top, 0.36 * inner_w, 0.6 * inner_w)
    w1 = inner_w - w2
    row = max(top(w2))
    intent_panel(cv, x0, y, w1, instance, images["reference"], height=row)
    request_panel(cv, x0 + w1, y, w2, instance, images["request"], height=row)
    overlap = PANEL_PAD - 0.04  # the redacted details start inside the row's bottom padding
    hr = redacted_panel(cv, x0, y + row - overlap, inner_w, redacted)
    cv.box(x0, y, inner_w, row + hr - overlap, "white", LINE, radius=0.14, zorder=0.5)
    y += row + hr - overlap + GAP
    cv.arrow(x0 + inner_w / 2, y - GAP + 0.06, x0 + inner_w / 2, y - 0.06)

    # middle band: the dialogue
    h3 = dialogue_panel(cv, x0, y, inner_w, episode)
    cv.box(x0, y, inner_w, h3, "white", LINE, radius=0.14, zorder=0.5)
    y += h3 + GAP

    # bottom row: the split between output and judge searched so that the two panels come out the same height,
    # the judge never narrower than its utility box needs
    def heights(w4):
        return (output_panel(cv.scratch(), x0, y, w4, episode, images[key], max_words),
                judge_panel(cv.scratch(), x0 + w4 + GAP, y, inner_w - w4 - GAP, redacted, judgment))

    hi = inner_w - GAP - utility_min_width(cv, judgment["u_rec"]) - 2 * PANEL_PAD
    w4 = balance(heights, min(0.47 * inner_w, hi), hi)
    w5 = inner_w - w4 - GAP
    x4, x5 = x0, x0 + w4 + GAP
    cv.arrow(x4 + w4 / 2, y - GAP + 0.06, x4 + w4 / 2, y - 0.06)
    row = max(heights(w4))
    output_panel(cv, x4, y, w4, episode, images[key], max_words, height=row)
    judge_panel(cv, x5, y, w5, redacted, judgment, height=row)
    cv.box(x4, y, w4, row, "white", LINE, radius=0.14, zorder=0.5)
    cv.box(x5, y, w5, row, "white", LINE, radius=0.14, zorder=0.5)
    cv.arrow(x4 + w4 + 0.06, y + row / 2, x5 - 0.06, y + row / 2)
    y += row

    cv.render(out_base, y + MARGIN)
    print(f"saved {out_base}.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("instance", help="instance id, e.g. test_00337.k3")
    parser.add_argument("eval_dir", help="evaluation run of the agent whose episode the figure walks through")
    parser.add_argument("--images", default="visuals_images",
                        help="directory with the images from scripts/visualize_images.py")
    parser.add_argument("--max-words", type=int, default=150,
                        help="cut the final description after this many words in the figure (0 shows it whole)")
    parser.add_argument("--out", help="output directory (default scripts/figures/<experiment>)")
    args = parser.parse_args()

    instances, episodes, judgments, config = load_eval(args.eval_dir)
    if args.instance not in instances or args.instance not in episodes:
        raise SystemExit(f"{args.instance} is not in the evaluation run {args.eval_dir}")
    key = describe(config)
    images = {"reference": load_image(os.path.join(args.images, f"{args.instance}_enhanced.jpg"))
              or load_image(os.path.join(args.images, f"{args.instance}_original.jpg")),
              "request": load_image(os.path.join(args.images, f"{args.instance}_request-edit.jpg")),
              key: load_image(os.path.join(args.images, f"{args.instance}_{key}-edit.jpg"))}
    missing = [name for name, image in images.items() if image is None]
    if missing:
        print(f"images missing in {args.images} for {args.instance}: {', '.join(missing)} (run scripts/visualize_images.py first)")

    out_dir = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", experiment_name(config))
    os.makedirs(out_dir, exist_ok=True)
    visualize(instances[args.instance], episodes[args.instance], judgments[args.instance], config, images,
              os.path.join(out_dir, f"{args.instance}_visual"), args.max_words)


if __name__ == "__main__":
    main()
