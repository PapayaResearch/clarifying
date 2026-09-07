"""The paper's overview figure of one episode: ground truth and user request side by side with the redacted details
under them, the clarification dialogue, then the agent's output beside the judge's verdict. Images are read from
<synthesis_dir>/images/ (generate_images_for_data.py) and <eval_dir>/images/ (generate_images_for_eval.py), a card
reads "missing" where one is absent.

Usage:
    python scripts/visualize_instance.py <instance_id> <eval_dir> [--max-words 150] [--out <dir>]

Output: <instance_id>_visual.png and .pdf in scripts/figures/k<k>_<agent>-<label>/ (e.g. k15_open-asking-patient) or
in --out. A final description longer than --max-words is cut with an ellipsis, the judge saw the whole text.
"""
import argparse
import os
from functools import partial

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Polygon  # noqa: E402

from visualize_images import SCRIPTS, draw_card, load_eval, load_image, run_label  # noqa: E402

INK, MUTED, LINE = "#3c4043", "#80868b", "#dadce0"
CAPTION_BG, USER_BG, AGENT_BG = "#f8f9fa", "#f1f3f4", "#e8f0fe"
DELIVERED, MISSED = "#188038", "#c5221f"

# layout, everything in inches
FIG_W = 10.9
MARGIN = 0.4
GAP = 0.36  # between panels along the pipeline, the arrows span it
PANEL_PAD = 0.22  # inside a panel
TITLE_PAD = 0.16  # above a panel title
HEADER_GAP = 0.1  # under a panel title
CARD_SIDE = 2.1  # image cards
FONT, SMALL, TINY = 7.4, 6.5, 5.8
PAD = 0.1  # inside text boxes
RADIUS = 0.08
UTILITY_LABEL = "Utility recovered"
UTILITY_INSET = PAD + 0.04


def line_height(size):
    return 1.38 * size / 72


class Canvas:
    """Drawing in inches with y running down from the top, rendered once the total height is known."""

    def __init__(self, width, probe=None):
        self.W = width
        self.ops = []  # drawing closures taking the page axes
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
            line = ""
            for word in para.split():
                if line and self.measure(f"{line} {word}", size) > width:
                    lines.append(line)
                    line = word
                else:
                    line = f"{line} {word}" if line else word
            lines.append(line)
        return lines

    def text(self, x, y, s, size=FONT, color=INK, ha="left", va="center", weight="normal"):
        self.ops.append(lambda ax: ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, fontweight=weight))

    def box(self, x, y, w, h, bg="white", edge=LINE, radius=RADIUS, dashed=False, tail=None, zorder=1):
        def op(ax):
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
        self.ops.append(lambda ax: ax.add_patch(Circle((x, y), r, facecolor=color, edgecolor="none")))

    def arrow(self, x0, y0, x1, y1):
        self.ops.append(lambda ax: ax.annotate(
            "", xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(arrowstyle="-|>", color=MUTED, linewidth=1.1, mutation_scale=9, shrinkA=0, shrinkB=0)))

    def image(self, x, y, side, image):
        self.ops.append(lambda ax: draw_card(
            ax.figure.add_axes([x / self.W, 1 - (y + side) / self.H, side / self.W, side / self.H]), image))

    def render(self, out_base, height):
        self.H = height
        fig = plt.figure(figsize=(self.W, height))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, self.W)
        ax.set_ylim(height, 0)
        ax.axis("off")
        for op in self.ops:
            op(ax)
        for ext in ("png", "pdf"):
            fig.savefig(f"{out_base}.{ext}", dpi=200, facecolor="white")
        plt.close(fig)
        plt.close(self._probe)


# Blocks: each draws at (x, y) within width w and returns the height it used


def paragraph(cv, x, y, w, text, bg, tail=None, tag=None, tag_side="right"):
    """Wrapped text in a rounded box, an optional small tag above one of its top corners."""
    lines = cv.wrap(text, w - 2 * PAD, FONT)
    step = line_height(FONT)
    h = len(lines) * step + 2 * PAD
    cv.box(x, y, w, h, bg, tail=tail)
    for i, line in enumerate(lines):
        cv.text(x + PAD, y + PAD + (i + 0.5) * step, line)
    if tag:
        tx, ha = (x + w, "right") if tag_side == "right" else (x, "left")
        cv.text(tx, y - 0.03, tag, SMALL, MUTED, ha=ha, va="bottom")
    return h


def header(cv, x, y, number, title):
    """Panel title, numbered in a dark disc when a number is given."""
    r = 0.095
    if number is not None:
        cv.circle(x + r, y + r, r, INK)
        cv.text(x + r, y + r + 0.005, str(number), 6.4, "white", ha="center", weight="bold")
        x += 2 * r + 0.1
    cv.text(x, y + r, title.upper(), 7.0, INK, weight="bold")
    return 2 * r


def detail_card(cv, x, y, w, detail, height=None):
    """A dashed card with the detail's category on top and its text wrapped below, at the given height when set so
    that a row of cards lines up."""
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
    """Utility recovered under both weightings in one soft box: the label at the left, the two values with their
    weighting notes centered beside it, on one line when they fit and stacked otherwise, the label moving above the
    stacked values when even that is too wide."""
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


def image_text_panel(cv, x, y, w, number, title, image, text, bg, tail=None, height=None):
    """A title, then an image card beside a text box, each centered vertically in the space under the title, which
    fills the panel when its height is given."""
    px, py, pw = x + PANEL_PAD, y + TITLE_PAD, w - 2 * PANEL_PAD
    hh = header(cv, px, py, number, title) + HEADER_GAP
    frame = 0.035 * CARD_SIDE  # the card frame reaches this far outside the card
    image_h = CARD_SIDE + 2 * frame
    tx, tw = px + CARD_SIDE + 0.3, pw - CARD_SIDE - 0.3
    text_h = paragraph(cv.scratch(), tx, 0, tw, text, bg)
    area = max(image_h, text_h, (height or 0) - TITLE_PAD - hh - PANEL_PAD)
    cv.image(px, py + hh + (area - image_h) / 2 + frame, CARD_SIDE, image)
    paragraph(cv, tx, py + hh + (area - text_h) / 2, tw, text, bg, tail=tail)
    return TITLE_PAD + hh + area + PANEL_PAD


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
    columns = {"agent_question": (px, AGENT_BG, "left"), "user_answer": (px + pw - bw, USER_BG, "right")}
    bottoms = {role: cy for role in columns}  # the last bubble's bottom per column
    prev_top, prev_h, end, q = cy, 0.0, cy, 0
    for turn in episode["turns"]:
        role = turn["role"]
        if role not in columns:
            continue
        if role == "agent_question":
            q += 1
        bx, bg, side = columns[role]
        top = max(prev_top + prev_h / 2, bottoms[role] + 0.1) + 0.17  # 0.17 for the tag above the bubble
        tag = f"Agent question {q}" if role == "agent_question" else f"Simulated user answer {q}"
        h = paragraph(cv, bx, top, bw, turn["text"], bg, tail=side, tag=tag, tag_side=side)
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
    def gap(split):
        first, second = pair(split)
        return first - second

    split = bisect(lambda s: gap(s) <= 0, lo, hi)
    return min((split, max(lo, split - 0.02)), key=lambda s: abs(gap(s)))


def experiment_name(config):
    """k<k>_<agent>-<label> of an evaluation run (the label from run_label), e.g. k15_open-asking-patient."""
    return f"k{config['general']['k']}_{config['agent']['name']}-{run_label(config)}"


def visualize(instance, episode, judgment, images, out_base, max_words):
    """images: reference, request and agent, any of them None when missing."""
    details = {d["id"]: d for d in instance["details"]}
    redacted = [details[i] for i in instance["redacted_ids"]]
    intent = partial(image_text_panel, number=1, title="Ground truth", image=images["reference"],
                     text=instance["full_spec"], bg=CAPTION_BG)
    request = partial(image_text_panel, number=2, title="User request", image=images["request"],
                      text=instance["eroded_prompt"], bg=USER_BG)
    output = partial(image_text_panel, number=4, title="Agent output", image=images["agent"],
                     text=truncated(episode["final_output"], max_words), bg=AGENT_BG, tail="right")
    judge = partial(judge_panel, redacted=redacted, judgment=judgment)

    cv = Canvas(FIG_W)
    x0, inner_w = MARGIN, FIG_W - 2 * MARGIN
    y = MARGIN

    # the instance as one panel: ground truth and request side by side, their split searched so that the two come
    # out the same height, the redacted details under both
    def top(w2):
        return (request(cv.scratch(), x0 + inner_w - w2, y, w2), intent(cv.scratch(), x0, y, inner_w - w2))

    w2 = balance(top, 0.36 * inner_w, 0.6 * inner_w)
    w1 = inner_w - w2
    row = max(top(w2))
    intent(cv, x0, y, w1, height=row)
    request(cv, x0 + w1, y, w2, height=row)
    overlap = PANEL_PAD - 0.04  # the redacted details start inside the row's bottom padding
    hr = redacted_panel(cv, x0, y + row - overlap, inner_w, redacted)
    cv.box(x0, y, inner_w, row + hr - overlap, "white", LINE, radius=0.14, zorder=0.5)
    y += row + hr - overlap + GAP
    cv.arrow(x0 + inner_w / 2, y - GAP + 0.06, x0 + inner_w / 2, y - 0.06)

    # the dialogue
    h3 = dialogue_panel(cv, x0, y, inner_w, episode)
    cv.box(x0, y, inner_w, h3, "white", LINE, radius=0.14, zorder=0.5)
    y += h3 + GAP

    # output beside judge, their split searched the same way, the judge never narrower than its utility box needs
    def bottom(w4):
        return (output(cv.scratch(), x0, y, w4), judge(cv.scratch(), x0 + w4 + GAP, y, inner_w - w4 - GAP))

    hi = inner_w - GAP - utility_min_width(cv, judgment["u_rec"]) - 2 * PANEL_PAD
    w4 = balance(bottom, min(0.47 * inner_w, hi), hi)
    w5 = inner_w - w4 - GAP
    x4, x5 = x0, x0 + w4 + GAP
    cv.arrow(x4 + w4 / 2, y - GAP + 0.06, x4 + w4 / 2, y - 0.06)
    row = max(bottom(w4))
    output(cv, x4, y, w4, height=row)
    judge(cv, x5, y, w5, height=row)
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
    parser.add_argument("--max-words", type=int, default=150,
                        help="cut the final description after this many words in the figure (0 shows it whole)")
    parser.add_argument("--out", help="output directory (default scripts/figures/<experiment>)")
    args = parser.parse_args()

    instances, episodes, judgments, config, instances_dir = load_eval(args.eval_dir)
    if args.instance not in instances or args.instance not in episodes:
        raise SystemExit(f"{args.instance} is not in the evaluation run {args.eval_dir}")
    data_images = os.path.join(instances_dir, "images")
    images = {"reference": load_image(os.path.join(data_images, f"{args.instance}_enhanced.jpg"))
              or load_image(os.path.join(data_images, f"{args.instance}_original.jpg")),
              "request": load_image(os.path.join(data_images, f"{args.instance}_request.jpg")),
              "agent": load_image(os.path.join(args.eval_dir, "images", f"{args.instance}.jpg"))}
    missing = [name for name, image in images.items() if image is None]
    if missing:
        print(f"images missing for {args.instance}: {', '.join(missing)} "
              "(run scripts/generate_images_for_data.py and scripts/generate_images_for_eval.py first)")

    out_dir = args.out or os.path.join(SCRIPTS, "figures", experiment_name(config))
    os.makedirs(out_dir, exist_ok=True)
    visualize(instances[args.instance], episodes[args.instance], judgments[args.instance], images,
              os.path.join(out_dir, f"{args.instance}_visual"), args.max_words)


if __name__ == "__main__":
    main()
