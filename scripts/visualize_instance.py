"""The paper's walk-through of one episode: the full description beside the user request with the redacted details
under both, the clarification dialogue, then the final output with the utility recovered under it and the judge's checks
beside them. Images come from <synthesis_dir>/images/ (generate_images_for_data.py) and <eval_dir>/images/
(generate_images_for_eval.py), a card reads "missing" where one is absent.

Usage:
    python scripts/visualize_instance.py <instance_id> [<eval_dir>] [--max-words 150] [--out <dir>]

Writes <instance_id>_visual as png and pdf to --out or scripts/figures/k<k>_<agent>-<label>. The full description and
the output are cut with an ellipsis to the height of their image (the output after --max-words first), the judge saw
the whole text.
"""
import argparse
import math
import os

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Circle, FancyBboxPatch, Polygon

from utils import AGENT_BG, CAPTION_BG, DELIVERED, EXAMPLE_RUN, FIGURES, INK, LINE, MISSED, USER_BG, draw_card, \
    load_eval, load_image, run_label

MUTED = "#5f6368"  # a shade darker than the muted grey of the other figures

# layout in inches
FIG_W, MARGIN = 12.5, 0.4
GAP = 0.3  # between panels, spanned by the arrows
PANEL_PAD, TITLE_PAD, HEADER_GAP = 0.18, 0.13, 0.08  # inside a panel, over its title, under its title
CARD_SIDE = 2.1
TOP_SPLIT = 0.5  # width share of the full description beside the request
OUTPUT_SHARE = 0.42  # width share of the final output beside the judge
QUESTION_SHARE = 0.67  # width share of the questions beside the answers
CHAT_GAP, CHAT_INSET = 0.14, 0.06  # between the chat columns, and their inset so the tails clear the panel edge
CHECK_COLUMNS = 2
UTILITY_GAP = 0.14  # between the output panel and the utility box
FONT, TITLE_SIZE, SUBTITLE_SIZE = 10.0, 11.5, 9.0  # body text, numbered panel titles, the redacted details heading
TAG_SIZE, UTILITY_SIZE = 8.0, 10.5  # the tags over the bubbles, the utility label and values
PAD = 0.09  # inside text boxes
LABEL_GAP = 0.06  # between a detail's text and its (category, weight)
CHECK_PAD = 0.06  # over and under a check's text
RADIUS = 0.08
TAIL = 0.13  # reach of a chat bubble's tail
UTILITY_INSET = PAD + 0.14


def line_height(size):
    return 1.32 * size / 72


class Canvas:
    """Drawing in inches with y running down from the top, rendered once the total height is known."""

    def __init__(self, width, probe=None):
        self.W = width
        self.ops = []  # drawing closures taking the page axes
        self._probe = probe or plt.figure(figsize=(1, 1))
        self._renderer = self._probe.canvas.get_renderer()

    def scratch(self):
        """A canvas that measures like this one but draws nothing, for dry-run layouts."""
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
            if tail:  # a wedge flush with the box top, squaring off that corner
                bx, sign = (x, -1) if tail == "left" else (x + w, 1)
                inner, apex = bx - sign * 0.11, bx + sign * TAIL
                ax.add_patch(Polygon([(inner, y), (inner, y + TAIL), (bx, y + TAIL), (apex, y)], closed=True,
                                     facecolor=bg, edgecolor="none", zorder=2))
                ax.plot([inner, apex, bx], [y, y, y + TAIL], color=edge, linewidth=1.0, zorder=2, solid_capstyle="round")
        self.ops.append(op)

    def circle(self, x, y, r, color):
        self.ops.append(lambda ax: ax.add_patch(Circle((x, y), r, facecolor=color, edgecolor="none")))

    def arrow(self, x0, y0, x1, y1):
        self.ops.append(lambda ax: ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(
            arrowstyle="-|>", color=MUTED, linewidth=1.1, mutation_scale=9, shrinkA=0, shrinkB=0)))

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


def paragraph(cv, x, y, w, text, bg, side=None, tag=None):
    """Wrapped text in a rounded box, with a chat tail and a small tag over the corner on the given side."""
    lines = cv.wrap(text, w - 2 * PAD, FONT)
    step = line_height(FONT)
    h = len(lines) * step + 2 * PAD
    cv.box(x, y, w, h, bg, tail=side)
    for i, line in enumerate(lines):
        cv.text(x + PAD, y + PAD + (i + 0.5) * step, line)
    if tag:
        tx, ha = (x + w, "right") if side == "right" else (x, "left")
        cv.text(tx, y - 0.03, tag, TAG_SIZE, MUTED, ha=ha, va="bottom")
    return h


def fitted(cv, text, w, max_h):
    """The text cut at a word with an ellipsis, so that its paragraph box of width w is at most max_h tall."""
    def fits(t):
        return paragraph(cv.scratch(), 0, 0, w, t, "white") <= max_h

    if fits(text):
        return text
    words = text.split()
    lo, hi = 0, len(words)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if fits(" ".join(words[:mid]) + " …") else (lo, mid)
    return " ".join(words[:lo]).rstrip(",;.") + " …"


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


def header(cv, x, y, number, title, size=TITLE_SIZE):
    """Panel title in capitals, after a numbered dark disc when a number is given."""
    r = 0.0085 * size
    if number is not None:
        cv.circle(x + r, y + r, r, INK)
        cv.text(x + r, y + r + 0.004, str(number), 0.66 * size, "white", ha="center", weight="bold")
        x += 2 * r + 0.11
    cv.text(x, y + r, title.upper(), size, INK, weight="bold")
    return 2 * r


def detail_label(detail):
    return f"({detail['category']}, {detail['importance']:.1f})"


def detail_card(cv, x, y, w, detail):
    """A dashed one-line card: the detail's text, then its category and weight in muted type."""
    h = line_height(FONT) + 2 * PAD
    cv.box(x, y, w, h, "white", LINE, dashed=True)
    cv.text(x + PAD, y + h / 2, detail["text"], FONT, INK)
    cv.text(x + PAD + cv.measure(detail["text"], FONT) + LABEL_GAP, y + h / 2, detail_label(detail), FONT, MUTED)


def check_row(cv, x, y, w, text, delivered, height=None):
    """The judge's check after a tick or a cross, edged in the verdict's color, its text centered in the given height."""
    color, glyph = (DELIVERED, "✓") if delivered else (MISSED, "✗")
    lead = PAD + cv.measure(glyph, FONT, "bold") + 0.07
    lines = cv.wrap(text, w - lead - PAD, FONT)
    step = line_height(FONT)
    h = height or len(lines) * step + 2 * CHECK_PAD
    top = y + (h - len(lines) * step) / 2
    cv.box(x, y, w, h, "white", color)
    cv.text(x + PAD, top + 0.5 * step, glyph, FONT, color, weight="bold")
    for i, line in enumerate(lines):
        cv.text(x + lead, top + (i + 0.5) * step, line, FONT, INK)
    return h


def utility_box(cv, x, y, w, u_rec):
    """Utility recovered under both weightings: the label at the left, the values with their notes centered beside it."""
    rows = [(f"{u_rec['equal']:.2f}", "(details weighed equally)"),
            (f"{u_rec['llm']:.2f}", "(details weighed by importance)")]
    gaps = [cv.measure(value, UTILITY_SIZE, "bold") + 0.06 for value, _ in rows]  # from a value to its note
    block = max(gap + cv.measure(note, FONT) for gap, (_, note) in zip(gaps, rows))
    left = x + UTILITY_INSET + cv.measure("UTILITY RECOVERED", UTILITY_SIZE, "bold") + 0.3
    right = x + w - UTILITY_INSET
    h = 0.58
    cv.box(x, y, w, h, CAPTION_BG, LINE)
    cv.text(x + UTILITY_INSET, y + h / 2, "UTILITY RECOVERED", UTILITY_SIZE, INK, weight="bold")
    for i, ((value, note), gap) in enumerate(zip(rows, gaps)):
        vx, vy = left + (right - left - block) / 2, y + h / 2 + (i - 0.5) * 0.22
        cv.text(vx, vy, value, UTILITY_SIZE, INK, weight="bold")
        cv.text(vx + gap, vy, note, FONT, MUTED)
    return h


# Panels: each lays out its content from the panel's top-left corner and returns the panel height


def image_text_panel(cv, x, y, w, number, title, image, text, bg):
    """A title, then an image card beside a text box cut to the image's height and centered on it."""
    px, py, pw = x + PANEL_PAD, y + TITLE_PAD, w - 2 * PANEL_PAD
    hh = header(cv, px, py, number, title) + HEADER_GAP
    frame = 0.035 * CARD_SIDE  # the card frame reaches this far outside the card
    image_h = CARD_SIDE + 2 * frame
    tx, tw = px + CARD_SIDE + 0.3, pw - CARD_SIDE - 0.3
    text = fitted(cv, text, tw, image_h)
    text_h = paragraph(cv.scratch(), tx, 0, tw, text, bg)
    cv.image(px, py + hh + frame, CARD_SIDE, image)
    paragraph(cv, tx, py + hh + (image_h - text_h) / 2, tw, text, bg)
    return TITLE_PAD + hh + image_h + PANEL_PAD


def redacted_panel(cv, x, y, w, redacted):
    """The redacted details as dashed one-line cards over the fewest rows that hold them, the widest texts placed first
    into the emptiest row, each row then stretched to the panel width with its cards in instance order."""
    px, py, pw = x + PANEL_PAD, y + TITLE_PAD, w - 2 * PANEL_PAD
    n = len(redacted)
    cy = py + header(cv, px, py, None, "Redacted details", SUBTITLE_SIZE) + HEADER_GAP
    gap = 0.12
    text_w = [cv.measure(d["text"], FONT) + 2 * PAD + 0.02 for d in redacted]  # the packing follows the texts alone
    widths = [cv.measure(d["text"], FONT) + LABEL_GAP + cv.measure(detail_label(d), FONT) + 2 * PAD + 0.02
              for d in redacted]
    n_rows = max(1, math.ceil((sum(text_w) + gap * n) / (pw + gap)))
    while True:  # more rows until every row holds its cards on one line, or every card is alone
        rows, totals = [[] for _ in range(n_rows)], [0.0] * n_rows
        for i in sorted(range(n), key=lambda i: -text_w[i]):
            r = totals.index(min(totals))
            rows[r].append(i)
            totals[r] += text_w[i]
        rows = [row for row in rows if row]
        if n_rows >= n or all(sum(widths[i] for i in row) + gap * (len(row) - 1) <= pw for row in rows):
            break
        n_rows += 1
    h = line_height(FONT) + 2 * PAD
    for row in rows:
        row.sort()
        scale = (pw - gap * (len(row) - 1)) / sum(widths[i] for i in row)
        cx = px
        for i in row:
            detail_card(cv, cx, cy, widths[i] * scale, redacted[i])
            cx += widths[i] * scale + gap
        cy += h + gap
    return cy - gap + PANEL_PAD - y


def dialogue_panel(cv, x, y, w, episode):
    """The chat in two columns, questions left and answers right, each answer level with its question and each question
    under the lower of the two before it, the columns inset so the tails clear the panel edge."""
    px, py, pw = x + PANEL_PAD, y + TITLE_PAD, w - 2 * PANEL_PAD
    cy = py + header(cv, px, py, 3, "Clarification") + 0.06
    usable = pw - 2 * CHAT_INSET - CHAT_GAP
    qw, aw = QUESTION_SHARE * usable, (1 - QUESTION_SHARE) * usable
    columns = {"agent_question": (px + CHAT_INSET, qw, AGENT_BG, "left", "ASSISTANT QUESTION"),
               "user_answer": (px + pw - CHAT_INSET - aw, aw, USER_BG, "right", "USER ANSWER")}
    top = end = cy
    q = 0
    for turn in episode["turns"]:
        if turn["role"] not in columns:
            continue
        bx, bw, bg, side, label = columns[turn["role"]]
        if side == "left":  # a question opens the next exchange
            q += 1
            top = end + 0.06 + 0.17  # 0.17 for the tag over the bubble
        end = max(end, top + paragraph(cv, bx, top, bw, turn["text"], bg, side, f"{label} {q}"))
    if q == 0:
        cv.text(px, cy + 0.25, "The assistant asks no questions.", FONT, MUTED)
        end = cy + 0.5
    return end + PANEL_PAD - y


def judge_panel(cv, x, y, w, redacted, judgment, height=None):
    """The checks in CHECK_COLUMNS columns, the tallest first so that checks of one height share a row, the spare room
    of a given panel height split between the gaps over and under the checks."""
    px, py, pw = x + PANEL_PAD, y + TITLE_PAD, w - 2 * PANEL_PAD
    hh = header(cv, px, py, 5, "Judge")
    verdicts = [(d["check"], bool(judgment["per_detail"].get(str(d["id"]), {}).get("delivered"))) for d in redacted]
    col_gap = 0.14
    cw = (pw - col_gap * (CHECK_COLUMNS - 1)) / CHECK_COLUMNS
    heights = [check_row(cv.scratch(), px, 0, cw, text, ok) for text, ok in verdicts]
    order = sorted(range(len(verdicts)), key=lambda i: -heights[i])  # stable, so equal heights keep instance order
    rows = [order[j:j + CHECK_COLUMNS] for j in range(0, len(order), CHECK_COLUMNS)]
    row_heights = [max(heights[i] for i in row) for row in rows]
    fixed = TITLE_PAD + hh + (sum(row_heights) + 0.06 * (len(rows) - 1))
    gap = max(0.12, (height - fixed) / 2) if height else 0.12
    cy = py + hh + gap
    for row, rh in zip(rows, row_heights):
        for j, i in enumerate(row):
            check_row(cv, px + j * (cw + col_gap), cy, cw, *verdicts[i], height=rh)
        cy += rh + 0.06
    return cy + (gap - 0.06) - y


def visualize(instance, episode, judgment, images, out_base, max_words):
    """images: reference, request and agent, any of them None when missing."""
    details = {d["id"]: d for d in instance["details"]}
    redacted = [details[i] for i in instance["redacted_ids"]]
    cv = Canvas(FIG_W)
    x0, inner_w = MARGIN, FIG_W - 2 * MARGIN
    y = MARGIN

    # the full description and the request side by side with the redacted details under both, in one panel
    w1 = TOP_SPLIT * inner_w
    top = image_text_panel(cv, x0, y, w1, 1, "Full Description", images["reference"], instance["full_spec"], CAPTION_BG)
    image_text_panel(cv, x0 + w1, y, inner_w - w1, 2, "User Request", images["request"], instance["eroded_prompt"],
                     USER_BG)
    overlap = PANEL_PAD - 0.04  # the redacted details start inside the row's bottom padding
    hr = redacted_panel(cv, x0, y + top - overlap, inner_w, redacted)
    cv.box(x0, y, inner_w, top + hr - overlap, "white", LINE, radius=0.14, zorder=0.5)
    y += top + hr - overlap + GAP
    cv.arrow(x0 + inner_w / 2, y - GAP + 0.06, x0 + inner_w / 2, y - 0.06)

    h3 = dialogue_panel(cv, x0, y, inner_w, episode)
    cv.box(x0, y, inner_w, h3, "white", LINE, radius=0.14, zorder=0.5)
    y += h3 + GAP

    # the output with the utility box under it, the judge stretched to end level with the utility box
    w4 = OUTPUT_SHARE * inner_w
    w5 = inner_w - w4 - GAP
    x4, x5 = x0, x0 + w4 + GAP
    cv.arrow(x4 + w4 / 2, y - GAP + 0.06, x4 + w4 / 2, y - 0.06)
    ho = image_text_panel(cv, x4, y, w4, 4, "Final Output", images["agent"],
                          truncated(episode["final_output"], max_words), AGENT_BG)
    ub = utility_box(cv.scratch(), x4, 0, w4, judgment["u_rec"])
    bottom = max(judge_panel(cv.scratch(), x5, y, w5, redacted, judgment), ho + UTILITY_GAP + ub)
    judge_panel(cv, x5, y, w5, redacted, judgment, height=bottom)
    cv.box(x4, y, w4, ho, "white", LINE, radius=0.14, zorder=0.5)
    cv.box(x5, y, w5, bottom, "white", LINE, radius=0.14, zorder=0.5)
    uy = y + bottom - ub
    utility_box(cv, x4, uy, w4, judgment["u_rec"])
    cv.arrow(x4 + w4 + 0.06, y + ho / 2, x5 - 0.06, y + ho / 2)
    cv.arrow(x5 - 0.06, uy + ub / 2, x4 + w4 + 0.06, uy + ub / 2)
    cv.render(out_base, y + bottom + MARGIN)
    print(f"saved {out_base}.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("instance", help="instance id, e.g. test_00337.k3")
    parser.add_argument("eval_dir", nargs="?", default=EXAMPLE_RUN,
                        help="evaluation run holding the episode (default: the paper's patient-trained open-asking run under the patient user)")
    parser.add_argument("--max-words", type=int, default=150,
                        help="cut the final description after this many words (0 shows it whole)")
    parser.add_argument("--out", help="output directory")
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
    name = f"k{config['general']['k']}_{config['agent']['name']}-{run_label(config)}"
    out_dir = args.out or os.path.join(FIGURES, name)
    os.makedirs(out_dir, exist_ok=True)
    visualize(instances[args.instance], episodes[args.instance], judgments[args.instance], images,
              os.path.join(out_dir, f"{args.instance}_visual"), args.max_words)


if __name__ == "__main__":
    main()
