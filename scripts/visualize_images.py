"""The paper's image comparison, one row per instance:

    [reference] || [user request] || [base model] || [trained model] ([trained model, open-asking])

Each assistant column is titled with its weights over the asking mode of its run, the user it was trained against
belongs in the caption. Under every image the number of details redacted or of questions asked. Images come from
<synthesis_dir>/images/ (generate_images_for_data.py) and <eval_dir>/images/ (generate_images_for_eval.py), a card
reads "missing" where one is absent.

Usage:
    python scripts/visualize_images.py [--naive <eval_dir>] [--fixed <eval_dir>] [--open <eval_dir>]
        [-n 5] [--seed 42] [--instances <id> ...] [--out <dir>]

The runs default to the paper's under paper_results/ (the base weights and the patient-trained checkpoint under the
patient user) and must share the instance set. Writes <instance_id>_figure and figure_all as png and pdf to --out or
scripts/figures/k<k>[-thinking]_naive-<label>_fixed-<label>[_open-<label>].
"""
import argparse
import os
import random
import textwrap

import matplotlib.pyplot as plt

from utils import FIGURES, LINE, MUTED, draw_card, load_eval, load_image, paper_run, run_label

CARD_SIDE, MARGIN, GAP = 2.44, 0.25, 0.52  # inches: card side, page margin, gap between rows
DIVIDERS, DIVIDER_GAP, NARROW_GAP = 3, 0.36, 0.4  # divider lines in the first column gaps, half of such a gap, the rest
TITLE_H, TOP, BOTTOM = 0.86, 0.12, 0.3
SUBTITLE = "#5f6368"
TITLE_FONT = ["Segoe UI", "Arial", "DejaVu Sans"]  # the installed Roboto has no italic face
AGENTS = ["naive-asking", "fixed-asking", "open-asking"]  # the columns of --naive, --fixed and --open
BASE_TITLE, TRAINED_TITLE = "Base Model", "Trained Model"  # the user a checkpoint was trained against stays in the caption
MODES = {"fixed": "Fixed-Asking", "dynamic": "Open-Asking"}  # by the config's asking mode


def plural(n, noun):
    return f"{n} {noun}{'' if n == 1 else 's'}"


def render(titles, rows, out_base, caption=None):
    """Columns titled (name, mode) over rows of (images, subtitles) as cards, the titles on the first row only, divider
    lines in the first DIVIDERS column gaps, an optional caption (lines) under the last row. Writes png and pdf."""
    n, card = len(rows), CARD_SIDE
    xs = [MARGIN]
    for k in range(len(titles) - 1):
        xs.append(xs[-1] + card + (2 * DIVIDER_GAP if k < DIVIDERS else NARROW_GAP))
    fig_w = xs[-1] + card + MARGIN
    caption_h = 0.16 * len(caption) + 0.5 if caption else 0.0
    fig_h = TOP + TITLE_H + n * card + (n - 1) * GAP + caption_h + BOTTOM
    fig = plt.figure(figsize=(fig_w, fig_h))
    face = dict(ha="center", va="baseline", fontweight="bold", fontstyle="italic", fontfamily=TITLE_FONT)
    for i, (images, subtitles) in enumerate(rows):
        y = fig_h - TOP - TITLE_H - i * (card + GAP) - card
        for x, image, (name, mode), sub in zip(xs, images, titles, subtitles):
            ax = fig.add_axes([x / fig_w, y / fig_h, card / fig_w, card / fig_h])
            draw_card(ax, image)
            if i == 0:  # the name on a shared baseline, the mode just over the frame
                ax.text(0.5, 1.035 + 0.42 / card, name, fontsize=23, color="black", transform=ax.transAxes, **face)
                if mode:
                    ax.text(0.5, 1.035 + 0.11 / card, mode, fontsize=17, color=SUBTITLE, transform=ax.transAxes, **face)
            ax.text(0.5, -0.035 - 0.05 / card, sub, ha="center", va="top", fontsize=15, color=SUBTITLE,
                    transform=ax.transAxes)
    for k in range(min(DIVIDERS, len(xs) - 1)):
        x = (xs[k] + card + DIVIDER_GAP) / fig_w
        fig.add_artist(plt.Line2D([x, x], [(BOTTOM + caption_h) / fig_h, (fig_h - TOP) / fig_h], color=LINE,
                                  linewidth=1.2, transform=fig.transFigure))
    if caption:
        fig.text(0.5, (BOTTOM + caption_h - 0.45) / fig_h, "\n".join(caption), ha="center", va="top", fontsize=7,
                 color=MUTED)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_base}.{ext}", dpi=200, facecolor="white")
    plt.close(fig)
    print(f"saved {out_base}.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--naive", default=paper_run("fixed-asking", "patient", "base_model"),
                        help="evaluation run of the base weights, either mode (default: the paper's, fixed-asking under the patient user)")
    parser.add_argument("--fixed", default=paper_run("fixed-asking", "patient", "patient_trained"),
                        help="evaluation run of a trained checkpoint in fixed-asking mode (default: the paper's patient-trained run)")
    parser.add_argument("--open", default=paper_run("open-asking", "patient", "patient_trained"),
                        help="the same in open-asking mode (default: the paper's patient-trained run), pass '' to leave the column out")
    parser.add_argument("-n", type=int, default=5, help="number of random instances, ignored with --instances")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--instances", nargs="+", metavar="ID", help="instance ids to render, the rows in this order")
    parser.add_argument("--out", help="output directory")
    args = parser.parse_args()

    eval_dirs = {key: path for key, path in zip(AGENTS, (args.naive, args.fixed, args.open)) if path}
    evals = {key: load_eval(path) for key, path in eval_dirs.items()}
    instances, _, _, config, instances_dir = evals["naive-asking"]
    common = set.intersection(*(set(episodes) for _, episodes, _, _, _ in evals.values()))
    if not common:
        raise SystemExit("the evaluation runs share no instance, check that they use the same synthesis directory and split")
    if args.instances:
        missing = [i for i in args.instances if i not in common]
        if missing:
            raise SystemExit(f"not in every evaluation run: {', '.join(missing)}")
        chosen = args.instances
    else:
        chosen = random.Random(args.seed).sample(sorted(common), min(args.n, len(common)))
    configs = {key: evals[key][3] for key in eval_dirs}
    labels = {key: run_label(cfg) for key, cfg in configs.items()}
    name = f"k{config['general']['k']}{'-thinking' if config['model'].get('thinking') else ''}_" + \
        "_".join(f"{key.split('-')[0]}-{labels[key]}" for key in eval_dirs)
    out_dir = args.out or os.path.join(FIGURES, name)
    os.makedirs(out_dir, exist_ok=True)
    data_images = os.path.join(instances_dir, "images")
    titles = [("Reference", ""), ("User Request", "")] + \
        [(BASE_TITLE if labels[key] == "base" else TRAINED_TITLE, MODES[cfg["evaluation"]["agent"]["mode"]])
         for key, cfg in configs.items()]

    rows = []
    for instance_id in chosen:
        reference = load_image(os.path.join(data_images, f"{instance_id}_enhanced.jpg")) \
            or load_image(os.path.join(data_images, f"{instance_id}_original.jpg"))
        images = [reference, load_image(os.path.join(data_images, f"{instance_id}_request.jpg"))]
        subtitles = [plural(len(instances[instance_id]["redacted_ids"]), "detail") + " redacted", "no questions asked"]
        for key, (_, episodes, _, _, _) in evals.items():
            images.append(load_image(os.path.join(eval_dirs[key], "images", f"{instance_id}.jpg")))
            subtitles.append(plural(episodes[instance_id]["n_questions"], "question") + " asked")
        if any(image is None for image in images):
            print(f"images missing for {instance_id} (run scripts/generate_images_for_data.py and _for_eval.py first)")
        caption = textwrap.wrap(f"User request (eroded): {instances[instance_id]['eroded_prompt']}", 150 + 20 * len(evals))
        render(titles, [(images, subtitles)], os.path.join(out_dir, f"{instance_id}_figure"), caption)
        rows.append((images, subtitles))
    if rows:
        render(titles, rows, os.path.join(out_dir, "figure_all"))


if __name__ == "__main__":
    main()
