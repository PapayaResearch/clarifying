"""Merge per-agent episode reports into one comparison table.

Usage:
    python -m uaug.analysis.compare results/docci/episodes/<model>            # newest run per agent
    python -m uaug.analysis.compare <run_dir> <run_dir> ...                   # explicit run dirs
"""

import argparse
import glob
import json
import os


def collect_reports(paths):
    """Newest report per agent across the given episode-run directories (or trees)."""
    reports = {}
    for base in paths:
        for path in sorted(glob.glob(os.path.join(base, "**", "report.*.json"), recursive=True)):
            with open(path, "r", encoding="utf-8") as f:
                report = json.load(f)
            key = (report.get("agent"), report.get("user"), report.get("split"))
            reports[key] = (report, path)  # sorted order -> newest run wins
    return reports


def main(args):
    reports = collect_reports(args.paths)
    if not reports:
        print("No report.*.json found under given paths.")
        return

    columns = ["agent", "user", "split", "n_episodes", "u_rec_mean", "u_rec_ci95", "eps_c_hat_mean", "eps_e_hat_mean", "preserved_mean", "n_questions_mean", "planning_gap_mean", "full_recovery_rate"]
    header = ["agent", "user", "split", "n", "u_rec", "ci95", "eps_C", "eps_E", "preserved", "questions", "gap", "full_rec"]

    def fmt(report, column):
        value = report.get(column, "")
        if column == "u_rec_ci95" and isinstance(value, list):
            return f"[{value[0]}, {value[1]}]"
        return "" if value is None else str(value)

    rows = [[fmt(report, c) for c in columns] for report, _ in sorted(reports.values(), key=lambda rp: -(rp[0].get("u_rec_mean") or 0))]

    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(header)]
    print("  ".join(h.ljust(w) for h, w in zip(header, widths)))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(v.ljust(w) for v, w in zip(row, widths)))

    if args.verbose:
        for _, path in sorted(reports.values(), key=lambda rp: rp[0].get("agent") or ""):
            print(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="Episode run directories (or parent trees) to scan")
    parser.add_argument("-v", "--verbose", action="store_true", help="Also print source report paths")
    main(parser.parse_args())
