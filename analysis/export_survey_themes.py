import pandas as pd
from pathlib import Path


def main():
    # SETTINGS
    directory = Path("analysis/survey/autointerp")
    themes = pd.read_csv(directory / "themes.csv")
    display_level = 2
    outputs = {"questions": "survey_question_themes.tex", "answers": "survey_answer_themes.tex"}

    # TABULARS: SELECTED LEVEL FROM THE ROOT
    for corpus, filename in outputs.items():
        rows = themes.loc[themes.corpus.eq(corpus)]
        selected_level = rows.level.max() - display_level + 1
        rows = rows.loc[rows.level.eq(selected_level)].sort_values(["cluster", "theme"])
        lines = [
            r"\begin{tabular}{@{}p{\dimexpr0.23\linewidth-\tabcolsep\relax}p{\dimexpr0.77\linewidth-\tabcolsep\relax}@{}}",
            r"\hline",
            r"\textbf{Theme} & \textbf{Description}\tabularnewline[0.2em]",
            r"\hline",
            r"\noalign{\vskip 0.3em}"
        ]
        for row in rows.itertuples():
            description = row.description.removeprefix(row.name + ": ")
            lines.append(r"\raggedright\textbf{%s} & \raggedright %s\tabularnewline[0.1em]" % (escape_latex(row.name), escape_latex(description)))
        lines.extend([r"\hline", r"\end{tabular}", ""])
        (directory / filename).write_text("\n".join(lines))
        print("Saved %s (%s level-%s themes)." % (directory / filename, len(rows), display_level))


def escape_latex(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "“": "``",
        "”": "''",
        "‘": "`",
        "’": "'",
        "—": "---",
        "–": "--"
    }
    return "".join(replacements.get(character, character) for character in text)


if __name__ == "__main__":
    main()
