"""
Code to plot barplot of panther results
"""

import os
import sys
from typing import *

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


def sanitize_label_lenth(l: List[str]) -> List[str]:
    """
    Given a list of labels, sanitize them to remove items with too large length
    """

    def insert_newlines(line: str, limit: int) -> str:
        lines = []
        curr_line = []
        for word in line.split():
            word = word.strip()
            if len(" ".join(curr_line + [word])) <= limit:
                curr_line.append(word)
            else:
                lines.append(" ".join(curr_line))
                curr_line = [word]
        lines.append(r"{}".format(" ".join(curr_line)))
        return r"{}".format("\n".join(lines))

    cutoff = 55
    l_clean = [insert_newlines(line, cutoff) for line in l]
    return l_clean


def main():
    panther_results = pd.read_csv(
        os.path.join(os.getcwd(), "panther_analysis.txt"),
        # sys.argv[1],
        delimiter="\t",
        skiprows=7,
        names=[
            "term",
            "ref",
            "count",
            "expected",
            "direction",
            "fold_enrichment",
            "p_raw",
            "p_adj",
        ],
    )
    print(panther_results)

    panther_results.drop(
        index=panther_results.index[np.where(panther_results["direction"] != "+")],
        inplace=True,
    )
    panther_results["log_p_adj"] = -np.log10(panther_results["p_adj"])
    panther_results.sort_values(by="log_p_adj", ascending=False, inplace=True)
    panther_results["term_clean"] = [
        idx.split("(")[0].strip() for idx in panther_results["term"]
    ]
    # panther_results["term_clean"] = sanitize_label_lenth(panther_results["term_clean"])
    print(panther_results)

    top_n = 15
    panther_results = panther_results.iloc[:top_n]  # Plot top N
    panther_results = panther_results.iloc[::-1]

    fig, ax = plt.subplots(figsize=(7, 5))
    panther_results.plot.barh(
        x="term_clean",
        y="log_p_adj",
        ax=ax,
        rot=0,
        color="grey",
        legend=False,
        width=0.7,
    )
    ax.set(
        ylabel="",
        xlabel="$-log_{10}$ adjusted $p$-value",
        title=f"Top {top_n} significantly enriched reactome terms",
    )
    fig.savefig("panther_analysis.pdf", bbox_inches="tight", transparent=True)


if __name__ == "__main__":
    main()
