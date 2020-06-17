"""
Code for plotting violinplot
"""
import os
import sys

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from baseline import load_apex_full_dataset

SRC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rnagps",
)
assert os.path.isdir(SRC_DIR)
sys.path.append(SRC_DIR)
import data_loader


def main():
    mito_transcripts = pd.Index(pd.read_csv(sys.argv[1], squeeze=True, names=["tx"]))

    apex_dset = load_apex_full_dataset()
    full_table = apex_dset.full_deseq_table.copy(deep=True)
    full_table.index = [apex_dset.get_ith_trans_name(i) for i in range(len(apex_dset))]
    # print(full_table)

    mito_transcript_table = full_table.loc[mito_transcripts]
    # print(mito_transcript_table)

    mito_logfc_table = mito_transcript_table.loc[
        :,
        [k + "_log2FoldChange" for k in data_loader.LOCALIZATION_FULL_NAME_DICT.keys()],
    ]
    mito_logfc_table.columns = data_loader.LOCALIZATION_FULL_NAME_DICT.values()
    print(mito_logfc_table)

    with plt.style.context("seaborn-notebook", after_reset=True):
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.violinplot(
            mito_logfc_table.T, widths=0.8, showextrema=False, bw_method=0.8,
        )
        ax.boxplot(
            mito_logfc_table.T,
            widths=0.4,
            medianprops=dict(color="k"),
            showfliers=False,
        )
        ax.axhline(0.0, color="grey", linestyle="--", alpha=0.5)
        ax.set_xticklabels(mito_logfc_table.columns, rotation=45, ha="right")
        ax.set(
            ylabel="Enrichment ($log_2$ fold change)",
            title=f"Nuclear-encoded mito-enriched genes (n={mito_logfc_table.shape[0]})",
            ylim=(-5, 7.5),
        )
        fig.savefig("mito_violinplot.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
