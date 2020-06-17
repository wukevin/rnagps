"""
Code for generating the gene lists that go into panther
"""

import os
import sys
from typing import List

import pandas as pd

from baseline import load_apex_full_dataset


def generate_panther_input() -> List[str]:
    x = load_apex_full_dataset()
    print(x.full_deseq_table)

    mito_genes = pd.Index(
        [
            x.full_deseq_table.index[i]
            for i in range(len(x))
            if x.get_ith_labels(i)[:, 2]
            and "MT" not in x.get_ith_trans_chrom(i).upper()
            and "pseudogene" not in x.get_ith_trans_type(i)
        ]
    )
    mito_table = x.full_deseq_table.loc[
        mito_genes,
        ["gene_name", "gene_type", "Mito_tpm", "Mito_log2FoldChange", "Mito_padj"],
    ]
    mito_table.sort_values(by="Mito_padj", inplace=True)
    print(mito_table)
    mito_top_genes = set()
    # Gather the top 100 genes
    for gene_name in mito_table["gene_name"]:
        # This code is fairly naive but works
        if len(mito_top_genes) >= 100:
            break
        if gene_name not in mito_top_genes:
            mito_top_genes.add(gene_name)
    with open("rnagps_mito_genes.txt", "w") as sink:
        sink.write("\n".join(mito_top_genes) + "\n")
    return mito_top_genes


if __name__ == "__main__":
    generate_panther_input()
