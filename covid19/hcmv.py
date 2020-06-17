"""
Code for running HCMV localization sanity check

beta2.7 has mitochondrial localization
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4435936/pdf/nihms-689135.pdf

Genome
https://www.ncbi.nlm.nih.gov/nuccore/NC_006273
"""

import os
import sys
import logging

import numpy as np
import pandas as pd

import tqdm

from covid19 import (
    query_genbank,
    fetch_genbank,
    pred_feature_dict,
    mean_sd_missing_vals,
    genbank_to_feature_dict,
    rev_comp,
    featurize,
    RNAGPS,
)
import baseline
import plotting

SRC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rnagps",
)
assert os.path.isdir(SRC_DIR)
sys.path.append(SRC_DIR)
import model_utils
import data_loader

HCMV_ACCESSION = "NC_006273"


def analyze_hcmv_beta27():
    """
    [2488:5000](-) full

    [4994:5000](-) TATA box leader, really 4969:5000
    [2488:4969](-) body, really 2510:4969
    [2504:2510](-) polyA, really 2488:2510
    """
    hcmv_record = fetch_genbank(HCMV_ACCESSION)
    hcmv_features = genbank_to_feature_dict(hcmv_record)

    # Pull out all seqfeatures pertaining to RNA2.7
    relevant_seqfeatures = []
    for feature_type in hcmv_features.keys():
        for gene in hcmv_features[feature_type]:
            if ("gene" in gene.qualifiers and "RNA2.7" in gene.qualifiers["gene"]) or (
                "note" in gene.qualifiers and "RNA2.7" in gene.qualifiers["note"][0]
            ):
                print(gene)
                relevant_seqfeatures.append(gene)

    genome_sequence = str(hcmv_record.seq)

    # Based on manual examination/annotation
    five_utr_seq = rev_comp(genome_sequence[4969:5000])
    print("Five prime", five_utr_seq)
    cds_seq = rev_comp(genome_sequence[2510:4969])
    print("Coding", cds_seq)
    three_utr_seq = rev_comp(genome_sequence[2488:2510])
    print("Three prime", three_utr_seq)

    print(
        f"Total length: {len(five_utr_seq) + len(cds_seq) + len(three_utr_seq)} bases"
    )

    featurized = featurize(
        [five_utr_seq, cds_seq, three_utr_seq], [False, False, False]
    )
    preds = model_utils.list_preds_to_array_preds(
        RNAGPS.predict_proba(featurized)
    ).flatten()

    human_pos_baseline = baseline.Baseline(
        baseline.HUMAN_BASELINE_FILE, baseline.HUMAN_BASELINE_LABELS_FILE
    )
    ranks = pd.DataFrame(
        human_pos_baseline.score_vec(preds),
        index=data_loader.LOCALIZATION_FULL_NAME_DICT.values(),
        columns=["$\\beta$2.7"],
    ).T
    print(ranks)

    # Plot heatmap
    plotting.heatmap(
        ranks,
        annot=True,
        val_range=(0, 1),
        cbar=True,
        figsize=(7, 1),
        fname="hcmv_localizations.pdf",
        title="Human cytomegalovirus rank (vs. localized human transcripts)",
    )


if __name__ == "__main__":
    analyze_hcmv_beta27()
