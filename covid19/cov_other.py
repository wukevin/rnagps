"""
Code for looking at other coronaviurses the same way we look at COVID-19

The primary objective here is to figure out if the stronger mito matrix
and nucleolus localization are a commmon trait amongst coronaviruses
"""

import os
import sys
import logging

import numpy as np
import pandas as pd

import tqdm

from covid19 import query_genbank, fetch_genbank, pred_feature_dict, mean_sd_missing_vals

from baseline import COV_BASELINE_QUERY, COV_CONSERVED

def analyze_general_corona(outdir="general_corona_localization_full"):
    """
    Run analysis
    """
    record_ids = query_genbank(COV_BASELINE_QUERY)
    records = (fetch_genbank(r) for r in record_ids)
    coronavirus_preds = {}
    for rec in tqdm.tqdm(records, total=len(record_ids)):
        try:
            local = pred_feature_dict(rec)
            local = local.loc[[i for i in local.index if i in COV_CONSERVED], :]
            if local.empty:
                continue
            assert rec.name not in coronavirus_preds
            coronavirus_preds[rec.name] = local
        except (AssertionError, ValueError, KeyError) as _err:
            continue

    assert coronavirus_preds
    logging.info(f"Number of general coronavirus records with predictions: {len(coronavirus_preds)}")

    preds_mean, preds_sd = mean_sd_missing_vals(coronavirus_preds.values())

    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    preds_mean.to_csv(os.path.join(outdir, "mean.csv"))
    preds_sd.to_csv(os.path.join(outdir, "sd.csv"))

if __name__ == "__main__":
    analyze_general_corona()

