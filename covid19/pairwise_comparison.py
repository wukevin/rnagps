"""
Code for doing pariwise comparions of two different folders of predictions

Example usages:
python pairwise_comparison.py covid19_localization_full/single_preds covid19_localization_full_targeted_ablation/single_preds -p covid19_localization_full_targeted_ablation/rf_ablation_pvalues.png
python pairwise_comparison.py gru_covid19_localization_full/single_preds gru_covid19_localization_full_targeted_ablation/single_preds -p gru_covid19_localization_full_targeted_ablation/gru_ablation_pvalues.png
"""

import os
import sys
import glob
import logging
import argparse
from typing import *
import collections

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats import multitest

PLOTTED_ORFS = [
    "orf1ab",
    "s",
    "orf3a",
    "e",
    "m",
    "orf6",
    "orf7a",
    "orf7b",
    "orf8",
    "n",
    "orf10",
]

def infer_reference_folder(path:str) -> str:
    """
    Given a path try to infer a reference folder
    """
    # https://stackoverflow.com/questions/3167154/how-to-split-a-dos-path-into-its-components-in-python
    orig_path = path
    path = os.path.abspath(path)
    path_parts = []
    while True:
        path, tail = os.path.split(path)
        if tail:
            path_parts.append(tail)
        else:
            if path:
                path_parts.append(path)
            break
    path_parts.reverse()

    bname = ""
    for i, part in enumerate(path_parts):
        if part.startswith("gru_covid19_localization_"):
            bname = "gru_covid19_localization_full"
        elif part.startswith("covid19_localization_"):
            bname = "covid19_localization_full"

    if bname:
        retval = os.path.join(*path_parts[:i-1], bname, "single_preds")
        logging.debug(f"Inferred reference path: {retval}")
        assert os.path.isdir(retval), f"Inferred path is not a dir: {retval}"
    else:
        retval = ""
        logging.warning(f"Could not infer reference dir from: {orig_path}")
    return retval

def get_file_dict(directory:str) -> Dict[str, str]:
    """Return a dictionary files"""
    assert os.path.isdir(directory)
    fnames = sorted(glob.glob(os.path.join(directory, "*.csv")))
    retval = {os.path.splitext(os.path.basename(fname))[0]: os.path.abspath(fname) for fname in fnames}
    assert retval
    return retval

def get_aggregated_preds(fnames:List[str], min_count:int=5) -> Dict[str, pd.DataFrame]:
    """
    Given files, read and aggregate into a series of dataframes
    Each dataframe contains all the localization predictions relevant to a gene (n_obs x 8)
    """
    accum = collections.defaultdict(dict)
    for fname in sorted(fnames):
        df = pd.read_csv(fname, index_col=0)
        df = df.iloc[[i for i, idx_name in enumerate(df.index) if idx_name in PLOTTED_ORFS], :]
        for idx, row in df.iterrows():
            accum[idx.lower()][os.path.splitext(os.path.basename(fname))[0]] = row
    if min_count > 0:
        accum = {k: v for k, v in accum.items() if len(v) >= min_count}

    # Consoldate
    retval = {k: pd.DataFrame(v).T for k, v in accum.items()}
    return retval

def compare_preds(dict_of_dfs1:Dict[str, pd.DataFrame], dict_of_dfs2:Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Compares the predictions, returning a dataframe of FDR adjusted p-values with sign denoting direction of change
    """
    # First level of dict maps to sgRNA/gene
    assert dict_of_dfs1.keys() == dict_of_dfs2.keys(), f"Got different keys of length {len(dict_of_dfs1)} and {len(dict_of_dfs2)}"
    accum = {}  # accumulates pvalues
    accum_delta = {}  # accumulates deltas
    for k in dict_of_dfs1.keys():
        preds1 = dict_of_dfs1[k]
        preds2 = dict_of_dfs2[k]
        idx = preds1.index.intersection(preds2.index)
        preds1 = preds1.loc[idx]
        preds2 = preds2.loc[idx]
        _t_stats, p_vals = stats.ttest_rel(preds1, preds2, axis=0)
        delta = np.mean(preds2 - preds1, axis=0)
        accum[k] = pd.Series(p_vals, index=preds1.columns)
        accum_delta[k] = delta

    deltas = pd.DataFrame(accum_delta).T
    retval_uncorr = pd.DataFrame(accum).T
    _reject, pvals_corrected, _alphacSidak, _alphacBonf = multitest.multipletests(
        retval_uncorr.values.flatten(),
        method='holm',  # More powerful than bonferroni, no independence assumption
    )
    retval = pd.DataFrame(
        np.sign(deltas.values) * pvals_corrected.reshape(retval_uncorr.shape),
        index=retval_uncorr.index,
        columns=retval_uncorr.columns,
    )

    return retval

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("dir1", type=str, help="First directory of predictions to compare - 'control'")
    parser.add_argument("dir2", type=str, help="Second directoiry of predictions to compare - 'experimental condition'")
    parser.add_argument("--plot", "-p", type=str, default="", help="File to save heatmap plot to")
    parser.add_argument("--fname", "-f", type=str, default="", help="File to save pvalues to")
    return parser

def do_comparison(dir1:str, dir2:str, fname:str="", plotname:str=""):
    fdict_1 = get_file_dict(dir1)
    fdict_2 = get_file_dict(dir2)
    assert fdict_1.keys() == fdict_2.keys(), f"Got different keys of length {len(fdict_1)} and {len(fdict_2)}"

    preds_1 = get_aggregated_preds(fdict_1.values())
    preds_2 = get_aggregated_preds(fdict_2.values())

    pvalue_df = compare_preds(preds_1, preds_2)
    pvalue_sig = np.abs(pvalue_df.values) <= 0.05  # Booleans
    pvalue_sign_sig = pd.DataFrame(
        np.sign(pvalue_df.values) * pvalue_sig,
        index=pvalue_df.index,
        columns=pvalue_df.columns,
    )

    if fname:
        pvalue_df.to_csv(fname)
    if plotname:
        raise NotImplementedError

def main():
    """Run script"""
    parser = build_parser()
    args = parser.parse_args()

    do_comparison(
        dir1=args.dir1,
        dir2=args.dir2,
        fname=args.fname,
        plotname=args.plot,
    )

if __name__ == "__main__":
    main()

