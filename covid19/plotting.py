"""
Code for plotting COVID-19 analysis results
"""

import os
import sys
import logging
import multiprocessing
import json
from typing import *
import collections

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from covid19 import LOCALIZATION_VALID_CUTOFFS, PREFIXES_TO_ABLATIONS
from covid19 import get_species, query_genbank, fetch_genbank
import baseline
import pairwise_comparison as pairwise

DPI = 600

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

def piechart(items:List[str], fname:str="", **axkwargs):
    """
    Plot a piechart of the items, which is a list of strings *with duplicates*
    """
    cnt = collections.Counter(items)
    fig, ax = plt.subplots(dpi=DPI)
    ax.pie(
        x=list(cnt.values()),
        labels=cnt.keys()
    )
    if axkwargs:
        ax.set(**axkwargs)
    if fname:
        fig.savefig(fname, bbox_inches='tight')
    return fig

def heatmap(mat:pd.DataFrame, annot=True, annot_fmt:str=".2f", annot_kws={"size": 13}, val_range:Tuple[int, int]=None, zero_center:bool=False, cbar:bool=False, figsize=(7.0, 5.0), fname:str="", **axkwargs):
    """
    Plot a heatmap of the matrix
    """
    if not isinstance(mat, pd.DataFrame):
        mat = pd.DataFrame(mat)
    if mat.empty:
        logging.warning(f"Got empty input - skipping plotting")
        return None
    fig, ax = plt.subplots(dpi=DPI, figsize=figsize)
    if val_range:
        vmin, vmax = val_range
    else:
        vmin = -np.nanmax(np.abs(mat.values)) if zero_center else np.nanmin(mat.values)
        vmax = np.nanmax(np.abs(mat.values)) if zero_center else np.nanmax(mat.values)

    sns.heatmap(
        mat,
        vmin=vmin,
        vmax=vmax,
        annot=annot,
        annot_kws=annot_kws,
        fmt=annot_fmt,
        cmap='coolwarm',
        cbar=cbar,
        ax=ax,
    )
    axis_fontsize = 16
    ax.set_yticklabels(mat.index, rotation=0, ha='right', size=axis_fontsize, weight='normal')
    ax.set_xticklabels(mat.columns, rotation=45, ha='right', size=axis_fontsize, weight='normal')
    if axkwargs:
        ax.set(**axkwargs)
    if fname:
        fig.savefig(fname, bbox_inches='tight')
    return fig

def group_idx_to_ranges(idx) -> List[Tuple[int, int]]:
    """Return a list of tuples of (start, stop)"""
    retval = []
    curr = [idx[0], idx[0] + 1]
    for i in idx[1:]:
        if i == curr[1]:
            curr[1] += 1
        else:
            retval.append(tuple(curr))
            curr = [i, i + 1]
    retval.append(tuple(curr))
    return retval

def highlighted_lineplot(vals:np.ndarray, highlight_idx:np.ndarray, highlight_str:str="highlight", figsize=(8, 6), fname:str="", include_hit_cov:bool=True, **axkwargs) -> mpl.figure.Figure:
    """
    Plot a lineplot
    """
    assert np.all(highlight_idx) < len(vals)
    assert len(highlight_idx) == len(set(highlight_idx))

    coverage = len(highlight_idx) / vals.size
    if include_hit_cov:
        highlight_str = highlight_str.strip() + f" (base coverage={coverage:.4f})"

    fig, ax = plt.subplots(dpi=DPI, figsize=figsize)
    ax.plot(np.arange(len(vals)), vals)
    for i, interval in enumerate(group_idx_to_ranges(highlight_idx)):
        ax.plot(np.arange(*interval), vals[np.arange(*interval)], color='blue', label=highlight_str if i == 0 else None)
    ax.legend()

    ax.set(**axkwargs)

    if fname:
        fig.savefig(fname, bbox_inches='tight')
    return fig

def highlighted_text(txt:str, highlight_idx:np.ndarray, fname:str="") -> mpl.figure.Figure:
    """Display the text with highlight"""
    raise NotImplementedError

def general_log(x):
    return np.sign(x) * np.log1p(np.abs(x))

### General plotting functions live above this line. Below is code
### used to generate specific plots we need

def plot_5_3_localization(fname:str="") -> mpl.figure.Figure:
    """
    Plot the figure looking at localization of the 5' leader and 3' UTR alone
    """
    human_pos_baseline = baseline.Baseline(baseline.HUMAN_BASELINE_FILE, baseline.HUMAN_BASELINE_LABELS_FILE)
    # Load in appropriate data
    only5_localization_fname = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "covid19_localization_only5", "mean.csv",
    )
    only5_localization = pd.read_csv(only5_localization_fname, index_col=0)

    only_cds_localization_fname = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "covid19_localization_onlycds", "mean.csv",
    )
    only_cds_localization = pd.read_csv(only_cds_localization_fname, index_col=0)

    only3_localization_fname = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "covid19_localization_only3", "mean.csv",
    )
    only3_localization = pd.read_csv(only3_localization_fname, index_col=0)

    probs = pd.DataFrame(
        {"5' leader": only5_localization.iloc[0], "CDS (mean)": only_cds_localization.mean(axis=0), "3' UTR": only3_localization.iloc[0]}
    ).T
    plot_df = pd.DataFrame(
        human_pos_baseline.score_mat(probs),
        columns=probs.columns,
        index=probs.index,
    )

    fig = heatmap(
        plot_df,
        annot=True,
        val_range=(0, 1),
        fname=fname,
        figsize=(7, 2),
        cbar=False,
        title="Per-segment rank (vs. localized human transcripts)",
    )
    return fig

def do_plotting(prefix:str, baseline_obj, addtl_prefix:str="", addtl_save_prefix:str="", title:str="") -> None:
    """
    Plots heatmaps based on how values compare to baseline values
    Addtl prefix should probably end in a underscore
    This is way more interpretable
    """
    def filter_df(df):
        """Filter df to contain only the orfs we plot"""
        idx = [i for i, gene in enumerate(df.index) if gene in PLOTTED_ORFS]
        retval = df.iloc[idx, :]
        return retval

    local_means = pd.read_csv(f"{prefix}/{addtl_prefix}mean.csv", index_col=0)
    local_sds = pd.read_csv(f"{prefix}/{addtl_prefix}sd.csv", index_col=0)

    local_means_scored = filter_df(pd.DataFrame(
        baseline_obj.score_mat(local_means),
        index=local_means.index,
        columns=local_means.columns
    ))
    local_sds_scored = filter_df(pd.DataFrame(
        baseline_obj.score_mat(local_sds),
        index=local_sds.index,
        columns=local_sds.columns,
    ))
    # drop NaN rows
    local_means_scored.dropna(axis=0, how='all', inplace=True)
    local_sds_scored.dropna(axis=0, how='all', inplace=True)
    # print(local_means_scored)

    fig = heatmap(
        local_means_scored,
        title=title,
        fname=f"{prefix}/{addtl_save_prefix}compartment_heatmap.pdf",
        val_range=(0, 1),
    )
    plt.close(fig)
    fig = heatmap(
        local_sds,
        title=title,
        fname=f"{prefix}/{addtl_save_prefix}compartment_heatmap_sd.pdf",
    )
    plt.close(fig)

    # Create pairwise comparison heatmap

def main():
    # RNA GPS random forest based baselines
    human_baseline = baseline.Baseline(baseline.HUMAN_BASELINE_FILE, "")
    human_pos_baseline = baseline.Baseline(baseline.HUMAN_BASELINE_FILE, baseline.HUMAN_BASELINE_LABELS_FILE)
    cov_baseline = baseline.CoronavirusBaseline(baseline.COV_BASELINE_FILE)

    # GRU baselines
    gru_human_baseline = baseline.Baseline(baseline.HUMAN_BASELINE_FILE_GRU, "")
    gru_human_pos_baseline = baseline.Baseline(baseline.HUMAN_BASELINE_FILE_GRU, baseline.HUMAN_BASELINE_LABELS_FILE)
    gru_cov_baseline = baseline.CoronavirusBaseline(baseline.COV_BASELINE_FILE_GRU)

    if len(sys.argv) > 1:  # Handle specific arguments
        for prefix in sys.argv[1:]:
            do_plotting(prefix, baseline_obj=human_baseline, title="Percentile (human transcripts)")
    else:  # Run all plotting by default
        # Create plot of constituent species in coronavirus baseline
        with open("baselines/cov_metadata.json") as source:
            cov_baseline_metadata = json.load(source)
        species_counts = collections.Counter(cov_baseline_metadata['species'])
        species_df = pd.DataFrame(
            species_counts.values(),
            index=species_counts.keys(),
            columns=['Count'],
        )
        species_sars = [i for i in species_df.index if i.startswith("SARS") or i.startswith("Severe acute respiratory syndrome")]
        species_sars_df = pd.DataFrame((species_df.loc[species_sars]).sum(axis=0))
        species_sars_df.index = ['SARS coronavirus (2003)']
        species_sars_df.columns = ['Count']
        species_df = species_df.drop(index=species_sars).append(species_sars_df)
        species_df['Proportion'] = species_df.values / np.sum(species_df.values)
        species_df.to_csv("baselines/baseline_species.csv", float_format="%.4f")

        # Create plot of localization of 5' and 3' regions alone
        plot_5_3_localization("localization_5_cds_3.pdf")

        # Create plot of where ablations occur
        three_prime_ablations = []
        motif_ablation_json_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "covid19_localization_full_targeted_ablation", "ablations.json",
        )
        with open(motif_ablation_json_file, 'r') as source:
            motif_ablation_positions = json.load(source)
        for genome_name, genome_dict in motif_ablation_positions.items():
            three_prime_ablations.append(genome_dict["3'UTR"])
        # Right justify
        three_prime_ablations_agg = np.zeros((len(three_prime_ablations), max([len(a) for a in three_prime_ablations])), dtype=bool)
        for i, row in enumerate(three_prime_ablations):
            three_prime_ablations_agg[i, -len(row):] = row
            # assert not np.any(three_prime_ablations_agg[i, :len(row)-1])
        three_prime_ablation_prob = np.mean(three_prime_ablations_agg, axis=0)
        heatmap(
            mat=pd.DataFrame(
                np.atleast_2d(three_prime_ablation_prob),
                # index=["3'UTR"],
                # columns=np.arange(len(three_prime_ablation_prob)),
            ),
            annot=False,
            val_range=(0, 1),
            cbar=False,
            figsize=(6, 0.6),
            fname="covid19_localization_full_targeted_ablation/ablation_regions.pdf",
            title="Locations of ablations made to 3' UTR",
            xlabel="Base in 3' UTR (5' to 3')",
            xticks=[],
            yticks=[],
            ylabel="",
        )

        # Create plots of heatmaps of localization rank scores
        for prefix in ['togaviridae_localization_full', 'general_corona_localization_full'] + list(PREFIXES_TO_ABLATIONS):
            print(prefix)
            if prefix.startswith("gru_"):
                blines = [gru_human_baseline, gru_human_pos_baseline, gru_cov_baseline]
            else:
                blines = [human_baseline, human_pos_baseline, cov_baseline]
            zipped_args = zip(
                blines,
                ["baseline_human_", "baseline_humanpos_", "baseline_cov_"],
                ["all human transcripts", "localized human transcripts", "human coronavirus transcripts"],
            )
            for bline, save_prefix, title_blurb in zipped_args:
                title_blurb += ", GRU" if prefix.startswith("gru_") else ""
                try:
                    do_plotting(
                        prefix,
                        baseline_obj=bline,
                        addtl_save_prefix=os.path.basename(prefix).split("_")[-1] + "_" + save_prefix,
                        title=f"Rank (vs. {title_blurb})",
                    )
                except FileNotFoundError:
                    logging.warning(f"Plotting failed for {prefix}")
                    continue

            single_preds_dir = os.path.abspath(os.path.join(prefix, "single_preds"))
            reference_preds_dir = pairwise.infer_reference_folder(single_preds_dir)
            if not reference_preds_dir:
                continue
            if single_preds_dir != reference_preds_dir:
                fdict_1 = pairwise.get_file_dict(reference_preds_dir)
                fdict_2 = pairwise.get_file_dict(single_preds_dir)
                preds_ref = pairwise.get_aggregated_preds(fdict_1.values())
                preds_exp = pairwise.get_aggregated_preds(fdict_2.values())

                pvalue_df = pairwise.compare_preds(preds_ref, preds_exp)
                pvalue_sig = np.abs(pvalue_df.values) <= 0.05
                pvalue_sign_sig = pd.DataFrame(
                    np.sign(pvalue_df.values) * pvalue_sig,
                    index=pvalue_df.index,
                    columns=pvalue_df.columns,
                )
                comp_pval_title = "Changes to localization"
                if prefix.endswith("ablation"):
                    comp_pval_title += " upon ablation"
                fig = heatmap(
                    pvalue_sign_sig,
                    annot=np.abs(pvalue_df.values),
                    annot_fmt=".2g",
                    annot_kws={},
                    title=comp_pval_title,
                    fname=f"{prefix}/comparison_pvalues.pdf",
                )
                plt.close(fig)

if __name__ == "__main__":
    main()

