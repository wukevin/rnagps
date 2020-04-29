"""
Code to do windowed ablations

Example MEME call after we've identified high-impact sequences
- There are 402 total sequences that we do window ablation across - hence maxsites 402
- We want to have motifs present in at least half of them - the 5' and 3' UTRs should be fairly well conserved
meme aggregated_3utr.fa -dna -oc aggregated_3utr_meme -mod anr -nmotifs 10 -minw 6 -maxw 15 -objfun classic -minsites 100 -maxsites 402 -markov_order 1 -evt 1e-6
meme aggregated_5utr.fa -dna -oc aggregated_5utr_meme -mod anr -nmotifs 10 -minw 6 -maxw 15 -objfun classic -minsites 100 -maxsites 402 -markov_order 1 -evt 1e-6
"""

import os
import sys
import logging
import multiprocessing
import functools
from typing import *
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tqdm
import seaborn as sns
import intervaltree as itree

import covid19
import plotting

SRC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "rnagps",
)
sys.path.append(SRC_DIR)
from pwm import load_attract_ppm, load_meme_ppm, find_ppm_hits
from fasta import write_sequence_dict_to_file
from data_loader import LOCALIZATIONS, LOCALIZATION_FULL_NAME_DICT

assert os.path.isdir(SRC_DIR)
sys.path.append(SRC_DIR)

# Default values
GENOME_ID = "NC_045512.2"
GENOME = covid19.fetch_genbank(GENOME_ID)
GENOME_FEATURES = covid19.genbank_to_feature_dict(GENOME)

def ablate_and_eval_positional_prop(genome_id:str=GENOME_ID, num_steps:int=10, mode:str="delta") -> pd.DataFrame:
    """Ablate chunks of step_size and predict genome-wide localization"""
    assert mode in ['delta', 'raw', 'rel'], f"Unrecognized mode: {mode}"
    assert num_steps > 0
    genome = covid19.fetch_genbank(genome_id)
    genome_features = covid19.genbank_to_feature_dict(genome)
    three_utr = str(covid19.seq_feature_to_sequence(genome_features["3'UTR"][0], genome.seq))
    five_utr = str(covid19.seq_feature_to_sequence(genome_features["5'UTR"][0], genome.seq))
    cds = str(genome.seq)[len(five_utr) : -len(three_utr)]
    assert five_utr + cds + three_utr == str(genome.seq)

    featurizations = []
    indexes = []
    for seq_idx in range(3):
        l = len([five_utr, cds, three_utr][seq_idx])
        l_cum = [0, len(five_utr), len(five_utr) + len(cds)][seq_idx]  # Len of all preceding parts
        pts = np.linspace(0, l, num=num_steps+1, endpoint=True, dtype=int)
        for i, j in zip(pts[:-1], pts[1:]):  # Ablate i:j
            assert i < j
            seq_parts = [five_utr, cds, three_utr]
            updated_seq = seq_parts[seq_idx][:i] + ("N" * (j - i)) + seq_parts[seq_idx][j:]
            seq_parts[seq_idx] = updated_seq
            ft = covid19.featurize(seq_parts, ablation_parts=[False, False, False])
            featurizations.append(ft)
            indexes.append((i + l_cum, j + l_cum))
    featurizations = np.vstack(featurizations)  # Stack the row vectors
    retval = covid19.list_preds_to_array_preds(covid19.RNAGPS.predict_proba(featurizations))
    assert retval.shape[1] == 8
    assert np.all(retval <= 1.0) and np.all(retval >= 0.0)
    if mode == 'delta':
        retval -= covid19.list_preds_to_array_preds(
            covid19.RNAGPS.predict_proba(covid19.featurize((five_utr, cds, three_utr), ablation_parts=[False, False, False]))
        )
    elif mode == 'rel':
        retval /= covid19.list_preds_to_array_preds(
            covid19.RNAGPS.predict_proba(covid19.featurize((five_utr, cds, three_utr), ablation_parts=[False, False, False]))
        )
        retval = retval - 1.
    else:
        pass
    retval = pd.DataFrame(
        retval,
        index=['-'.join(map(str, i)) for i in indexes],
        columns=LOCALIZATION_FULL_NAME_DICT.values(),
    )
    return retval

def ablate_and_eval_positional(genome_id:str=GENOME_ID, step_size:int=25, mode:str='delta') -> pd.DataFrame:
    """Ablate chunks of step_size across genome, ignoring the boundaries between leader/UTR/cds"""
    assert mode in ['delta', 'raw', 'rel'], f"Unrecognized mode: {mode}"
    genome = covid19.fetch_genbank(genome_id)
    genome_features = covid19.genbank_to_feature_dict(genome)
    three_utr = str(covid19.seq_feature_to_sequence(genome_features["3'UTR"][0], genome.seq))
    five_utr = str(covid19.seq_feature_to_sequence(genome_features["5'UTR"][0], genome.seq))
    cds = str(genome.seq)[len(five_utr) : -len(three_utr)]
    assert five_utr + cds + three_utr == str(genome.seq)

    featurizations = []
    indexes = []
    for i in range(0, len(genome.seq), step_size):
        j = min(i + step_size, len(genome.seq))
        modified_genome = str(genome.seq)[:i] + ("N" * (j - i)) + str(genome.seq)[j:]
        assert len(modified_genome) == len(genome.seq)
        modified_parts = modified_genome[:len(five_utr)], modified_genome[len(five_utr) : -len(three_utr)], modified_genome[-len(three_utr):]
        assert ''.join(modified_parts) == modified_genome
        ft = covid19.featurize(modified_parts, ablation_parts=[False, False, False])
        featurizations.append(ft)
        indexes.append((i, j))
        # preds.append(covid19.list_preds_to_array_preds(covid19.RNAGPS.predict_proba(ft)).flatten())
    featurizations = np.vstack(featurizations)
    assert featurizations.shape[1] == 4032
    retval = covid19.list_preds_to_array_preds(covid19.RNAGPS.predict_proba(featurizations))
    assert retval.shape[1] == 8
    assert np.all(retval <= 1.0) and np.all(retval >= 0.0)
    if mode == 'delta':
        retval -= covid19.list_preds_to_array_preds(
            covid19.RNAGPS.predict_proba(covid19.featurize((five_utr, cds, three_utr), ablation_parts=[False, False, False]))
        )
    elif mode == 'rel':
        retval /= covid19.list_preds_to_array_preds(
            covid19.RNAGPS.predict_proba(covid19.featurize((five_utr, cds, three_utr), ablation_parts=[False, False, False]))
        )
        retval = retval - 1.
    else:
        pass
    retval = pd.DataFrame(
        retval,
        index=['-'.join(map(str, i)) for i in indexes],
        columns=LOCALIZATION_FULL_NAME_DICT.values(),
    )
    return retval

def extract_high_impact_seqs(ablation_df:pd.DataFrame, genome_id:str=GENOME_ID, impact_thresh:float=0.1) -> Dict[str, str]:
    """
    ablation_df is assumed to be a dataframe where each row represents a region that is ablated
    and each column represents the relative proportion change in
    impact_thresh is the percentage chane for "high impact" - 0.1 would correspond to a 10 percent change
    """
    genome = covid19.fetch_genbank(genome_id)
    genome_features = covid19.genbank_to_feature_dict(genome)
    three_utr = str(covid19.seq_feature_to_sequence(genome_features["3'UTR"][0], genome.seq))
    five_utr = str(covid19.seq_feature_to_sequence(genome_features["5'UTR"][0], genome.seq))
    cds = str(genome.seq)[len(five_utr) : -len(three_utr)]

    impact_locs = ablation_df.index[np.where(np.any(np.abs(ablation_df) >= impact_thresh, axis=1))]  # strings describing intervals
    impact_locs = [tuple(map(int, i.split("-"))) for i in impact_locs]  # List of range tuples
    impact_locs = itree.IntervalTree.from_tuples([(p[0], p[1] + 1) for p in impact_locs])  # Intervaltree of intervals, adding 1 to merge overlaps
    impact_locs.merge_overlaps()
    impact_fa_dict = {}
    for interval in sorted(impact_locs):
        seq = str(GENOME.seq[interval.begin:interval.end-1])  # Subtract one because we added one previously
        if len(seq) < 8:
            continue
        if interval.begin < len(five_utr):  # Determine part based on start position
            genome_part = "5utr"
        elif interval.begin < len(five_utr) + len(cds):
            genome_part = "cds"
        else:
            genome_part = "3utr"
        impact_fa_dict[f"{genome_id}_{genome_part}_{interval.begin}-{interval.end}"] = seq
    return impact_fa_dict

def _aggregate_ablate_helper(i:str, num_steps=50, mode='rel'):
    try:
        ablation_df = ablate_and_eval_positional_prop(i, num_steps=num_steps, mode=mode)
        return ablation_df
    except (AssertionError, ValueError, KeyError) as _err:
        return None

def aggregate_window_ablation(query_string=covid19.COVID19_QUERY_SEQUENCE) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Perform window ablation for multiple genomes and aggregate results"""
    genome_ids = covid19.query_genbank(query_string)

    # Run the window ablations in parallel
    pfunc = functools.partial(_aggregate_ablate_helper, num_steps=50, mode='rel')
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    abl = pool.map(pfunc, genome_ids)
    pool.close()
    pool.join()
    abl_dict = {k: v for k, v in zip(genome_ids, abl) if v is not None}

    abl_tensor = np.stack([df.values for df in abl_dict.values()])
    assert abl_tensor.shape[0] == len(abl_dict)
    logging.info(f"Aggregating ablations from {abl_tensor.shape[0]} genomes")
    ablation_df = abl_dict[list(abl_dict.keys())[0]]
    abl_mean = pd.DataFrame(
        np.mean(abl_tensor, axis=0),
        columns=ablation_df.columns,
        index=ablation_df.index,
    )
    abl_std = pd.DataFrame(
        np.std(abl_tensor, axis=0),
        columns=ablation_df.columns,
        index=ablation_df.index,
    )
    return abl_mean, abl_std, abl_dict

def main():
    ### Sliding window ablation
    if not os.path.isdir("window_ablation"):
        os.mkdir("window_ablation")

    three_utr = str(covid19.seq_feature_to_sequence(GENOME_FEATURES["3'UTR"][0], GENOME.seq))
    five_utr = str(covid19.seq_feature_to_sequence(GENOME_FEATURES["5'UTR"][0], GENOME.seq))
    full_genome = str(GENOME.seq)

    sliding_step_size = 10
    sliding_num_steps = 50
    sliding_absolute = False  # Whether to use number of base pairs to ablate, or to use relative ablation window
    sliding_ablation_df = ablate_and_eval_positional(mode='rel', step_size=sliding_step_size) if sliding_absolute else ablate_and_eval_positional_prop(mode='rel', num_steps=sliding_num_steps)
    sliding_ablation_df.to_csv("window_ablation/sliding_window_ablation.csv")
    fig, ax = plt.subplots(figsize=(16, 6))
    sliding_ablation_df.plot.line(ax=ax)
    ax.axhline(0.0, linestyle="--", color='gray')
    if sliding_absolute:
        ax.axvline(len(five_utr) // sliding_step_size, linestyle='--', color='black')
        ax.axvline((len(full_genome) - len(five_utr)) // sliding_step_size, linestyle='--', color='black')
    else:  # Relative ablation such that each ablation is a fixed percentiel of the ablation space
        ax.axvline(sliding_num_steps, linestyle='--', color='black')
        ax.axvline(sliding_num_steps * 2, linestyle='--', color='black')
        print(
            # Total deviation from 1
            np.sum(np.abs(sliding_ablation_df.values[:sliding_num_steps].flatten())),
            np.sum(np.abs(sliding_ablation_df.values[sliding_num_steps:2*sliding_num_steps].flatten())),
            np.sum(np.abs(sliding_ablation_df.values[2*sliding_num_steps:].flatten())),
        )
    ax.set(
        title=f"Sliding window ablation ({sliding_step_size} bp)",
        ylabel="Relative localization prediction",
        xlabel="Ablation window (5' leader, CDS, 3' UTR regions separated by black lines)",
    )
    fig.savefig('window_ablation/sliding_window_ablation.pdf')

    # Save the bins that are corresponding to the three and five utr
    sliding_ablation_df_five = sliding_ablation_df.iloc[:(len(five_utr) // sliding_step_size)]
    fig, ax = plt.subplots()
    sliding_ablation_df_five.plot.line(ax=ax)
    ax.set(
        title=f"Sliding window ablation - 5' header ({sliding_step_size} bp)",
        ylabel="Relative localization prediction",
        xlabel="Ablation window (5' to 3')",
    )
    fig.savefig("window_ablation/sliding_window_ablation_5utr.pdf")

    sliding_ablation_df_three = sliding_ablation_df.iloc[-(len(three_utr) // sliding_step_size):]
    fig, ax = plt.subplots()
    sliding_ablation_df_three.plot.line(ax=ax)
    ax.set(
        title=f"SLiding window ablation - 3' UTR ({sliding_step_size} bp)",
        ylabel="Relative localization prediction",
        xlabel="Ablation window (5' to 3')",
    )
    fig.savefig("window_ablation/sliding_window_ablation_3utr.pdf")

    ### Extract the regions tht have high impact
    impact_fa_dict = extract_high_impact_seqs(sliding_ablation_df)
    write_sequence_dict_to_file(impact_fa_dict, "window_ablation/high_impact_sequences.fa")

    ### Do the above, but for many sequences
    window_mean, window_sd, window_dict = aggregate_window_ablation()
    fig, ax = plt.subplots(figsize=(20, 6))
    window_mean.plot.line(ax=ax, yerr=window_sd, elinewidth=0.5)
    ax.axvline(sliding_num_steps, linestyle='--', color='black')
    ax.axvline(sliding_num_steps * 2, linestyle='--', color='black')
    ax.set(
        title=f"Sliding window ablation mean impact",
        ylabel="Relative localization prediction",
        xlabel="Ablation window (5' leader, CDS, 3' UTR regions separated by black lines)",
    )
    fig.savefig("window_ablation/sliding_window_mean.pdf", bbox_inches='tight')

    # Walk through the window dict and create fast files of ssequences
    outdir = os.path.join("window_ablation", "high_impact_seqs")
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    logging.info(f"Writing high impact sequence ablations to {outdir}")
    for genome_id, ablation_df in window_dict.items():
        impact_fa_dict = extract_high_impact_seqs(ablation_df=ablation_df, genome_id=genome_id, impact_thresh=0.1)
        outpath = os.path.join(outdir, f"{genome_id}_impact_seqs.fa")
        write_sequence_dict_to_file(seq_dict=impact_fa_dict, fname=outpath)

if __name__ == "__main__":
    main()

