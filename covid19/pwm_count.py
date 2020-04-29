"""
Code for counting PWMs in sequence parts
"""

import os
import sys
import argparse
import multiprocessing
import functools
import logging
from typing import Dict

import numpy as np
import pandas as pd

from Bio import SeqRecord
from Bio.SeqFeature import SeqFeature, FeatureLocation

import tqdm

from covid19 import query_genbank, fetch_genbank, genbank_to_feature_dict, get_feature_labels, seq_feature_to_sequence, mean_sd_missing_vals
from baseline import load_apex_test_dataset, rank
from plotting import heatmap

SRC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "rnagps",
)
assert os.path.isdir(SRC_DIR)
sys.path.append(SRC_DIR)
import pwm
import data_loader

logging.basicConfig(level=logging.INFO)

def count_baseline(localization_idx:int, pwms:Dict[str, np.ndarray], length_norm:bool=False) -> pd.DataFrame:
    """
    Count PWMs in sequences that significantly localize to the given localization (index)
    """
    test_dataset = load_apex_test_dataset(fold=5)
    test_data, test_labels = data_loader.load_data_as_np(test_dataset, progress_bar=False)

    # Indicies that we want to fetch transcripts for
    if isinstance(localization_idx, int):
        keep_idx = np.where(test_labels[:, localization_idx] > 0)[0]
    elif localization_idx is None:
        keep_idx = np.arange(test_labels.shape[0])
    else:
        raise TypeError(f"Unrecognized localization index: {localization_idx} of type {type(localization_idx)}")
    logging.info(f"Found {len(keep_idx)} matches for idx {localization_idx}")

    transcripts = (''.join(test_dataset.get_ith_trans_parts(i)) for i in keep_idx)
    transcript_labels = (test_dataset.get_ith_trans_name(i) for i in keep_idx)

    pwm_counts = {}
    for trans, label in zip(transcripts, transcript_labels):
        c = np.array([len(pwm.find_ppm_hits(trans, p, prop=0.9)) for p in pwms.values()], np.float)
        if length_norm:
            c /= float(len(trans))
        pwm_counts[label] = c

    retval = pd.DataFrame(
        pwm_counts,
        index=pwms.keys(),
    ).T
    return retval

def count_record(record:SeqRecord.SeqRecord, pwms:Dict[str, np.ndarray], length_norm:bool=False) -> pd.DataFrame:
    """
    Count the PWMs in the record
    """
    ft_dict = genbank_to_feature_dict(record)
    gene_names = get_feature_labels(record)
    assert len(gene_names) == len(ft_dict['CDS']), f"Got differing lengths of genes and CDS {len(gene_names)} {len(ft_dict['CDS'])}\n{record}"

    five_utr = ft_dict["5'UTR"]
    three_utr = ft_dict["3'UTR"]
    genome = str(record.seq)

    if not five_utr and ft_dict["CDS"]:
        # Empirically we see that that 5' UTR is just the bases going up to the first CDS
        # so we assume that's what it is if we aren't given an explicit 5' UTR
        five_utr_end = ft_dict['CDS'][0].location.start
        if five_utr_end < 100:
            raise ValueError(f"Inferred 5' UTR has anomalous end position {five_utr_end}")
        five_utr = [SeqFeature(FeatureLocation(start=0, end=five_utr_end, strand=+1))]
    if not three_utr and ft_dict['CDS']:
        # Empirically we see that the 3' UTR is just the remaining bases after the last CDS
        three_utr = [SeqFeature(FeatureLocation(start=ft_dict['CDS'][-1].location.end, end=len(genome), strand=+1))]

    assert len(five_utr) == 1
    assert len(three_utr) == 1
    five_utr_seq = seq_feature_to_sequence(five_utr[0], genome, three_stop=None)
    three_utr_seq = seq_feature_to_sequence(three_utr[0], genome, three_stop=None)
    assert five_utr_seq and three_utr_seq

    pwm_counts = {}
    pwm_counts["5UTR"] = np.array([len(pwm.find_ppm_hits(five_utr_seq, p, prop=0.9)) for p in pwms.values()])
    pwm_counts["3UTR"] = np.array([len(pwm.find_ppm_hits(three_utr_seq, p, prop=0.9)) for p in pwms.values()])
    if length_norm:
        pwm_counts['5UTR'] = pwm_counts['5UTR'] / len(five_utr_seq)
        pwm_counts['3UTR'] = pwm_counts['3UTR'] / len(three_utr_seq)

    for gene, cds in zip(gene_names, ft_dict['CDS']):
        assert gene not in pwm_counts
        cds_seq = seq_feature_to_sequence(cds, genome, three_stop=three_utr[0])
        cds_seq_trunc = seq_feature_to_sequence(cds, genome, three_stop=None)  # version that doesn't run through
        assert len(cds_seq_trunc) <= len(cds_seq), f"{gene} truncated sequence longer than seq: {len(cds_seq_trunc)} {len(cds_seq)}"
        # We count the TRUNCATED sequence to avoid overlap
        c = np.array([len(pwm.find_ppm_hits(cds_seq_trunc, p, prop=0.9)) for p in pwms.values()], np.float)
        if length_norm:
            c /= float(len(cds_seq_trunc))
        pwm_counts[gene] = c

    retval = pd.DataFrame(
        pwm_counts,
        index=pwms.keys(),
    ).T
    return retval

def flatten_dict_vals(d:Dict[str, list], sep:str="|") -> dict:
    """Flatten dict so values are unique, adding identifiers to keys"""
    retval = {}
    for k, v_list in d.items():
        for i, v in enumerate(v_list):
            retval[k + sep + str(i)] = v
    return retval

def build_parser() -> argparse.ArgumentParser:
    """Build commandline argument parser"""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("rbpdb", type=str, choices=['meme', 'attract'], help="RBP database to use")
    parser.add_argument("-l", "--lengthnorm", action="store_true", help="Normalize by length")
    parser.add_argument("-o", "--output", type=str, default="rbp_pwm_counts", help="Directory to output to")
    parser.add_argument("-t", "--truncate", type=int, default=0, help="Truncate plot to top n RBPs")
    parser.add_argument("--threads", type=int, default=8, help="Number of threads to run")
    return parser

def _count_record_helper(record, pwms, length_norm):
    """Helper function to parallelize calling count_record"""
    try:
        return record, count_record(record=record, pwms=pwms, length_norm=length_norm)
    except (AssertionError, ValueError, KeyError) as err:
        return record, None

def main():
    """
    Run as script
    """
    # Declare some parameters upfront
    parser = build_parser()
    args = parser.parse_args()

    normalize_by_length = args.lengthnorm
    outdir = args.output
    if not os.path.isdir(outdir):
        logging.info(f"Creating directory: {outdir}")
        os.mkdir(outdir)

    if args.rbpdb == "meme":
        pwm_db = pwm.load_meme_ppm()
    elif args.rbpdb == "attract":
        pwm_db = flatten_dict_vals(pwm.load_attract_ppm())
    else:
        raise ValueError(f"Unrecognized RBP database: {args.rbpdb}")

    identifiers = query_genbank()
    genbank_records = (fetch_genbank(i) for i in identifiers)

    pfunc = functools.partial(_count_record_helper, pwms=pwm_db, length_norm=normalize_by_length)
    pool = multiprocessing.Pool(args.threads)
    all_count_tuples = pool.imap(pfunc, genbank_records, chunksize=5)
    pool.close()
    pool.join()
    per_record_counts = {record.name: c for record, c in all_count_tuples if c is not None and not c.empty}
    assert per_record_counts

    # Aggregate and write output
    counts_mean, counts_sd = mean_sd_missing_vals(per_record_counts.values())
    counts_mean.to_csv(os.path.join(outdir, f"overall_rbp_counts_mean.csv"))
    counts_sd.to_csv(os.path.join(outdir, f"overall_rbp_counts_sd.csv"))
    for record_name, record_counts in per_record_counts.items():
        record_counts.to_csv(os.path.join(outdir, f"{record_name}_rbp_counts.csv"))

    # Construct a baseline for occurrences
    pfunc = functools.partial(count_baseline, pwms=pwm_db, length_norm=normalize_by_length)
    _x = load_apex_test_dataset(fold=5)  # Populate the cache
    # Figure out rank

    pool = multiprocessing.Pool(args.threads)
    full_bline, *blines = pool.map(pfunc, [None] + list(range(8)))
    pool.close()
    pool.join()

    ranked_vals = np.zeros_like(counts_mean.values)
    for i in range(counts_mean.shape[0]):
        row = counts_mean.iloc[i]
        for j, val in enumerate(row.values):
            ranked_vals[i, j] = rank(val, full_bline.iloc[:, j].values)

    counts_rank = pd.DataFrame(
        ranked_vals,
        index=counts_mean.index,
        columns=counts_mean.columns,
    )
    counts_rank.to_csv(os.path.join(outdir, f"overall_rbp_mean_ranks.csv"))

    # print(counts_mean.mean(axis=1))
    genes_to_plot = counts_mean.columns
    if args.truncate > 0:
        per_gene_maxes = np.nanmax(counts_rank, axis=0)
        idx = np.argsort(per_gene_maxes)[::-1][:args.truncate]  # Index of biggest element first
        # genes_to_plot = [col for i, col in enumerate(counts_mean.columns) if i in idx]
        genes_to_plot = [counts_mean.columns[i] for i in idx]

    heatmap(
        counts_mean.loc[:, genes_to_plot],
        title="PWM count means" + (", length normalized" if normalize_by_length else ""),
        fname=os.path.join(outdir, "overall_rbp_counts_mean.pdf"),
        figsize=(30, 6),
        annot=False,
        cbar=True,
    )
    heatmap(
        counts_rank.loc[:, genes_to_plot],
        title="PWM count rank" + (", length normalized" if normalize_by_length else ""),
        fname=os.path.join(outdir, "overall_rbp_ranks.pdf"),
        figsize=(30, 6),
        annot=False,
        cbar=True,
    )

    for i, bline in enumerate(blines):
        heatmap(
            bline.loc[:, genes_to_plot],
            title=f"PWM counts, {i} localization" + (", length normalized" if normalize_by_length else ""),
            fname=os.path.join(outdir, f"baseline_localization_{i}.pdf"),
            figsize=(30, 6),
            annot=False,
            cbar=True,
        )

if __name__ == '__main__':
    main()

