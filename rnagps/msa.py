"""
Utilities for handling multiple sequence alignments

When run as a standalone script, consumes msa file and outputs sequence
motifs that pass specified support and length cutoffs.
"""

import os
import sys
import logging
import argparse
import collections
import itertools

import numpy as np
import pandas as pd

def read_msa(fname, combine_lines=True):
    """
    Read the given MSA, assumed to be in the format clustalW provides
    """
    retval = collections.defaultdict(list)
    with open(fname) as source:
        for line in source:
            if line.startswith("CLUSTAL") or not line.strip():
                continue
            tokens = line.strip().split()
            seq_id, aln, num = tokens
            retval[seq_id].append(aln)
    if combine_lines:
        retval = {k: ''.join(v) for k, v in retval.items()}
    return retval

def _fetch_kmer_from_msa_i(i, seed_seq, msa, min_len, min_reps):
    """
    Helper function for below
    Given the index in the msa, extend the kmer as long as possible while maintaining the minimum reps
    """
    # Filter out irrelevant sequences and chop off prefix that doesn't match
    relevant_seqs = [m[i:] for m in msa if m[i:i+min_len] == seed_seq]
    num_matching = len(relevant_seqs)
    msa_len = len(relevant_seqs[0])

    extended = []
    # Extend the seed sequence
    for combo in itertools.combinations(relevant_seqs, min_reps):
        # Enumerate all the ways we can find the given support
        this_seq = []
        for j in range(len(combo[0])):
            jth_bases = set([c[j] for c in combo if c[j] != '-'])
            if len(jth_bases) != 1:
                break
            this_seq.append(jth_bases.pop())
        extended.append(''.join(this_seq))
    # Tuples of length and num occurrences
    extended_properties = [(len(s), len([m for m in relevant_seqs if s in m])) for s in extended]
    extended_sorted = [seq for _prop, seq in sorted(zip(extended_properties, extended))]
    return extended_sorted[0]

def find_motifs_in_msa(msa, min_len=7, min_reps=3):
    """
    Find motifs in the msa with the given minimum length and reps
    msa is expected to be an iterable of sequences
    """
    unique_lenths = set([len(m) for m in msa])
    assert len(unique_lenths) == 1, "All MSA sequences must be of the same length"
    msa_len = unique_lenths.pop()

    # First find seeds where we know there are hits
    hits = []
    for i in range(msa_len - min_len):
        block = [m[i:i+min_len] for m in msa]
        block_no_gaps = [m for m in block if "-" not in m]
        if not block_no_gaps:
            continue
        block_counter = collections.Counter(block_no_gaps)
        for kmer, count in block_counter.items():
            if count >= min_reps:
                hits.append((i, kmer))
    # Extend each hit to maximum length
    hits_extended = [_fetch_kmer_from_msa_i(hit[0], hit[1], msa=msa, min_len=min_len, min_reps=min_reps) for hit in hits]
    # Filter out hits that are sub-sequences of other hits
    hits_extended.sort(key=len, reverse=True)  # Sort by descending length
    hits_extended_dedup = []
    for hit in hits_extended:  # Subsequences must be shorter
        if any([hit in longer for longer in hits_extended_dedup]):
            continue
        hits_extended_dedup.append(hit)
    return hits_extended_dedup

def build_parser():
    """Build argument parser"""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-l", "--len", default=7, type=int, help="Minimum length")
    parser.add_argument("-r", "--rep", default=3, type=int, help="Minimum number of times motif is repeated")
    parser.add_argument("-o", "--outdir", default=os.getcwd(), help="Directory to output fasta files to")
    parser.add_argument("msafile", type=str, nargs='*')
    return parser

def main():
    """Execute script to find motifs in msa file"""
    logging.basicConfig(level=logging.INFO)
    parser = build_parser()
    args = parser.parse_args()
    logging.info(f"Minimum length: {args.len}")
    logging.info(f"Minimum num occur: {args.rep}")
    for msafile in args.msafile:
        bname = os.path.basename(msafile).split(".")[0]
        msa = read_msa(msafile)
        logging.info(f"Read in {msafile} for {len(msa)} sequences")
        results = find_motifs_in_msa(list(msa.values()), min_len=args.len, min_reps=args.rep)
        logging.info(f"Found {len(results)} motifs")
        out_fname = os.path.join(args.outdir, bname + ".fa")
        logging.info(f"Writing to {out_fname}")
        with open(out_fname, 'w') as sink:
            for i, motif in enumerate(results):
                sink.write(f">{bname}_motif_{i}\n")
                sink.write(motif + "\n")

if __name__ == "__main__":
    main()

