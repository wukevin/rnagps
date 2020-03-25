"""
Code for gapped kmer stuffs
"""

import os
import sys
import collections
import multiprocessing
import functools
import itertools

import numpy as np
import pandas as pd

BASE_TO_INT = {
    "A": 0,
    "C": 1,
    "G": 2,
    "T": 3,
    "N": 4,
}
INT_TO_BASE = {v: k for k, v in BASE_TO_INT.items()}
BASE_TO_COMP = {
    "A": "T",
    "T": "A",
    "C": "G",
    "G": "C",
    "N": "N",
}

@functools.lru_cache()
def reverse_complement(seq):
    """
    Return the reverse complement

    >>> reverse_complement("ACTG")
    'CAGT'

    >>> reverse_complement("NAT")
    'ATN'
    """
    return ''.join([BASE_TO_COMP[b] for b in seq][::-1])

@functools.lru_cache(maxsize=128)
def generate_all_kmers(k, ignore_N=True):
    """
    Generate all kmers of length k, returning a dictionary mapping each kmer to an index in the list of kmers
    """
    alphabet = "ACGT"
    if not ignore_N:
        alphabet += "N"
    possible_kmers = itertools.product(alphabet, repeat=k)
    retval = collections.OrderedDict()
    for i, kmer in enumerate(possible_kmers):
        retval[''.join(kmer)] = i
    return retval

def sequence_to_kmer_freqs(seq, kmer_size=6, step_size=1, ignore_N=True, normalize=True):
    """
    Featurizes a sequence into a kmer count table. If ignore_N is true, we skip any
    kmers that have a N base, and the output is a array of length 4^kmer_size.
    Otherwise, we include N in the alphabet, and the output is a array of length
    5^kmer_size
    
    >>> list(sequence_to_kmer_freqs("AACCTTGGGCT", 2, 1))
    [0.1, 0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.2, 0.0, 0.1, 0.2, 0.0, 0.0, 0.0, 0.1, 0.1]
    >>> list(sequence_to_kmer_freqs("AAAA", 2, 1))
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    >>> list(sequence_to_kmer_freqs("AAAA", 1, 1, ignore_N=False))
    [1.0, 0.0, 0.0, 0.0, 0.0]
    """
    # Split into kmers
    seq_kmers = [seq[i:i+kmer_size] for i in range(0, len(seq) - kmer_size + 1, step_size)]
    # Generate all possible kmers
    possible_kmers = generate_all_kmers(kmer_size, ignore_N=ignore_N)
    if ignore_N:
        seq_kmers = [kmer for kmer in seq_kmers if "N" not in kmer]
    indices = [possible_kmers[kmer] for kmer in seq_kmers]
    retval = np.zeros(len(possible_kmers))
    if not seq or not seq_kmers or len(seq) < kmer_size:  # For an empty/too-short sequence return a matrix of 0's
        return retval
    np.add.at(retval, indices, 1)  # Increment all specified indices in place, increment multiple times if index occurs multiple times
    if normalize:
        retval /= len(seq_kmers)
        assert np.isclose(np.sum(retval), 1.0)
    return retval

def sequences_to_kmer_counts(sequences, kmer_size):
    """Return a dataframe where index is sequences and columns is kmers"""
    pfunc = functools.partial(sequence_to_kmer_freqs, kmer_size=kmer_size, normalize=False)
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    kmer_freqs = pool.map(pfunc, sequences, chunksize=250)
    pool.close()
    pool.join()
    mat = np.vstack(kmer_freqs).astype(int)
    colnames = ['' for _ in range(len(generate_all_kmers(kmer_size)))]
    for kmer, i in generate_all_kmers(kmer_size).items():
        colnames[i] = kmer
    retval = pd.DataFrame(mat, columns=colnames)
    return retval

def sequence_kmer_pileup(seq, query_kmers):
    """
    Given a sequence and query kmers, return a binary matrix where each row represents if that kmer is in that position

    >>> sequence_kmer_pileup("ACTGCGTACG", ['ACT', 'GCG', 'TACG'])
    array([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]])
    >>> sequence_kmer_pileup("ACTGCGTACG", ['ACT', 'GCGT', 'TACG'])
    array([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]])
    >>> sequence_kmer_pileup("AAAA", ["AAA"])
    array([[1, 2, 2, 1]])
    """
    assert isinstance(query_kmers, list)
    lengths = set([len(kmer) for kmer in query_kmers])
    retval = np.zeros((len(query_kmers), len(seq))).astype(int)
    for length in lengths:
        assert length <= len(seq), "Cannoty query a kmer against a seq shorter than that kmer"
        kmers = [seq[i:i+length] for i in range(len(seq) - length + 1)]
        kmer_to_idx = generate_all_kmers(length)
        # Row vector
        kmers_int = np.array([kmer_to_idx[k] for k in kmers if "N" not in k], dtype=int)
        # Column vector
        query_int = np.atleast_2d(np.array([kmer_to_idx[k] for k in query_kmers if len(k) == length and "N" not in k], dtype=int)).T
        # Array of where each query is found in the seq, by the first index of occurrence
        hits = np.where(query_int == kmers_int)  # Automatically broadcasts
        this_rows = np.zeros((len(query_int), len(seq)))
        for i in range(length):
            this_rows[hits[0], hits[1] + i] += 1
        retval_idx = np.array([i for i, k in enumerate(query_kmers) if len(k) == length], dtype=int)
        retval[retval_idx, ] = this_rows
    return retval

def find_long_runs(num_sequence, l):
    """
    Return the index and length of all run non zero elements of length greater than l
    Primarily intended as a helper function for the below function assemble_kmer_motifs
    """
    chunked = [(k, list(g)) for k, g in itertools.groupby(num_sequence)]
    retval = [(i, len(g)) for i, (k, g) in enumerate(chunked) if k and len(g) > l]
    return retval

def connect_nearby_runs(pileup_flat, allowed_gap_num):
    """
    Given a binary pileup (1 if has something, 0 if nothing), return a binary pileup
    where pileups with allowed_gap_num between them are connected

    >>> connect_nearby_runs(np.array([1, 1, 1, 0, 1, 1, 1]), 1)
    array([1, 1, 1, 1, 1, 1, 1])
    >>> connect_nearby_runs(np.array([1, 1, 1, 0, 0, 1, 1, 0]), 2)
    array([1, 1, 1, 1, 1, 1, 1, 0])
    >>> connect_nearby_runs(np.array([1, 1, 0, 0, 0, 1, 1]), 1)
    array([1, 1, 0, 0, 0, 1, 1])
    >>> connect_nearby_runs(np.array([0, 0, 1, 1, 0, 0, 1]), 2)
    array([0, 0, 1, 1, 1, 1, 1])
    """
    chunked = [(k, list(g)) for k, g in itertools.groupby(list(pileup_flat))]
    retval = []
    for i, (item, group) in enumerate(chunked):
        if not item and len(group) <= allowed_gap_num and 0 < i < len(chunked) - 1:
            retval.extend([1] * len(group))
        else:
            retval.extend(group)
    return np.array(retval, dtype=int)

def assemble_kmer_motifs(seq, kmers, min_len=10, gap_allowed=2):
    """Given a sequence and kmers, return a list of assembled kmers that are at least min_len"""
    try:
        pileup = sequence_kmer_pileup(seq, kmers)
    except AssertionError:
        return []
    pileup_flat = np.clip(np.sum(pileup, axis=0), 0, 1)  # Flatten
    pileup_flat = connect_nearby_runs(pileup_flat, gap_allowed)  # Connect runs that are separated by 1 gap
    motif_idx = find_long_runs(pileup_flat, l=min_len)
    retval = [seq[i:i+l] for i, l in motif_idx]
    # Sanity check against weird off by 1 indexing errors
    assert all([len(s) == l for s, (_i, l) in zip(retval, motif_idx)])
    return retval

@functools.lru_cache()
def gkm_name(l=4, k=3, rev_comp=False):
    """
    Generates the GKM name vector

    >>> gkm_name(l=3, k=2)
    ['NAA', 'NAC', 'NAG', 'NAT', 'NCA', 'NCC', 'NCG', 'NCT', 'NGA', 'NGC', 'NGG', 'NGT', 'NTA', 'NTC', 'NTG', 'NTT', 'ANA', 'ANC', 'ANG', 'ANT', 'CNA', 'CNC', 'CNG', 'CNT', 'GNA', 'GNC', 'GNG', 'GNT', 'TNA', 'TNC', 'TNG', 'TNT', 'AAN', 'ACN', 'AGN', 'ATN', 'CAN', 'CCN', 'CGN', 'CTN', 'GAN', 'GCN', 'GGN', 'GTN', 'TAN', 'TCN', 'TGN', 'TTN']

    >>> gkm_name(l=2, k=1)
    ['NA', 'NC', 'NG', 'NT', 'AN', 'CN', 'GN', 'TN']

    >>> gkm_name(l=2, k=1, rev_comp=True)
    ['NA/TN', 'NC/GN', 'NG/CN', 'NT/AN']
    """
    assert k < l
    ungapped_kmers = list(itertools.product(*["ACGT" for _ in range(k)]))
    discard_locs = list(itertools.combinations(range(l), l - k))
    retval = []
    for loc in discard_locs:  # Locations of N chars
        for kmer in ungapped_kmers:
            s = ['' for _ in range(l)]
            j = 0
            for i in range(l):
                if i in loc: s[i] = "N"
                else:
                    s[i] = kmer[j]
                    j += 1
            retval.append(''.join(s))
    if rev_comp:
        retval_first = [retval[i] for i in gkm_rc_indices(l=l, k=k)[0, :]]
        retval_second = [retval[i] for i in gkm_rc_indices(l=l, k=k)[1, :]]
        retval = ["/".join(pair) for pair in zip(retval_first, retval_second)]
    return retval

@functools.lru_cache()
def gkm_rc_indices(l=4, k=3):
    """Generate the indices that are reverse complements of each other"""
    names = gkm_name(l=l, k=k, rev_comp=False)
    collect_seqs = set()  # Contains the seqs and rev comps added thus far
    first_index, second_index = [], []
    for i, kmer in enumerate(names):
        if kmer not in collect_seqs:
            collect_seqs.add(kmer)  # Add kmer and its RC so we don't process it again
            collect_seqs.add(reverse_complement(kmer))
            first_index.append(i)  # Add the pair indices
            second_index.append(names.index(reverse_complement(kmer)))
    assert len(first_index) == len(second_index)
    return np.vstack((first_index, second_index)).astype(int)

def gkm_fv(seq, l=4, k=3, rev_comp=False, normalize=False):
    """
    Return the gapped kmer feature vector for the given sequence
    Reference:
    https://github.com/zhanglabtools/gkm-DNN/blob/master/R/gkmfv.R

    >>> gkm_fv("ACTGAGAATGATGCGATGC", l=2, k=1)
    array([5., 3., 6., 4., 6., 2., 6., 4.])

    >>> gkm_fv("ACTGAGAATGATGCGATGC", l=2, k=1, rev_comp=True)
    array([ 9.,  9.,  8., 10.])
    """
    assert k < l
    retval = np.zeros(len(gkm_name(l=l, k=k)))
    if not seq:
        if rev_comp:
            return np.zeros(gkm_rc_indices(l=l, k=k).shape[1])
        return retval  # Array of 0's

    # Transform base strings to int
    seq_int = [BASE_TO_INT[base] for base in seq]
    seq_int_lmers = [seq_int[i:i+l] for i in range(0, len(seq_int) - l + 1)]
    seq_int_lmers = [s for s in seq_int_lmers if 4 not in s]  # Ignore items with N
    if not seq_int_lmers:
        if rev_comp:
            return np.zeros(gkm_rc_indices(l=l, k=k).shape[1])
        return retval  # Array of 0's
    seq_int_lmers_stacked = np.vstack(seq_int_lmers)

    # Convert lmers to gapped kmers
    quad_base = np.power(4, np.arange(k))[::-1]
    keep_locs = list(itertools.combinations(range(l), k))
    for x, loc in zip(np.arange(len(keep_locs))[::-1], keep_locs):
        kept_cols = seq_int_lmers_stacked[:, loc]
        intified = kept_cols @ quad_base + x * 4**k
        np.add.at(retval, intified, 1)
    if rev_comp:  # Consolidate reverse complements
        rc_indices = gkm_rc_indices(l=l, k=k)
        retval = retval[rc_indices[0, :]] + retval[rc_indices[1, :]]
    if normalize:
        retval /= float(len(seq))
    return retval

def main():
    print(gkm_fv("ACTGAGAATGATGCGATGC", l=5, k=3, rev_comp=True))
    print(sequences_to_kmer_counts(['AACTGACGAGCTG', 'ACTGGAGCTGAGGCA'], 3))
    print(sequence_kmer_pileup("ACTGCGTACG", ['ACT', 'GCGT', 'TACG']))

if __name__ == "__main__":
    import doctest
    doctest.testmod()
    main()

