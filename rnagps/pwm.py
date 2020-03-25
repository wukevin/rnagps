"""
Code for handling PWMs
"""

import os
import sys
import warnings
import logging
import glob
import socket
import collections

import numpy as np
import pandas as pd
import scipy.signal

import seq

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))),
    "rnafinder_data",
)
assert os.path.isdir(DATA_DIR), "Cannot find data directory: {}".format(DATA_DIR)

LOCAL_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data"))
assert os.path.isdir(LOCAL_DATA_DIR), "Cannot find data directory: {}".format(LOCAL_DATA_DIR)

MEME_DB = os.path.join(LOCAL_DATA_DIR, "meme/Ray2013_rbp_Homo_sapiens.dna_encoded.meme")
assert os.path.isfile(MEME_DB), f"Cannot find meme db {MEME_DB}"

def calculate_base_background(sequence, alphabet='ACGT', pseudocount=True):
    """Given a sequence, calculate the proportion of each base"""
    assert sequence
    cnt = collections.Counter(sequence.upper())
    retval = np.array([cnt[base] for base in alphabet]).astype(float)
    if pseudocount:
        retval += 1
    retval /= float(len(sequence))
    return retval

def find_ppm_hits(sequence, ppm, prop=0.8, debug=False):
    """
    Finds the indices of positional probability matrix hits in the given sequence. Uses either prop or pval cutoff
    prop cutoff sets the maximum score as a proportion of the maximum attainable score
    pval cutoff sets the cutoff as everything below that pval
    """
    if debug:
        assert ppm.shape[1] == 4
        assert np.allclose(np.sum(ppm, axis=1), 1), "PWM should be supplied as probabilities of each base at each position"
        assert np.all(ppm >= 0)
    one_hot = seq.sequence_to_image(sequence)
    pwm_len = ppm.shape[0]
    if len(sequence) < pwm_len:
        return []  # no hits, return empty list

    background_dist = calculate_base_background(sequence)

    # Scores greater than 0 if more likely to be fucntional site (i.e. a hit)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Ignore the divide by zero that log2 throws
        position_weight_matrix = np.nan_to_num(np.log2(ppm / background_dist))  # No log2 1 plus func
    # Use masked array to ignore inf values in max
    # https://stackoverflow.com/questions/4485779/ignoring-inf-values-in-arrays-using-numpy-scipy-in-python
    threshold = prop * np.sum(np.ma.masked_invalid(ppm).max(axis=1))

    # Slide the ppm over the one hot encoded sequence
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate2d.html
    sliding_conv = scipy.signal.correlate2d(one_hot, position_weight_matrix, mode='valid').flatten()
    # assert len(sliding_conv) == len(sequence) - ppm.shape[0] + 1
    hits = list(np.where(sliding_conv >= threshold)[0])
    return hits

def score_sequence_with_ppm(sequence, ppm, min_prop=0.8):
    """
    Given a string sequence and a matrix PWM (with log(p) in each cell) in the order ACTG, return count of significant hits
    From deepLncRNA paper:
    Matches were counted using a sliding-window approach, and a match was scored
    if the sub-sequence obtained a log-likelihood position weight matrix (PWM)
    score greater than 80% of the maximal PWM score
    """
    hits = find_ppm_hits(sequence, ppm, prop=min_prop)
    return len(hits)

def load_ppm(fname, log_transform=False):
    """Read the position probability matrix (ppm) from the given filename. Assumes that the file's values are [0, 1]"""
    parsed = pd.read_csv(fname, delimiter="\t", index_col=0, header=0)
    # Sanity check the header
    assert all([col == expected for col, expected in zip(parsed.columns, ["A", "C", "G", "U"])])
    mat = parsed.to_numpy()  # As a numpy array
    assert np.all(mat >= 0) and np.all(mat <= 1)  # sanity check
    if log_transform:  # Log transform if requested
        mat = np.log2(mat)
    return mat

def load_all_ppm_in_dir(dirname=os.path.join(DATA_DIR, "Homo_sapiens_2019_04_15_1_44_pm/pwms_all_motifs"), log_transform=True):
    """Loads all the pwms in the given directory, returning as a dictionary"""
    matches = glob.glob(os.path.join(dirname, "*.txt"))
    retval = collections.OrderedDict()
    for match in matches:
        if os.stat(match).st_size == 0:  # Skip empty files
            continue
        # retval.append(load_ppm(match, log_transform=log_transform))
        fname_base = os.path.basename(match).split(".")[0]
        assert fname_base not in retval, "Found duplicated fname: {}".format(fname_base)
        retval[fname_base] = load_ppm(match, log_transform=log_transform)
    logging.info("Read in {} PWMs".format(len(retval)))
    return retval

def load_meme_ppm(fname=MEME_DB, log_transform=False):
    """Loads in the position probability matrices from the given meme file"""
    retval = collections.OrderedDict()
    curr_name = None
    curr_vals = []
    curr_expected_shape = None
    alphabet = None
    with open(fname) as source:
        for line in source:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith("ALPHABET"):
                alphabet = line.split()[-1]
            elif line.startswith("MOTIF"):
                if curr_name is not None:
                    retval[curr_name] = np.vstack(curr_vals)
                    assert retval[curr_name].shape == curr_expected_shape
                    assert np.all(retval[curr_name] <= 1.0) and np.all(retval[curr_name] >= 0.0)
                curr_name = line.replace("MOTIF", "").strip()
                curr_vals = []  # Reset
                curr_expected_shape = None
            elif line.startswith('letter-probability matrix'):
                line = line.replace("letter-probability matrix:", "").strip()
                line_tokens = line.split()
                line_dict = {line_tokens[i].strip("="):line_tokens[i+1] for i in range(0, len(line_tokens), 2)}
                curr_expected_shape = (int(line_dict['w']), int(line_dict['alength']))
            elif line.startswith(" "):  # Starts with blank space
                v = [float(x) for x in line.strip().split()]
                assert len(v) == 4
                curr_vals.append(v)
        assert curr_name is not None and curr_vals
        retval[curr_name] = np.vstack(curr_vals)  # Add in the last chunk
    if log_transform:
        retval = {k: np.log2(v) for k, v in retval.items()}
    return retval

def main():
    """On the fly testing"""
    x = load_meme_ppm()
    #print(score_sequence_with_pwm("ACTAGCGTGACTGACTGACTGACGTGAC", list(x.values())[0], min_prop=0.9))
    print(find_ppm_hits("ATAATTGACTGATCGTAGCTAGCTAC", x["RNCMPT00001 A1CF"], prop=0.9))
    print(find_ppm_hits("ATAATTGACTGATCGTAGCTAGCTAC", x["RNCMPT00001 A1CF"], prop=None, pval=0.01))

if __name__ == "__main__":
    main()

