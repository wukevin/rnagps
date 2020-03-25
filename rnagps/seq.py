"""
Code for manipulating sequences
"""

import functools

import numpy as np

BASE_TO_INT = {
    "A": 0,
    "C": 1,
    "G": 2,
    "T": 3,
    "N": 4,
}

BASE_TO_COMP = {
    "A": "T",
    "T": "A",
    "C": "G",
    "G": "C",
    "N": "N",
}

CHROMOSOMES = ["chr" + str(i + 1) for i in range(22)] + ['chrX', 'chrY']

@functools.lru_cache()
def sequence_to_image(seq, N_strategy='ignore', return_type=int, channel_first=False):
    """
    Convert the given sequence to a 'image' of 4xlen(seq) or 5xlen(seq)

    N_strategy denotes how we deal with N bases:
    - ignore = exclude, so that they are a column of 0's
    - onehot = include, so they form a 5th row
    - quarters = include, so they are [0.25, 0.25, 0.25, 0.25]
    """
    assert N_strategy in ['ignore', 'onehot', 'quarters']
    numeric = np.array([BASE_TO_INT[base] for base in seq], dtype=int)
    encoded = np.zeros((len(numeric), 5), dtype=float)  # Empty array of 0's to populate later
    encoded[np.arange(len(numeric)), numeric] = 1  # One-hot encode the sequence
    if N_strategy == 'ignore':
        encoded = encoded[:, :4]
    elif N_strategy == 'quarters':
        encoded = encoded[:, :4]
        n_idx = np.array([i for i, base in enumerate(seq) if base == "N"])  # Positions where Ns are
        if n_idx.size > 0:  # Nonempty
            encoded[n_idx, :] = np.array([0.25, 0.25, 0.25, 0.25])
    else:
        pass  # under onehot we just leave as is

    retval = encoded.astype(return_type)
    if channel_first:
        retval = retval.T
    return retval

def trim_or_pad(sequence, length, right_align=True):
    """
    Trims or pads the sequence to be of the specified length. Padding is done with N
    Right align is equivalent to left-padding

    >>> trim_or_pad("ACGG", 5)
    'NACGG'
    >>> trim_or_pad("ACGG", 6)
    'NNACGG'
    >>> trim_or_pad("ACGG", 3)
    'CGG'
    >>> trim_or_pad("CGATCGATCG", 5)
    'GATCG'
    >>> trim_or_pad("ACGG", 3, right_align=False)
    'ACG'
    """
    assert length > 0
    if len(sequence) < length:
        n_prefix = "N" * (length - len(sequence))
        sequence = n_prefix + sequence if right_align else sequence + n_prefix
    elif len(sequence) > length:
        sequence = sequence[-length:] if right_align else sequence[:length]
    assert len(sequence) == length
    return sequence

def normalize_chrom_string(s):
    """
    Given a chromosome string that may be weirdly formatted, return chr<X>
    >>> normalize_chrom_string('13')
    'chr13'
    >>> normalize_chrom_string('CHR_HSCHR14_3_CTG1')
    'chr14'
    >>> normalize_chrom_string('CHR_HSCHR7_2_CTG6')
    'chr7'
    >>> normalize_chrom_string('CHR_HSCHR19KIR_G085_A_HAP_CTG3_1')
    'chr19'
    """
    assert isinstance(s, str)
    suffixes = [c.strip('chr') for c in CHROMOSOMES]
    if s in suffixes:
        return 'chr' + s
    else:
        s = s.lower()
        for suffix in ["_", "kir"]:
            matches = [c + suffix in s for c in CHROMOSOMES]
            if sum(matches):
                i = np.where(matches)[0][0]
                return CHROMOSOMES[i]
        raise ValueError(f"Could not match {s}")

if __name__ == "__main__":
    import doctest
    doctest.testmod()

