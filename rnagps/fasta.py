"""
Functions for handling fasta files
"""

import os
import sys
import logging
import collections
import glob
import functools
from typing import List, Dict
import gzip

from Bio.SeqFeature import SeqFeature, FeatureLocation

def write_sequence_dict_to_file(seq_dict:Dict[str, str], fname:str) -> str:
    """Writes the given sequences to the filename"""
    with open(fname, 'w') as sink:
        for header, seq in seq_dict.items():
            if not header.startswith(">"):
                header = ">" + header
            sink.write(header.strip() + "\n")
            sink.write(seq.strip() + "\n")
    return fname

@functools.lru_cache(128)
def read_file_as_sequence_dict(fname:str) -> Dict[str, str]:
    """Reads the file in as a sequence dict"""
    retval = {}
    curr_key = None
    opener = gzip.open if fname.endswith(".gz") else open
    with opener(fname) as source:
        for line in source:
            line = line.strip()
            if not isinstance(line, str):
                line = line.decode()
            if line.startswith(">"):
                curr_key = line[1:].strip()  # Remove >
                retval[curr_key] = []
            else:
                assert curr_key
                retval[curr_key].append(line)
    return {k: ''.join(v) for k, v in retval.items()}

def read_file_as_seqfeature_dict(fname:str, delim:str="-") -> Dict[SeqFeature, str]:
    """
    Reads in file as seqfeature object to string
    """
    naive_dict = read_file_as_sequence_dict(fname)
    retval = {}
    for identifier, sequence in naive_dict.items():
        refdb, start, stop = identifier.split(delim)
        seqfeature = SeqFeature(
            FeatureLocation(int(start), int(stop), ref=refdb),
            ref=refdb,
            type='seq_ft_ablation',
        )
        retval[seqfeature] = sequence
    return retval

def combine_fasta_files(fnames:List[str], dedup_sequences:bool=True) -> Dict[str, str]:
    """
    Combine the given fasta files optionally without writing duplicated sequences
    """
    fasta_dicts = [read_file_as_sequence_dict(fname) for fname in fnames]
    retval = {}
    for seq_dict in fasta_dicts:
        for seq_name, seq in seq_dict.items():
            if not dedup_sequences:
                assert seq_name not in retval
                retval[seq_name] = seq
            else:
                # This could be a lot more efficient but less readable
                # For now not worth it
                if seq not in set(retval.values()):
                    assert seq_name not in retval
                    retval[seq_name] = seq
    if dedup_sequences:
        assert len(list(retval.values())) == len(set(retval.values()))
    return retval

def main():
    """
    Run as script to merge fasta files for COVID19
    Intended to run in the same directory as the fasta files to aggregate
    Outputs to a folder named "aggregated" with filenames "aggregated_gene.fa"
    """
    genes = collections.defaultdict(list)
    for item in glob.glob("*.fa"):
        bname = os.path.basename(item).split(".")[0]
        identifier, gene = bname.split("_")
        genes[gene].append(identifier)

    outdir = os.path.join(os.getcwd(), "aggregated")
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    for gene, identifiers in genes.items():
        fa_files = [f"{i}_{gene}.fa" for i in identifiers]
        combined_seq_dict = combine_fasta_files(fa_files)
        write_sequence_dict_to_file(combined_seq_dict, os.path.join(outdir, f"aggregated_{gene}.fa"))

if __name__ == "__main__":
    main()

