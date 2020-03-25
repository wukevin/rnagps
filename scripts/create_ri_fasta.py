"""
Code for generating retained intron fasta file for aligning reads to
Augments an existing fasta reference "transcriptome"
"""

import os, sys
import logging
import collections

import pandas as pd
import numpy as np

from pyfaidx import Fasta
from intervaltree import Interval, IntervalTree

SRC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "rnagps"
)
assert os.path.isdir(SRC_DIR), f"Could not find {SRC_DIR}"
sys.path.append(SRC_DIR)

import data_loader
import fasta
import utils

LOCAL_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data"
)
assert os.path.isdir(LOCAL_DATA_DIR), f"Could not find {LOCAL_DATA_DIR}"

TRANS_TO_EXONS = utils.read_gtf_trans_to_exons()
GENOME_FA = Fasta(
    os.path.expanduser("~/genomes/GRCh38/Homo_sapiens.GRCh38.dna_sm.primary_assembly.fa"),
    sequence_always_upper=True,
)

def seq_from_exons_introns(exons, introns, join=True):
    """
    Merges exons and introns and returns sequence
    Note that exons and introns are in different formats
    Exons are tuples of (exn_num, chrom, start, stop, strand, gene) (should be nonoverlapping)
    Introns are tuples of (intron_seq, intron_gcoords) that are nonoverlappping by construction
    We ignore the intron sequence IN CASE the coords overlap
    """
    itree = IntervalTree()
    chroms = set()
    strands = set()
    for exn_num, chrom, start, stop, strand, gene in exons:
        chroms.add(chrom)
        strands.add(strand)
        itree[start:stop] = f"exon_{exn_num}"

    assert len(chroms) == 1
    chrom = chroms.pop()
    assert len(strands) == 1
    strand = strands.pop()

    for i, gcoord in enumerate(introns[1]):
        chrom, startstop, strand = gcoord.split(":")
        start, stop = map(int, startstop.split("-"))
        itree[start:stop] = f"ri_{i}"

    itree_orig = itree.copy()
    itree.merge_overlaps(lambda x, y: ";".join([x, y]))
    if len(itree) != len(itree_orig):
        logging.warn(f"Contains overlaps: {itree_orig}")

    # The itree sorts everything in 5' to 3' regardless of strand
    seqs = []
    for interval in itree:
        # Actual sequences are rev comped properly
        # seq = GENOME_FA[chrom][interval.begin:interval.end]
        seq = GENOME_FA.get_seq(chrom, interval.begin, interval.end, strand == "-")
        assert seq.seq
        seqs.append(seq.seq)

    return ''.join(seqs) if join else seqs

def get_gene_to_trans_map(fa_dict: dict) -> (dict, dict):
    """
    Given a transcript fasta dict with annotated gene info return
    a mapping for gene to transcript, and a mapping from transcript to gene
    """
    gene_to_trans = collections.defaultdict(list)
    trans_to_gene = {}
    for seq_name in fa_dict.keys():
        trans_name = seq_name.split()[0]
        tokens = [t for t in seq_name.split() if t.startswith("gene:ENSG")]
        assert len(tokens) == 1
        gene = tokens[0].strip("gene:")
        gene_to_trans[gene].append(seq_name)
        trans_to_gene[trans_name] = gene
    return gene_to_trans, trans_to_gene

def main():
    """Run the script, writing output to given first argument"""
    logging.basicConfig(level=logging.INFO)
    assert len(sys.argv) == 2
    ri_dataset = data_loader.NLSvNESRetainedIntronDataset()
    # List of ENST*
    ri_transcripts = [ri_dataset.get_most_common_transcript(ri_dataset.gene_ids[i]) for i in range(len(ri_dataset) // 2)]

    # List of tuples (intron_seq, intron_itree)
    ri_introns = [ri_dataset.get_retained_introns(ri_dataset.gene_ids[i], join=False) for i in range(len(ri_dataset) // 2)]

    # read in list of existing sequences and add the RI version to them
    ri_full_seqs = fasta.read_file_as_sequence_dict(os.path.join(LOCAL_DATA_DIR, "Homo_sapiens.GRCh38.merge.90.fa.gz"))
    gene_to_trans, trans_to_gene = get_gene_to_trans_map(ri_full_seqs)
    ri_full_seqs_simple_key = {k.split()[0]:v for k, v in ri_full_seqs.items()}  # Re-key to exclude metadata
    ri_seq_lens, no_ri_seq_lens = [], []
    ri_isoform_seqs = {}
    for trans, introns in zip(ri_transcripts, ri_introns):
        ri_gene = trans_to_gene[trans]
        trans_abridge = trans.split(".").pop(0)
        # List of tuples (exn_num, chrom, start, stop, strand, gene)
        exons = TRANS_TO_EXONS[trans_abridge]
        assert exons, f"Found no matches to {trans_abridge}"

        full_seq = seq_from_exons_introns(exons, introns)
        ri_key = f"{trans}_retained_intron"
        assert ri_key not in ri_full_seqs
        ri_full_seqs[ri_key] = full_seq
        ri_full_seqs[trans] = ri_full_seqs_simple_key[trans]

        ri_seq_lens.append(len(full_seq))
        no_ri_seq_lens.append(len(ri_full_seqs[trans]))

        # Remove any other transcripts from this gene
        for t in gene_to_trans[ri_gene]:
            ri_full_seqs.pop(t)

    logging.info(f"Average length of RI sequences:    {np.mean(ri_seq_lens)}")
    logging.info(f"Average length of no RI sequences: {np.mean(no_ri_seq_lens)}")
    fasta.write_sequence_dict_to_file(ri_full_seqs, sys.argv[1])

if __name__ == "__main__":
    main()

