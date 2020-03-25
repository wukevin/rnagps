"""
Code for loading in data

USE THIS FOR FIRST STAB
control = overall cell = unlabelled

Alternatively, look at each compartment separately, and label sequences that way

In sleuth
q values are adjusted p values
b value is estimate of log2 fold change, instead of raw log2foldchange
That said, transcript quantifications are kind of noisy (limited by annotation quality)
Anecdotally, sometimes kallisto can be wonky

Notes on u5 cds u3
- lncRNA is only present in the u3 field
- maybe featurize poly A length separate instead of within 3' UTR
- maybe featurize length of u5 cds and u3?
- adjust kmer count by p(kmer|per-base-abundance)

Also: GKSVM
consider clustering of kmers for large kmers (conslidates counts)

Legend for abbreviations:
ERM - ER membrane
KDEL - negative control (not a dataset that was analyzed, just a biological validation of the method)
LMA - nuclear lamina
Mito - inner mitochondrial matrix
NES - cytosol
NIK - nucleolus
NLS - nucleus
NucPore/NUP - nuclear pore
OMM - outer mitochondrial matrix
"""
import os
import sys
from typing import List
import multiprocessing
import math
import random
import gzip
import socket
import glob
import logging
import collections
import itertools
import functools

import tqdm
import scipy.spatial
import scipy.signal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns

from pyfaidx import Fasta

from intervaltree import Interval, IntervalTree

import torch
from torch.utils import data

from kmer import generate_all_kmers, sequence_to_kmer_freqs, gkm_fv, reverse_complement
from seq import sequence_to_image, trim_or_pad, normalize_chrom_string, BASE_TO_INT, CHROMOSOMES
import pwm
import interpretation
import utils

# Figure out some paths
DATA_DIR = os.path.join(  # Expects this data dir external to src dir
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))),
    "rnafinder_data",
)
assert os.path.isdir(DATA_DIR), "Cannot find data directory: {}".format(DATA_DIR)
LOCAL_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data"))
assert os.path.isdir(LOCAL_DATA_DIR), "Cannot find data directory: {}".format(LOCAL_DATA_DIR)
PLOT_DIR = os.path.join(os.path.dirname(LOCAL_DATA_DIR), "plots")
assert os.path.isdir(PLOT_DIR)
GENOME_FA = os.path.expanduser("~/genomes/GRCh38/Homo_sapiens.GRCh38.dna_sm.primary_assembly.fa")
assert os.path.isfile(GENOME_FA)

K_FOLDS = 10  # Total number of k-folds

LOCALIZATION_FULL_NAME_DICT = {
    "Erm": "ER membrane",
    "Lma": "Nuclear lamina",
    "Mito": "Mito matrix",
    "Nes": "Cytosol",
    "Nik": "Nucleolus",
    "Nls": "Nucleus",
    "NucPore": "Nuclear pore",
    "Omm": "Outer mito membrane",
}
LOCALIZATIONS = tuple(sorted(list(LOCALIZATION_FULL_NAME_DICT.keys())))

class LocalizationTranscriptClassiifcationKmers(data.Dataset):
    """
    Loads in the dataset of transcripts
    """
    def __init__(self, split='train', k_fold:int=0, min_tpm:int=50, max_qval:float=0.05, trans_parts:List[str]=['u5', 'cds', 'u3'], kmer_sizes:List[int]=[3, 4, 5], localizations:List[str]=LOCALIZATIONS, dup_only:bool=False, sort_by_gene:bool=False):
        self.full_deseq_table = pd.read_csv(os.path.join(LOCAL_DATA_DIR, "sleuth_merged.txt.gz"), sep='\t', index_col='target_id')
        self.trans_parts = trans_parts
        self.kmer_sizes = kmer_sizes
        self.dup_only = dup_only
        self.localizations = localizations
        self.qval_colnames = [l + "_qval" for l in self.localizations]
        self.tpm_colnames = [l + "_tpm" for l in self.localizations]

        # Drop rows that fail tpm and qval cutoff for all localizations
        qval_pass_idx = self.full_deseq_table.loc[:, self.qval_colnames] <= max_qval
        tpm_pass_idx = self.full_deseq_table.loc[:, self.tpm_colnames] >= min_tpm
        pass_idx = np.logical_and(qval_pass_idx, tpm_pass_idx)
        at_least_one_pass = np.any(pass_idx, axis=1)
        at_least_one_pass_idx = np.where(at_least_one_pass)
        logging.info(f"Dropping {np.sum(at_least_one_pass == 0)}/{at_least_one_pass.size} rows for having no localizations that pass significance")
        self.full_deseq_table = self.full_deseq_table.iloc[at_least_one_pass_idx]

        # Split by train/valid/test
        np.random.seed(332133)  # Do not change
        indices = np.arange(self.full_deseq_table.shape[0])
        np.random.shuffle(indices)
        # NOTE it may be better to split by chromosome, but that's a lot more code
        indices_split = np.array_split(indices, K_FOLDS)  # Splits into K_FOLDS partitions of approx equal size
        valid_partition = k_fold % K_FOLDS # Define train/valid/test partitions
        test_partition = (k_fold + 1) % K_FOLDS
        assert 0 <= valid_partition < K_FOLDS
        assert 0 <= test_partition < K_FOLDS
        train_partitions = [i for i in range(K_FOLDS) if i != valid_partition and i != test_partition]

        # Figure out which indices to keep based on train/test/valid/all argument
        if split == 'valid':
            keep_indices = indices_split[valid_partition]
        elif split == 'test':
            keep_indices = indices_split[test_partition]
        elif split == 'train':
            keep_indices = np.concatenate([indices_split[i] for i in train_partitions])
        else:
            keep_indices = indices  # Just keep everything, but still shuffle
        self.full_deseq_table = self.full_deseq_table.iloc[keep_indices]
        assert self.full_deseq_table.shape[0] == len(keep_indices)

        if self.dup_only:
            # keep=False indicates we mark all duplicates as true
            self.full_deseq_table = self.full_deseq_table[self.full_deseq_table.duplicated(subset='ensembl_gene', keep=False)]

        if sort_by_gene:
            self.full_deseq_table = self.full_deseq_table.sort_values(by=['ensembl_gene'])

        ### DO NOT MODIFY DESEQ TABLE AFTER THIS POINT
        self.truth_matrix = np.logical_and(
            self.full_deseq_table.loc[:, self.qval_colnames] <= max_qval,
            self.full_deseq_table.loc[:, self.tpm_colnames] >= min_tpm,
        )
        assert self.truth_matrix.shape[1] == len(self.localizations)

        # Load in reference files
        self.fasta_dict = fasta_to_dict(os.path.join(LOCAL_DATA_DIR, "Homo_sapiens.GRCh38.merge.90.fa.gz"))  # dict
        fasta_table = os.path.join(LOCAL_DATA_DIR, "parsed_fasta_rel90.txt.gz")
        self.fasta_table = load_fasta_table(fasta_table)  # This is a dataframe

    def get_ith_trans_parts(self, i):
        """Return the parts of the ith transcript"""
        transcript = self.full_deseq_table.index[i]
        transcript = transcript.split(".")[0]  # Remove the training suffix after period
        transcript_row = self.fasta_table.loc[transcript]
        retval = [transcript_row[p] for p in self.trans_parts]
        return retval

    def get_ith_labels(self, i):
        """Return the ith labels, considering reverse complementing status"""
        retval = self.truth_matrix.iloc[i].to_numpy().astype(int)
        assert retval.size == len(self.localizations)
        return retval

    def __len__(self):
        """Total number of samples"""
        return self.full_deseq_table.shape[0]

    def __getitem__(self, i):
        """Get one sample of data"""
        # X features
        transcript_parts = self.get_ith_trans_parts(i)
        encodings = [sequence_to_kmer_freqs(part, kmer_size=s) for s in self.kmer_sizes for part in transcript_parts]
        seq_encoded = np.hstack(encodings)
        # Y labels (negative genes have their log2fc values zeroed out, so require no special handling)
        labels = self.get_ith_labels(i)
        return torch.from_numpy(seq_encoded).type(torch.FloatTensor), torch.from_numpy(labels).type(torch.FloatTensor)

class LocalizationClassificationKmers(data.Dataset):
    """
    Loads in the dataset containing given localizations
    If localizations is provided, it will determine the order of features
    """
    def __init__(self, split='train', localizations=[], trans_parts=['u5', 'cds', 'u3'], k_fold=0, pval_cutoff=0.05, kmer_sizes=[3, 4, 5], addtl_negatives=False, rc_aug=False, include_len=False, fasta_table=os.path.join(DATA_DIR, "parsed_fasta_rel90.txt")):
        """
        fasta_table is where we read transcript segments
        """
        if split not in ['train', 'valid', 'test', 'all']:
            raise ValueError("Unrecognized split: {}".format(split))
        assert k_fold < K_FOLDS and k_fold >= -1 * K_FOLDS
        compartment_from_label = lambda s: s.split("_")[0]

        self.full_deseq_table = pd.read_csv(os.path.join(LOCAL_DATA_DIR, "190423_deseq_all.txt.gz"), index_col=0, delimiter="\t")
        # Collect column names and optionally filter on them
        self.log2fc_colnames = {compartment_from_label(col): col for col in self.full_deseq_table.columns if col.endswith("_log2FoldChange")}
        self.padj_colnames = {compartment_from_label(col): col for col in self.full_deseq_table.columns if col.endswith("_padj")}
        self.tpm_colnames = {compartment_from_label(col):col for col in self.full_deseq_table.columns if col.endswith("_tpm")}
        self.agnostic_colnames = ["gene_name", "gene_type", "ctrl_avg"]
        assert tuple(sorted(self.log2fc_colnames.keys())) == tuple(sorted(self.padj_colnames.keys())) == tuple(sorted(self.tpm_colnames.keys()))
        self.compartments = list(self.log2fc_colnames.keys())
        logging.info("Read deseq table containing results for localizations: {}".format(" ".join(self.compartments)))
        if not localizations:
            localizations = [l for l in self.compartments if l != "Kdel"]
        assert "Kdel" not in localizations, "Cannot include Kdel control"
        if localizations:
            assert all([l in self.compartments for l in localizations])
            logging.info("Retaining data only for {}".format(" ".join(localizations)))
            columns_to_keep = list(self.agnostic_colnames)
            for compartment in localizations:
                columns_to_keep.append(self.tpm_colnames[compartment])
                columns_to_keep.append(self.log2fc_colnames[compartment])
                columns_to_keep.append(self.padj_colnames[compartment])
            columns_to_drop = [col for col in self.full_deseq_table.columns if col not in columns_to_keep]
            self.full_deseq_table.drop(columns=columns_to_drop, inplace=True)
            self.compartments = list(localizations)
            filt_dict_keys = lambda dictionary, desired_keys: {k: v for k, v in dictionary.items() if k in desired_keys}
            self.log2fc_colnames = filt_dict_keys(self.log2fc_colnames, self.compartments)
            self.padj_colnames = filt_dict_keys(self.padj_colnames, self.compartments)
            self.tpm_colnames = filt_dict_keys(self.tpm_colnames, self.compartments)

        # Drop rows where values are always NaN
        self.full_deseq_table.dropna(axis='index', how='all', inplace=True, subset=self.log2fc_colnames.values())
        self.full_deseq_table.dropna(axis='index', how='all', inplace=True, subset=self.tpm_colnames.values())
        self.full_deseq_table.dropna(axis='index', how='all', inplace=True, subset=self.padj_colnames.values())

        # Drop rows that don't pass p value cutoff for any localization, and aren't positive enrichment
        padj_matrix = self.full_deseq_table.loc[:, [self.padj_colnames[c] for c in self.compartments]].values
        padj_matrix[np.isnan(padj_matrix)] = 1  # Set nan to 1 - these will not pass cutoff anyway and avoids error thrown
        log2fc_matrix = self.full_deseq_table.loc[:, [self.log2fc_colnames[c] for c in self.compartments]]
        log2fc_matrix[np.isnan(log2fc_matrix)] = 0  # Set nan to 0 as this will not trip the cutoff
        is_significant = np.any(np.logical_and(padj_matrix <= pval_cutoff, log2fc_matrix > 0), axis=1)  # Just needs significant in at least 1 column
        logging.info("Retaining {}/{} genes as significant".format(np.sum(is_significant), len(is_significant)))
        discard_genes = [gene for i, gene in enumerate(self.full_deseq_table.index) if not is_significant[i]]
        # Pick some genes to keep even though they're not significantly localized anywhere
        random.seed(1234)
        if addtl_negatives:
            self.negative_genes = set(random.sample(discard_genes, np.sum(is_significant)))  # Sample, w/o replacement, equal number of negatives
            discard_genes = [gene for gene in discard_genes if gene not in self.negative_genes]
            logging.info("Retaining {} genes with no significant enrichment".format(len(self.negative_genes)))
        else:
            self.negative_genes = set()  # Empty set
        self.full_deseq_table.drop(inplace=True, index=discard_genes)
        assert self.full_deseq_table.shape[0] == np.sum(is_significant) + len(self.negative_genes)

        # randomize, drop based on train/valid/test
        np.random.seed(332133)  # Do not change
        indices = np.arange(self.full_deseq_table.shape[0])
        np.random.shuffle(indices)
        # NOTE it may be better to split by chromosome, but that's a lot more code
        indices_split = np.array_split(indices, K_FOLDS)  # Splits into K_FOLDS partitions of approx equal size
        valid_partition = k_fold % K_FOLDS # Define train/valid/test partitions
        test_partition = (k_fold + 1) % K_FOLDS
        assert 0 <= valid_partition < K_FOLDS
        assert 0 <= test_partition < K_FOLDS
        train_partitions = [i for i in range(K_FOLDS) if i != valid_partition and i != test_partition]

        # Figure out which indices to keep based on train/test/valid/all argument
        if split == 'valid':
            keep_indices = indices_split[valid_partition]
        elif split == 'test':
            keep_indices = indices_split[test_partition]
        elif split == 'train':
            keep_indices = np.concatenate([indices_split[i] for i in train_partitions])
        else:
            keep_indices = indices  # Just keep everything, but still shuffle
        self.full_deseq_table = self.full_deseq_table.iloc[keep_indices]
        assert self.full_deseq_table.shape[0] == len(keep_indices)

        self.rc_aug = rc_aug
        if self.rc_aug:
            logging.info("Using reverse complement for data augmentation")
        ### DO NOT MODIFY self.full_deseq_table AFTER THIS POINT ###

        # Store the padj values in a separate matrix
        self.padj_matrix = self.full_deseq_table.loc[:, [col for col in self.full_deseq_table.columns if col.endswith("_padj")]].values
        self.padj_matrix[np.isnan(self.padj_matrix)] = 1  # Set nan to 1 - this won't pass significance anyway
        # Store the log2 fold change values in a separate matrix
        self.log2fc_matrix = self.full_deseq_table.loc[:, [col for col in self.full_deseq_table.columns if col.endswith("_log2FoldChange")]].values
        self.log2fc_matrix[np.isnan(self.log2fc_matrix)] = 0  # Set nan to 0 - this won't pass the cutoff
        if addtl_negatives:
            negative_gene_indices = [i for i, gene in enumerate(self.full_deseq_table.index) if gene in self.negative_genes]
            self.log2fc_matrix[negative_gene_indices, :] = 0  # Zero out negative genes
        self.truth_matrix = np.logical_and(self.log2fc_matrix > 0, self.padj_matrix <= pval_cutoff)
        per_category_positives = np.sum(self.truth_matrix.astype(int), axis=0)
        for compartment, count in zip(self.compartments, per_category_positives):
            logging.info(f"{compartment} - {count}/{self.log2fc_matrix.shape[0]} = {count / self.log2fc_matrix.shape[0]} positive")
        if np.any(np.all(self.truth_matrix, axis=1)):
            logging.warn("Encountered rows with positive labels in all compartments")

        # Store other vars
        self.kmer_sizes = kmer_sizes
        self.trans_parts = trans_parts
        self.pval_cutoff = pval_cutoff
        self.split = split
        self.include_len = include_len

        # Load in auxillary files
        self.fasta_dict = fasta_to_dict(os.path.join(DATA_DIR, "Homo_sapiens.GRCh38.merge.90.fa"))  # dict
        self.fasta_table = load_fasta_table(fasta_table)  # This is a dataframe
        genes_to_transcripts_table = pd.read_csv(
            os.path.join(DATA_DIR, "Homo_sapiens.GRCh38.90.merge_transcriptome.fa.processed.txt"),
            sep="\t",
        )
        self.genes_to_transcripts = collections.defaultdict(list)
        for row_tuple in genes_to_transcripts_table.itertuples():
            self.genes_to_transcripts[row_tuple.gene].append(row_tuple.transcript)
        self.kallisto_table = load_kallisto_table()
        self.kallisto_ctrl_colnames = [col for col in self.kallisto_table.columns if col.split("_")[1] == 'control']

        # Calculate metrics for transcript proportion
        tc = TranscriptClassifier()
        tcounter = collections.Counter(self.full_deseq_table['gene_type'])
        ttotal = self.full_deseq_table.shape[0]
        for tt, tcount in tcounter.most_common():
            logging.info("{}\t{}\t{}".format(tcount, tcount/ttotal, tt))

    def __len__(self):
        """Total number of samples"""
        return self.full_deseq_table.shape[0] if not self.rc_aug else self.full_deseq_table.shape[0] * 2

    def get_representative_trans(self, gene, transcripts_only=True):
        """Return the name of the representative transcript for the given gene"""
        transcript_ids = self.genes_to_transcripts[gene]  # Find transcripts belonging to this gene
        if transcripts_only:
            transcript_ids = [t for t in transcript_ids if t.startswith("ENST")]
        # Figure out the most common transcript. Given the same gene, this should always return the same transcript.
        # Use only control to determine most common transcript
        # Average over ALL controls, including KDEL
        kallisto_subset_summed = self.kallisto_table.loc[transcript_ids, self.kallisto_ctrl_colnames].sum(axis=1)
        most_common_transcript = kallisto_subset_summed.idxmax()
        return most_common_transcript

    def get_ith_trans_seq(self, i, transcripts_only=True):
        """Return the sequece of the ith transcript"""
        gene = self.full_deseq_table.index[i if not self.rc_aug else i // 2]
        transcript = self.get_representative_trans(gene)
        seq = self.fasta_dict[transcript]
        if self.rc_aug and i % 2 == 1:  # With RC augmentation, odd indices are RCs
            seq = reverse_complement(seq)
        return seq

    def get_ith_trans_parts(self, i, transcripts_only=True):
        """Return the parts of the ith transcript"""
        gene = self.full_deseq_table.index[i if not self.rc_aug else i // 2]
        transcript = self.get_representative_trans(gene)
        transcript = transcript.split(".")[0]  # Remove the training suffix after period
        transcript_row = self.fasta_table.loc[transcript]
        retval = [transcript_row[p] for p in self.trans_parts]
        if self.rc_aug and i % 2 == 1:
            # We don't reverse order of u5 cds and u3, just the sequences within those parts
            retval = tuple([reverse_complement(s) for s in retval])
        return retval

    def get_ith_labels(self, i):
        """Return the ith labels, considering reverse complementing status"""
        effective_index = i if not self.rc_aug else i // 2
        retval = self.truth_matrix[np.newaxis, effective_index, :].astype(int)
        assert len(retval.shape) == 2
        return retval

    def __getitem__(self, i):
        """Get one sample of data"""
        # X features
        transcript_parts = self.get_ith_trans_parts(i)
        encodings = [sequence_to_kmer_freqs(part, kmer_size=s) for s in self.kmer_sizes for part in transcript_parts]
        seq_encoded = np.hstack(encodings)
        if self.include_len:
            seq_encoded = np.insert(seq_encoded, 0, sum([len(t) for t in transcript_parts]))
        # Y labels (negative genes have their log2fc values zeroed out, so require no special handling)
        labels = self.get_ith_labels(i)
        return torch.from_numpy(seq_encoded).type(torch.FloatTensor), torch.from_numpy(labels).type(torch.FloatTensor)

    def get_feature_labels(self):
        """Return feature labels"""
        labels_from_part_and_k = lambda p, k: ["{}_{}".format(p, "".join(x)) for x in itertools.product("ACGT", repeat=k)]
        feature_labels = [labels_from_part_and_k(part, s) for s in self.kmer_sizes for part in self.trans_parts]
        feature_labels_flat = list(itertools.chain.from_iterable(feature_labels))
        return feature_labels_flat

    def compute_localization_similarity_matrix(self, metric=scipy.spatial.distance.cosine):
        """Return a matrix where each cell represents the distance in localization pattern between two compartments"""
        mat = np.zeros((len(self.compartments), len(self.compartments)))
        for i, x in enumerate(self.compartments):
            for j, y in enumerate(self.compartments):
                dist = metric(self.truth_matrix[:, i], self.truth_matrix[:, j])
                mat[i, j] = dist
        retval = pd.DataFrame(
            mat,
            index=self.compartments,
            columns=self.compartments,
        )
        if not np.allclose(retval.values, retval.values.T):
            logging.warn("Localization similarity matrix does not appear to be symmetric!")
        return retval

    def plot_truth_histogram(self, fname):
        """Plots a histogram of how many compartments a transcript is localized in"""
        counts = np.sum(self.truth_matrix, axis=1).astype(int)
        ax = sns.distplot(counts, kde=False, bins=max(counts))
        ax.set(
            xlabel="Number of significant localizations",
            ylabel="Frequency",
            title="Histogram of transcript localizations ({})".format(self.split),
        )
        fig = ax.get_figure()
        logging.info("Saving histogram to {}".format(fname))
        fig.savefig(fname, dpi=600)

    def plot_most_common_colocalizations(self, fname, co=2, n=10):
        """Plot the top n colocalizations of co localizations"""
        combos = []
        for i in range(self.truth_matrix.shape[0]):
            row = self.truth_matrix[i, :].flatten()
            if np.sum(row) != co:
                continue
            combo = tuple(sorted([l for i, l in enumerate(self.compartments) if row[i]]))
            combos.append(combo)
        counter = collections.Counter(combos)
        most_common = counter.most_common(n=n)
        labels, vals = zip(*most_common)
        df = pd.DataFrame({"localization": ["/".join(l) for l in labels], "counts": vals})

        fig, ax = plt.subplots(dpi=300)
        ax = sns.barplot(
            x='localization',
            y='counts',
            data=df,
            ax=ax,
        )
        ax.set_xticklabels(labels=["/".join(l) for l in labels], rotation=90)
        ax.set(
            title="{} most common {}-way colocalizations".format(len(vals), co),
            xlabel="Colocalizations",
            ylabel="Count",
        )
        plt.tight_layout()
        fig.savefig(fname)

class LocalizationClassificationGKmers(LocalizationClassificationKmers):
    """
    Loads in the dataset containing given localizations
    If localizations is provided, it will determine order of features
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        for k in self.kmer_sizes:
            assert isinstance(k, tuple) and k[1] < k[0], "For GKmers, kmer sizes must be a list of tuples"

    def __getitem__(self, i):
        # X features
        transcript_parts = self.get_ith_trans_parts(i)
        gkm_encodings = [gkm_fv(part, l=l, k=k) for l, k in self.kmer_sizes for part in transcript_parts]
        seq_encoded = np.hstack(gkm_encodings)
        # Ylabels
        labels = self.get_ith_labels(i)
        return torch.from_numpy(seq_encoded).type(torch.FloatTensor), torch.from_numpy(labels).type(torch.FloatTensor)

class LocalizationClassificationGKmersNormalized(LocalizationClassificationGKmers):
    """
    Loads in dataset containing given localizations
    Same as above GKmers, but will normalize counts by sequence length
    """
    def __getitem__(self, i):
        # X features
        transcript_parts = self.get_ith_trans_parts(i)
        gkm_encodings = [gkm_fv(part, l=l, k=k, normalize=True) for l, k in self.kmer_sizes for part in transcript_parts]
        seq_encoded = np.hstack(gkm_encodings)
        # Ylabels
        labels = self.get_ith_labels(i)
        return torch.from_numpy(seq_encoded).type(torch.FloatTensor), torch.from_numpy(labels).type(torch.FloatTensor)

class LocalizationClassificationGKmersRevComp(LocalizationClassificationGKmers):
    """
    Loads in dataset containing given localizations
    Uses reverse complemented gapped kmers in order to reduce feature set
    """
    def __getitem__(self, i):
        # x features
        transcript_parts = self.get_ith_trans_parts(i)
        gkm_encodings = [gkm_fv(part, l=l, k=k, rev_comp=True) for l, k in self.kmer_sizes for part in transcript_parts]
        seq_encoded = np.hstack(gkm_encodings)
        # ylabels
        labels = self.get_ith_labels(i)
        return torch.from_numpy(seq_encoded).type(torch.FloatTensor), torch.from_numpy(labels).type(torch.FloatTensor)

class LocalizationClassificationGKmersRevCompTensor(LocalizationClassificationGKmers):
    """
    Loads in dataset containing given localizations
    Uses reverse complemented gapped kmers in order to reduce feature set, but does so
    by vertically stacking into a tensor instead of horizontally concatenating
    """
    def __getitem__(self, i):
        # x features
        transcript_parts = self.get_ith_trans_parts(i)
        gkm_encodings = [gkm_fv(part, l=l, k=k, rev_comp=True) for l, k in self.kmer_sizes for part in transcript_parts]
        horiz_parts = [np.hstack([gkm_encodings[i] for i in range(j, len(gkm_encodings), 3)]) for j in range(3)]
        seq_encoded = np.vstack(horiz_parts)
        # ylabels
        labels = self.get_ith_labels(i)
        return torch.from_numpy(seq_encoded).type(torch.FloatTensor), torch.from_numpy(labels).type(torch.FloatTensor)


class LocalizationClassificationOneHot(LocalizationClassificationKmers):
    """
    Lads in dataset containg given localizations.
    Returns features as one-hot-encoded sequence
    """
    def __getitem__(self, i):
        """Returns one-hot"""
        seq = self.get_ith_trans_seq(i)
        seq = trim_or_pad(seq, 1000)
        seq_encoded = sequence_to_image(seq)
        labels = self.get_ith_labels(i)
        return torch.from_numpy(seq_encoded).type(torch.FloatTensor), torch.from_numpy(labels).type(torch.FloatTensor)

class LocalizationClassificationOneHotPartitioned(LocalizationClassificationOneHot):
    """
    Loads in dataset containing given localizations
    Returns features as a tensor of one-hot-encoded sequence per portion of transcript
    Meant for CNN based architectures
    """
    def __getitem__(self, i):
        """Returns one-hot tensor"""
        seq_len = 250
        transcript_parts = self.get_ith_trans_parts(i)
        #  One hot encode each part of the transcript, trimming or padding each to 250 bp
        seq_encoded = np.stack([sequence_to_image(trim_or_pad(part, seq_len)) for part in transcript_parts], axis=0)
        assert seq_encoded.shape == (3, seq_len, 4)

        labels = self.get_ith_labels(i)
        return torch.from_numpy(seq_encoded).type(torch.FloatTensor), torch.from_numpy(labels).type(torch.FloatTensor)

class LocalizationClassificationOneHotUntrimmed(LocalizationClassificationKmers):
    """
    Loads in dataset containing given localizations
    Returns features as one-hot-encoded sequence that is untrimmed
    This is meant for RNN input
    """
    def __getitem__(self, i):
        seq = self.get_ith_trans_seq(i)
        seq_encoded = sequence_to_image(seq)  # This should be len(seq) x 1 x 4
        seq_encoded = np.expand_dims(seq_encoded, axis=1)
        assert seq_encoded.shape == (len(seq), 1, 4)
        labels = self.get_ith_labels(i)
        return torch.from_numpy(seq_encoded).type(torch.FloatTensor), torch.from_numpy(labels).type(torch.FloatTensor)

class LocalizationClassificationOneHotQuarters(LocalizationClassificationOneHot):
    """
    Loads in dataset containing given loocalizations
    Returns features as one-hot-encoded sequence with N bases represented as [0.25, 0.25, 0.25, 0.25]
    Meant for reproducing RNAtracker architecture
    """
    def __getitem__(self, i):
        seq = self.get_ith_trans_seq(i)
        seq = trim_or_pad(seq, 4000, right_align=True)
        seq_encoded = sequence_to_image(seq, N_strategy='quarters', return_type=float, channel_first=True)
        labels = self.get_ith_labels(i)
        labels = np.squeeze(labels)

        x = torch.from_numpy(seq_encoded).type(torch.FloatTensor)
        y = torch.from_numpy(labels).type(torch.FloatTensor)
        return x, y

class LocalizationClassificationIdxUntrimmed(LocalizationClassificationKmers):
    """
    Loads in dataset contain given localizations
    Returns features as indices of one hot encoding that is untrimmed
    This is meant for input to a LSTM network with embeddings since the embedding layer
    essentially adds another dimension to the end
    """
    def __getitem__(self, i):
        seq = self.get_ith_trans_seq(i)
        # Add 1 becfause we consider 0 to be a padding index
        seq_encoded = np.array([BASE_TO_INT[b] + 1 for b in seq])
        # seq_encoded = np.expand_dims(seq_encoded, axis=1)
        seq_encoded = seq_encoded[:, np.newaxis, np.newaxis]
        assert seq_encoded.shape == (len(seq), 1, 1), f"Got unexpected shape {seq_encoded.shape}"
        labels = self.get_ith_labels(i)
        return torch.from_numpy(seq_encoded).type(torch.FloatTensor), torch.from_numpy(labels).type(torch.FloatTensor)

class NLSvNESDataset(data.Dataset):
    """
    Loads in nucleus vs cytosol gene-level dataset
    Internally partitions the data into 10 folds, and uses the given k_fold argument to return a subset of the data
    """
    def __init__(self, split='train', mode='padj', k_fold=0, deseq_fname=os.path.join(LOCAL_DATA_DIR, "deseq_nls_nes.txt"), pval_cutoff=0.05, kallisto_table=os.path.join(DATA_DIR, "apex_kallisto_tpm.txt"), fixed_seq_len=0, kmer_sizes=[], trans_types=[], seq_by_parts=False, pwms=None, chr_ft=False):
        if split not in ['train', 'valid', 'test', 'all']:
            raise ValueError("Unrecognized split: {}".format(split))
        assert k_fold < K_FOLDS

        self.full_deseq_table = pd.read_csv(deseq_fname, index_col=0, delimiter="\t")
        self.full_deseq_table.dropna(subset=['log2FoldChange'], inplace=True)
        if mode == 'padj':
            deseq_table_significant = self.full_deseq_table[self.full_deseq_table['padj'] < pval_cutoff]  # Only take significant genes
        elif mode == 'quartile':
            log2fc_values = self.full_deseq_table['log2FoldChange']
            log2fc_values = log2fc_values[~np.isnan(log2fc_values)]
            first_quartile, third_quartile = np.quantile(log2fc_values, [0.25, 0.75])
            logging.info("log2fc first and third quartiles: {} {}".format(first_quartile, third_quartile))
            deseq_table_low = self.full_deseq_table[self.full_deseq_table['log2FoldChange'] < first_quartile]
            deseq_table_high = self.full_deseq_table[self.full_deseq_table['log2FoldChange'] > third_quartile]
            deseq_table_significant = deseq_table_low.append(deseq_table_high)
        else:
            raise ValueError("Unrecognized mode: {}".format(mode))
        logging.info("{}/{} entries in table retained as significant".format(deseq_table_significant.shape[0], self.full_deseq_table.shape[0]))

        # randomize, drop based on train/valid/test
        np.random.seed(332133)  # Do not change
        indices = np.arange(deseq_table_significant.shape[0])
        np.random.shuffle(indices)
        # NOTE it may be better to split by chromosome, but that's a lot more code
        indices_split = np.array_split(indices, K_FOLDS)  # Splits into K_FOLDS partitions of approx equal size
        valid_partition = k_fold  # Define train/valid/test partitions
        test_partition = (k_fold + 1) % K_FOLDS
        train_partitions = [i for i in range(K_FOLDS) if i != valid_partition and i != test_partition]

        # Figure out which indices to keep based on train/test/valid/all argument
        if split == 'valid':
            keep_indices = indices_split[valid_partition]
        elif split == 'test':
            keep_indices = indices_split[test_partition]
        elif split == 'train':
            keep_indices = np.concatenate([indices_split[i] for i in train_partitions])
        else:
            keep_indices = indices  # Just keep everything, but still shuffle
        # Passes significance check and is in the data split partition
        deseq_table_subsetted = deseq_table_significant.iloc[keep_indices]

        # Load in auxillary files
        self.fasta_dict = fasta_to_dict(os.path.join(DATA_DIR, "Homo_sapiens.GRCh38.merge.90.fa"))  # dict
        self.fasta_table = load_fasta_table(no_periods=False)  # This is a dataframe
        genes_to_transcripts_table = pd.read_csv(
            os.path.join(DATA_DIR, "Homo_sapiens.GRCh38.90.merge_transcriptome.fa.processed.txt"),
            sep="\t",
        )
        self.genes_to_transcripts = collections.defaultdict(list)
        for row_tuple in genes_to_transcripts_table.itertuples():
            self.genes_to_transcripts[row_tuple.gene].append(row_tuple.transcript)
        # Normalize for counts (use TPM instead of counts)
        self.kallisto_table = load_kallisto_table(kallisto_table)
        # Drop columns that don't pertain to our interest
        kallisto_cols_to_drop = [col for col in self.kallisto_table.columns if not col.lower().startswith("nes") and not col.lower().startswith('nls')]
        self.kallisto_table.drop(columns=kallisto_cols_to_drop, inplace=True)
        # Select correct transcript types
        if trans_types:
            tc = TranscriptClassifier()
            ttypes = [tc.get_transcript_type(self.get_most_common_transcript(g)) for g in deseq_table_subsetted.index]
            ttype_match_indices = [i for i, t in enumerate(ttypes) if t in trans_types]
            deseq_table_subsetted = deseq_table_subsetted.iloc[ttype_match_indices]
            assert deseq_table_subsetted.shape[0] == len(ttype_match_indices)
            logging.info(f"Retained {len(ttype_match_indices)} transcript matching types {trans_types}")

        self.deseq_table_subsetted = deseq_table_subsetted

        ##### DO NOT MODIFY THE TABLE AFTER THIS POINT #####
        self.gene_ids = deseq_table_subsetted.index
        # Log2fc > 0.75
        # try quartiles
        # Or try taking padj < 0.05 and take log2foldchange > 0.5 as "positive" set, select matching "negative" set
        # Check for genes found in both (because they are compared against cytoplasm, so you can have enrichment in both)
        # drop < 10 bp transcripts
        self.labels = deseq_table_subsetted['log2FoldChange'] > 0  # Indicates positive enrichment
        self.kmer_sizes = [kmer_sizes] if isinstance(kmer_sizes, int) else kmer_sizes  # List of kmer sizes we use to featurize
        self.seq_by_parts = seq_by_parts  # Binary flag
        self.fixed_seq_len = fixed_seq_len
        logging.info("Proportion of positives in {}: {}".format(split, np.sum(self.labels) / len(self.labels)))

        self.chr_features = chr_ft
        self.transcript_to_chrom = {}
        for row_tuple in genes_to_transcripts_table.itertuples():
            try:
                norm_chrom = normalize_chrom_string(row_tuple.chr)
                self.transcript_to_chrom[row_tuple.transcript] = norm_chrom
            except ValueError:  # Try to ignore these
                pass
        # Check that we have everything we need
        # for i in range(len(self)):
        #     t = self.get_most_common_transcript(self.gene_ids[i])
        #     assert t in self.transcript_to_chrom, f"{t} does not have an associated chromosome"

        self.pwms = pwms
        if self.pwms:
            assert self.kmer_sizes, "Cannot use PWMs unless we are also using kmer featurization"
            assert isinstance(self.pwms, collections.OrderedDict)
            # Create the feature matrix ahead of time to improve training performance
            pwm_features = []
            sequences = [self.__get_most_common_transcript_sequence(self.gene_ids[i]) for i in range(len(self))]
            pool = multiprocessing.Pool(8)
            for seq in sequences:
                pwm_vals = np.array(pool.starmap(pwm.score_sequence_with_ppm, iterable=[(seq, p) for p in self.pwms.values()], chunksize=10))
                pwm_features.append(pwm_vals)
            pool.close()
            pool.join()
            self.pwm_matrix = np.vstack(pwm_features)

        # Report metrics on proportion of transcripts in each category
        tc = TranscriptClassifier()
        ttypes = [tc.get_transcript_type(self.get_most_common_transcript(g)) for g in self.gene_ids]
        self.tcounter = collections.Counter(ttypes)
        ttotal = sum(self.tcounter.values())
        for tt, tcount in self.tcounter.most_common():
            logging.info("{}\t{}\t{}".format(tcount, tcount/ttotal, tt))

    def __len__(self):
        """Total number of samples"""
        return len(self.labels)

    def __getitem__(self, i):
        """Get one sample of data"""
        gene = self.gene_ids[i]  # Desired gene
        if self.seq_by_parts:
            seq = self.__get_most_common_transcript_sequence_parts(gene)
        else:
            seq = self.__get_most_common_transcript_sequence(gene)  # Fetch the sequence for the most common transcript
        # Note that fixed seq len trimming occurs regardless of featurization
        if self.fixed_seq_len != 0:  # Default value here is 0, resulting in no trimming
            trim_amount = abs(self.fixed_seq_len)
            trim_from_right = True if self.fixed_seq_len < 0 else False
            if self.seq_by_parts:
                seq = tuple([trim_or_pad(part, trim_amount, right_align=trim_from_right) for part in seq])
            else:
                seq = trim_or_pad(seq, trim_amount, right_align=trim_from_right)
        # Return as sequence, or as kmer table based on input
        if not self.kmer_sizes:  # One-hot encode the transcript
            if self.seq_by_parts:
                raise NotImplementedError("Cannot one hot encode when splitting sequence by parts")
            else:
                seq_encoded = sequence_to_image(seq)
        else:  # kmer table
            if self.seq_by_parts:
                # Iterates all the 4-mers for all parts before moving to 5-mers
                encodings = [sequence_to_kmer_freqs(part, kmer_size=s) for s in self.kmer_sizes for part in seq]
            else:
                encodings = [sequence_to_kmer_freqs(seq, kmer_size=s) for s in self.kmer_sizes]
            seq_encoded = np.hstack(encodings)
            # If we are using kmers, we can optionally also use pwms as well.
            if self.pwms:
                pwm_vals = self.pwm_matrix[i, :]
                seq_encoded = np.hstack([seq_encoded, pwm_vals])
        if self.chr_features:
            seq_encoded = np.append(seq_encoded, CHROMOSOMES.index[self.transcript_to_chrom[gene]])
        label = np.atleast_1d(np.array(self.labels[gene])).astype(int)
        x = torch.from_numpy(seq_encoded).type(torch.FloatTensor)
        y = torch.from_numpy(label).type(torch.FloatTensor)
        return x, y

    @functools.lru_cache()
    def get_most_common_transcript(self, gene, transcripts_only=True):
        """Return the name of the most common transcript for the given gene"""
        transcript_ids = self.genes_to_transcripts[gene]  # Find transcripts belonging to this gene
        if transcripts_only:
            transcript_ids = [t for t in transcript_ids if t.startswith("ENST")]
        # Figure out the most common transcript. Given the same gene, this should always return the same transcript.
        # Sum over all compartments (in this case, NES and NLS, since we dropped the others)
        # transcript = u5 + cds + u3
        # protein trunc is the first 100 bases (usually thought to be important in localization)
        kallisto_subset_summed = self.kallisto_table.loc[transcript_ids].sum(axis=1)
        most_common_transcript = kallisto_subset_summed.idxmax()
        return most_common_transcript

    def __get_most_common_transcript_sequence(self, gene):
        """Return the most common transcript's sequence based on kallisto tpm counts"""
        most_common_transcript = self.get_most_common_transcript(gene)
        return self.fasta_dict[most_common_transcript]

    def __get_most_common_transcript_sequence_parts(self, gene):
        """Return the most common transcript's sequence components based on kallisto tpm counts"""
        most_common_transcript = self.get_most_common_transcript(gene)
        row = self.fasta_table.loc[most_common_transcript]
        return (row['u5'], row['cds'], row['u3'])

    def get_feature_labels(self):
        """Return the feature labels as a list of strings"""
        if not self.kmer_sizes:
            raise NotImplementedError("Cannot get feature labels for one-hot encoded sequence")
        # if self.pwms:
        #     raise NotImplementedError("Cannot get feature labels when PWMs are involved")
        labels = []
        if self.seq_by_parts:
            part_names = ['u5', 'cds', 'u3']
            for kmer_size in self.kmer_sizes:
                kmer_dict = generate_all_kmers(kmer_size)  # Maps kmer to index
                kmer_list = ["" for _ in range(len(kmer_dict))]
                for kmer, index in kmer_dict.items():
                    kmer_list[index] = kmer
                for part in part_names:
                    for kmer in kmer_list:
                        labels.append("{}_{}".format(part, kmer))
        else:
            for kmer_size in self.kmer_sizes:
                kmer_dict = generate_all_kmers(kmer_size)  # Maps kmer to index
                kmer_list = ["" for _ in range(len(kmer_dict))]
                for kmer, index in kmer_dict.items():
                    kmer_list[index] = kmer
                labels.extend(kmer_list)
        # Insert the PWM labels
        if self.pwms:
            labels.extend(self.pwms.keys())
        return labels

    def plot_log2_fold_change(self, fname=None):
        """Plot the log2 fold change values (unfiltered) as a histogram, saving to fname if provided"""
        fig, ax = plt.subplots()
        values = self.full_deseq_table['log2FoldChange']
        values = values[~np.isnan(values)]
        ax.hist(values, bins=30)
        ax.set(
            title='log2 fold change histogram',
            ylabel='Count',
            xlabel='log2 fold change'
        )
        if fname:
            fig.savefig(fname)
        else:
            fig.show()

    def plot_sequence_lengths(self, fname=None, log_scale=True):
        """Plot a histogram of the sequence lengths (AFTER filtering), saving to fname if provided"""
        sequences = [self._NLSvNESDataset__get_most_common_transcript_sequence(g) for g in self.gene_ids]
        lengths = np.array([len(s) for s in sequences], dtype=int)
        if log_scale:
            lengths = np.log(lengths)
        fig, ax = plt.subplots()
        ax.hist(lengths, bins=30)
        ax.set(
            title="Sequence lengths histogram",
            xlabel='log (ln) sequence length' if log_scale else 'sequence length',
            ylabel="Count",
        )
        if fname:
            fig.savefig(fname)
        else:
            fig.show()

class NLSvNESRetainedIntronDataset(data.Dataset):
    """
    Take transcripts with retained introns and return versions with and without the retained introns
    Presence of retained introns should correlate with increased nuclear localization
    EVEN queries are without retained intron, with label of 0
    ODD queries are WITH retained intor, with label of 1 indicating nuclear retention/positive intron retention
    """
    def __init__(self, split:str='all', k_fold:int=0, pval_cutoff:float=0.05, kmer_sizes:List[int]=[3, 4, 5]):
        if split != 'all':
            raise NotImplementedError("Data splits for retained intron dataset is not supported")
        assert k_fold < K_FOLDS
        self.kmer_sizes = kmer_sizes
        kallisto_table = os.path.join(DATA_DIR, "apex_kallisto_tpm.txt")
        self.kallisto_table = load_kallisto_table(kallisto_table)

        # Load in resource files
        self.faidx = Fasta(GENOME_FA, sequence_always_upper=True)
        self.fasta_table = load_fasta_table(no_periods=False)  # This is a dataframe
        genes_to_transcripts_table = pd.read_csv(
            os.path.join(DATA_DIR, "Homo_sapiens.GRCh38.90.merge_transcriptome.fa.processed.txt"),
            sep="\t",
        )
        self.genes_to_transcripts = collections.defaultdict(list)
        for row_tuple in genes_to_transcripts_table.itertuples():
            self.genes_to_transcripts[row_tuple.gene].append(row_tuple.transcript)


        ri_table = os.path.join(LOCAL_DATA_DIR, "apex_retained_introns", "apex_nes-nls_RI.MATS.JCEC.txt")
        self.ri_table = pd.read_csv(ri_table, sep="\t", engine="c", low_memory=False, index_col=0)
        # Nuclear side has bigger tail
        keep_idx = np.logical_and(
            self.ri_table['FDR'] <= pval_cutoff,
            self.ri_table['IncLevelDifference'] < 0,  # Using > yields 138 rows, using < yields 2154 rows
        )
        self.ri_table = self.ri_table.iloc[np.where(keep_idx)]
        assert np.all(self.ri_table['FDR'] <= pval_cutoff)
        # Remove entries with no transcripts

        ### DO NOT MODIFY ri_table AFTER THIS POINT
        gene_ids = sorted(list(set(self.ri_table['GeneID'])))
        self.gene_ids = [g for g in gene_ids if self.genes_to_transcripts[g]]
        delta = len(gene_ids) - len(self.gene_ids)
        if delta:
            logging.warn(f"Omitted {delta} genes with no transcript information")
        random.seed(934)
        random.shuffle(self.gene_ids)

    def get_most_common_transcript(self, gene, transcripts_only=True):
        """Return the name of the most common transcript for the given gene"""
        transcript_ids = self.genes_to_transcripts[gene]  # Find transcripts belonging to this gene
        assert transcript_ids, f"Got no transcript matches for gene {gene}"
        if transcripts_only:
            transcript_ids = [t for t in transcript_ids if t.startswith("ENST")]
        # Figure out the most common transcript. Given the same gene, this should always return the same transcript.
        # Sum over all compartments (in this case, NES and NLS, since we dropped the others)
        # transcript = u5 + cds + u3
        # protein trunc is the first 100 bases (usually thought to be important in localization)
        kallisto_subset_summed = self.kallisto_table.loc[transcript_ids].sum(axis=1)
        most_common_transcript = kallisto_subset_summed.idxmax()
        return most_common_transcript

    def get_most_common_transcript_expression(self, gene, transcripts_only=True):
        """Return the expression TPM of the most common transcript for the given gene"""
        transcript_ids = self.genes_to_transcripts[gene]
        assert transcript_ids, f"Got no transcript matches for gene {gene}"
        if transcripts_only:
            transcript_ids = [t for t in transcript_ids if t.startswith("ENST")]
        kallisto_subset_summed = self.kallisto_table.loc[transcript_ids].sum(axis=1)
        return np.max(kallisto_subset_summed)

    def get_most_common_transcript_sequence_parts(self, gene):
        """Return the most common transcript's sequence components based on kallisto tpm counts"""
        most_common_transcript = self.get_most_common_transcript(gene)
        row = self.fasta_table.loc[most_common_transcript]
        return row['u5'], row['cds'], row['u3']

    def get_retained_introns(self, gene, join=True):
        """Return the retained intron for the given gene"""
        def get_strand(s):
            """Gets strand information from string"""
            strands = set()
            for token in s.split(";"):
                strand = token.split(":")[-1]
                assert strand in ('-', "+")
                strands.add(strand)
            assert len(strands) == 1, f"Multiple conflicting strands found in {s}"
            return strands.pop()

        this_ri_table = self.ri_table[self.ri_table['GeneID'] == gene]
        ranges = collections.defaultdict(IntervalTree)
        for row in this_ri_table.itertuples():
            chrom = row.chr.strip("chr")
            start = row.upstreamEE
            end = row.downstreamES
            strand = row.strand
            assert end > start
            ranges[chrom][start:end] = f"{chrom}:{start}-{end}:{strand}"

        assert len(ranges) == 1, f"Got more than one chromosome for {gene}: {ranges.keys()}"
        chrom = list(ranges.keys())[0]
        itree = ranges[chrom]
        itree.merge_overlaps(data_reducer=utils.gcoord_str_merger)
        intronic_sequences = []
        gcoord_strs = []
        strands = set()
        for interval in itree:
            strand = get_strand(interval.data)
            gcoord_strs.append(interval.data)
            seq = self.faidx.get_seq(chrom, interval.begin, interval.end, strand=="-")
            intronic_sequences.append(seq.seq)
            strands.add(strand)

        assert len(strands) == 1
        if strands.pop() == "-":
            intronic_sequences = intronic_sequences[::-1]
            gcoord_strs = gcoord_strs[::-1]
        assert len(intronic_sequences) == len(gcoord_strs)

        return ''.join(intronic_sequences) if join else (intronic_sequences, gcoord_strs)

    def __len__(self):
        return len(self.gene_ids) * 2

    def __getitem__(self, i):
        """Return the ith example"""
        # Everything operates with genes
        gene_idx = math.floor(i / 2)
        label = np.array(i % 2)

        gene = self.gene_ids[gene_idx]
        u5, cds, u3 = self.get_most_common_transcript_sequence_parts(gene)
        if label:
            retained_intron = self.get_retained_introns(gene, join=True)
            cds += retained_intron  # Somewhat hacky since intron isn't actually appended to CDS but approximately works
        transcript_parts = [u5, cds, u3]

        encodings = [sequence_to_kmer_freqs(part, kmer_size=s) for s in self.kmer_sizes for part in transcript_parts]
        seq_encoded = np.hstack(encodings)

        return torch.from_numpy(seq_encoded).type(torch.FloatTensor), torch.from_numpy(label).type(torch.FloatTensor)

class NLSvNESEncodeDataset(data.Dataset):
    def __init__(self, deseq_fname, rsem_files, split='train', mode='padj', k_fold=0, pval_cutoff=0.05, fixed_seq_len=0, kmer_sizes=[], seq_by_parts=False, chr_ft=False):
        if split not in ['train', 'valid', 'test', 'all']:
            raise ValueError("Unrecognized split: {}".format(split))
        assert k_fold < K_FOLDS

        # Load in the rsem files
        rsem_tables = []
        for rsem_file in rsem_files:
            assert os.path.isfile(rsem_file)
            rsem_table = pd.read_csv(rsem_file, sep="\t")
            rsem_tables.append(rsem_table)
        assert len(set([t.shape for t in rsem_tables])) == 1, "Got multiple shapes for RSEM tables"
        self.trans_counts = collections.defaultdict(float)
        for tab in rsem_tables:
            assert len(set(tab['transcript_id'])) == len(tab['transcript_id'])  # Unique names
            for row in tab.itertuples():
                self.trans_counts[row.transcript_id] += row.TPM  # Increment TPM across all RSEM files

        self.full_deseq_table = pd.read_csv(deseq_fname, index_col=0, delimiter="\t")
        self.full_deseq_table.dropna(subset=['log2FoldChange'], inplace=True)
        if mode == 'padj':
            deseq_table_significant = self.full_deseq_table[self.full_deseq_table['padj'] < pval_cutoff]  # Only take significant genes
        else:
            raise NotImplementedError
        logging.info("{}/{} entries in table retained as significant".format(deseq_table_significant.shape[0], self.full_deseq_table.shape[0]))

        np.random.seed(332133)  # Do not change
        indices = np.arange(deseq_table_significant.shape[0])
        np.random.shuffle(indices)
        # NOTE it may be better to split by chromosome, but that's a lot more code
        indices_split = np.array_split(indices, K_FOLDS)  # Splits into K_FOLDS partitions of approx equal size
        valid_partition = k_fold  # Define train/valid/test partitions
        test_partition = (k_fold + 1) % K_FOLDS
        train_partitions = [i for i in range(K_FOLDS) if i != valid_partition and i != test_partition]

        # Figure out which indices to keep based on train/test/valid/all argument
        if split == 'valid':
            keep_indices = indices_split[valid_partition]
        elif split == 'test':
            keep_indices = indices_split[test_partition]
        elif split == 'train':
            keep_indices = np.concatenate([indices_split[i] for i in train_partitions])
        else:
            keep_indices = indices  # Just keep everything, but still shuffle
        # Passes significance check and is in the data split partition
        deseq_table_subsetted = deseq_table_significant.iloc[keep_indices]

        # Load in auxillary files
        self.fasta_dict = fasta_to_dict(os.path.join(DATA_DIR, "Homo_sapiens.GRCh38.merge.90.fa"))  # dict
        self.fasta_table = load_fasta_table(no_periods=False)  # This is a dataframe
        genes_to_transcripts_table = pd.read_csv(
            os.path.join(DATA_DIR, "Homo_sapiens.GRCh38.90.merge_transcriptome.fa.processed.txt"),
            sep="\t",
        )
        self.genes_to_transcripts = collections.defaultdict(list)
        for row_tuple in genes_to_transcripts_table.itertuples():
            self.genes_to_transcripts[row_tuple.gene].append(row_tuple.transcript)

        self.deseq_table_subsetted = deseq_table_subsetted

        ##### DO NOT MODIFY THE TABLE AFTER THIS POINT #####
        self.gene_ids = deseq_table_subsetted.index
        # Log2fc > 0.75
        # try quartiles
        # Or try taking padj < 0.05 and take log2foldchange > 0.5 as "positive" set, select matching "negative" set
        # Check for genes found in both (because they are compared against cytoplasm, so you can have enrichment in both)
        # drop < 10 bp transcripts
        self.labels = deseq_table_subsetted['log2FoldChange'] > 0  # Indicates positive enrichment
        self.kmer_sizes = [kmer_sizes] if isinstance(kmer_sizes, int) else kmer_sizes  # List of kmer sizes we use to featurize
        self.seq_by_parts = seq_by_parts  # Binary flag
        self.fixed_seq_len = fixed_seq_len
        logging.info("Proportion of positives in {}: {}".format(split, np.sum(self.labels) / len(self.labels)))

        # Report metrics on proportion of transcripts in each category
        tc = TranscriptClassifier()
        ttypes = [tc.get_transcript_type(self.get_most_common_transcript(g)) for g in self.gene_ids]
        self.tcounter = collections.Counter(ttypes)
        ttotal = sum(self.tcounter.values())
        for tt, tcount in self.tcounter.most_common():
            logging.info("{}\t{}\t{}".format(tcount, tcount/ttotal, tt))

    def get_most_common_transcript(self, gene):
        """Return the name (e.g. ENSTxxxxxx) of the most common transcript for the given gene"""
        transcript_ids = self.genes_to_transcripts[gene]  # Find transcripts belonging to this gene
        transcript_ids = [t for t in transcript_ids if t.startswith("ENST")]
        # Figure out the most common transcript. Given the same gene, this should always return the same transcript.
        # Sum over all compartments (in this case, NES and NLS, since we dropped the others)
        # transcript = u5 + cds + u3
        # protein trunc is the first 100 bases (usually thought to be important in localization)
        trans_counts = [self.trans_counts[t] for t in transcript_ids]
        most_common_transcript_idx = np.argmax(trans_counts)
        most_common_transcript = transcript_ids[most_common_transcript_idx]
        return most_common_transcript

    def __get_most_common_transcript_sequence_parts(self, gene):
        """Return the most common transcript's sequence components based on total tpm counts"""
        most_common_transcript = self.get_most_common_transcript(gene)
        row = self.fasta_table.loc[most_common_transcript]
        return (row['u5'], row['cds'], row['u3'])

    def __get_most_common_transcript_sequence(self, gene):
        """Return the most common transcript's sequence based on kallisto tpm counts"""
        most_common_transcript = self.get_most_common_transcript(gene)
        return self.fasta_dict[most_common_transcript]

    def __len__(self):
        """Total number of samples"""
        return len(self.labels)

    def __getitem__(self, i):
        """Get one sample of data"""
        gene = self.gene_ids[i]  # Desired gene
        if self.seq_by_parts:
            seq = self.__get_most_common_transcript_sequence_parts(gene)
        else:
            seq = self.__get_most_common_transcript_sequence(gene)  # Fetch the sequence for the most common transcript
        # Note that fixed seq len trimming occurs regardless of featurization
        if self.fixed_seq_len != 0:  # Default value here is 0, resulting in no trimming
            trim_amount = abs(self.fixed_seq_len)
            trim_from_right = True if self.fixed_seq_len < 0 else False
            if self.seq_by_parts:
                seq = tuple([trim_or_pad(part, trim_amount, right_align=trim_from_right) for part in seq])
            else:
                seq = trim_or_pad(seq, trim_amount, right_align=trim_from_right)
        # Return as sequence, or as kmer table based on input
        if not self.kmer_sizes:  # One-hot encode the transcript
            if self.seq_by_parts:
                raise NotImplementedError("Cannot one hot encode when splitting sequence by parts")
            else:
                seq_encoded = sequence_to_image(seq)
        else:  # kmer table
            if self.seq_by_parts:
                # Iterates all the 4-mers for all parts before moving to 5-mers
                encodings = [sequence_to_kmer_freqs(part, kmer_size=s) for s in self.kmer_sizes for part in seq]
            else:
                encodings = [sequence_to_kmer_freqs(seq, kmer_size=s) for s in self.kmer_sizes]
            seq_encoded = np.hstack(encodings)
            # If we are using kmers, we can optionally also use pwms as well.
        label = np.atleast_1d(np.array(self.labels[gene])).astype(int)
        x = torch.from_numpy(seq_encoded).type(torch.FloatTensor)
        y = torch.from_numpy(label).type(torch.FloatTensor)
        return x, y

class NLSvNESLubelskyDataset(data.Dataset):
    """Loads in the Lubelsky 2018 dataset, retaining only significant localizations"""
    def __init__(self, split:str='all', k_fold:int=0, kmer_sizes:List[int]=[3, 4, 5], padj_cutoff:float=0.05, binarize:bool=True):
        assert split == 'all'
        self.kmer_sizes = kmer_sizes
        self.padj_cutoff = padj_cutoff
        self.binarize = binarize
        self.diff_exp_table = pd.read_csv(os.path.join(LOCAL_DATA_DIR, "lubelsky_et_al_2018", "Ulitsky_edger_output.txt"), sep='\t')

        keep_idx = np.where(self.diff_exp_table['padj.C1-Cyt.C1-Nuc'] <= padj_cutoff)
        self.diff_exp_table = self.diff_exp_table.iloc[keep_idx]

        self.seq_table = pd.read_csv(os.path.join(LOCAL_DATA_DIR, "lubelsky_et_al_2018", "ulitsky_count_table.csv"))
        self.seq_table = self.seq_table.iloc[keep_idx]

        assert self.seq_table.shape[0] == self.diff_exp_table.shape[0]

    def __len__(self):
        return self.seq_table.shape[0]

    def get_ith_trans_parts(self, i):
        u5 = ""
        cds = "ATGGTGAGCAAGGGCGCCGAGCTGTTCACCGGCATCGTGCCCATCCTGATCGAGCTGAATGGCGATGTGAATGGCCACAAGTTCAGCGTGAGCGGCGAGGGCGAGGGCGATGCCACCTACGGCAAGCTGACCCTGAAGTTCATCTGCACCACCGGCAAGCTGCCTGTGCCCTGGCCCACCCTGGTGACCACCCTGAGCTACGGCGTGCAGTGCTTCTCACGCTACCCCGATCACATGAAGCAGCACGACTTCTTCAAGAGCGCCATGCCTGAGGGCTACATCCAGGAGCGCACCATCTTCTTCGAGGATGACGGCAACTACAAGTCGCGCGCCGAGGTGAAGTTCGAGGGCGATACCCTGGTGAATCGCATCGAGCTGACCGGCACCGATTTCAAGGAGGATGGCAACATCCTGGGCAATAAGATGGAGTACAACTACAACGCCCACAATGTGTACATCATGACCGACAAGGCCAAGAATGGCATCAAGGTGAACTTCAAGATCCGCCACAACATCGAGGATGGCAGCGTGCAGCTGGCCGACCACTACCAGCAGAATACCCCCATCGGCGATGGCCCTGTGCTGCTGCCCGATAACCACTACCTGTCCACCCAGAGCGCCCTGTCCAAGGACCCCAACGAGAAGCGCGATCACATGATCTACTTCGGCTTCGTGACCGCCGCCGCCATCACCCACGGCATGGATGAGCTGTACAAG"
        u3_parts = ["TCCGGACTCAGATCT", "foo", "GAATTCTGCAGTCGACGGTACCGCGGGCCCGGGATCCACCGGATCTAGATAACTGATCATAATCAGCCATACCACATTTGTAGAGGTTTTACTTGCTTTAAAAAACCTCCCACACCTCCCCCTGAACCTGAAACATAAAATGAATGCAATTGTTGTTGTTAACTTGTTTATTGCAGCTTATAATGGTTACAAATAAAGCAATAGCATCACAAATTTCACAAATAAA"]
        var_seq = self.seq_table.iloc[i]['Sequence']
        assert isinstance(var_seq, str)
        u3_parts[1] = var_seq
        u3 = ''.join(u3_parts)
        return u5, cds, u3

    def get_ith_labels(self, i):
        """Return the ith truth label"""
        val = self.diff_exp_table['logFC.C1-Cyt.C1-Nuc'].iloc[i]
        if self.binarize:
            val = val >= 0
        return np.array(val)

    def __getitem__(self, i):
        trans_parts = self.get_ith_trans_parts(i)
        encodings = [sequence_to_kmer_freqs(part, kmer_size=s) for s in self.kmer_sizes for part in trans_parts]
        seq_encoded = np.hstack(encodings)
        y = self.get_ith_labels(i)

        return torch.from_numpy(seq_encoded).type(torch.FloatTensor), torch.from_numpy(y).type(torch.FloatTensor)

class NLSvNESParkerDataset(NLSvNESLubelskyDataset):
    """Loads in Kevin Parker's dataset (currently unpublished)"""
    # TODO code to select only inserted sequences based on a filter func
    # filter func may be just "includes some PWM"
    def __init__(self, modality:str="NES-T.NES-C", split:str='all', filter_func=None, k_fold:int=0, kmer_sizes:List[int]=[3, 4, 5], padj_cutoff:float=0.05, logfc_cutoff:float=0, binarize:bool=True):
        if split != 'all':
            raise NotImplementedError(f"Data splits other than all are not currently supported, but got {split}")
        self.kmer_sizes = kmer_sizes
        self.modality = modality
        self.padj_cutoff = padj_cutoff
        self.filter_func = filter_func
        self.logfc_cutoff = logfc_cutoff
        self.binarize = binarize
        self.diff_exp_table = pd.read_csv(
            os.path.join(LOCAL_DATA_DIR, "krparker_tiling", "200207_edger-output-for-kevin-wu.txt"),
            sep="\t",
            index_col=0,
            engine='c',
            low_memory=False,
        )
        # print(self.diff_exp_table)
        self.padj_colnames = [c for c in self.diff_exp_table.columns if c.startswith("padj")]
        # print(self.padj_colnames)
        self.logfc_colnames = [c for c in self.diff_exp_table.columns if c.startswith('logFC')]
        # print(self.logfc_colnames)

        self.seq_table = pd.read_csv(
            os.path.join(LOCAL_DATA_DIR, "krparker_tiling", "200207_kevinwu_zipcode-tile-annotation.txt"),
            index_col=0,
            sep="\t",
            engine='c',
            low_memory=False,
        )
        assert self.seq_table.shape[0] == self.diff_exp_table.shape[0]

        passes = self.diff_exp_table['padj' + "." + self.modality] <= padj_cutoff
        if logfc_cutoff:
            assert logfc_cutoff > 0
            passes = np.logical_and(
                passes,
                np.abs(self.diff_exp_table['logFC.' + self.modality]) >= logfc_cutoff,
            )
        keep_idx = self.diff_exp_table.index[np.where(passes)]
        self.seq_table = self.seq_table.loc[keep_idx]
        self.diff_exp_table = self.diff_exp_table.loc[keep_idx]

        # Filter based on filter_func
        if filter_func is not None:
            filter_pass_idx = [i for i, seq in enumerate(self.seq_table['sequence_trimmed']) if filter_func(seq)]
            self.diff_exp_table = self.diff_exp_table.iloc[filter_pass_idx]
            self.seq_table = self.seq_table.iloc[filter_pass_idx]

        assert self.seq_table.shape[0] == self.diff_exp_table.shape[0]

    def get_ith_trans_parts(self, i):
        u5 = "GGTGCTAGTCCAGTGTGGTGGAATTCTGCAGATATCAACAAGTTTGTACAAAAAAGCAGGCTTCGAAGGAGATAGAACCATGG"
        cds = "ATGTACAACATGATGGAGACGGAGCTGAAGCCGCCGGGCCCGCAGCAAACTTCGGGGGGCGGCGGCGGCAACTCCACCGCGGCGGCGGCCGGCGGCAACCAGAAAAACAGCCCGGACCGCGTCAAGCGGCCCATGAATGCCTTCATGGTGTGGTCCCGCGGGCAGCGGCGCAAGATGGCCCAGGAGAACCCCAAGATGCACAACTCGGAGATCAGCAAGCGCCTGGGCGCCGAGTGGAAACTTTTGTCGGAGACGGAGAAGCGGCCGTTCATCGACGAGGCTAAGCGGCTGCGAGCGCTGCACATGAAGGAGCACCCGGATTATAAATACCGGCCCCGGCGGAAAACCAAGACGCTCATGAAGAAGGATAAGTACACGCTGCCCGGCGGGCTGCTGGCCCCCGGCGGCAATAGCATGGCGAGCGGGGTCGGGGTGGGCGCCGGCCTGGGCGCGGGCGTGAACCAGCGCATGGACAGTTACGCGCACATGAACGGCTGGAGCAACGGCAGCTACAGCATGATGCAGGACCAGCTGGGCTACCCGCAGCACCCGGGCCTCAATGCGCACGGCGCAGCGCAGATGCAGCCCATGCACCGCTACGACGTGAGCGCCCTGCAGTACAACTCCATGACCAGCTCGCAGACCTACATGAACGGCTCGCCCACCTACAGCATGTCCTACTCGCAGCAGGGCACCCCTGGCATGGCTCTTGGCTCCATGGGTTCGGTGGTCAAGTCCGAGGCCAGCTCCAGCCCCCCTGTGGTTACCTCTTCCTCCCACTCCAGGGCGCCCTGCCAGGCCGGGGACCTCCGGGACATGATCAGCATGTATCTCCCCGGCGCCGAGGTGCCGGAACCCGCCGCCCCCAGCAGACTTCACATGTCCCAGCACTACCAGAGCGGCCCGGTGCCCGGCACGGCCATTAACGGCACACTGCCCCTCTCACACATGTGA"
        u3_parts = ["TAGGACCGGTACTGGCCGCTGGCGCGCCATAC", "foo", "GTATGCGGCCGCTTCGAGCAGACATGATAAGATACATTGATGAGTTTGGACAAACCACAACTAGAATGCAGTGAAAAAAATGCTTTATTTGTGAAATTTGTGATGCTATTGCTTTATTTGTAACCATTATAAGCTGCAATAAACAAGTTAACAACAACAATTGCATTCATTTTATGTTTCAGGTTCAGGGGGAGGTGTGGGAGGTTTTTTAAAGCAAGTAAAACCTCTACAAATGTGGT"]

        tile_sequence = self.seq_table.loc[self.seq_table.index[i], "sequence_trimmed"]
        assert isinstance(tile_sequence, str)
        tile_sequence = tile_sequence.replace("U", "T", len(tile_sequence))  # Replace all U with T
        u3_parts[1] = tile_sequence
        u3 = ''.join(u3_parts)
        assert 'foo' not in u3
        return u5, cds, u3

    def get_ith_labels(self, i):
        val = self.diff_exp_table.loc[self.diff_exp_table.index[i], 'logFC.' + self.modality]
        if self.binarize:
            val = val >= 0
        return np.array(val)

class NLSvNESShuklaDataset(data.Dataset):
    """Loads in the Shukla 2018 dataset, retaining only significant localizations"""
    def __init__(self, split:str='all', k_fold:int=0, kmer_sizes:List[int]=[3, 4, 5], padj_cutoff:float=0.05, binarize:bool=True):
        self.kmer_sizes = kmer_sizes
        self.padj_cutoff = padj_cutoff
        self.binarize = binarize
        self.diff_exp_table = pd.read_csv(os.path.join(LOCAL_DATA_DIR, "shukla_et_al_2018", "Rinn_edger_output.txt"), sep="\t")
        self.diff_exp_table = self.diff_exp_table.iloc[np.where(self.diff_exp_table['padj.Nuclei.Total'] <= self.padj_cutoff)]

        np.random.seed(332133)  # Do not change
        indices = np.arange(self.diff_exp_table.shape[0])
        np.random.shuffle(indices)
        # NOTE it may be better to split by chromosome, but that's a lot more code
        indices_split = np.array_split(indices, K_FOLDS)  # Splits into K_FOLDS partitions of approx equal size
        valid_partition = k_fold % K_FOLDS # Define train/valid/test partitions
        test_partition = (k_fold + 1) % K_FOLDS
        assert 0 <= valid_partition < K_FOLDS
        assert 0 <= test_partition < K_FOLDS

        train_partitions = [i for i in range(K_FOLDS) if i != valid_partition and i != test_partition]
        if split == 'valid':
            keep_indices = indices_split[valid_partition]
        elif split == 'test':
            keep_indices = indices_split[test_partition]
        elif split == 'train':
            keep_indices = np.concatenate([indices_split[i] for i in train_partitions])
        elif split == 'all':
            keep_indices = indices  # Just keep everything, but still shuffle
        else:
            raise ValueError(f"Unrecognized split: {split}")
        self.diff_exp_table = self.diff_exp_table.iloc[keep_indices]

        fa = fasta_to_dict(os.path.join(LOCAL_DATA_DIR, "shukla_et_al_2018", "OligoArrayDesign.fa"))
        self.variable_sequence = {}
        for k, seq in fa.items():
            txn_num = int(k.split("_")[1])
            idx = int(k.split("_")[2])
            assert (txn_num, idx) not in self.variable_sequence
            self.variable_sequence[(txn_num, idx)] = seq
        self.metadata = pd.read_csv(os.path.join(LOCAL_DATA_DIR, "shukla_et_al_2018", "oligo_pool_anno.csv"))

    def _deseq_name_to_fasta_seq(self, x):
        """Convert the deseq name to name in fasta file"""
        name2 = "_".join(x.split("_")[:-2])
        idx = int(x.split("_")[-2])
        barcode = x.split("_")[-1]
        metarow = self.metadata[self.metadata['name2'] == name2]
        assert not metarow.empty
        assert idx < metarow['numOfOligos'].item()
        txn_num = int(metarow['txnNum'].item())
        new_key = (txn_num, idx)
        retval = self.variable_sequence[new_key]
        assert retval
        return retval

    def get_ith_trans_parts(self, i):
        """Combine variablke seq with backbone seq"""
        cds = "ATGTACAACATGATGGAGACGGAGCTGAAGCCGCCGGGCCCGCAGCAAACTTCGGGGGGCGGCGGCGGCAACTCCACCGCGGCGGCGGCCGGCGGCAACCAGAAAAACAGCCCGGACCGCGTCAAGCGGCCCATGAATGCCTTCATGGTGTGGTCCCGCGGGCAGCGGCGCAAGATGGCCCAGGAGAACCCCAAGATGCACAACTCGGAGATCAGCAAGCGCCTGGGCGCCGAGTGGAAACTTTTGTCGGAGACGGAGAAGCGGCCGTTCATCGACGAGGCTAAGCGGCTGCGAGCGCTGCACATGAAGGAGCACCCGGATTATAAATACCGGCCCCGGCGGAAAACCAAGACGCTCATGAAGAAGGATAAGTACACGCTGCCCGGCGGGCTGCTGGCCCCCGGCGGCAATAGCATGGCGAGCGGGGTCGGGGTGGGCGCCGGCCTGGGCGCGGGCGTGAACCAGCGCATGGACAGTTACGCGCACATGAACGGCTGGAGCAACGGCAGCTACAGCATGATGCAGGACCAGCTGGGCTACCCGCAGCACCCGGGCCTCAATGCGCACGGCGCAGCGCAGATGCAGCCCATGCACCGCTACGACGTGAGCGCCCTGCAGTACAACTCCATGACCAGCTCGCAGACCTACATGAACGGCTCGCCCACCTACAGCATGTCCTACTCGCAGCAGGGCACCCCTGGCATGGCTCTTGGCTCCATGGGTTCGGTGGTCAAGTCCGAGGCCAGCTCCAGCCCCCCTGTGGTTACCTCTTCCTCCCACTCCAGGGCGCCCTGCCAGGCCGGGGACCTCCGGGACATGATCAGCATGTATCTCCCCGGCGCCGAGGTGCCGGAACCCGCCGCCCCCAGCAGACTTCACATGTCCCAGCACTACCAGAGCGGCCCGGTGCCCGGCACGGCCATTAACGGCACACTGCCCCTCTCACACATGTGA"
        u5 = "GGTGCTAGTCCAGTGTGGTGGAATTCTGCAGATATCAACAAGTTTGTACAAAAAAGCAGGCTTCGAAGGAGATAGAACCATGG"
        # foo is a placeholder here
        u3_parts = ["TAGGACCGGTACTGGCCGCTTCACTG", "foo", "AGATCGGAAGAGCGTCGGCGGCCGCTTCGAGCAGACATGATAAGATACATTGATGAGTTTGGACAAACCACAACTAGAATGCAGTGAAAAAAATGCTTTATTTGTGAAATTTGTGATGCTATTGCTTTATTTGTAACCATTATAAGCTGCAATAAACAAGTTAACAACAACAATTGCATTCATTTTATGTTTCAGGTTCAGGGGGAGGTGTGGGAGGTTTTTTAAAGCAAGTAAAACCTCTACAAATGTGGTA"]
        var_seq = self._deseq_name_to_fasta_seq(self.diff_exp_table.index[i])
        u3_parts[1] = var_seq
        u3 = ''.join(u3_parts)
        return u5, cds, u3

    def get_ith_labels(self, i):
        """Return the ith truth label"""
        val = self.diff_exp_table['logFC.Nuclei.Total'][i]
        if self.binarize:
            val = val >= 0
        return np.array(val)

    def __len__(self):
        return self.diff_exp_table.shape[0]

    def __getitem__(self, i):
        trans_parts = self.get_ith_trans_parts(i)
        encodings = [sequence_to_kmer_freqs(part, kmer_size=s) for s in self.kmer_sizes for part in trans_parts]
        seq_encoded = np.hstack(encodings)
        y = self.get_ith_labels(i)

        return torch.from_numpy(seq_encoded).type(torch.FloatTensor), torch.from_numpy(y).type(torch.FloatTensor)

class NLSvNESShuklaDatasetTopN(NLSvNESShuklaDataset):
    """Loads in shukla dataset, but takes the top N strongest logFC values instead of using p-values"""
    def __init__(self, split:str='all', kmer_sizes:List[int]=[3, 4, 5], top_n:int=20, binarize:bool=True):
        assert split == 'all'
        self.kmer_sizes = kmer_sizes
        self.binarize = binarize
        self.diff_exp_table = pd.read_csv(os.path.join(LOCAL_DATA_DIR, "shukla_et_al_2018", "Rinn_edger_output.txt"), sep="\t")
        self.diff_exp_table = self.diff_exp_table.sort_values(by=['logFC.Nuclei.Total'])
        self.diff_exp_table = pd.concat([self.diff_exp_table.iloc[:top_n], self.diff_exp_table.iloc[-top_n:]])

        fa = fasta_to_dict(os.path.join(LOCAL_DATA_DIR, "shukla_et_al_2018", "OligoArrayDesign.fa"))
        self.variable_sequence = {}
        for k, seq in fa.items():
            txn_num = int(k.split("_")[1])
            idx = int(k.split("_")[2])
            assert (txn_num, idx) not in self.variable_sequence
            self.variable_sequence[(txn_num, idx)] = seq
        self.metadata = pd.read_csv(os.path.join(LOCAL_DATA_DIR, "shukla_et_al_2018", "oligo_pool_anno.csv"))

class NLSvNESShuklaDatasetAbsCutoff(NLSvNESShuklaDataset):
    """Loads in shukla dataset, but takes the top N strongest logFC values instead of using p-values"""
    def __init__(self, split:str='all', kmer_sizes:List[int]=[3, 4, 5], abs_cutoff:float=7.5, binarize:bool=True):
        assert split == 'all'
        self.kmer_sizes = kmer_sizes
        self.binarize = binarize
        self.diff_exp_table = pd.read_csv(os.path.join(LOCAL_DATA_DIR, "shukla_et_al_2018", "Rinn_edger_output.txt"), sep="\t")
        self.diff_exp_table = self.diff_exp_table.iloc[np.where(np.abs(self.diff_exp_table['logFC.Nuclei.Total']) >= abs_cutoff)]

        fa = fasta_to_dict(os.path.join(LOCAL_DATA_DIR, "shukla_et_al_2018", "OligoArrayDesign.fa"))
        self.variable_sequence = {}
        for k, seq in fa.items():
            txn_num = int(k.split("_")[1])
            idx = int(k.split("_")[2])
            assert (txn_num, idx) not in self.variable_sequence
            self.variable_sequence[(txn_num, idx)] = seq
        self.metadata = pd.read_csv(os.path.join(LOCAL_DATA_DIR, "shukla_et_al_2018", "oligo_pool_anno.csv"))

class LocalizationRegressionKmers(LocalizationClassificationKmers):
    def get_ith_fc_values(self, i):
        """Return the ith log2fc values for regression task"""
        effective_index = i if not self.rc_aug else i // 2
        # print(self.log2fc_matrix)
        retval = self.log2fc_matrix[np.newaxis, effective_index, :]
        assert len(retval.shape) == 2
        return retval

    def __getitem__(self, i):
        transcript_parts = self.get_ith_trans_parts(i)
        encodings = [sequence_to_kmer_freqs(part, kmer_size=s) for s in self.kmer_sizes for part in transcript_parts]
        seq_encoded = np.hstack(encodings)

        if self.include_len:
            seq_encoded = np.insert(seq_encoded, 0, sum([len(t) for t in transcript_parts]))
        target = self.get_ith_fc_values(i)
        return torch.from_numpy(seq_encoded).type(torch.FloatTensor), torch.from_numpy(target).type(torch.FloatTensor)

class TranscriptClassifier(object):
    """Lookup for what category a transcript is"""
    def __init__(self, fasta_table=os.path.join(DATA_DIR, "Homo_sapiens.GRCh38.90.merge_transcriptome.fa.processed.txt"), map_file=os.path.join(LOCAL_DATA_DIR, "gencode_biotypes.txt")):
        self.biotype_to_label = {}
        with open(map_file) as source:  # Similar to fasta file format
            curr_label = None
            for line in source:
                if not line.strip() or line.startswith("#"):  # These lines denote comments
                    continue
                if line.startswith(">"):
                    curr_label = line.strip(">").strip()
                else:
                    assert curr_label
                    biotype = line.strip()
                    assert biotype not in self.biotype_to_label, "Duplicated biotype: {}".format(biotype)
                    self.biotype_to_label[biotype] = curr_label
        self.fasta_table = pd.read_csv(fasta_table, delimiter='\t', engine='c')
        self.trans_to_label = {}
        for i, row in self.fasta_table.iterrows():
            self.trans_to_label[row['transcript']] = row['transcript_type']

    def get_transcript_type(self, query):
        """Return the transcript type of the given query"""
        retval = self.biotype_to_label[self.trans_to_label[query]]
        return retval


@functools.lru_cache(1)
def fasta_to_dict(fname=os.path.join(DATA_DIR, "Homo_sapiens.GRCh38.merge.90.fa")):
    """Load the fasta file into a dictionary"""
    curr_key = None
    retval = {}
    opener = gzip.open if fname.endswith(".gz") else open
    with opener(fname) as source:
        for line in source:
            if isinstance(line, bytes):  # Decode if necessary
                line = line.decode("utf-8")
            if line.startswith(">"):
                curr_key = line.rstrip().strip(">").split()[0]
                assert curr_key not in retval, "Found duplicated entry: {}".format(curr_key)
                retval[curr_key] = []
            else:
                retval[curr_key].append(line.rstrip())
    return {k: "".join(v) for k, v in retval.items()}

def load_kallisto_table(fname=os.path.join(DATA_DIR, "apex_kallisto_tpm.txt")):
    """Loads the given kallisto table"""
    retval = pd.read_csv(fname, index_col=0, delimiter="\t")
    return retval

@functools.lru_cache(2)
def load_fasta_table(fname=os.path.join(DATA_DIR, "parsed_fasta_rel90.txt"), no_periods=True):
    """Load a table of fasta sequences (i.e. not a canonical fasta file)"""
    parsed = pd.read_csv(fname, index_col=0, delimiter="\t", dtype=str)
    parsed.replace(np.nan, "", inplace=True)  # Replace NaN with empty strings
    if no_periods:
        indices_stripped = [idx.split(".")[0] for idx in parsed.index]
        assert len(indices_stripped) == len(set(indices_stripped)), "Cannot strip periods when names are non-unique without"
        parsed.index = indices_stripped
    return parsed

def load_deseq(fname=os.path.join(DATA_DIR, "deseq_merged.txt")):
    """Load and return the deseq table"""
    retval = pd.read_csv(fname, index_col=0, delimiter="\t")
    return retval

def load_data_as_np(dataset, progress_bar=True, ignore_missing=False, check_dims=True):
    """Given a dataset, load all its data into np arrays. Useful for re-using torch code to load into non-torch models"""
    xs = []
    ys = []
    miss_counter = 0
    pbar = tqdm.tqdm_notebook if utils.isnotebook() else tqdm.tqdm
    for i in pbar(range(len(dataset)), disable=not progress_bar):
        try:
            x, y = dataset[i]
            xs.append(np.atleast_2d(x.numpy()))
            ys.append(y.numpy())
        except KeyError as error:
            if not ignore_missing:
                raise error
            else:
                miss_counter += 1
    if ignore_missing:
        logging.warn(f"{miss_counter} missing transcript names")

    x_combined = np.vstack(xs)
    if len(ys[0].shape) < 2:
        y_combined = np.vstack(np.expand_dims(ys, axis=0))
    else:
        y_combined = np.vstack(ys)

    if y_combined.shape == (x_combined.shape[0], 1) or y_combined.shape == (1, x_combined.shape[0]):
        y_combined = np.squeeze(y_combined)
    if check_dims:
        assert x_combined.shape[0] == y_combined.shape[0], "Got differing shapes: {} {}".format(x_combined.shape, y_combined.shape)
    return x_combined, y_combined

def _loader_helper(fold, dataset, dataset_kwargs, as_np=True, ignore_missing=False):
    train_d = dataset(split="train", k_fold=fold, **dataset_kwargs)
    valid_d = dataset(split='valid', k_fold=fold, **dataset_kwargs)
    test_d = dataset(split='test', k_fold=fold, **dataset_kwargs)
    if as_np:
        train = load_data_as_np(train_d, progress_bar=False, ignore_missing=ignore_missing)
        valid = load_data_as_np(valid_d, progress_bar=False, ignore_missing=ignore_missing)
        test = load_data_as_np(test_d, progress_bar=False, ignore_missing=ignore_missing)
        return train, valid, test
    else:
        return train_d, valid_d, test_d

def load_dataset_all_folds(dataset, dataset_kwargs, as_np=True, ignore_missing=False):
    """
    Returns all k-folds of the dataset as tuples of (train, valid, test)
    If as_np is False, we return the datasets
    """
    assert "k_fold" not in dataset_kwargs
    assert "split" not in dataset_kwargs
    pfunc = functools.partial(_loader_helper, dataset=dataset, dataset_kwargs=dataset_kwargs, as_np=as_np, ignore_missing=ignore_missing)
    pool = multiprocessing.Pool(min(multiprocessing.cpu_count(), K_FOLDS))
    retval = list(pool.imap(pfunc, range(K_FOLDS)))
    pool.close()
    pool.join()
    return retval

def load_data_as_np_by_trans_type(dataset):
    """Given a dataset, load all of its data into np arrays by transcript type"""
    xs_by_type = collections.defaultdict(list)
    ys_by_type = collections.defaultdict(list)
    for i in tqdm.tqdm(range(len(dataset))):
        x, y = dataset[i]
        transcript_type = dataset.full_deseq_table.iloc[i]['gene_type']
        xs_by_type[transcript_type].append(x.numpy())
        ys_by_type[transcript_type].append(y.numpy())
    retval = {}
    for tt in xs_by_type.keys():
        x_combined = np.vstack(np.expand_dims(xs_by_type[tt], axis=0))
        y_combined = np.vstack(np.expand_dims(ys_by_type[tt], axis=0))
        retval[tt] = (x_combined, y_combined)
    return retval

def load_kmer_dataset_ablated(dataset, ablate_seqs, mutate=True, random_ablation=False, pwms=None, seed=None, ablation_strategy="N"):
    """
    Loads the kmer dataset, retaining only things where ablate_seqs applies
    ablate_seqs is a dictionary of localization to transcript part to sequences to ablate
    if random_ablation is set to True, instead of ablating the actual sequence, we choose
    a random subsequences to ablate. This preserves which transcripts are selected in the
    dataset, and preserves how many sequences are ablated per transcript, and varies only
    the actual kmer being ablated out.
    """
    assert str(type(dataset)).split(".")[-1].strip(">").strip("'") == "LocalizationClassificationKmers", f"Unrecognized dataset type: {type(dataset)}"
    if seed is not None:
        random.seed(seed)
    if pwms:
        assert isinstance(pwms, dict) or isinstance(pwms, collections.OrderedDict)
        pwm_list = list(pwms.values())
    x_vals = []
    y_vals = []
    for i in range(len(dataset)):
        transcript_parts = list(dataset.get_ith_trans_parts(i))
        labels = np.squeeze(dataset.get_ith_labels(i))
        true_localizations = [compartment for k, compartment in enumerate(dataset.compartments) if labels[k]]
        matches_ablation = False
        for l in true_localizations:
            for j, trans_part in enumerate(['u5', 'cds', 'u3']):  # j is index of trans parts
                if trans_part not in ablate_seqs[l]: continue
                relevant_ablations = ablate_seqs[l][trans_part]  # pertinent to localization and transcript part
                trans_seq = transcript_parts[j]
                if any([ab in trans_seq for ab in relevant_ablations]):
                    matches_ablation = True
                    if mutate:
                        for ab in relevant_ablations:
                            if ab not in transcript_parts[j]: continue
                            if not random_ablation:
                                transcript_parts[j] = interpretation.ablate(
                                    transcript_parts[j],
                                    ab,
                                    method=ablation_strategy,
                                )
                            else:
                                if pwms is None:
                                    # Pick a random subsequence in the transcipt part to ablate
                                    random_ab = "N" * len(ab)  # Default null sequence
                                    while random_ab == "N" * len(ab):  # Iterate until we get a non-null sequence
                                        random_ab_start = random.randint(0, len(transcript_parts[j]) - len(ab) - 1)
                                        random_ab = transcript_parts[j][random_ab_start:random_ab_start + len(ab)]
                                    assert len(random_ab) == len(ab)
                                else:
                                    # Pick a random PWM that matches this transcript part
                                    random.shuffle(pwm_list)  # Reshuffle each time
                                    k = 0
                                    random_ab = "N"
                                    while random_ab == "N" * len(random_ab):
                                        hits = pwm.find_ppm_hits(transcript_parts[j], pwm_list[k], prop=0.8)
                                        if hits:
                                            random_ab_start = random.choice(hits)
                                            random_ab = transcript_parts[j][random_ab_start:random_ab_start + pwm_list[k].shape[0]]
                                        k += 1
                                assert random_ab
                                assert random_ab != "N" * len(random_ab), "Cannot have ablation of all N"
                                transcript_parts[j] = interpretation.ablate(
                                    transcript_parts[j],
                                    random_ab,
                                    method=ablation_strategy,
                                    # max_iter=transcript_parts[j].count(ab),  # Do not ablate any more than the original
                                )
        if matches_ablation:  # Ignore aeverything that does not match an ablation sequence
            encodings = [sequence_to_kmer_freqs(part, kmer_size=s) for s in dataset.kmer_sizes for part in transcript_parts]
            seq_encoded = np.hstack(encodings)
            x_vals.append(seq_encoded)
            y_vals.append(labels)
    x_retval = np.vstack(x_vals)
    y_retval = np.vstack(np.expand_dims(y_vals, axis=0))
    assert x_retval.shape[0] == y_retval.shape[0]  # Same number of rows
    return x_retval, y_retval

def main():
    """On the fly testing"""
    logging.basicConfig(level=logging.INFO)
    tc = TranscriptClassifier()
    print(tc.get_transcript_type("ENST00000631435.1"))
    d = NLSvNESDataset(split='all', mode='padj', seq_by_parts=True, kmer_sizes=[3, 4, 5], pwms=pwm.load_all_ppm_in_dir())
    print(d[0][0].shape)
    print(len(d.get_feature_labels()))
    x = load_data_as_np_by_trans_type(d)
    d.plot_log2_fold_change(os.path.join(PLOT_DIR, "nes_vs_nls_log2fc.png"))
    d.plot_sequence_lengths(os.path.join(PLOT_DIR, "nes_vs_nls_seq_len.png"))
    # print(load_data_as_np(d)[0].shape)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
    logging.basicConfig(level=logging.INFO)
    x = NLSvNESRetainedIntronDataset()
    print(x[0])
    print(x.get_most_common_transcript_expression(x.gene_ids[0]))
    sys.exit()
    x = NLSvNESEncodeDataset(
        "/Users/kevin/Documents/Stanford/zou/chang_rna_localization/rnafinder/data/encode/HeLa/deseq_hela.txt",
        glob.glob(os.path.join("/Users/kevin/Documents/Stanford/zou/chang_rna_localization/rnafinder/data/encode/HeLa", "*.tsv.gz")),
        kmer_sizes=[3, 4, 5],
        seq_by_parts=True,
    )
    print(x[0][0].shape, x[0][1].shape)
    sys.exit()
    x = LocalizationClassificationKmers(split='all', kmer_sizes=[3, 4, 5], rc_aug=False)
    x.plot_truth_histogram(os.path.join(PLOT_DIR, "trans_localization_hist.png"))
    for i in range(2, 7):
        x.plot_most_common_colocalizations(os.path.join(PLOT_DIR, "common_{}_colocal.png".format(i)), co=i)
    sys.exit(1)
    x.get_representative_trans("ENSG00000004779.9")
    print(len(x))
    print(x[1][0].shape)
    load_data_as_np(x)
    main()
