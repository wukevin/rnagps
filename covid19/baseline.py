"""
Code for generating baselines to compare localization predictions to
"""

import os
import sys
import json
import logging
import glob
import functools
import pickle
from typing import Dict, List
import collections

import numpy as np
import pandas as pd
import scipy.spatial.distance as distance

import matplotlib as mpl
import matplotlib.pyplot as plt

import tqdm

import covid19

SRC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "rnagps",
)
assert os.path.isdir(SRC_DIR)
sys.path.append(SRC_DIR)
from fasta import read_file_as_sequence_dict, write_sequence_dict_to_file
from model_utils import list_preds_to_array_preds
import kmer
import model_utils
import data_loader
import utils

MODEL_PATH = os.path.join(
    os.path.dirname(SRC_DIR),
    "models/rf_8way_fold5.0.21.3.skmodel",
)
assert os.path.isfile(MODEL_PATH)
RNAGPS = utils.load_sklearn_model(MODEL_PATH, strict=True)

NUC_CYTO_COMPOUND_MODEL = model_utils.CompoundModel()

# Explicitly enumerates all 6 human coronaviruses
COV_BASELINE_QUERY = '"Human coronavirus 229E"[Organism] OR "Human coronavirus NL63"[Organism] OR "Human coronavirus OC43"[Organism] OR "Human coronavirus HKU1"[Organism] OR "Middle East respiratory syndrome-related coronavirus"[Organism] OR ("txid694009"[Organism] AND "complete genome"[Title] NOT "Bat"[Title] NOT txid2697049[Organism] AND ("2000/01/01"[PDAT] : "2010/12/31"[PDAT]) AND "SARS"[Title]) AND "coronavirus"[Title] AND "complete genome"[Title]'
COV_BASELINE_FILE = 'baselines/baseline_cov.pkl'
COV_BASELINE_FILE_COMPOUND = 'baselines/baseline_cov_compound.pkl'
COV_BASELINE_FILE_GRU = 'baselines/baseline_cov_gru.pkl'
HUMAN_BASELINE_FILE = 'baselines/baseline_human_apex.csv'
HUMAN_BASELINE_FILE_COMPOUND = 'baselines/baseline_human_apex_compound.csv'
HUMAN_BASELINE_FILE_GRU = 'baselines/baseline_human_apex_gru.csv'
HUMAN_BASELINE_LABELS_FILE = 'baselines/baseline_human_apex_labels.csv'

COV_CONSERVED = [  # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3185738/
    "orf1ab",
    "s",
    "e",
    "m",
    "n",
]

def rank(value:float, baseline:np.ndarray) -> float:
    """Rank of value relative to baseline (0.5 would be median)"""
    return np.sum(baseline.flatten() <= value) / baseline.size

def zscore(value:float, baseline:np.ndarray) -> float:
    """Z score of value relative to baseline"""
    mean = np.mean(baseline.flatten())
    sd = np.mean(baseline.flatten())
    z = (value - mean) / sd
    return z

class Baseline(object):
    """
    Given matrices of (obs x variables) score incoming values
    """
    def __init__(self, values_fname:str=HUMAN_BASELINE_FILE, labels_fname:str="", score_func=rank):
        self.values = np.loadtxt(values_fname)
        assert len(self.values.shape) == 2
        self.labels = np.loadtxt(labels_fname) if labels_fname else None
        if self.labels is not None:
            assert self.labels.shape == self.values.shape, f"Got mismatched shapes"
        self.score_func = score_func

    def score_single(self, idx:int, value:float):
        """Score a single value against a given column"""
        baseline_values = self.values[:, idx]
        if self.labels is not None:  # Select only positive examples
            baseline_labels = self.labels[:, idx]
            baseline_values = baseline_values[baseline_labels > 0]
        return self.score_func(value, baseline_values)

    def score_vec(self, values:np.ndarray):
        """Score a vector against all columns"""
        assert len(values.shape) == 1
        assert values.size == self.values.shape[1], f"Got differing sizes: {values.size} {self.values.shape[1]}"
        return np.array([self.score_single(i, x) for i, x in enumerate(values)])

    def score_mat(self, values:pd.DataFrame):
        """Score a matrix against all columns"""
        assert len(values.shape) == 2
        return np.vstack([self.score_vec(row) for row in values.values])

    def best_match(self, vec:np.ndarray, top_n:int=5, max_dist:float=None, dist_func=distance.euclidean, hist_file:str="") -> List[int]:
        """Returns the index of the best n matches disregarding labels"""
        # These indexes correspond to the i-th transcripts
        assert top_n is not None or max_dist is not None
        assert len(vec.shape) == 1
        d = np.array([dist_func(vec, row) for row in self.values])
        idx = np.argsort(d)
        if top_n is not None:
            idx = idx[:top_n]
        if max_dist is not None:
            idx_values = d[idx]
            idx = list(idx[idx_values <= max_dist])
        retval = list(idx)

        if hist_file:
            fig, ax = plt.subplots()
            ax.hist(d, bins=25, alpha=0.8)
            ax.axvline(d[retval[-1]], linestyle='--', label=f"Cutoff at {d[retval[-1]]:.4f}")
            ax.set(
                title="Histogram of distances",
                xlabel="Euclidean distance between predictions",
            )
            ax.legend()
            fig.savefig(hist_file)

        return retval

class CoronavirusBaseline(object):
    """
    Given matrices of (obs/gene x localization) score incoming values
    """
    def __init__(self, values_fname:str=COV_BASELINE_FILE, score_func=rank):
        with open(values_fname, 'rb') as source:
            self.values = pickle.load(source)  # Dictionary of protein names to baseline preds
        self.values = {k: v for k, v in self.values.items() if k in COV_CONSERVED}
        self.score_func = score_func

    def score_single(self, gene:str, idx:int, value:float) -> float:
        if gene.lower() not in self.values:
            return np.NaN
        df = self.values[gene.lower()]
        baseline_values = df.values[:, idx]
        return self.score_func(value, baseline_values)

    def score_vec(self, gene:str, values:np.ndarray) -> np.ndarray:
        assert len(values.shape) == 1
        return np.array([self.score_single(gene, i, x) for i, x in enumerate(values)])

    def score_mat(self, values:pd.DataFrame) -> np.ndarray:
        """"""
        assert len(values.shape) == 2
        return np.vstack([self.score_vec(gene, row) for gene, row in values.iterrows()])

@functools.lru_cache(maxsize=10)
def load_apex_test_dataset(fold:int=5):
    """Loads in the k-th fold dataset"""
    test_dataset = data_loader.LocalizationClassificationKmers(
        split='test',
        k_fold=fold,
    )
    return test_dataset

def get_human_apex_baseline(values_fname:str, labels_fname:str, fold:int=5, model=RNAGPS):
    """
    Return human baseline based on test set predictions
    """
    test_dataset = load_apex_test_dataset(fold)
    test_data, test_labels = data_loader.load_data_as_np(test_dataset, progress_bar=True)
    test_preds = model.predict_proba(test_data)
    if not isinstance(test_preds, np.ndarray):
        test_preds = model_utils.list_preds_to_array_preds(test_preds)
    np.savetxt(values_fname, test_preds)
    np.savetxt(labels_fname, test_labels)
    return test_preds

def get_cov_baseline(values_fname:str, use_gru:bool=False, model=RNAGPS, metadata_dict:dict=collections.defaultdict(list)) -> Dict[str, pd.DataFrame]:
    """
    Returns baseline based on other coronaviruses that infect humans
    Writes to values_fname and returns
    Metadata_dict is not returned, but is modified such that if one is provided
    it will contain the given metadata
    """
    # Human betacoronavirus sequences that are not COVID-19
    query_string = COV_BASELINE_QUERY
    identifiers = covid19.query_genbank(query_string)
    genbank_records = (covid19.fetch_genbank(i) for i in identifiers)
    # print(len(identifiers))

    genbank_preds = {}
    for record in tqdm.tqdm(genbank_records, total=len(identifiers), disable=False):
        try:
            preds = covid19.pred_feature_dict(record, use_gru=use_gru, model=model)  # Produces a df (genes x localization)
            genbank_preds[record.name] = preds
            metadata_dict['species'].append(covid19.get_species(record))
        except (AssertionError, ValueError, KeyError) as _err:
            continue
    logging.info(f"Got coronavirus baseline of {len(genbank_preds)} genomes")

    bline = collections.defaultdict(dict)
    for record, preds in genbank_preds.items():
        for gene, row in preds.iterrows():
            bline[gene.lower()][record] = row
    retval = {k: pd.DataFrame(v).T for k, v in bline.items()}
    with open(values_fname, 'wb') as sink:
        pickle.dump(retval, sink)
    return retval

def generate_baselines():
    """
    Generate and save baselines
    """
    if not os.path.isdir('baselines'):
        os.mkdir("baselines")
    cov_metadata_dict = collections.defaultdict(list)
    # RF baselines
    get_cov_baseline(COV_BASELINE_FILE, use_gru=False, metadata_dict=cov_metadata_dict)
    get_human_apex_baseline(HUMAN_BASELINE_FILE, HUMAN_BASELINE_LABELS_FILE)

    # Dump the baseline metadata
    with open("baselines/cov_metadata.json", 'w') as sink:
        json.dump(cov_metadata_dict, sink, indent=4)

    # GRU baselines (human one is already done)
    get_cov_baseline(COV_BASELINE_FILE_GRU, use_gru=True)

    # Compound baselines (where we have separate nuc cyto predictors)
    get_human_apex_baseline(HUMAN_BASELINE_FILE_COMPOUND, HUMAN_BASELINE_LABELS_FILE, model=NUC_CYTO_COMPOUND_MODEL)
    get_cov_baseline(COV_BASELINE_FILE_COMPOUND, use_gru=False, model=NUC_CYTO_COMPOUND_MODEL)

def get_ith_apex_test_sequence_parts(i:int, fold:int=5):
    dataset = load_apex_test_dataset(fold=fold)
    return dataset.get_ith_trans_parts(i)

def find_similar_apex_sequences(identifier:str, top_n:int=5, max_dist:float=0.075) -> Dict[str, List[List[str]]]:
    """
    Given an identifier, load in the predictions, and find sequences that are similar
    Maps each gene to a list of top n transcript parts
    """
    preds_file = os.path.join("covid19_localization_full", "single_preds", f"{identifier}.csv")
    preds = pd.read_csv(preds_file, index_col=0, header=0)
    human_baseline = Baseline()

    retval = {}
    for gene in preds.index:
        idx = human_baseline.best_match(preds.loc[gene], top_n=top_n, max_dist=max_dist)
        retval[gene] = [get_ith_apex_test_sequence_parts(i) for i in idx]
    return retval

def load_viral_parts(identifier:str) -> Dict[str, List[str]]:
    fasta_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "covid19_localization_full", "sequences.fa",
    )
    fasta_dict = {tuple(k.split("|")): v for k, v in read_file_as_sequence_dict(fasta_file).items()}
    fasta_dict_sub = {k: v for k, v in fasta_dict.items() if k[0] == identifier}
    genes = set([k[1] for k in fasta_dict_sub.keys()])
    retval = {}
    for gene in genes:
        retval[gene] = (
            fasta_dict_sub[(identifier, gene, '5')],
            fasta_dict_sub[(identifier, gene, 'CDS')],
            fasta_dict_sub[(identifier, gene, '3')],
        )
    return retval

def make_fasta_for_meme(identifier:str, out_fa_prefix:str, include_5:bool=True, include_cds:bool=True, include_3:bool=True):
    """
    Find similar APEX sequences and create a fasta file for feeding into clustalO msa
    Writes multiple output files, one for each gene associated with the identifier
    """
    def concat_seq(five_utr, cds, three_utr):
        """Helper to concatenate sequences based on flags"""
        retval = ""
        if include_5: retval += five_utr
        if include_cds: retval += cds
        if include_3: retval += three_utr
        return retval

    assert "_" not in os.path.basename(out_fa_prefix)
    viral_sequences = load_viral_parts(identifier)
    apex_sequences = find_similar_apex_sequences(identifier)
    assert viral_sequences.keys() == apex_sequences.keys()

    for gene in viral_sequences.keys():
        out_fa = f"{out_fa_prefix.strip('_')}_{gene}.fa"
        seq_dict = {}
        seq_dict[f"{identifier}|{gene}|viral"] = concat_seq(*viral_sequences[gene])
        for i, seq_tuple in enumerate(apex_sequences[gene]):
            assert len(seq_tuple) == 3
            seq_dict[f"{identifier}|{gene}|apex_{i}"] = concat_seq(*seq_tuple)
        write_sequence_dict_to_file(seq_dict, out_fa)

def create_meme_fasta():
    """Run as script"""
    # For all COVID19 genomes/predicted localizations write out the sequences we need for MSA
    identifiers = [os.path.basename(fname).split(".")[0] for fname in glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "covid19_localization_full", "single_preds", "*.csv"))]
    for i in tqdm.tqdm(identifiers):
        # We exclude the 5' and 3' UTRs because they are all the same
        make_fasta_for_meme(i, f"msa_fasta/{i}", include_5=False, include_cds=True, include_3=False)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    generate_baselines()

