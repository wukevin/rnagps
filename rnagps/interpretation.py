"""
Functions related directly to interpreting sequences
"""

import os
import sys
import shutil
import logging
import glob
import random
import re
import logging
import collections
import itertools

import numpy as np
import pandas as pd
import scipy

import torch
import torch.nn as nn

import sklearn.metrics as metrics

import tqdm

import model_utils
import kmer
import pwm
import seq
import fasta
import utils

LOCAL_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data"))
assert os.path.isdir(LOCAL_DATA_DIR), "Cannot find data directory: {}".format(LOCAL_DATA_DIR)

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

if torch.cuda.is_available():
    logging.info(torch.cuda.get_device_name(torch.cuda.current_device()))
    DEVICE = torch.device(f"cuda:{torch.cuda.device_count() - 1}")
else:
    logging.info("CPU")
    DEVICE = torch.device("cpu")

def ablate(sequence, target, method='N', max_iter=None):
    """Removes all occurrences of target from the sequence"""
    assert target in sequence, f"Could not find target {target} in sequence"
    assert target != "N" * len(target)
    retval = sequence
    if max_iter is None:
        max_iter = len(sequence)
    assert isinstance(max_iter, int), "max_iter must be an integer"
    match_indices = [m.start() for m in re.finditer(target, sequence)]
    random.shuffle(match_indices)  # Shuffle in case it's a subset
    match_indices = match_indices[:max_iter]  # Limit to the maximum iterations
    for start_idx in match_indices:
        end_idx = start_idx + len(target)
        if method == 'random':
            replace = ''.join([random.choice("ACGT") for _i in range(len(target))])
        elif method == "N":
            replace = "N" * len(target)
        else:
            raise NotImplementedError(f"Unrecognized method: {method}")
        retval = retval[:start_idx] + replace + retval[end_idx:]
    assert len(retval) == len(sequence)  # Make sure we don't lose bases
    # assert target not in retval, f"Found target {target} in final sequence"
    return retval

def ablate_ppm(sequence, ppm, method='N', prop=0.8, max_iter=None):
    """Permute the sequence until the PPM is no longer found"""
    hits = pwm.find_ppm_hits(sequence, ppm, prop=prop)
    if not hits:  # If no matches, return sequence as is
        return sequence
    if max_iter is None:
        max_iter = int(len(sequence) / len(ppm))
    assert isinstance(max_iter, int)
    retval = sequence
    for start_idx in hits:
        end_idx = start_idx + ppm.shape[0]
        if method == "N":
            replace = "N" * ppm.shape[0]
        elif method == 'shuffle':
            chunk = list(retval[start_idx:end_idx])
            random.shuffle(chunk)
            replace = ''.join(chunk)
        elif method == 'random':
            replace = ''.join([random.choice("ACGT") for _i in range(ppm.shape[0])])
        else:
            raise NotImplementedError(f"Unrecognized method: {method}")
        retval = retval[:start_idx] + replace + retval[end_idx:]
    assert len(retval) == len(sequence)
    return retval

def generate_random_ablation_seqs(n=5, l=7, seed=1234, localizations=['Erm', 'Lma', 'Mito', 'Nes', 'Nik', 'Nls', 'NucPore', 'Omm'], transcript_parts=['u5', 'cds', 'u3']):
    """
    Generate random sequences to ablate
    Each localization/transcript part will have n random l-mers
    """
    random.seed(seed)
    random_sequences_to_ablate = {}
    for localization in localizations:
        random_sequences_to_ablate[localization] = {}
        for trans_part in transcript_parts:
            seqs = set()
            while len(seqs) < n:
                random_seq = ''.join([random.choice(list("ACGT")) for _i in range(7)])
                seqs.add(random_seq)
            random_sequences_to_ablate[localization][trans_part] = seqs
    return random_sequences_to_ablate

def read_motifs_from_tomtom(tomtom_tsv_files, q_cutoff=None):
    """
    Given TomTom tsv output, read in sequences into a dictionary of <localization: < transcript part: seqs>>
    This output format is compatible with the ablation method from above
    """
    sequences_to_ablate = collections.defaultdict(dict)  # localization to transcript part
    for fname in tomtom_tsv_files:
        localization, transcript_part, _x, _y = os.path.basename(os.path.dirname(fname)).split(".")[0].split("_")
        tomtom_results = read_tomtom_tsv(fname, q_cutoff=q_cutoff)
        motif_sequences = set(tomtom_results['Query_consensus'])  # This step tends to reduce the number of actual sequences compared to motif hits
        sequences_to_ablate[localization][transcript_part] = motif_sequences
    return sequences_to_ablate

def read_motifs_from_fa(fa_files):
    """
    Given fasta files, read in sequences into a dictionary of <localization: <transcript_part:seq>>
    Fasta file name is assumed to be _ delimited with localization and transcript part as first two parts
    This output format is compatible with the ablation method above
    """
    sequences = collections.defaultdict(dict)
    for fname in fa_files:
        localization, transcript_part, _x, _y = os.path.basename(fname).split(".")[0].split("_")
        fa_contents = fasta.read_file_as_sequence_dict(fname)
        motifs = set(fa_contents.values())
        sequences[localization][transcript_part] = motifs
    return sequences

def get_feature_importance_z_scores(perturbation_tensor, reference_performance):
    """
    Return a matrix of z scores
    """
    avg = np.mean(perturbation_tensor, axis=0) - reference_performance
    sd = np.std(perturbation_tensor, axis=0)
    sd[sd == 0] = np.min(sd[sd != 0])  # Cap the 0 SD's to the smallest nonzero SD
    z = avg / sd
    assert not np.any(np.isnan(z))
    assert not np.any(np.isinf(z))
    return z

def get_significant_features(perturbation_tensor, reference_performance, z_cutoff=-2):
    """
    Return a boolean matrix indicating if a feature is significant
    """
    assert z_cutoff < 0
    z = get_feature_importance_z_scores(perturbation_tensor, reference_performance)
    retval = z <= z_cutoff
    logging.info("Number of significant features per localization: {}".format(np.sum(retval, axis=0)))
    return retval

def get_explanatory_motifs(valid_prob, truth_labels, significant_ft, valid_dataset, gap_allowed=2, min_len=6, youden=True, stdout=False):
    """
    Get explanatory motifs from each segment of the transcript
    valid_prob/truth_labels are *single-class* predictions and labels
    significant_ft is a list/iterable of significant features for consideration
    Returns three dictionaries, each representing the "fasta" of u5, cds, and u3 assembled kmers
    """
    cutoff = 0.5
    if youden:
        if isinstance(youden, bool):
            cutoff = model_utils.youden_threshold(valid_prob, truth_labels)
            logging.info(f"Computed cutoff of {cutoff}")
        else:
            cutoff = youden

    tp_indices = np.where(np.logical_and(truth_labels.flatten(), valid_prob.flatten() >= cutoff))
    retval = [{}, {}, {}]
    for ith_part, prefix in enumerate(['u5', 'cds', 'u3']):
        # Kmer features, without the u5/cds/u3 prefix, so they can be assembled
        sig_ft_part = [ft.split("_")[1] for ft in significant_ft if ft.startswith(prefix)]
        for i in range(len(tp_indices[0])):
            assembled_kmers = kmer.assemble_kmer_motifs(
                valid_dataset.get_ith_trans_parts(tp_indices[0][i])[ith_part],
                sig_ft_part,
                min_len=min_len,  # This represents 2 3-mers
                gap_allowed=gap_allowed,
            )
            for j, ak in enumerate(assembled_kmers):
                fasta_header = f">tp_{i}_{prefix}_assembled_kmer_{j}"
                retval[ith_part][fasta_header] = ak
                if stdout: print(f"{fasta_header}\n{ak}")
    return retval

def evaluate_pwm_importance(model, dataset, ppms, prop=0.9, ablation_strat="N", progressbar=True, device=DEVICE):
    """
    Given a dataset and PWMs, evaluate which PPMs, when ablated, impact the model the most
    Works for both pytorch and sklearn models. For sklearn models, assumes the dataset is
    per-transcript-part kmer featurization with kmer sizes of 3, 4, 5
    """
    def seq_parts_to_ft_vec(transcript_parts, kmer_sizes=[3, 4, 5]):
        """Helper function to translate seq parts to feature vector of one sample"""
        return np.atleast_2d(np.hstack([kmer.sequence_to_kmer_freqs(part, kmer_size=s) for s in kmer_sizes for part in transcript_parts]))

    def seq_parts_to_ft_vec_torch(transcript_parts, kmer_sizes=[3, 4, 5], with_len=False):
        """Helper function to translate seq parts to feature vector of one sample"""
        seq_encoded = np.array([seq.BASE_TO_INT[b] for b in ''.join(transcript_parts)])
        retval = torch.from_numpy(seq_encoded[:, np.newaxis, np.newaxis]).type(torch.LongTensor).to(device)
        length_retval = torch.LongTensor([len(''.join(transcript_parts))])
        if with_len:
            return retval, length_retval
        else:
            return retval
    def sigmoid(x):
        """Calculate the sigmoid function"""
        return 1.0 / (1.0 + np.exp(-x))

    is_torch_model = isinstance(model, nn.Module)
    is_batched_torch_model = is_torch_model and model.forward.__code__.co_argcount==3
    retval = {localization: pd.DataFrame(  # Create empty dataframe that we'll fill in later
        np.nan,
        index=np.arange(len(dataset)),
        columns=[f"{trans_part}_{ppm_name}" for trans_part, ppm_name in itertools.product(['u5', 'cds', 'u3'], ppms.keys())]
    ) for localization in dataset.compartments}

    if is_torch_model:
        model.to(device)
    ppm_names = list(ppms.keys())
    ppm_mats = [ppms[k] for k in ppm_names]
    pbar = tqdm.tqdm if not utils.isnotebook() else tqdm.tqdm_notebook
    for i in pbar(range(len(dataset)), disable=not progressbar):
        seq_parts = dataset.get_ith_trans_parts(i)
        if is_torch_model:
            if not is_batched_torch_model:
                orig_preds = model(seq_parts_to_ft_vec_torch(seq_parts))
            else:
                orig_preds = model(*seq_parts_to_ft_vec_torch(seq_parts, with_len=True))
            orig_preds = sigmoid(orig_preds.detach().cpu().numpy()) # Make it a numpy array
        else:
            orig_preds = model_utils.list_preds_to_array_preds(model.predict_proba(seq_parts_to_ft_vec(seq_parts)))
        assert np.all(orig_preds <= 1.0) and np.all(orig_preds >= 0), "Preds are expected to be probabilties, so cannot exceed [0, 1]"
        for j, (part_name, part) in enumerate(zip(['u5', 'cds', 'u3'], seq_parts)):
            ablations = [ablate_ppm(part, p, ablation_strat, prop) for p in ppm_mats]
            for ppm_name, ppm, ablated_part in zip(ppm_names, ppm_mats, ablations):
                if ablated_part != part:  # There was *something* to ablate
                    seq_parts_ablated = list(seq_parts)  # Make a copy of the original and sub in the part we just ablated
                    seq_parts_ablated[j] = ablated_part
                    if is_torch_model:
                        if not is_batched_torch_model:
                            ablated_preds = model(seq_parts_to_ft_vec_torch(seq_parts_ablated))
                        else:
                            ablated_preds = model(*seq_parts_to_ft_vec_torch(seq_parts_ablated, with_len=True))
                        ablated_preds = sigmoid(ablated_preds.detach().cpu().numpy())
                    else:
                        ablated_preds = model_utils.list_preds_to_array_preds(model.predict_proba(seq_parts_to_ft_vec(seq_parts_ablated)))
                    assert np.all(ablated_preds <= 1.0) and np.all(ablated_preds >= 0.0), "Preds are expected to be probabilities, so cannot exceed [0, 1]"
                    delta = ablated_preds - orig_preds
                    for localization, impact, true_label in zip(dataset.compartments, delta.flatten(), np.squeeze(dataset.get_ith_labels(i))):
                        if true_label:  # only insert if it's suppose to be positive localization
                            retval[localization].loc[i, f"{part_name}_{ppm_name}"] = impact
    return retval

def pwm_importance_to_significant_hits(dict_of_tables, avg_impact_cutoff:float=-0.01, min_count:int=5, override_count:int=72):
    """
    Given a dictioanry of dataframes denoting importance (rows are obs, cols are features)
    process it and return a dataframe of significant RBPs per localization
    """
    assert isinstance(dict_of_tables, dict)
    pwm_importance_neg_counts = []
    pwm_importance_avg_impact = []
    pwm_importance_sd_impact = []
    for localization in LOCALIZATIONS:
        this_local_pwm_importance = dict_of_tables[localization]
        assert isinstance(this_local_pwm_importance, pd.DataFrame)
        this_local_ft_neg = np.logical_and(
            ~np.isnan(this_local_pwm_importance.values),
            this_local_pwm_importance.values < 0
        )
        this_local_ft_neg_count = np.sum(this_local_ft_neg, axis=0)

        pwm_importance_neg_counts.append(this_local_ft_neg_count)
        pwm_importance_avg_impact.append(np.nanmean(this_local_pwm_importance, axis=0))
        pwm_importance_sd_impact.append(np.nanstd(this_local_pwm_importance, axis=0))

    pwm_importance_avg_impact = pd.DataFrame(
        pwm_importance_avg_impact,
        index=LOCALIZATIONS,
        columns=this_local_pwm_importance.columns,
    )
    pwm_importance_sd_impact = pd.DataFrame(
        pwm_importance_sd_impact,
        index=LOCALIZATIONS,
        columns=this_local_pwm_importance.columns,
    )
    pwm_importance_neg_counts = pd.DataFrame(
        pwm_importance_neg_counts,
        index=LOCALIZATIONS,
        columns=this_local_pwm_importance.columns,
    )

    desired_idx = np.logical_and(
        np.logical_and(
            pwm_importance_avg_impact.values <= avg_impact_cutoff,
            pwm_importance_neg_counts >= min_count,
        ),
        ~np.isnan(pwm_importance_avg_impact.values)
    )
    desired_idx = np.logical_or(
        desired_idx,
        pwm_importance_neg_counts >= override_count,
    )
    # print(np.sum(desired_idx.values))

    pwm_importance_rbps = {l: list() for l in LOCALIZATIONS}
    pwm_importance_rbps_ids = {l: list() for  l in LOCALIZATIONS}
    pwm_importance_rbps_ids_by_trans_part = collections.defaultdict(list)
    pwm_importance_hit_locations = pd.DataFrame(
        0,
        index=LOCALIZATIONS,
        columns=['u5', 'cds', 'u3'],
    )
    for local_idx, ft_idx in zip(*np.where(desired_idx.values)):
        localization = pwm_importance_avg_impact.index[local_idx]
        trans_part, pwm_name = pwm_importance_avg_impact.columns[ft_idx].split("_")
        id_name, gene_name = pwm_name.split()
        pwm_importance_rbps[localization].append(gene_name)
        pwm_importance_rbps_ids[localization].append(id_name)
        pwm_importance_rbps_ids_by_trans_part[localization + "_" + trans_part].append(id_name)
        pwm_importance_hit_locations.loc[localization, trans_part] += 1

    all_genes = sorted(list(set(itertools.chain.from_iterable(pwm_importance_rbps.values()))))
    all_genes_counts = pd.DataFrame(
        0,
        index=LOCALIZATIONS,
        columns=all_genes,
    )
    for localization, genes in pwm_importance_rbps.items():
        for gene in genes:
            all_genes_counts.loc[localization, gene] += 1
    all_genes_counts.index = [LOCALIZATION_FULL_NAME_DICT[l] for l in LOCALIZATIONS]
    return all_genes_counts

def pwm_to_kmer_probs(pwm, k=4, reduce=True):
    """
    Given a PWM, return the probability of every kmer size
    PWM are assumed to be a dataframe where columns are bases, and index is each position
    """
    def _get_kmer_probs(sub_pwm):
        """Enumerate all the kmers possible from the sub_pwm"""
        combos = itertools.product(*sub_pwm.itertuples(index=False))
        probs = [np.product(c) for c in combos]
        return probs
    alphabet = pwm.columns
    kmers = list(itertools.product(alphabet, repeat=k))
    probs = []
    for start_idx in range(pwm.shape[0] - k + 1):
        sub_pwm = pwm.iloc[start_idx:start_idx + k, :]
        sub_probs = _get_kmer_probs(sub_pwm)  # These sum to 1
        assert len(sub_probs) == len(kmers)
        probs.append(sub_probs)
    probs = np.vstack(probs)
    if reduce:
        probs = np.sum(probs, axis=0, keepdims=True) / probs.shape[0]
    retval = pd.DataFrame(probs, columns=[''.join(k) for k in kmers])
    return retval

def compare_probs(x, y, idx=None, test=scipy.stats.ttest_rel):
    """
    Given two lists of predicted probabilities, test at truths to see if the probabilities at those positions
    are statistically different
    test must take in x a dy and return a paired

    Return a test statistic and p-value
    """
    if isinstance(x, list):
        x = model_utils.list_preds_to_array_preds(x)
    if isinstance(y, list):
        y = model_utils.list_preds_to_array_preds(y)
    assert x.shape == y.shape
    # Just flattens - what about comparing per localization?
    x_vals = x[idx].flatten() if idx is not None else x.flatten()
    y_vals = y[idx].flatten() if idx is not None else y.flatten()
    # assert not np.any(np.isnan(x_vals))
    # assert not np.any(np.isnan(y_vals))
    stat, pvalue = test(x_vals, y_vals)
    return stat, pvalue

def read_meme_pwms(fname, incl_gene=True):
    """Reads the meme file as a dictionary of motifs"""
    def is_numeric(x):
        """Returnss true if x is numeric"""
        try:
            _x = float(x)
            return True
        except ValueError:
            return False

    pwms = {}
    curr_item = None
    alphabet = None
    with open(fname) as source:
        for line in source:
            line = line.strip()
            if not line:
                continue
            if line.startswith("ALPHABET"):
                assert not alphabet  # Can only be set once
                alphabet = line.split("=")[1].strip()
            elif line.startswith("MOTIF"):
                curr_item = line.replace("MOTIF", "").strip()
                if not incl_gene:
                    curr_item = curr_item.split()[0]
                pwms[curr_item] = []
            elif all([is_numeric(t.strip()) for t in line.split()]):
                v = np.array([float(t.strip()) for t in line.split()])
                assert len(v) == 4
                pwms[curr_item].append(v)
    return {k:pd.DataFrame(np.vstack(v), columns=list(alphabet)) for k, v in pwms.items()}

def read_rna_complete_html_table(fname=os.path.join(LOCAL_DATA_DIR, "meme", "RNAcompete_report_index.html"), human_only=True):
    """Reads the html table mapping RNCMPT* ids to gene names"""
    # https://pandas.pydata.org/pandas-docs/version/0.23.4/generated/pandas.read_html.html
    with open(fname) as source:
        retval = pd.read_html(source)[0]
    if human_only:
        retval = retval[retval['Species'] == 'Homo_sapiens']
    return retval

def read_tomtom_tsv(fname, q_cutoff=None, e_cutoff=None, p_cutoff=None):
    """Reads the tomtom table"""
    tab = pd.read_csv(fname, delimiter='\t', engine='c', comment='#')
    if q_cutoff is not None:
        tab = tab[tab.loc[:, "q-value"] <= q_cutoff]
    if e_cutoff is not None:
        tab = tab[tab.loc[:, "E-value"] <= e_cutoff]
    if p_cutoff is not None:
        tab = tab[tab.loc[:, "p-value"] <= p_cutoff]
    tab = tab.reset_index(drop=True)
    return tab

def save_dict_of_tables(dict_of_tables, tar_fname):
    """Saves the dictionary of tables to a tar file"""
    assert tar_fname.endswith(".tar")
    temp_folder = '.'.join(tar_fname.split(".")[:-1])
    assert not os.path.exists(temp_folder)
    os.makedirs(temp_folder)

    for k, v in dict_of_tables.items():
        v.to_csv(os.path.join(temp_folder, f"{k}.csv"))

    shutil.make_archive(tar_fname.rstrip(".tar"), 'tar', temp_folder)
    shutil.rmtree(temp_folder)

    return os.path.abspath(tar_fname)

def read_dict_of_tables(tar_fname):
    """Read a dictionary of dfs from a tar file"""
    temp_folder = '.'.join(tar_fname.split(".")[:-1])
    assert not os.path.exists(temp_folder), f"Folder {temp_folder} already exists!"
    os.makedirs(temp_folder)

    shutil.unpack_archive(tar_fname, temp_folder)
    retval = {}
    for df_fname in glob.glob(os.path.join(temp_folder, "*.csv")):
        k = os.path.basename(df_fname).split(".")[0]
        v = pd.read_csv(df_fname, index_col=0, engine='c', low_memory=False)
        retval[k] = v
    shutil.rmtree(temp_folder)
    return retval

if __name__ == "__main__":
    # read_rna_complete_html_table()
    # pwms = read_meme_pwms("/Users/kevin/Documents/Stanford/zou/chang_rna_localization/rnafinder/data/meme/Ray2013_rbp_Homo_sapiens.dna_encoded.meme")
    # print(pwm_to_kmer_probs(pwms['RNCMPT00079 U2AF2']))
    # print(read_tomtom_tsv("/Users/kevin/Documents/Stanford/zou/chang_rna_localization/rnafinder/data/intermediate/kmer_assembly_0_bp_gap/localization_tomtom/Mito_u3_assembled_kmers/tomtom.tsv", q_cutoff=0.05))
    print(pwm_importance_to_significant_hits(
        read_dict_of_tables("/storage/wukevin/projects/rnagps/data/intermediate_kfold_5/pwm_test_importance_N.tar")
    ))

