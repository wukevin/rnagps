"""
Deep dive on the mitochondrial signal

Specifically, we want to compare the predictions from covid19 to:
- The ~400 non-canonical transcripts at the mito matrix
- The ~15 canonical transcripts at the mito matrix

To do:
- Do a PCA-based analysis on feature space to see what SARS-CoV-2 is more
  similar to in feature space.
  - Or more simply, checking things like GC content might work too
  - Also compare to nucleus

Example command to convert pdf to png
convert -density 500 cov_mito_clustering.pdf -resample 300x300 -units pixelsperinch cov_mito_clustering.png
"""

import os
import sys
import logging
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
from scipy import spatial
from sklearn.decomposition import PCA
from umap import UMAP

import matplotlib.pyplot as plt
import seaborn as sns

import tqdm

from baseline import load_apex_test_dataset, load_apex_full_dataset, RNAGPS
import covid19

SRC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rnagps",
)
assert os.path.isdir(SRC_DIR)
sys.path.append(SRC_DIR)
import data_loader
from model_utils import list_preds_to_array_preds
from utils import read_gtf_trans_to_exons

TRANS_MAP = read_gtf_trans_to_exons()


def gc_content(seq: str) -> float:
    """
    Calculate the GC content of a given sequence
    >>> gc_content("ACGT")
    0.5
    >>> gc_content("ACCCG")
    0.8
    """
    seq = seq.upper()
    bases = set(seq)
    assert bases.issubset({"A", "C", "G", "T", "N"}), f"Unrecognized alphabet: {bases}"
    return (seq.count("G") + seq.count("C")) / len(seq)


def get_covid_mito_preds() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Return the mito matrix predictions from covid
    """
    identifiers = covid19.query_genbank()

    genome_seqs = []
    featurizations, predictions, names = [], [], []
    for identifier in tqdm.tqdm(identifiers):
        try:
            record = covid19.fetch_genbank(identifier)
            record_ft_dict = covid19.genbank_to_feature_dict(record)
            gene_names = covid19.get_feature_labels(record)
            assert len(gene_names) == len(record_ft_dict["CDS"])
            assert record_ft_dict["CDS"]

            five_utr = record_ft_dict["5'UTR"]
            assert len(five_utr) == 1
            three_utr = record_ft_dict["3'UTR"]
            assert len(three_utr) == 1
            genome = str(record.seq)
            five_utr_seq = covid19.seq_feature_to_sequence(five_utr[0], genome)
            three_utr_seq = covid19.seq_feature_to_sequence(three_utr[0], genome)
            cds_seq = genome[five_utr[0].location.end : three_utr[0].location.start]

            ft = covid19.featurize(
                [five_utr_seq, cds_seq, three_utr_seq], [False, False, False]
            )
            preds = list_preds_to_array_preds(RNAGPS.predict_proba(ft)).squeeze()
            names.append(record.name)
            featurizations.append(ft)
            predictions.append(preds)
            genome_seqs.append(five_utr_seq + cds_seq + three_utr_seq)

        except (AssertionError) as _err:
            continue

    featurizations = pd.DataFrame(data=np.vstack(featurizations), index=names)
    predictions = pd.DataFrame(
        data=np.vstack(predictions),
        index=names,
        columns=data_loader.LOCALIZATION_FULL_NAME_DICT.values(),
    )

    covid_gc_contents = np.array([gc_content(s) for s in genome_seqs])
    logging.info(
        f"SARS-CoV-2 genomes: GC content mean/sd: {np.mean(covid_gc_contents)} / {np.std(covid_gc_contents)}"
    )

    return featurizations, predictions["Mito matrix"]


def get_human_apex_mito_preds(
    compartment: str = "Mito matrix", test_only: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns the featurization and the output localization probabilities for mito
    """
    # Test dataset actually contains either full or test data
    if test_only:
        test_dataset = load_apex_test_dataset()
    else:
        test_dataset = load_apex_full_dataset()

    test_data, test_labels = data_loader.load_data_as_np(
        test_dataset, progress_bar=True
    )
    test_labels = pd.DataFrame(
        test_labels,
        index=[
            test_dataset.get_ith_trans_name(i).split(".")[0]
            for i in range(len(test_labels))
        ],
        columns=data_loader.LOCALIZATION_FULL_NAME_DICT.values(),
    )
    test_preds = pd.DataFrame(
        list_preds_to_array_preds(RNAGPS.predict_proba(test_data)),
        index=[
            test_dataset.get_ith_trans_name(i).split(".")[0]
            for i in range(len(test_data))
        ],
        columns=data_loader.LOCALIZATION_FULL_NAME_DICT.values(),
    )
    test_data = pd.DataFrame(test_data, index=test_labels.index,)
    test_transcripts = pd.Series(
        [
            "".join(test_dataset.get_ith_trans_parts(i))
            for i in range(len(test_dataset))
        ],
        index=test_labels.index,
    )

    mito_transcripts = test_labels.index[test_labels.loc[:, compartment] > 0]
    test_data = test_data.loc[mito_transcripts]
    test_transcripts = test_transcripts.loc[mito_transcripts]

    transcript_chroms = []
    for transcript in mito_transcripts:
        chroms = {exon[1] for exon in TRANS_MAP[transcript]}
        if chroms:
            assert len(chroms) == 1
            transcript_chroms.append(chroms.pop())
        else:
            transcript_chroms.append(None)
    assert len(transcript_chroms) == len(mito_transcripts)

    noncanon_test_gc_contents = np.array(
        [
            gc_content(s)
            for chrom, s in zip(transcript_chroms, test_transcripts)
            if chrom != "MT"
        ]
    )
    canon_test_gc_contents = [
        gc_content(s)
        for chrom, s in zip(transcript_chroms, test_transcripts)
        if chrom == "MT"
    ]
    logging.info(
        f"{'TEST' if test_only else 'FULL'} APEX non-MT {compartment} transcripts: GC content mean/sd: {np.mean(noncanon_test_gc_contents)} / {np.std(noncanon_test_gc_contents)}"
    )
    if len(canon_test_gc_contents) > 0:
        logging.info(
            f"{'TEST' if test_only else 'FULL'} APEX MT {compartment} transcripts: GC content mean/sd: {np.mean(canon_test_gc_contents)} / {np.std(canon_test_gc_contents)}"
        )

    retval = pd.DataFrame(
        data={
            "prediction": test_preds.loc[mito_transcripts, compartment],
            "chromosome": transcript_chroms,
        }
    )
    assert len(test_data) == len(
        retval
    ), f"Got mismatched lengths: {len(test_data)} {len(retval)}"
    return test_data, retval


def plot_mito_hist(covid_preds, apex_preds):
    """
    Plot histogram
    """
    fig, ax = plt.subplots()
    hist_bins = np.arange(start=0, stop=1, step=0.05)

    nonmito_idx = [chrom != "MT" for chrom in apex_preds["chromosome"]]
    ax.hist(
        apex_preds.loc[nonmito_idx, "prediction"],
        alpha=0.5,
        bins=hist_bins,
        label=f"Noncanonical mito test preds (n={sum(nonmito_idx)})",
        density=True,
    )
    mito_idx = [chrom == "MT" for chrom in apex_preds["chromosome"]]
    ax.hist(
        apex_preds.loc[mito_idx, "prediction"],
        alpha=0.5,
        bins=hist_bins,
        label=f"Canoncal mito RNA test preds (n={sum(mito_idx)})",
        density=True,
    )
    ax.hist(
        covid_preds.values,
        alpha=0.5,
        bins=hist_bins,
        label=f"SARS-CoV-2 mito preds (n={len(covid_preds.values)})",
        density=True,
    )
    ax.legend()
    ax.set(
        xlabel="Predicted mito matrix localization",
        title="Mito matrix localization predictions",
    )
    fig.savefig("cov_mito.pdf")


def plot_ft(
    ft_dict: Dict[str, pd.DataFrame], fname: str, method=PCA, method_name: str = "PCA"
):
    """
    Apply and plot PCA of the given features, assuming no repeated feature vectors
    """
    logging.info(f"Plotting {method_name}")
    # Apply PCA transformation
    n_comps = 10
    all_ft = np.vstack([df.values for df in ft_dict.values()])
    assert all_ft.shape[1] == 4032
    reducer = method(n_components=n_comps)  # Single pca shared by all feature spaces
    all_transformed = reducer.fit_transform(all_ft)

    # https://stackoverflow.com/questions/51210955/seaborn-jointplot-add-colors-for-each-class
    g = sns.JointGrid(all_transformed[:, 0], all_transformed[:, 1])
    for ft_name, ft in ft_dict.items():
        ft_pca = pd.DataFrame(
            reducer.transform(ft.values),
            index=ft.index,
            columns=[f"PC{i+1}" for i in range(n_comps)],
        )
        g.ax_joint.scatter(
            ft_pca.loc[:, "PC1"],
            ft_pca.loc[:, "PC2"],
            alpha=0.75,
            label=f"{ft_name} (n={len(ft_pca)})",
        )
        sns.kdeplot(ft_pca.loc[:, "PC1"], ax=g.ax_marg_x, legend=False)
        sns.kdeplot(ft_pca.loc[:, "PC2"], ax=g.ax_marg_y, vertical=True, legend=False)
    g.ax_joint.legend()
    if hasattr(reducer, "explained_variance_"):
        g.ax_joint.set(
            xlabel=f"PC1 - {reducer.explained_variance_[0]:.4f} variance",
            ylabel=f"PC2 - {reducer.explained_variance_[1]:.4f} variance",
            # title=f"RNA-GPS feature space {method_name}",
        )
        g.ax_marg_x.set(title=f"RNA-GPS feature space {method_name}")
    else:
        g.ax_marg_x.set(title=f"RNA-GPS feature space {method_name}")
    g.savefig(fname)
    # logging.info(f"Explaind variances: {pca.explained_variance_}")


def compute_ft_distance(
    ft_dict: Dict[str, pd.DataFrame], p: int = 2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute the average and SD pairwise distance between all groups
    """
    keys = list(ft_dict.keys())
    n = len(ft_dict)
    means, stdevs = np.zeros((n, n)), np.zeros((n, n))
    for i in range(len(keys)):
        for j in range(i, len(keys)):
            vals = spatial.distance_matrix(
                ft_dict[keys[i]], ft_dict[keys[j]], p=p
            ).flatten()
            m = np.mean(vals)
            s = np.std(vals)
            means[i, j] = m
            means[j, i] = m
            stdevs[i, j] = s
            stdevs[j, i] = s
    means = pd.DataFrame(means, index=keys, columns=keys,)
    stdevs = pd.DataFrame(stdevs, index=keys, columns=keys,)
    return means, stdevs


def main():
    """
    Run analysis
    """
    covid_ft, covid_mito_preds = get_covid_mito_preds()

    apex_ft, apex_mito_preds = get_human_apex_mito_preds()
    full_apex_ft, full_apex_mito_preds = get_human_apex_mito_preds(test_only=False)
    full_apex_omm_ft, full_apex_omm_preds = get_human_apex_mito_preds(
        compartment="Outer mito membrane", test_only=False
    )

    plot_mito_hist(covid_mito_preds, apex_mito_preds)  # This is only test set

    ft_dict = {
        "SARS-CoV-2": covid_ft,
        "APEX noncanonical mito": full_apex_ft.drop(
            index=full_apex_ft.index[full_apex_mito_preds["chromosome"] == "MT"]
        ),
        "APEX canonical mito": full_apex_ft.loc[
            full_apex_ft.index[full_apex_mito_preds["chromosome"] == "MT"]
        ],
        "APEX outer mito membrane": full_apex_omm_ft,
    }
    plot_ft(ft_dict, "cov_mito_clustering.pdf")
    # plot_ft(ft_dict, "cov_mito_clustering_umap.pdf", method=UMAP, method_name="UMAP")
    ft_dist_mean, ft_dist_sd = compute_ft_distance(ft_dict)
    ft_dist_mean.to_csv("cov_mit_ft_dist_mean.csv")
    ft_dist_sd.to_csv("cov_mit_ft_dist_sd.csv")
    print(ft_dist_mean)
    print(ft_dist_sd)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
    main()
