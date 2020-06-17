"""
Code for looking at COVID-19
"""

import os
import sys
import logging
import itertools
import string
import random
import json
import shelve
import functools
from typing import *
import collections

import numpy as np
import pandas as pd

import regex

import torch

from Bio import GenBank, Entrez, SeqIO, SeqRecord
from Bio.SeqFeature import SeqFeature, FeatureLocation, CompoundLocation

import tqdm

logging.basicConfig(level=logging.INFO)

SRC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rnagps",
)
assert os.path.isdir(SRC_DIR)
sys.path.append(SRC_DIR)
from model_utils import list_preds_to_array_preds, lstm_eval
from interpretation import ablate_ppm
from pwm import load_attract_ppm, load_meme_ppm, load_meme_results, find_ppm_hits
from fasta import read_file_as_seqfeature_dict
import kmer
import data_loader
import utils

sys.path.append(os.path.join(SRC_DIR, "models"))
import recurrent

DEVICE = utils.get_device(i=-1)

RESOURCE_DIR = os.path.join(os.path.dirname(SRC_DIR), "covid19",)
assert os.path.isdir(RESOURCE_DIR)

MODEL_DIR = os.path.join(os.path.dirname(SRC_DIR), "models")
assert os.path.isdir(MODEL_DIR)
RNAGPS_PATH = os.path.join(MODEL_DIR, "rf_8way_fold5.0.21.3.skmodel")
RNAGPS_MITO_DROP_PATH = os.path.join(
    MODEL_DIR, "rf_8way_mito_drop_fold5.0.21.3.skmodel"
)
RNAGRU_PATH = os.path.join(MODEL_DIR, "gru_double_best.pt")
RNAGPS = utils.load_sklearn_model(
    RNAGPS_PATH, strict=True, disable_multiprocessing=True
)
RNAGPS_MITO_DROP = utils.load_sklearn_model(
    RNAGPS_MITO_DROP_PATH, strict=True, disable_multiprocessing=True
)
RNAGRU = utils.load_pytorch_model(
    recurrent.GRULocalizationClassifier, RNAGRU_PATH, device=DEVICE
)
RNAGRU.eval()

GENBANK_QUERY_CACHE_FILE = os.path.join(
    os.path.dirname(__file__), "genbank_query_cache.json"
)
GENBANK_FETCH_CACHE_FILE = os.path.join(
    os.path.dirname(__file__), "genbank_fetch_cache"
)  # OS adds extension

COVID19_QUERY_SEQUENCE = 'txid2697049[Orgn] AND (viruses[filter] AND biomol_genomic[PROP] AND ddbj_embl_genbank[filter] AND is_nuccore[filter] AND ("20000"[SLEN] : "40000"[SLEN]))'

PROTEIN_SUFFIXES = [
    "polyprotein",
    "polypeptide",
    "glycoprotein",
    "phosphoprotein",
    "protein",  # This has to be last because otherwise we do partial strips
]
# Map non-canonical names to canonical names
PROTEIN_NAME_MAP = {
    "surface": "S",
    "spike": "S",
    "envelope": "E",
    "membrane": "M",
    "nucleocapsid": "N",
}

# Controls ablation of 5' UTR, CDS, and 3' UTR respectively
ABLATION = [False, False, False]

PREFIXES_TO_ABLATIONS = {
    "covid19_localization_full_rc": [False, False, False],  # Reverse complement
    "covid19_localization_full": [False, False, False],
    "covid19_localization_no5": [True, False, False],
    "covid19_localization_nocds": [False, True, False],
    "covid19_localization_no3": [False, False, True],
    "covid19_localization_only5": [False, True, True],
    "covid19_localization_only3": [True, True, False],
    "covid19_localization_onlycds": [True, False, True],
}
PREFIXES_TO_ABLATIONS.update(
    {
        ("mito_drop_" + k): v
        for k, v in PREFIXES_TO_ABLATIONS.items()
        if not k.startswith("gru_")
    }
)
PREFIXES_TO_ABLATIONS.update(
    {
        ("gru_" + k): v
        for k, v in PREFIXES_TO_ABLATIONS.items()
        if not k.startswith("mito_drop_")
    }
)

COMPLEMENTS = {
    "A": "T",
    "T": "A",
    "C": "G",
    "G": "C",
    "N": "N",
}

random.seed(1234)


def rev_comp(seq: str) -> str:
    """
    Return the reverse COMPLEMENTS
    >>> rev_comp("ACGT")
    'ACGT'
    >>> rev_comp("ACNNAG")
    'CTNNGT'
    """
    return "".join((COMPLEMENTS[base] for base in seq))[::-1]


def normalize_feature_labels(
    n: str, suffixes: List[str] = PROTEIN_SUFFIXES, name_map=PROTEIN_NAME_MAP
):
    """
    Normalize the feature labels to canonical names
    >>> normalize_feature_labels("surface protein")
    'S'
    """
    n = n.lower()
    for suffix in suffixes:
        n = n.replace(suffix, "").strip()
    if n in name_map:
        n = name_map[n]
    return n


def get_country(rec: SeqRecord.SeqRecord) -> str:
    """Attempt to pull out country information"""
    ft_dict = genbank_to_feature_dict(rec)
    source_data = ft_dict["source"][0]
    if "country" not in source_data.qualifiers:
        return ""
    country_string = source_data.qualifiers["country"][0].split(":")[0]
    country_string = country_string.replace(" ", "_")
    return country_string


def get_species(rec: SeqRecord.SeqRecord) -> str:
    """Attempt to pull out species information"""
    ft_dict = genbank_to_feature_dict(rec)
    source_data = ft_dict["source"][0]
    if "organism" not in source_data.qualifiers:
        return ""
    organism = source_data.qualifiers["organism"]
    if not isinstance(organism, str):  # Is a list
        assert len(organism) == 1
        organism = organism[0]
    return organism


def featurize(
    transcript_parts: List[str], ablation_parts: List[bool] = ABLATION
) -> np.ndarray:
    """Generate feature vector for the given transcript parts"""
    assert len(transcript_parts) == len(ablation_parts) == 3
    transcript_parts = [
        "" if ablate else part.replace("X", "N", len(part))
        for part, ablate in zip(transcript_parts, ablation_parts)
    ]
    encodings = [
        kmer.sequence_to_kmer_freqs(part, kmer_size=s)
        for s in [3, 4, 5]
        for part in transcript_parts
    ]
    seq_encoded = np.hstack(encodings)
    return seq_encoded.reshape(1, -1)  # Single example


def featurize_gru(
    transcript_parts: List[str], ablation_parts: List[bool]
) -> np.ndarray:
    """Featurize for GRU - one hot encoding"""
    transcript_parts = [
        "" if ablate else part.replace("X", "N", len(part))
        for part, ablate in zip(transcript_parts, ablation_parts)
    ]
    combined_seq = "".join(transcript_parts)
    featurization = np.array([data_loader.BASE_TO_INT[b] + 1 for b in combined_seq])
    featurization = featurization[:, np.newaxis, np.newaxis]
    return featurization


@functools.lru_cache(maxsize=32)
def query_genbank(
    query: str = COVID19_QUERY_SEQUENCE,
    retmax: int = 9999,
    use_file_cache: bool = True,
    file_cache: str = GENBANK_QUERY_CACHE_FILE,
) -> List[str]:
    """
    Query genbank using the given query
    Returns a list of identifiers
    txid2697049 is the coronavirus for covid19
    https://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi?id=2697049

    Cache stores string keys to list of outputs
    """
    # query based on https://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi?id=2697049
    # http://biopython.org/DIST/docs/tutorial/Tutorial.html#chapter:entrez
    query = query.strip()

    cached_results = {}
    if use_file_cache and os.path.isfile(file_cache):
        with open(file_cache, "r") as source:
            cached_results = json.load(source)
            logging.info(f"Loaded genk query disk cache: {file_cache}")

    if query not in cached_results:
        logging.info("Query not in disk cache, fetching from GenBank")
        handle = Entrez.esearch(
            db="nucleotide", term=query, idtype="acc", retmax=retmax
        )
        record = Entrez.read(handle)
        retval = record["IdList"]
        assert retval
        logging.info(f"Number of returned query matches: {len(retval)}")

        if use_file_cache:
            cached_results[query] = retval
            with open(file_cache, "w") as sink:
                logging.info(f"Updated genbank query disk cache: {file_cache}")
                json.dump(cached_results, sink, indent=4)
    else:
        logging.info("Query found in disk cache, returning cached results")
        retval = cached_results[query]

    assert retval
    return retval


def fetch_genbank(
    identifier: str, file_cache: str = GENBANK_FETCH_CACHE_FILE
) -> SeqRecord.SeqRecord:
    """
    Fetch from genbank (identifier should be similar to MN908947.3)
    Setting file_cache to a blank string disables cache functionality
    """
    # http://biopython.org/DIST/docs/tutorial/Tutorial.html#htoc56
    if file_cache:
        with shelve.open(
            file_cache, writeback=False
        ) as cache:  # Writeback need not be true to store new entries
            if identifier in cache:
                seq_record = cache[identifier]
            else:
                with Entrez.efetch(
                    db="nucleotide", rettype="gb", retmode="text", id=identifier
                ) as handle:
                    seq_record = SeqIO.read(handle, format="gb")
                cache[identifier] = seq_record  # Write into cache
    else:
        with Entrez.efetch(
            db="nucleotide", rettype="gb", retmode="text", id=identifier
        ) as handle:
            seq_record = SeqIO.read(handle, format="gb")

    assert seq_record, f"Got empty result for {identifier}"
    return seq_record


def genbank_to_feature_dict(
    gb_record: SeqRecord.SeqRecord, infer_utr: bool = True
) -> dict:
    """Extract the features from genbank record, inferring 5' and 3' UTR if not present"""
    retval = collections.defaultdict(list)
    for ft in gb_record.features:
        retval[ft.type].append(ft)
    assert "CDS" in retval
    # Infer 5/3' UTR if not present
    five_utr = retval["5'UTR"]
    three_utr = retval["3'UTR"]
    if not five_utr and infer_utr:
        # Empirically we see that that 5' UTR is just the bases going up to the first CDS
        # so we assume that's what it is if we aren't given an explicit 5' UTR
        five_utr_end = retval["CDS"][0].location.start
        five_utr = SeqFeature(FeatureLocation(start=0, end=five_utr_end, strand=+1))
        retval["5'UTR"].append(five_utr)
    if not three_utr and infer_utr:
        # Empirically we see that the 3' UTR is just the remaining bases after the last CDS
        three_utr = SeqFeature(
            FeatureLocation(
                start=retval["CDS"][-1].location.end, end=len(gb_record.seq), strand=+1
            )
        )
        retval["3'UTR"].append(three_utr)
    return retval


def seq_feature_to_sequence(
    seq_feature: SeqFeature,
    genome: str,
    three_stop: SeqFeature = None,
    ablation: list = [],
) -> str:
    """
    Given a seq_feature which could have Feature Location or compound location, return sequence
    Since the coronavirus shares 3' UTR (meaning that it runs through) other ORFs until 3' UTR
    via subgenomic RNA (but only in some cases), we have special logic to handle this when
    three_stop is provided
    In addition, this supports ablations, but ablations are done using "X" in order to differentiate
    from bases that are naturally "N"
    """
    # TODO maybe look into randomly shuffling these to serve as a "baseline"
    def handle_feature_location(x: FeatureLocation, three_loc: FeatureLocation = None):
        # x.start and x.end are "ExactPosition" types but behave like ints
        end_pos = x.end if three_loc is None else three_loc.start
        s = genome[x.start : end_pos]
        if x.strand == 1:  # Positive strand
            pass
        else:
            raise ValueError(f"Unrecognized strand: {x.strand}")
        # Do ablations if they are given
        for motif in ablation:
            if isinstance(motif, np.ndarray):
                num_hits = find_ppm_hits(s, motif, prop=0.9)
                if num_hits:
                    global RBP_ABLATION_COUNTER
                    RBP_ABLATION_COUNTER += 1
                    s = ablate_ppm(s, motif, method="X", prop=0.9)
            elif isinstance(motif, tuple) and tuple(map(type, motif)) == (
                SeqFeature,
                str,
            ):
                s = s.replace(
                    motif[1], "X" * len(motif[1]), len(motif[1])
                )  # Simple version
            else:
                raise TypeError(f"Unrecognized type for ablation {type(motif)}")

        return s

    def handle_compound_location(
        x: CompoundLocation, three_loc: FeatureLocation = None
    ):
        # For first few parts, we do not read through to end
        s = "".join([handle_feature_location(part, None) for part in x.parts[:-1]])
        s += handle_feature_location(x.parts[-1], three_loc)
        return s

    loc = seq_feature.location
    three_loc = three_stop.location if three_stop is not None else None
    if isinstance(loc, FeatureLocation):
        retval = handle_feature_location(loc, three_loc)
    elif isinstance(loc, CompoundLocation):
        retval = handle_compound_location(loc, three_loc)
    else:
        raise TypeError(f"Unrecognized type: {type(seq_feature)}")
    assert isinstance(retval, str), f"Unrecognized return type: {type(retval)}"
    return retval


def get_feature_labels(
    record: SeqRecord.SeqRecord,
    suffixes: List[str] = PROTEIN_SUFFIXES,
    name_map: dict = PROTEIN_NAME_MAP,
) -> List[str]:
    """Given a record, try to extract cds gene names"""
    # TODO add functionality to handle gene name in other records
    ft_dict = genbank_to_feature_dict(record)
    retval = [g.qualifiers["gene"][0] for g in ft_dict["gene"]]
    if not retval:
        try:
            retval = [c.qualifiers["product"][0] for c in ft_dict["CDS"]]
        except KeyError or IndexError:
            pass
    retval = [
        normalize_feature_labels(l, suffixes=suffixes, name_map=name_map)
        for l in retval
    ]
    # We should not have duplicated feature labels
    assert len(set(retval)) == len(retval)
    return retval


def gru_predict(featurization: np.ndarray) -> np.ndarray:
    """Generate GRU predictions"""
    assert featurization.size > 0
    featurization = torch.from_numpy(featurization).type(torch.LongTensor).to(DEVICE)
    preds = (
        torch.sigmoid(RNAGRU(featurization)).detach().cpu().numpy()
    )  # model outputs logits!
    return preds


def pred_feature_dict(
    record: SeqRecord.SeqRecord,
    fa_file: str = "",
    write_trunc_seq: bool = False,
    ablation_motifs: Dict[str, list] = collections.defaultdict(list),
    use_gru: bool = False,
    model=RNAGPS,
    name_map: dict = PROTEIN_NAME_MAP,
    rc: bool = False,
    shuffle: bool = False,
    ablation_pos: Dict[str, List[bool]] = {},
) -> pd.DataFrame:
    """
    Given a feature dict, predict on its parts and return a dataframe of resultant floats
    If use_gru then we specifically use the hardcoded GRU, otherwise we use the given model
    which is assumed to take in a feature space of 4032
    If given a fasta file, then we append the sequences we predict on to that file
    Format of fasta sequence name is >recordname|gene|seqpart

    If ablation pos is given, it is modified (not returned) to include a record of whether
    the i-th base in the 5 UTR, CDS, and 3 UTR are mutated
    Structure is a dict of [str, List[bool]]
    """

    def shuffle_str(x: str) -> str:
        """Shuffles string"""
        l = list(x)
        random.shuffle(l)
        return "".join(l)

    ft_dict = genbank_to_feature_dict(record)
    gene_names = get_feature_labels(record, name_map=name_map)
    assert len(gene_names) == len(
        ft_dict["CDS"]
    ), f"Got differing lengths of genes and CDS {len(gene_names)} {len(ft_dict['CDS'])}\n{record}"
    assert ft_dict["CDS"]
    five_utr = ft_dict["5'UTR"]
    three_utr = ft_dict["3'UTR"]
    genome = str(record.seq)
    if "X" in genome:
        raise RuntimeError(f"X base found in {record.name}")
    # assert len(genome) > 20000, f"Got anomalous length for coronavirus genome: {len(genome)}"

    assert len(five_utr) == 1
    assert len(three_utr) == 1
    five_utr_seq = seq_feature_to_sequence(
        five_utr[0],
        genome,
        three_stop=None,
        ablation=ablation_motifs["5'UTR"] + ablation_motifs["all"],
    )
    three_utr_seq = seq_feature_to_sequence(
        three_utr[0],
        genome,
        three_stop=None,
        ablation=ablation_motifs["3'UTR"] + ablation_motifs["all"],
    )
    ablation_pos["5'UTR"] = [base == "X" for base in five_utr_seq]
    ablation_pos["3'UTR"] = [base == "X" for base in three_utr_seq]

    if shuffle:
        five_utr_seq = shuffle_str(five_utr_seq)
        three_utr_seq = shuffle_str(three_utr_seq)

    pred_localization = {}
    for gene, cds in zip(gene_names, ft_dict["CDS"]):
        assert gene not in pred_localization
        assert gene.lower() != "orf1b", f"ORF1B does not exist in isolation"

        cds_seq = seq_feature_to_sequence(
            cds,
            genome,
            three_stop=three_utr[0],
            ablation=ablation_motifs["CDS"] + ablation_motifs["all"],
        )
        cds_seq_trunc = seq_feature_to_sequence(
            cds,
            genome,
            three_stop=None,
            ablation=ablation_motifs["CDS"] + ablation_motifs["all"],
        )  # version that doesn't run through
        ablation_pos[gene.lower()] = [
            base == "X" for base in cds_seq_trunc
        ]  # We do this on truncated version
        assert len(cds_seq_trunc) <= len(
            cds_seq
        ), f"{gene} truncated sequence longer than seq: {len(cds_seq_trunc)} {len(cds_seq)}"

        if shuffle:
            cds_seq = shuffle_str(cds_seq)
            cds_seq_trunc = shuffle_str(cds_seq_trunc)

        # under the featurize calls, we convert the X ablations to N
        if use_gru:
            if rc:
                featurization = featurize_gru(
                    [
                        rev_comp(three_utr_seq),
                        rev_comp(cds_seq),
                        rev_comp(five_utr_seq),
                    ],
                    ablation_parts=ABLATION,
                )
            else:
                featurization = featurize_gru(
                    [five_utr_seq, cds_seq, three_utr_seq], ablation_parts=ABLATION
                )
            preds = gru_predict(featurization)
        else:
            if rc:
                featurization = featurize(
                    (
                        rev_comp(three_utr_seq),
                        rev_comp(cds_seq),
                        rev_comp(five_utr_seq),
                    ),
                    ablation_parts=ABLATION,
                )
            else:
                featurization = featurize(
                    (five_utr_seq, cds_seq, three_utr_seq), ablation_parts=ABLATION
                )
            preds = list_preds_to_array_preds(
                model.predict_proba(featurization)
            ).squeeze()
        assert np.all(preds >= 0.0) and np.all(preds <= 1.0)
        pred_localization[gene] = preds

        if fa_file:
            with open(fa_file, "a") as sink:
                seqs = [five_utr_seq, cds_seq, three_utr_seq]
                if write_trunc_seq:
                    seqs[1] = cds_seq_trunc
                for seq_part_name, seq_part in zip(["5", "CDS", "3"], seqs):
                    sink.write(f">{record.name}|{gene}|{seq_part_name}\n")
                    sink.write(seq_part + "\n")

    pred_localization = pd.DataFrame(
        data=pred_localization,
        index=[
            data_loader.LOCALIZATION_FULL_NAME_DICT[l]
            for l in data_loader.LOCALIZATIONS
        ],
    ).T
    return pred_localization


def pred_full_genome(record: SeqRecord.SeqRecord, use_gru: bool = False) -> pd.Series:
    """
    Given a record, predict localization assuming full uninterrupted genome
    """
    ft_dict = genbank_to_feature_dict(record)
    assert len(ft_dict["gene"]) == len(
        ft_dict["CDS"]
    ), f"Got differing lengths of genes and CDS in {record}"
    assert ft_dict["CDS"]
    five_utr = ft_dict["5'UTR"]
    three_utr = ft_dict["3'UTR"]
    genome = str(record.seq)
    assert (
        len(genome) > 20000
    ), f"Got anomalous length for coronavirus genome: {len(genome)}"
    assert len(five_utr) == 1
    assert len(three_utr) == 1

    five_utr_seq = seq_feature_to_sequence(five_utr[0], genome)
    three_utr_seq = seq_feature_to_sequence(three_utr[0], genome)
    full_cds = genome[five_utr[0].location.end : three_utr[0].location.start]

    if use_gru:
        featurization = featurize_gru(
            [five_utr_seq, full_cds, three_utr_seq], ablation_parts=ABLATION
        )
        preds = gru_predict(featurization)
    else:
        featuriation = featurize(
            (five_utr_seq, full_cds, three_utr_seq), ablation_parts=ABLATION
        )
        preds = list_preds_to_array_preds(RNAGPS.predict_proba(featuriation)).squeeze()

    pred_localization = pd.Series(
        data=preds,
        index=[
            data_loader.LOCALIZATION_FULL_NAME_DICT[l]
            for l in data_loader.LOCALIZATIONS
        ],
    )
    return pred_localization


def mean_sd_missing_vals(
    preds: list, min_count: int = 5, verbose: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate mean and SD given preds with mismatched/unmatched indices
    """
    assert preds
    accum = collections.defaultdict(list)
    for p in preds:
        for idx in p.index:
            accum[idx.lower()].append(p.loc[idx].values)
    if verbose:
        for k, vals in accum.items():
            print(f"{k}\t{len(vals)}")

    if min_count > 0:
        accum = {k: v for k, v in accum.items() if len(v) >= min_count}
    accum = {k: np.vstack(v) for k, v in accum.items()}
    means = {k: np.mean(v, axis=0) for k, v in accum.items()}
    stds = {k: np.std(v, axis=0) for k, v in accum.items()}

    colnames = next(iter(preds)).columns
    means_df = pd.DataFrame(means, index=colnames).T  # All cols are same
    stds_df = pd.DataFrame(stds, index=colnames).T
    return means_df, stds_df


def make_aggregate_predictions(
    genbank_query: str = COVID19_QUERY_SEQUENCE,
    ablation: List[bool] = [False, False, False],
    ablation_motifs: Dict[str, list] = collections.defaultdict(list),
    use_gru: bool = False,
    mito_drop: bool = False,
    rc: bool = False,
    shuffle: bool = False,
    prefix: str = "covid19_localization_full",
):
    """
    Run the given ablation and save to the files with the given prefix (which denotes subfolder name)
    """
    # Skip if results already exist
    mean_file = os.path.join(prefix, "mean.csv")
    sd_file = os.path.join(prefix, "sd.csv")
    gmean_file = os.path.join(prefix, "genome_mean.csv")
    gsd_file = os.path.join(prefix, "genome_sd.csv")
    if (
        os.path.isfile(mean_file)
        and os.path.isfile(sd_file)
        and os.path.isfile(gmean_file)
        and os.path.isfile(gsd_file)
    ):
        logging.warning(
            f"Results appear to be complete at {prefix} - skipping (delete folder to rerun)"
        )
        return

    global ABLATION
    ABLATION = ablation  # Modify this for ablation study
    global RBP_ABLATION_COUNTER
    RBP_ABLATION_COUNTER = 0

    for k in ablation_motifs:
        assert k in ["5'UTR", "CDS", "3'UTR", "all"]

    logging.info(f"Prefix {prefix}\tAblating {ABLATION}")
    if rc:
        logging.info(f"Predicting on reverse complements")
    if not os.path.isdir(prefix):
        os.mkdir(prefix)
    preds_subdir = os.path.join(prefix, "single_preds")
    if not os.path.isdir(preds_subdir):
        os.mkdir(preds_subdir)

    # identifiers = ["MT192765.1", "MN908947.3"]  # For quick debugging runs
    identifiers = query_genbank(genbank_query)
    genbank_records = (fetch_genbank(i) for i in identifiers)
    genbank_preds = {}
    genbank_full_genome_preds = []  # List of Series

    fa_file = f"{prefix}/sequences.fa"
    with open(fa_file, "w") as sink:  # Clear the file
        pass

    ablation_record = {}
    for record in tqdm.tqdm(genbank_records, total=len(identifiers)):
        try:
            this_ablation_record = {}
            preds = pred_feature_dict(
                record,
                ablation_motifs=ablation_motifs,
                fa_file=fa_file,
                write_trunc_seq=True,
                use_gru=use_gru,
                rc=rc,
                model=RNAGPS if not mito_drop else RNAGPS_MITO_DROP,
                shuffle=shuffle,
                ablation_pos=this_ablation_record,
            )
            preds.to_csv(os.path.join(preds_subdir, f"{record.name}.csv"))
            # print(get_country(record))
            genbank_preds[record.name] = preds
            genbank_full_genome_preds.append(pred_full_genome(record, use_gru=use_gru))
            ablation_record[record.name] = this_ablation_record
        except (AssertionError, ValueError, KeyError) as e:
            continue
    assert genbank_preds

    logging.info(f"Number of records with predictions: {len(genbank_preds)}")
    logging.info(f"Ablated number of PWM occurrences:  {RBP_ABLATION_COUNTER}")

    # print(len(genbank_preds))
    preds_mean, preds_sd = mean_sd_missing_vals(genbank_preds.values())
    # print(preds_mean)
    # print(preds_sd)
    preds_mean.to_csv(f"{prefix}/mean.csv")
    preds_sd.to_csv(f"{prefix}/sd.csv")

    genome_preds_stacked = np.vstack([p.values for p in genbank_full_genome_preds])
    genome_preds_mean = pd.Series(
        np.mean(genome_preds_stacked, axis=0),
        index=[
            data_loader.LOCALIZATION_FULL_NAME_DICT[l]
            for l in data_loader.LOCALIZATIONS
        ],
    )
    genome_preds_mean.to_csv(f"{prefix}/genome_mean.csv", header=False)

    genome_preds_sd = pd.Series(
        np.std(genome_preds_stacked, axis=0),
        index=[
            data_loader.LOCALIZATION_FULL_NAME_DICT[l]
            for l in data_loader.LOCALIZATIONS
        ],
    )
    genome_preds_sd.to_csv(f"{prefix}/genome_sd.csv", header=False)

    with open(os.path.join(prefix, "ablations.json"), "w") as sink:
        logging.info(f"Writing json of ablations to {sink.name}")
        json.dump(ablation_record, sink, indent=4)


def make_regional_predictions(prefix: str = "covid19_localization_by_country"):
    """
    Produce predictions per country
    """
    if not os.path.isdir(prefix):
        os.mkdir(prefix)
    identifiers = query_genbank()
    genbank_records = (fetch_genbank(i) for i in identifiers)
    genbank_preds_by_country = collections.defaultdict(dict)
    genbank_full_genome_preds_by_country = collections.defaultdict(
        list
    )  # List of Series
    for record in tqdm.tqdm(genbank_records, total=len(identifiers)):
        try:
            preds = pred_feature_dict(record)
            country = get_country(record)
            genbank_preds_by_country[country][record.name] = preds
            genbank_full_genome_preds_by_country[country].append(
                pred_full_genome(record)
            )
        except (AssertionError, ValueError, KeyError) as e:
            continue
    for country in genbank_full_genome_preds_by_country.keys():
        if (
            not country or len(genbank_full_genome_preds_by_country[country]) < 5
        ):  # Skip unknown countries or those with too few data
            continue
        preds_mean, preds_sd = mean_sd_missing_vals(
            genbank_preds_by_country[country].values()
        )
        preds_mean.to_csv(f"{prefix}/{country}_mean.csv")
        preds_sd.to_csv(f"{prefix}/{country}_sd.csv")

        genome_preds_stacked = np.vstack(
            [p.values for p in genbank_full_genome_preds_by_country[country]]
        )
        genome_preds_mean = pd.Series(
            np.mean(genome_preds_stacked, axis=0),
            index=[
                data_loader.LOCALIZATION_FULL_NAME_DICT[l]
                for l in data_loader.LOCALIZATIONS
            ],
        )
        genome_preds_mean.to_csv(f"{prefix}/{country}_genome_mean.csv", header=False)

        genome_preds_sd = pd.Series(
            np.std(genome_preds_stacked, axis=0),
            index=[
                data_loader.LOCALIZATION_FULL_NAME_DICT[l]
                for l in data_loader.LOCALIZATIONS
            ],
        )
        genome_preds_sd.to_csv(f"{prefix}/{country}_genome_sd.csv", header=False)


def main():
    for prefix, abl_tuple in PREFIXES_TO_ABLATIONS.items():
        logging.info(f"Processing: {prefix}")
        is_gru = prefix.startswith("gru_")
        is_mito_drop = prefix.startswith("mito_drop_")
        genbank_query_string = COVID19_QUERY_SEQUENCE
        make_aggregate_predictions(
            genbank_query=genbank_query_string,
            ablation=abl_tuple,
            prefix=prefix,
            use_gru=is_gru,
            mito_drop=is_mito_drop,
            rc=prefix.endswith("_rc"),
            shuffle=prefix.endswith("_shuffle"),
        )


if __name__ == "__main__":
    import doctest

    doctest.testmod()
    main()
