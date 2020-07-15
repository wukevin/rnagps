"""
Short wrapper for running predictor as a Gradio app
"""

import os
import sys
import argparse

import numpy as np
import gradio as gr

SRC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rnagps"
)
assert os.path.isdir(SRC_DIR), f"Could not find source dir: {SRC_DIR}"
sys.path.append(SRC_DIR)
from kmer import sequence_to_kmer_freqs
from model_utils import list_preds_to_array_preds
from data_loader import LOCALIZATION_FULL_NAME_DICT
import utils

MODEL = utils.load_sklearn_model(
    os.path.join(os.path.dirname(SRC_DIR), "models", "rf_8way_fold5.0.21.3.skmodel")
)

LOCALIZATION_VALID_CUTOFFS = np.array(
    [
        0.35180464,
        0.21918939,
        0.08393867,
        0.05509824,
        0.34117096,
        0.22306494,
        0.07391534,
        0.24330075,
    ]
)


def pred_localization(five_prime: str, cds: str, three_prime: str) -> str:
    """
    Given three strings represnting 5' UTR, cds and 3' UTR,
    predict localization
    """
    ft = np.hstack(
        [
            sequence_to_kmer_freqs(part, kmer_size=s)
            for s in [3, 4, 5]
            for part in [five_prime, cds, three_prime]
        ]
    ).reshape(1, -1)
    vec = list_preds_to_array_preds(MODEL.predict_proba(ft)).flatten()
    vec_pos = vec > LOCALIZATION_VALID_CUTOFFS

    localizations = list(LOCALIZATION_FULL_NAME_DICT.values())
    retval = "No predicted localization"
    if np.any(vec_pos):
        retval = "\n".join([localizations[i] for i in np.where(vec_pos)[0]])
    return retval


def build_parser():
    """Very basic CLI parser for testing"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test run of pred function, no gradio function",
    )
    return parser


def main():
    """Run script"""
    parser = build_parser()
    args = parser.parse_args()

    if args.test:
        print(pred_localization("ACGT", "ACGT", "ACGT"))
    else:
        gr.Interface(
            pred_localization, inputs=["text", "text", "text"], outputs="text"
        ).launch(share=True)


if __name__ == "__main__":
    main()
