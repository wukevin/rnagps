"""
Script for running tomtom from a fasta file
"""

import os, sys
import logging
import argparse
import subprocess

TOMTOM_DB = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data/meme/Ray2013_rbp_Homo_sapiens.dna_encoded.meme"
)
assert os.path.isfile(TOMTOM_DB)

def run_tomtom(meme_input, cutoff, use_e_value=False, db=TOMTOM_DB):
    """Generate the tomtom command"""
    assert os.path.isfile(meme_input)
    assert os.path.isfile(db)
    outdir = meme_input.split('.')[0]
    cmd = f"tomtom -no-ssc -oc {outdir} -verbosity 1 -norc -min-overlap 5 -dist pearson {'-evalue' if use_e_value else ''} -thresh {cutoff} {meme_input} {db}"
    subprocess.call(cmd, shell=True)

def convert_fa_to_meme(fa_input, outdir):
    """Convert the fasta file to meme format, returning path to meme file"""
    assert os.path.isfile(fa_input)
    meme_file_destination = os.path.join(
        outdir,
        os.path.basename(fa_input).split(".")[0] + ".meme"
    )
    # assert not os.path.isfile(meme_file_destination)
    cmd = f"rna2meme -dna {fa_input} > {meme_file_destination}"
    subprocess.call(cmd, shell=True)
    return meme_file_destination

def build_parser():
    """Build argument parser"""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-o", "--outdir", type=str, default=os.getcwd())
    parser.add_argument("-k", "--keepintermediate", action="store_true", help="Do not delete intermediate files")
    parser.add_argument("-c", "--cutoff", type=float, default=0.1, help="Cutoff for significance, using q value unless otherwise specified")
    parser.add_argument("-e", "--evalue", action="store_true", help="Use E-value as cutoff instead of q value")
    parser.add_argument("-d", "--db", type=str, default=TOMTOM_DB, help="Database to query against")
    parser.add_argument("motif_file", nargs='*')
    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    for motif_file in args.motif_file:
        logging.info(f"Converting {motif_file} to meme format")
        meme_file = convert_fa_to_meme(motif_file, args.outdir)
        logging.info(f"Running tomtom on {meme_file}")
        run_tomtom(meme_file, cutoff=args.cutoff, use_e_value=args.evalue, db=args.db)
        if not args.keepintermediate:
            os.remove(meme_file)

if __name__ == "__main__":
    main()

