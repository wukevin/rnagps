"""
Script for running MSA on an input fasta

Example wrapped command:
clustalo --infile clustalo-I20190515-231900-0786-39557822-p1m.sequence --threads 8 --MAC-RAM 8000 --verbose --outfmt clustal --resno --outfile clustalo-I20190515-231900-0786-39557822-p1m.clustal_num --output-order tree-order --seqtype rna

Example usage:
python run_msa.py fasta1.fa fasta2.fa

Each fa gets its own msa output
"""

import sys, os
import logging
import argparse
import subprocess

def run_msa(fa_file, outdir=""):
    """Run the MSA, outputting to outdir if specified"""
    out_fname = os.path.join(
        os.path.abspath(outdir),
        os.path.basename(fa_file).split('.')[0] + ".msa",
    )
    logging.info(f"Running clustalo on {fa_file}")
    cmd = f"clustalo --infile {fa_file} --MAC-RAM 8000 --verbose --outfmt clustal --resno --outfile {out_fname} --output-order tree-order --seqtype rna"
    logging.info(cmd)
    subprocess.call(cmd, shell=True)

def build_parser():
    """Build argument parser"""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-o", "--outdir", type=str, default=os.getcwd(), help="Directory to output to")
    parser.add_argument("fasta", nargs="*")
    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()

    for fa_file in args.fasta:
        run_msa(fa_file, args.outdir)

if __name__ == "__main__":
    main()

