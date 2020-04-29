"""
Run meme on the given inputs
"""

# Reference command:
# meme aggregated_ns3.fa -dna -oc . -nostatus -time 18000 -mod zoops -nmotifs 5 -minw 6 -maxw 15 -objfun classic -markov_order 1

# Example usage:
# python run_meme.py aggregated_e.fa aggregated_m.fa aggregated_n.fa aggregated_s.fa aggregated_ns3.fa aggregated_orf1ab.fa

import os
import argparse
import shutil
import multiprocessing
import shlex
import subprocess

def run_meme(msa_file:str, outdir:str, dry:bool=False) -> str:
    """
    Run meme
    """
    assert os.path.isfile(msa_file), f"Cannot find {msa_file}"
    # Meme auto-clears out the output dir when using -oc
    command = f"meme {msa_file} -dna -oc {outdir} -nostatus -mod zoops -nmotifs 10 -minw 6 -maxw 15 -objfun classic -markov_order 1 -seed 1234"
    print(command)
    if dry:
        return None

    _p = subprocess.run(shlex.split(command), check=True)
    retval = os.path.join(outdir, "meme.html")
    assert os.path.isfile(retval)
    return retval

def build_parser() -> argparse.ArgumentParser:
    """Build commandline argument parser"""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("msa_file", type=str, nargs="+", help="MSA files to run on")
    parser.add_argument("-t", "--threads", type=int, default=multiprocessing.cpu_count(), help="Number of threads to run")
    parser.add_argument("--dry", action="store_true", help="Print command without running")
    return parser

def main():
    """
    Run script
    """
    assert shutil.which("meme"), f"Script requires meme to present in the path"
    parser = build_parser()
    args = parser.parse_args()

    arg_tuples = []
    for fname in args.msa_file:
        fname = os.path.abspath(fname)
        assert os.path.isfile(fname)
        # Remove extension and add folder suffix
        outdir = ".".join(fname.split(".")[:-1]) + "_meme_out"
        arg_tuples.append((fname, outdir, args.dry))

    pool = multiprocessing.Pool(min(args.threads, len(arg_tuples)))
    meme_output_htmls = pool.starmap(run_meme, arg_tuples)
    pool.close()
    pool.join()

    for fname in meme_output_htmls:
        print(fname)

if __name__ == "__main__":
    main()

