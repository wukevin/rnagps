"""
Various miscellaneous utility functions
"""
import os
import gzip
import collections
import pickle

import torch

import sklearn

LOCAL_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data"))

def isnotebook():
    """
    Returns True if the current execution environment is a jupyter notebook
    https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

def get_device(i=0):
    """
    Returns the i-th GPU if GPU is available, else CPU
    Specifying None also returns CPU
    """
    if torch.cuda.is_available() and i is not None:
        devices = list(range(torch.cuda.device_count()))
        device_idx = devices[i]
        torch.cuda.set_device(device_idx)
        d = torch.device(f"cuda:{device_idx}")
    else:
        d = torch.device('cpu')
    return d

def gcoord_str_merger(x, y):
    """
    Takes two intervals of frmat chrom:start-end:strand and merge
    """
    chrom1, int1, strand1 = x.split(":")
    chrom2, int2, strand2 = y.split(":")
    assert strand1 == strand2, f"Mismatched strands: {strand1} {strand2}"
    assert chrom1 == chrom2, f"Mismatched chrom: {chrom1} {chrom2}"

    start1, end1 = map(int, int1.split("-"))
    assert start1 < end1
    start2, end2 = map(int, int2.split("-"))
    assert start2 < end2

    start = min(start1, start2)
    end = max(end1, end2)
    return f"{chrom1}:{start}-{end}:{strand1}"

def read_gtf_trans_to_exons(fname:str=os.path.join(LOCAL_DATA_DIR, "Homo_sapiens.GRCh38.90.gtf.gz")):
    """
    Read gtf file into a dictionary mapping transcripts to exons
    Return value: dict<trans_id, <[(exon_num, chrom, start, stop, gene), ...]>
    """
    retval = collections.defaultdict(list)
    opener = gzip.open if fname.endswith(".gz") else open
    with opener(fname) as source:
        for line in source:
            line = line.decode('utf8')
            if line.startswith("#"):
                continue
            tokens = line.strip().split("\t")
            chrom, source, feature, start, end, score, strand, frame, attr = tokens
            if feature != "exon":
                continue
            attr_split = [chunk.strip().split(" ", 1) for chunk in attr.split(";") if chunk]
            attr_dict = {k: v.strip('"') for k, v in attr_split}

            if "transcript_id" not in attr_dict or "exon_number" not in attr_dict:
                continue
            trans_id = attr_dict['transcript_id']
            exon_num = int(attr_dict['exon_number'])
            # Use exon num first so we can easily sort
            gcoord = (exon_num, chrom, int(start), int(end), strand, attr_dict['gene_name'])
            retval[trans_id].append(gcoord)
    # Sanity check output
    for trans_id, coords in retval.items():
        assert len(set([c[1] for c in coords])) == 1, f"Got multiple chroms for {trans_id}"
        assert len(set([c[4] for c in coords])) == 1, f"Got multiple strands for {trans_id}"
        assert len(set([c[-1] for c in coords])) == 1, f"Got multiple chroms for {trans_id}"
        assert max([c[0] for c in coords]) == len(coords)  # No extra exons
        retval[trans_id] = sorted(coords)  # Ensure ordering of exons
    return retval

def save_sklearn_model(model, file_prefix):
    """
    Save the given model to the given file prefix, appending the sklearn version number and .skmodel extension
    """
    assert '.' not in os.path.basename(file_prefix)
    full_fname = ".".join([file_prefix, sklearn.__version__, 'skmodel'])
    with open(full_fname, 'wb') as sink:
        pickle.dump(model, sink)
    return full_fname

def load_sklearn_model(file_name, strict=True):
    """
    Load the sklearn model from the given filename
    """
    if strict:
        bname = os.path.basename(file_name)
        tokens = bname.split(".")
        version = '.'.join(tokens[1:-1])
        assert sklearn.__version__ == version, f"Got mismatched sklearn versions: {sklearn.__version__} {version}"
    with open(file_name, 'rb') as source:
        retval = pickle.load(source)
    return retval

if __name__ == "__main__":
    g = read_gtf_trans_to_exons()
    print(g['ENST00000614171'])

