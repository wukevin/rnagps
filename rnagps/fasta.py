"""
Functions for handling fasta files
"""

import os
import sys
import gzip

def write_sequence_dict_to_file(seq_dict, fname):
    """Writes the given sequences to the filename"""
    with open(fname, 'w') as sink:
        for header, seq in seq_dict.items():
            if not header.startswith(">"):
                header = ">" + header
            sink.write(header.strip() + "\n")
            sink.write(seq.strip() + "\n")

def read_file_as_sequence_dict(fname):
    """Reads the file in as a sequence dict"""
    retval = {}
    curr_key = None
    opener = gzip.open if fname.endswith(".gz") else open
    with opener(fname) as source:
        for line in source:
            line = line.strip()
            if not isinstance(line, str):
                line = line.decode()
            if line.startswith(">"):
                curr_key = line[1:].strip()  # Remove >
                retval[curr_key] = []
            else:
                assert curr_key
                retval[curr_key].append(line)
    return {k: ''.join(v) for k, v in retval.items()}

def main():
    """On the fly testing"""
    write_sequence_dict_to_file({">test": "ACGG"}, "test.fa")
    print(read_file_as_sequence_dict("test.fa"))

if __name__ == "__main__":
    main()

