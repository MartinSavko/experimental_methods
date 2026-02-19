#!/usr/bin/env python

import re

pdbline = "HETATM 4462  O   HOH S   2     -12.495  72.606  -3.175  1.00 11.42           O"
match = "HETATM ([\d]*) O.  HOH "
def clean_waters(pdb, result_chain):
    ilines = open(pdb).readlines()
    olines = []
    waters_sofar = 0
    for line in ilines:
        if "HOH" in line:
            waters_sofar += 1
            to_replace = re.findall(r"HOH \w[\s]*[\d]*", line)[0]
            replace_with = f"HOH {result_chain}{waters_sofar:4d}"
            line = line.replace(to_replace, replace_with)
        olines.append(line)
    f = open(pdb.replace(".pdb", "_clean.pdb"), "w")
    f.writelines(olines)
    f.close()

def main():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--pdb", default="/Users/academia/Documents/buster_refine_5_manual_prune12_after_refmac.pdb", type=str, help="pdb file to clean up")
    parser.add_argument("-c", "--result_chain", default="W", type=str, help="Name of the result water chain")

    args = parser.parse_args()

    clean_waters(args.pdb, args.result_chain)

if __name__ == "__main__":
    main()