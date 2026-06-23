#!/usr/bin/env python
# coding: utf-8

import os
from useful_routines import get_pickled_file
from diffraction_experiment import diffraction_experiment

def run_dozor(directory, name_pattern, binning=2, blocking=True, force=False):
    de = diffraction_experiment(
        directory=directory,
        name_pattern=name_pattern
    )
    de.run_dozor(blocking=True, binning=binning, force=force)
    
def main():
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-e",
        "--experiment",
        type=str,
        default="/nfs/data4/2026_Run3/20260017/2026-06-11/RAW_DATA/IRF5/IRF5-MT260973_H11-1_BX028A-04/IRF5-MT260973_H11-1_BX028A-04_1_master.h5",
        help="sufficient experiment description",
    )

    parser.add_argument(
        "-b",
        "--binning",
        default=2,
        type=int,
        help="binning",
    )
    
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="force execution",
    )
    
    parser.add_argument(
        "-u",
        "--unblock",
        action="store_true",
        help="do not block execution",
    )
    
    args = parser.parse_args()
    print(f"args {args}")
    
    if args.experiment.endswith("_master.h5") and os.path.isfile(args.experiment):
        directory = os.path.dirname(args.experiment)
        name_pattern = os.path.basename(args.experiment).replace("_master.h5", "")
    elif args.experiment.endswith(".pickle"):
        pars = get_pickled_file(args.experiment)
        directory = pars["directory"]
        name_pattern = pars["name_pattern"]
    
    run_dozor(directory, name_pattern, binning=args.binning, blocking=not args.unblock, force=args.force)

if __name__ == "__main__":
    main()
