#!/usr/bin/env python
# -*- coding: utf-8 -*-

from diffraction_experiment import diffraction_experiment


def main(method="dozor", bin=2):
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,)

    parser.add_argument(
        "-d",
        "--directory",
        default="/nfs/data4/2023_Run3/com-proxima2a/2023-06-02/RAW_DATA/Nastya/px2-0007/pos10/dosing_b/point_15_helical_0.152",
        type=str,
        help="directory",
    )
    parser.add_argument(
        "-n", "--name_pattern", default="pass_1", type=str, help="name pattern"
    )
    parser.add_argument(
        "-m", "--method", default="dozor", type=str, help="name_pattern"
    )
    parser.add_argument("-b", "--binning", default=2, type=int, help="dozor binning")

    args = parser.parse_args()

    print(args)

    dt = diffraction_experiment(
        directory=args.directory, name_pattern=args.name_pattern
    )

    print(dt.get_parameters_filename())
    getattr(dt, "run_%s" % args.method)(binning=args.binning, blocking=True)


if __name__ == "__main__":
    main()
