#!/usr/bin/python

import h5py
import numpy as np


def main(master_filename):
    m = h5py.File(master_filename, "r+")
    omega = m["entry/sample/goniometer/omega"][()]
    omega_end = m["entry/sample/goniometer/omega_end"][()]
    delta = omega_end[0] - omega[0]
    if delta < 0:
        print(
            "delta %.3f is negative" % delta,
            m["entry/sample/goniometer/omega_range_average"][()],
        )
        m["entry/sample/goniometer/omega"].write_direct(np.array(list(omega_end[::-1])))
        m["entry/sample/goniometer/omega_end"].write_direct(np.array(list(omega[::-1])))
        m["entry/sample/goniometer/omega_range_average"].write_direct(
            np.array([-delta])
        )
    else:
        print(
            "delta %.3f okay, moving on" % delta,
            m["entry/sample/goniometer/omega_range_average"][()],
        )
    # print('omega', m['entry/sample/goniometer/omega'][()])
    # print('omega_end', m['entry/sample/goniometer/omega_end'][()])
    m.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-m",
        "--master",
        type=str,
        default="./Lille_MannitouFab_01032023__G8-2_rscan_2024-04-10_10-10-30_master.h5",
        help="master file to fix",
    )

    args = parser.parse_args()
    main(args.master)
