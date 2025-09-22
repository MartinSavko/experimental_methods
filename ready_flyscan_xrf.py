#!/usr/bin/env python
# coding: utf-8

import os
import h5py
import pylab
import numpy as np
from energy_scan import energy_scan


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--directory",
        default="/nfs/ruche/proxima2a-soleil/com-proxima2a/flyscan/2023_Run3/plate/2023-06-06",
        type=str,
        help="directory",
    )
    parser.add_argument(
        "-f", "--filename", default="flyscan_3977.nxs", type=str, help="filename"
    )
    parser.add_argument("-e", "--element", default="Zn", type=str, help="element")
    parser.add_argument("-s", "--edge", default="K", type=str, help="edge")

    args = parser.parse_args()
    es = energy_scan(
        name_pattern=args.filename,
        directory=args.directory,
        element=args.element,
        edge=args.edge,
    )

    roi_start, roi_end = map(int, es.get_roi_start_end())
    print("roi_start, roi_end", roi_start, roi_end)

    m = h5py.File(os.path.join(args.directory, args.filename), "r")

    spectra = m["acq/scan_data/channel00"][()]
    energy = m["acq/scan_data/energy"][()]
    dead_time = m["acq/scan_data/deadtime00"][()]

    roi_counts = spectra[:, roi_start:roi_end].sum(axis=1)
    compton_counts = spectra[:, roi_start : roi_end + 250].sum(axis=1)

    normalized_counts = 1000.0 * roi_counts / compton_counts
    normalized_counts *= 1.0 + dead_time / 100

    pylab.plot(energy, normalized_counts)
    # pylab.plot(energy, compton_counts)
    # pylab.plot(energy, roi_counts)
    pylab.show()


if __name__ == "__main__":
    main()
