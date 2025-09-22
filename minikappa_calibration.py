#!/usr/bin/env python

import random
from goniometer import goniometer
import os


def get_random_selection(min_value, max_value, n):
    random_selection = [
        round(random.random() * (max_value - min_value) + min_value, 2)
        for k in range(n)
    ]
    return random_selection


def main():
    import optparse

    parser = optparse.OptionParser()

    parser.add_option(
        "-d",
        "--directory",
        type=str,
        default="/nfs/ruche/proxima2a-spool/Martin/Research/minikappa_callibration/2018-12-16_20um_random_pairs",
    )
    parser.add_option("-n", "--name_pattern", type=str, default="oa")
    parser.add_option("-P", "--n_phi", type=int, default=10)
    parser.add_option("-K", "--n_kappa", type=int, default=10)
    parser.add_option("-N", "--n_points", type=int, default=33)
    parser.add_option("-g", "--n_angles", type=int, default=75)
    options, args = parser.parse_args()

    kappa_min = 0
    kappa_max = 249
    phi_min = 0
    phi_max = 360

    g = goniometer()

    for k in range(options.n_points):
        # if k == 0:
        # kappa = 0.
        # phi = 0.
        # else:
        kappa = round(random.random() * (kappa_max - kappa_min) + kappa_min, 2)
        phi = round(random.random() * (phi_max - phi_min) + phi_min, 2)
        for zoom in [1, 10]:
            align_line = "optical_alignment.py -d {directory} -n kappa_{kappa:.2f}_phi_{phi:.2f}_zoom_{zoom:d} -K {kappa:.2f} -P {phi:.2f} -z {zoom:d} -A -C -R --rightmost -g {n_angles:d}".format(
                **{
                    "kappa": kappa,
                    "phi": phi,
                    "zoom": zoom,
                    "directory": options.directory,
                    "n_angles": options.n_angles,
                }
            )
            os.system(align_line)


if __name__ == "__main__":
    main()
