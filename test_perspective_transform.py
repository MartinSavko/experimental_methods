#!/usr/bin/env python

import os
import pylab

import seaborn as sns

sns.set(color_codes=True)
from matplotlib import rc

rc("font", **{"family": "serif", "serif": ["Palatino"]})
rc("text", usetex=True)

from useful_routines import _check_image
from mk3_calibration import get_image_at_datum


def project(i1, i2, okp1, okp2, calibration=0.0019, figsize=(16, 9)):
    i1 = _check_image(i1)
    i2 = _check_image(i2)
    i1_at_okp2 = get_image_at_datum(okp1, okp2, i1, pixel_calibration=calibration)

    fig, axes = pylab.subplots(1, 3, figsize=figsize)
    fig.suptitle("Projective transform vs real sample image after change in datum")
    axes[0].imshow(i1)
    axes[0].set_title(f"sample at $\omega={okp1[0]}, \kappa={okp1[1]}, \phi={okp1[2]}$")
    #axes[1].imshow(i1_at_okp2.mean(axis=2).T[:,::-1], cmap="gray")
    axes[1].imshow(i1_at_okp2)
    axes[2].set_title(
        f"i1 after transform to $\omega={okp2[0]}, \kappa={okp2[1]}, \phi={okp2[2]}$"
    )
    axes[2].imshow(i2)
    axes[2].set_title(f"sample at $\omega={okp2[0]}, \kappa={okp2[1]}, \phi={okp2[2]}$")
    for a in axes:
        a.grid(0)

    pylab.savefig("projective_transform_vs_real_sample_image_after_change_in_datum.png")

    pylab.show()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        default="/nfs/data4/2026_Run2/com-proxima2a/2026-03-18/ARCHIVE/opti",
        help="directory",
    )
    parser.add_argument(
        "-1",
        "--image1",
        type=str,
        default="manu_2_04_20260318_185145_click_1_omega_345.00_ay_-1.1309_az_0.0000_cx_1.0701_cy_0.9904_omega_57.001_kappa_0.000_phi_0.000_zoom_1_y_510_x_608.jpg",
        help="image 1",
    )
    parser.add_argument(
        "-2",
        "--image2",
        type=str,
        default="manu_2_04_20260318_185300_click_1_omega_345.00_ay_-1.1450_az_0.0000_cx_0.8936_cy_1.0546_omega_57.000_kappa_40.000_phi_0.000_zoom_1_y_506_x_608.jpg",
        help="image 2",
    )
    parser.add_argument("-o", "--omega1", type=float, default=345.0, help="omega 1")
    parser.add_argument("-k", "--kappa1", type=float, default=0.0, help="kappa 1")
    parser.add_argument("-p", "--phi1", type=float, default=0.0, help="phi 1")
    parser.add_argument("-O", "--omega2", type=float, default=345.0, help="omega 1")
    parser.add_argument("-K", "--kappa2", type=float, default=40.0, help="kappa 1")
    parser.add_argument("-P", "--phi2", type=float, default=0.0, help="phi 1")
    parser.add_argument(
        "-c",
        "--calibration",
        type=float,
        default=0.019,
        help="pixel calibration [mm/px]",
    )

    args = parser.parse_args()

    if args.directory is not None:
        i1 = os.path.join(args.directory, args.image1)
        i2 = os.path.join(args.directory, args.image2)
    else:
        i1 = args.image1
        i2 = args.image2
    
    okp1 = [args.omega1, args.kappa1, args.phi1]
    okp2 = [args.omega2, args.kappa2, args.phi2]
    project(i1, i2, okp1, okp2)

if __name__ == "__main__":
    main()
