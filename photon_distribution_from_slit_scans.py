#!/usr/bin/env python

import numpy as np
import pylab
import time
import pickle
from scipy.interpolate import interp1d
import scipy.signal
import scipy.ndimage as ndi
import os

import seaborn as sns

sns.set(color_codes=True)

from matplotlib import rc

rc("font", **{"family": "serif", "serif": ["Palatino"], "size": 20})
rc("text", usetex=True)

from mpl_toolkits.mplot3d import Axes3D

import skimage.io
import skimage.transform

from area import shift, scale


def main(results_filename="s3b_results.pickle"):
    s = pickle.load(open(results_filename, "rb"))

    blank_slate = np.ones((4000, 4000))
    distribution = blank_slate[::]

    if "s2" in results_filename or "s1" in results_filename:
        if "s1" in results_filename:
            labels = [
                "i11-ma-c02/ex/fent_h.1-mt_i",
                "i11-ma-c02/ex/fent_h.1-mt_o",
                "i11-ma-c02/ex/fent_v.1-mt_u",
                "i11-ma-c02/ex/fent_v.1-mt_d",
            ]
            tex_labels = {
                "i11-ma-c02/ex/fent_h.1-mt_i": "horizontal in",
                "i11-ma-c02/ex/fent_h.1-mt_o": "horizontal out",
                "i11-ma-c02/ex/fent_v.1-mt_u": "vertical up",
                "i11-ma-c02/ex/fent_v.1-mt_d": "vertical down",
            }

            hi = s["i11-ma-c02/ex/fent_h.1-mt_i"]
            ho = s["i11-ma-c02/ex/fent_h.1-mt_o"]
            vu = s["i11-ma-c02/ex/fent_v.1-mt_u"]
            vd = s["i11-ma-c02/ex/fent_v.1-mt_d"]

        elif "s2" in results_filename:
            labels = [
                "i11-ma-c04/ex/fent_h.2-mt_i",
                "i11-ma-c04/ex/fent_h.2-mt_o",
                "i11-ma-c04/ex/fent_v.2-mt_u",
                "i11-ma-c04/ex/fent_v.2-mt_d",
            ]
            tex_labels = {
                "i11-ma-c04/ex/fent_h.2-mt_i": "horizontal in",
                "i11-ma-c04/ex/fent_h.2-mt_o": "horizontal out",
                "i11-ma-c04/ex/fent_v.2-mt_u": "vertical up",
                "i11-ma-c04/ex/fent_v.2-mt_d": "vertical down",
            }

            hi = s["i11-ma-c04/ex/fent_h.2-mt_i"]
            ho = s["i11-ma-c04/ex/fent_h.2-mt_o"]
            vu = s["i11-ma-c04/ex/fent_v.2-mt_u"]
            vd = s["i11-ma-c04/ex/fent_v.2-mt_d"]

        for s, label in zip([hi, ho, vu, vd], labels):
            position = s["analysis"]["position"]
            transmission = s["analysis"]["transmission"]

            fill_value = np.array((0, 1))
            if "mt_d" in label or "mt_i" in label:
                position = position[::-1]
                fill_value = fill_value[::-1]
            ip = interp1d(
                position,
                transmission,
                fill_value=tuple(fill_value),
                bounds_error=False,
                kind="slinear",
            )

            extended_positions = np.linspace(-2, 2, 4000)
            if "mt_d" in label or "mt_i" in label:
                extended_positions = extended_positions[::-1]

            calculated_transmission = ip(extended_positions)

            pylab.figure(figsize=(16, 9))
            pylab.plot(
                extended_positions,
                calculated_transmission,
                label="%s" % tex_labels[label],
            )
            pylab.title("Slit scan", fontsize=24)
            pylab.ylabel("transmission", fontsize=20)
            pylab.xlabel("position", fontsize=20)
            pylab.legend(fontsize=20)
            pylab.savefig(
                "%s/slit_scan_%s_%s.png"
                % (
                    os.path.dirname(results_filename),
                    os.path.basename(results_filename).replace("_results.pickle", ""),
                    tex_labels[label].replace(" ", "_"),
                )
            )

            contribution = blank_slate * np.abs(
                np.gradient(ndi.gaussian_filter1d(calculated_transmission, 7))
            )

            if "mt_d" in label or "mt_i" in label:
                contribution = np.flip(contribution)

            if "mt_d" in label or "mt_u" in label:
                contribution = contribution.T

            distribution = distribution * contribution
    else:
        if "s3" in results_filename:
            labels = ["i11-ma-c05/ex/fent_v.3-mt_tz", "i11-ma-c05/ex/fent_h.3-mt_tx"]
            tex_labels = {
                "i11-ma-c05/ex/fent_v.3-mt_tz": "vertical",
                "i11-ma-c05/ex/fent_h.3-mt_tx": "horizontal",
            }

            h = s["i11-ma-c05/ex/fent_h.3-mt_tx"]
            v = s["i11-ma-c05/ex/fent_v.3-mt_tz"]
        elif "s5" in results_filename:
            labels = ["i11-ma-c06/ex/fent_v.5-mt_tz", "i11-ma-c06/ex/fent_h.5-mt_tx"]
            tex_labels = {
                "i11-ma-c06/ex/fent_v.5-mt_tz": "vertical",
                "i11-ma-c06/ex/fent_h.5-mt_tx": "horizontal",
            }

            h = s["i11-ma-c06/ex/fent_h.5-mt_tx"]
            v = s["i11-ma-c06/ex/fent_v.5-mt_tz"]
        elif "s6" in results_filename:
            labels = ["i11-ma-c06/ex/fent_v.6-mt_tz", "i11-ma-c06/ex/fent_h.6-mt_tx"]
            tex_labels = {
                "i11-ma-c06/ex/fent_v.6-mt_tz": "vertical",
                "i11-ma-c06/ex/fent_h.6-mt_tx": "horizontal",
            }

            h = s["i11-ma-c06/ex/fent_h.6-mt_tx"]
            v = s["i11-ma-c06/ex/fent_v.6-mt_tz"]
        for s, label in zip([v, h], labels):
            position = s["analysis"]["position"]
            transmission = s["analysis"]["transmission"]

            fill_value = np.array((0, 0))
            ip = interp1d(
                position,
                transmission,
                fill_value=tuple(fill_value),
                bounds_error=False,
                kind="slinear",
            )

            extended_positions = np.linspace(-2, 2, 4000)

            calculated_transmission = ip(extended_positions)

            pylab.figure(figsize=(16, 9))
            pylab.plot(
                extended_positions,
                calculated_transmission,
                label="%s" % tex_labels[label],
            )
            pylab.title("Slit scan", fontsize=24)
            pylab.ylabel("transmission", fontsize=20)
            pylab.xlabel("position", fontsize=20)
            pylab.legend(fontsize=20)
            pylab.savefig(
                "%s/slit_scan_%s_%s.png"
                % (
                    os.path.dirname(results_filename),
                    os.path.basename(results_filename).replace("_results.pickle", ""),
                ),
                tex_labels[label].replace(" ", "_"),
            )

            contribution = blank_slate * calculated_transmission

            if "mt_tz" in label:
                contribution = contribution.T

            distribution = distribution * contribution

    sum_distribution = distribution.sum()
    print("sum distribution %.3f" % sum_distribution)

    distribution /= sum_distribution

    np.save(
        "%s/distribution_%s.npy"
        % (
            os.path.dirname(results_filename),
            os.path.basename(results_filename).replace("_results.pickle", ""),
        ),
        distribution,
    )
    start_ii = time.time()
    ii = skimage.transform.integral.integral_image(distribution)
    end_ii = time.time()
    print("integral image generated in %.3f seconds" % (end_ii - start_ii))

    np.save(
        "%s/ii_%s.npy"
        % (
            os.path.dirname(results_filename),
            os.path.basename(results_filename).replace("_results.pickle", ""),
        ),
        ii,
    )

    def get_indices_for_slit_setting(
        horizontal_gap,
        vertical_gap,
        horizontal_center=0.0,
        vertical_center=0.0,
        npixels=4000,
        extent=(-2, 2),
    ):
        e = extent[1] - extent[0]
        pixels_per_mm = npixels / e
        horizontal_start = (-horizontal_gap / 2.0 - horizontal_center) - extent[0]
        horizontal_end = (horizontal_gap / 2.0 - horizontal_center) - extent[0]
        vertical_start = (-vertical_gap / 2.0 - vertical_center) - extent[0]
        vertical_end = (vertical_gap / 2.0 - vertical_center) - extent[0]

        horizontal_start *= pixels_per_mm
        horizontal_end *= pixels_per_mm
        vertical_start *= pixels_per_mm
        vertical_end *= pixels_per_mm

        return (int(vertical_start), int(horizontal_start)), (
            int(vertical_end),
            int(horizontal_end),
        )

    start, end = get_indices_for_slit_setting(1.0, 1.0)
    print("start, end", start, end)

    _start = time.time()
    print("opening 1., 1.", skimage.transform.integral.integrate(ii, start, end))
    _end = time.time()
    print("single integral took %.3f" % (_end - _start))

    _start = time.time()
    print(
        "opening 1., 1. from sum",
        distribution[start[0] : end[0] + 1, start[1] : end[1] + 1].sum(),
    )
    _end = time.time()
    print("single sum took %.3f" % (_end - _start))

    start, end = get_indices_for_slit_setting(1.0, 1.0, -0.5, -0.5)
    print("start, end", start, end)
    print("opening 1., 1.", skimage.transform.integral.integrate(ii, start, end))

    start, end = get_indices_for_slit_setting(1.0, 1.0, 0.5, 0.5)
    print("start, end", start, end)
    print("opening 1., 1.", skimage.transform.integral.integrate(ii, start, end))

    start, end = get_indices_for_slit_setting(1.0, 1.0, -1, -1)
    print("start, end", start, end)
    print("opening 1., 1.", skimage.transform.integral.integrate(ii, start, end))

    start, end = get_indices_for_slit_setting(1.0, 1.0, 1, 1)
    print("start, end", start, end)
    print("opening 1., 1.", skimage.transform.integral.integrate(ii, start, end))

    start, end = get_indices_for_slit_setting(0.5, 0.5)
    print("start, end", start, end)
    print("opening 0.5, 0.5", skimage.transform.integral.integrate(ii, start, end))

    start, end = get_indices_for_slit_setting(1.5, 0.8)
    print("start, end", start, end)
    print("opening 1.5, 0.8", skimage.transform.integral.integrate(ii, start, end))

    start, end = get_indices_for_slit_setting(0.01, 0.005)
    print("start, end", start, end)
    print("opening 0.01, 0.005", skimage.transform.integral.integrate(ii, start, end))
    _end = time.time()
    print("8 integrals took %.3f" % (_end - _start))

    pylab.figure(figsize=(9, 9))
    ax = pylab.gca()
    im = ax.imshow(distribution, extent=(-2, 2, -2, 2))
    im.axes.grid(False)
    ax.set_xlabel("horizontal", fontsize=20)
    ax.set_ylabel("vertical", fontsize=20)
    ax.set_title("Photon distribution from slit scans", fontsize=24)
    # pylab.savefig('photon_distribution_primary_slits_2d.png')
    pylab.savefig(
        "%s/photon_distribution_%s_2d.png"
        % (
            os.path.dirname(results_filename),
            os.path.basename(results_filename).replace("_results.pickle", ""),
        )
    )

    pylab.figure(figsize=(9, 9))
    ax = pylab.gca()
    im = ax.imshow(ii, extent=(-2, 2, -2, 2))
    im.axes.grid(False)
    ax.set_xlabel("horizontal", fontsize=20)
    ax.set_ylabel("vertical", fontsize=20)
    ax.set_title("Integral image of photon distribution", fontsize=24)
    # pylab.savefig('photon_distribution_primary_slits_2d.png')
    pylab.savefig(
        "%s/photon_distribution_%s_integral_image.png"
        % (
            os.path.dirname(results_filename),
            os.path.basename(results_filename).replace("_results.pickle", ""),
        )
    )

    sum_normalized_distribution = distribution.sum()
    print("sum normalized distribution %.3f" % sum_normalized_distribution)
    x = np.linspace(-2, 2, 4000)
    y = np.linspace(-2, 2, 4000)

    X, Y = np.meshgrid(x, y)

    fig = pylab.figure(figsize=(16, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_wireframe(X, Y, distribution * 5e5, cstride=75, rstride=75)
    ax.view_init(azim=-56, elev=42)
    ax.set_xlabel("horizontal", fontsize=20)
    ax.set_ylabel("vertical", fontsize=20)
    ax.set_zlabel("distribution", fontsize=20)
    ax.set_title("Photon distribution from slit scans", fontsize=24)
    pylab.savefig(
        "%s/photon_distribution_%s_3d_auto.png"
        % (
            os.path.dirname(results_filename),
            os.path.basename(results_filename).replace("_results.pickle", ""),
        )
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-r",
        "--results_filename",
        default="s1b_results.pickle",
        type=str,
        help="results file name",
    )
    parser.add_argument("-D", "--display", action="store_true", help="display graphs")
    args = parser.parse_args()
    print("args", args)
    main(results_filename=args.results_filename)
    if args.display:
        pylab.show()
