#!/usr/bin/env python
# -*- coding: utf-8 -*-

from volume_aware_diffraction_tomography import volume_aware_diffraction_tomography
from optical_alignment import optical_alignment

import pickle
import open3d as o3d
from colors import *
import pylab
import numpy as np


def normalize(lien):
    try:
        lien = lien / lien.max()
    except:
        pass

    return lien


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--raster",
        default="/nfs/data4/2024_Run3/com-proxima2a/Commissioning/automated_operation/px2-0042/puck_09_pos_12_g/tomo/tomo_a_puck_09_pos_12_g_parameters.pickle",
        type=str,
        help="raster",
    )
    parser.add_argument(
        "-v",
        "--volume",
        default="/nfs/data4/2024_Run3/com-proxima2a/Commissioning/automated_operation/px2-0042/puck_09_pos_12_g/opti/oa_puck_09_pos_12_g_zoom_X_careful_parameters.pickle",
        type=str,
        help="optical",
    )

    args = parser.parse_args()

    r = pickle.load(open(args.raster, "rb"))
    v = pickle.load(open(args.volume, "rb"))

    oa = optical_alignment(directory=v["directory"], name_pattern=v["name_pattern"])
    vadt = volume_aware_diffraction_tomography(
        directory=r["directory"],
        name_pattern=r["name_pattern"],
        volume=oa.get_pcd_mm_name(),
    )
    # pcd = o3d.io.read_point_cloud(oa.get_pcd_mm_name())
    # pcd.paint_uniform_color(yellow)

    # o3d.visualization.draw_geometries([pcd])

    print("ntrigger", vadt.get_ntrigger())
    print("nimages", vadt.get_nimages())
    print("total_number_of_images", vadt.get_total_number_of_images())
    tr = vadt.get_tioga_results()
    print(tr)
    xr = vadt.get_xds_results()
    dr = vadt.get_dozor_results()[:, -2]
    pylab.figure()
    pylab.plot(normalize(tr), label="tioga")
    pylab.plot(normalize(xr), label="colspot")
    pylab.plot(normalize(dr), label="dozor")
    pylab.legend()
    pylab.show()


if __name__ == "__main__":
    main()
