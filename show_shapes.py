#!/usr/bin/env python

import os
import open3d as o3d
import glob

yellow = (1.0, 0.706, 0)
cyan = (0, 0.706, 1)
magenta = (0.706, 0, 1)


def show_shapes(directory):
    opti = glob.glob(os.path.join(directory, "opti", "*_careful_mm.pcd"))[-1]
    optio = glob.glob(os.path.join(directory, "opti", "*_careful_mm.obj"))[-1]

    try:
        tomo = glob.glob(os.path.join(directory, "tomo", "*_tioga.pcd"))[0]
    except:
        tomo = None

    try:
        opti = o3d.io.read_point_cloud(opti)
        opti.paint_uniform_color(yellow)
    except:
        opti = None

    try:
        optio = o3d.io.read_triangle_mesh(optio)
        optio.compute_triangle_normals()
        optio.paint_uniform_color(yellow)
    except:
        optio = None

    try:
        tomo = o3d.io.read_point_cloud(tomo)
        tomo.paint_uniform_color(magenta)  # cyan)
    except:
        tomo = None
    # o3d.visualization.draw_geometries([opti, tomo])
    if optio is not None and tomo is not None:
        o3d.visualization.draw_geometries([optio, tomo])
    elif opti is not None:
        print("no diffraction detected")
        o3d.visualization.draw_geometries([opti])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,)

    parser.add_argument(
        "-d",
        "--directory",
        default="/nfs/data4/2025_Run2/20250023/2025-04-20/RAW_DATA/PTPN22/PTPN22-CD044623_H04-1_BX033A-12",
        type=str,
        help="directory",
    )

    args = parser.parse_args()

    show_shapes(args.directory)
