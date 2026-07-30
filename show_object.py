#!/usr/bin/env python

import os
import open3d as o3d
import glob

yellow = (1.0, 0.706, 0)
cyan = (0, 0.706, 1)
magenta = (0.706, 0, 1)


def show_object(obj):
    if obj.endswith(".pcd"):
        opti = o3d.io.read_point_cloud(obj)
        opti.paint_uniform_color(yellow)
    elif obj.endswith(".obj"):
        opti = o3d.io.read_triangle_mesh(obj)
        opti.compute_triangle_normals()
        opti.paint_uniform_color(magenta)
        
    o3d.visualization.draw_geometries([opti])
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-o",
        "--obj",
        default="/nfs/data4/2026_Run3/20100023/2026-06-14/ARCHIVE/opti/manu_-1_-1_20260614_234718_zoom_4_kappa_0.00_phi_360.00_mm.pcd",
        type=str,
        help="obj",
    )

    args = parser.parse_args()

    show_object(args.obj)
