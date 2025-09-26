#!/usr/bin/env python
# -*- coding: utf-8 -*-
####!/usr/local/conda/envs/murko_3.11/bin/python


import os
import re
import sys
import zmq
import time
import math
import pickle
import copy
import numpy as np
import open3d as o3d
import pylab
import glob
from pprint import pprint
import cv2 as cv
from skimage.measure import regionprops
from skimage.morphology import remove_small_objects, binary_closing
from skimage.transform import rotate
from scipy.interpolate import interp1d, RectBivariateSpline
from scipy.spatial import distance_matrix

from useful_routines import (
    get_shift_from_aligned_position_and_reference_position,
    get_aligned_position_from_reference_position_and_shift,
    get_shift_between_positions,
    get_points_in_goniometer_frame,
    get_origin,
    get_voxel_calibration,
    get_distance,
    get_reduced_point,
    get_position_from_vector,
    get_vector_from_position,
)

from volume_reconstruction_tools import (
    get_reconstruction,
    get_volume_from_reconstruction,
    get_pcd_px,
    get_mesh_px,
    get_mesh_or_pcd_mm,
)

from diffraction_experiment_analysis import diffraction_experiment_analysis
from optical_alignment import optical_alignment

# import seaborn as sns
# sns.set_color_codes()
# from reconstruct import principal_axes

from colors import (
    magenta,
    yellow,
    green,
    blue,
    unknown1,
    unknown2,
)


def get_line_as_image(line):
    v = len(line)
    h = v
    if h % 2 == 0:
        h += 1
    limg = np.zeros((v, h))
    limg[:, int(h / 2)] = line
    return limg


def get_profiles(
    lines,
    scan_start_angles,
    threshold_identity=0.999,
    # threshold_projection=0.7,
    threshold_projection=0.85,
):
    print("scan_start_angles", scan_start_angles)
    norientations = len(scan_start_angles)
    profiles = {}
    for orientation in scan_start_angles:
        profiles[orientation] = {}

    positions_indices = list(range(lines.shape[1]))

    for position in positions_indices:
        orientation = scan_start_angles[position % norientations]
        measured_line_at_position = lines[:, position]
        measured_line_as_image = get_line_as_image(measured_line_at_position)
        center = np.array(measured_line_as_image.shape) / 2.0
        integral_at_position = np.sum(measured_line_at_position)

        for angle in scan_start_angles:
            profiles[angle][position] = {}
            measured = None
            estimated = None
            projected = None

            if integral_at_position > 0:
                angle_difference = angle - orientation
                projection_factor = np.abs(np.cos(np.deg2rad(angle_difference)))

                rotated = rotate(
                    measured_line_as_image, angle_difference, center=center
                )
                projected = rotated.sum(axis=1)

                assert measured is None
                if projection_factor >= threshold_identity:
                    measured = measured_line_at_position
                assert estimated is None
                if projection_factor > threshold_projection:
                    estimated = projected
            else:
                estimated = np.zeros(measured_line_at_position.shape)
                measured = np.zeros(measured_line_at_position.shape)
                projected = np.zeros(measured_line_at_position.shape)

            profiles[angle][position]["measured"] = measured
            profiles[angle][position]["estimated"] = estimated
            profiles[angle][position]["projected"] = projected

            if measured is not None and estimated is None:
                print("This should not have been possible, please check")
                print("measured", measured)
                print("estimated", estimated)
                print(
                    "position",
                    position,
                    "angle",
                    angle,
                    "projection_factor",
                    projection_factor,
                    "integral_at_position",
                    integral_at_position,
                )

    return profiles


def get_rasters_from_profiles(
    profiles, seed_positions, max_bounding_ray, reference_position
):
    rasters = {}
    start_shifts_all = []
    stop_shifts_all = []
    for angle in profiles:
        start_shifts = []
        stop_shifts = []
        for sp in seed_positions:
            position = get_position_from_vector(
                sp, keys=["CentringX", "CentringY", "AlignmentY"]
            )
            position["AlignmentZ"] = reference_position["AlignmentZ"]
            position["Omega"] = angle
            start = get_aligned_position_from_reference_position_and_shift(
                position, max_bounding_ray, 0
            )
            stop = get_aligned_position_from_reference_position_and_shift(
                position, -max_bounding_ray, 0
            )
            start_shift = get_shift_between_positions(start, reference_position)[1]
            stop_shift = get_shift_between_positions(stop, reference_position)[1]
            start_shifts.append(start_shift)
            stop_shifts.append(stop_shift)

        start_shifts_all += start_shifts
        stop_shifts_all += stop_shifts

        rasters[angle] = {
            "start_shifts": start_shifts,
            "stop_shifts": stop_shifts,
        }
    rasters["start_shifts_all"] = start_shifts_all
    rasters["stop_shifts_all"] = stop_shifts_all

    return rasters


def get_overlay(
    optical_image,
    raster,
    optical_reference,
    raster_reference,
    optical_calibration,
    raster_calibration,
):
    # overlay = optical_image.copy() #.mean(axis=2)
    print("optical shape", optical_image.shape[:2])
    print("raster shape", raster.shape)
    print("optical_calibration", optical_calibration)
    print("raster_calibration", raster_calibration)
    overlay = optical_image.mean(axis=2)
    shift_mm = get_shift_from_aligned_position_and_reference_position(
        raster_reference, optical_reference
    )
    shift_px = shift_mm / optical_calibration
    print("shift (mm, px):", shift_mm, shift_px)
    rV, rH = raster.shape
    oV, oH = optical_image.shape[:2]
    scale = raster_calibration / optical_calibration  # /raster_calibration
    print("scale", scale)
    optical_raster = cv.resize(
        raster,
        (
            int(rH * scale[1]),
            int(rV * scale[0]),
        ),
    )
    center = np.array([oV / 2, oH / 2])
    or_shape = np.array(optical_raster.shape)
    print("optical raster shape", or_shape)
    raster_start = center - or_shape / 2 + shift_px
    sV, sH = raster_start.astype(int)
    # eV, eH = (raster_start + or_shape + 1).astype(int)
    print("raster_start", raster_start)
    print("raster_extent", or_shape)
    eV = sV + or_shape[0]
    eH = sH + or_shape[1]
    print("eV -sV, eH - sH", eV - sV, eH - sH)
    assert or_shape[0] == eV - sV
    assert or_shape[1] == eH - sH
    # overlay[sV: eV, sH: eH] = optical_raster
    opr = 255 * optical_raster / optical_raster.max()
    overlay[sV:eV, sH:eH][optical_raster > 25] = opr[optical_raster > 25]
    return overlay


def get_rectified_projection(rectified_interpolation, along_cells):
    ortho_cells = rectified_interpolation.shape[1]
    rectified_projection = cv.resize(
        rectified_interpolation,
        (
            ortho_cells,
            along_cells,
        ),
    )
    return rectified_projection


def get_projections_from_profiles(
    profiles,
    nimages,
    seed_positions,
    max_bounding_ray,
    reference_position,
    oa,
    along_cells,
    ortho_step,
):
    rasters = get_rasters_from_profiles(
        profiles, seed_positions, max_bounding_ray, reference_position
    )
    start_shifts_all = rasters["start_shifts_all"]
    stop_shifts_all = rasters["stop_shifts_all"]
    rectification_start = start_shifts_all[np.argmax(np.abs(start_shifts_all))]
    rectification_stop = stop_shifts_all[np.argmax(np.abs(stop_shifts_all))]

    omegas, images = oa._get_omegas_images()
    optical_reference = oa.get_reference_position()
    optical_calibration = oa.get_calibration()
    raster_calibration = np.array([ortho_step, ortho_step])

    projections = {}
    for angle in profiles:
        image = oa.get_image_at_angle(angle, omegas=omegas, images=images)
        (
            measurement,
            estimation,
            interpolation,
            rectified_interpolation,
        ) = get_projection(
            profiles[angle],
            nimages,
            len(seed_positions),
            rasters[angle],
            rectification_start,
            rectification_stop,
        )

        rectified_projection = get_rectified_projection(
            rectified_interpolation.T, along_cells
        )

        projections[angle] = {
            "measurement": measurement,
            "estimation": estimation,
            "interpolation": interpolation,
            "rectified_interpolation": rectified_interpolation,
            "rectified_projection": rectified_projection,
            "optical_image": image,
            "overlay": get_overlay(
                image,
                rectified_projection,
                optical_reference,
                reference_position,
                optical_calibration,
                raster_calibration,
            ),
        }

    return projections


def get_projection(
    profiles_at_angle,
    nimages,
    nlines,
    raster_at_angle,
    rectification_start,
    rectification_stop,
):
    measurement = np.zeros((nimages, nlines))
    estimation = np.zeros((nimages, nlines))

    rectified = []
    trusted = []
    pos = []

    start_shifts = raster_at_angle["start_shifts"]
    stop_shifts = raster_at_angle["stop_shifts"]

    sampled_range = abs(start_shifts[0]) + abs(stop_shifts[0])
    sampling = sampled_range / nimages

    rectified_range = abs(rectification_start) + abs(rectification_stop)
    kimages = int(math.ceil(rectified_range / sampling))
    print("nimages, kimages", nimages, kimages)
    rectification_points = np.linspace(rectification_start, rectification_stop, kimages)

    for position in profiles_at_angle:
        if profiles_at_angle[position]["measured"] is not None:
            measured = profiles_at_angle[position]["measured"]
            # print("measured", measured)
            measurement[:, position] = measured

        if profiles_at_angle[position]["estimated"] is not None:
            estimated = profiles_at_angle[position]["estimated"]
            # print("estimated", estimated)
            estimation[:, position] = estimated

            pos.append(position)
            trusted.append(estimated)

            measurement_points = np.linspace(
                start_shifts[position], stop_shifts[position], nimages
            )
            ip = interp1d(
                measurement_points,
                estimated,
                bounds_error=False,
                fill_value="extrapolate",
            )

            rectified.append(ip(rectification_points))

    trusted = np.array(trusted).T
    y, x = pos, np.arange(nimages)
    ip = RectBivariateSpline(x, y, trusted)
    y = np.arange(nlines)
    interpolation = ip(x, y)

    rectified = np.array(rectified).T
    y, x = pos, np.arange(kimages)
    ip2 = RectBivariateSpline(x, y, rectified)
    y = np.arange(nlines)
    rectified_interpolation = ip2(x, y)[::-1]

    return measurement, estimation, interpolation, rectified_interpolation


def plot_projections(projections, ntrigger, nimages, along_step, ortho_step):
    keys = [
        "measurement",
        "estimation",
        "interpolation",
        "rectified_interpolation",
        "optical_image",
        "overlay",
    ]
    columns = len(keys)
    norientations = len(projections)
    fig, axes = pylab.subplots(math.ceil(columns * norientations / columns), columns)
    fig.suptitle("evidence")
    axs = axes.ravel()
    k = 0
    for angle in projections:
        for key in keys:
            p = projections[angle][key]
            if key not in ["optical_image", "overlay"]:
                fimages = p.shape[0]
                p = cv.resize(
                    p,
                    (
                        int(ntrigger * (along_step / ortho_step)),
                        fimages,
                    ),
                )
                # axs[k].imshow(p.T/p.max() > 0.05)
                p[p < 1] = 0
                p = p.T

            axs[k].imshow(p)
            axs[k].set_title(f"{key.replace('_', ' ')} {angle:.1f}")
            # https://stackoverflow.com/questions/9295026/how-to-remove-axis-legends-and-white-padding
            axs[k].set_axis_off()
            k += 1


def get_rectified_projections(projections):  # , along_cells, ortho_cells):
    rectified_projections, angles, ortho_cells = [], [], []
    for angle in projections:
        angles.append(angle)
        # ri = projections[angle]["rectified_interpolation"].T
        # rp = get_rectified_projection(ri, along_cells)
        rp = projections[angle]["rectified_projection"]
        rectified_projections.append(rp)
        ortho_cells.append(rp.shape[1])
    assert np.allclose(ortho_cells, np.mean(ortho_cells))
    return rectified_projections, angles, ortho_cells[0]


def get_max_bounding_ray(parameters):
    if "max_bounding_ray" in parameters:
        max_bounding_ray = parameters["max_bounding_ray"]
    elif "initial_raster" in parameters:
        ir = parameters["initial_raster"]
        start, stop = ir[0][:2]
        s = get_vector_from_position(start)
        p = get_vector_from_position(stop)
        max_bounding_ray = np.linalg.norm(s - p) / 2
    return max_bounding_ray


def get_opti(directory, ext="obj"):
    # opti = o3d.io.read_point_cloud(os.path.join(os.path.dirname(args.directory), "opti", "zoom_X_careful_mm.pcd"))
    print("directory", directory)
    assert ext in ["pcd", "obj"]
    opti_meshes = glob.glob(
        os.path.join(os.path.dirname(directory), "opti", f"*careful_mm.{ext}")
    )
    print("opti_meshes", opti_meshes)
    winner = None
    if len(opti_meshes) == 1:
        winner = opti_meshes[0]
    else:
        for item in opti_meshes:
            if "zoom_X" in item:
                winner = item
    print("winner", winner)
    opti = optical_alignment(
        directory=os.path.dirname(winner), name_pattern=os.path.basename(winner)[:-7]
    )
    return opti


def get_scan_start_angles(parameters):
    ir = parameters["initial_raster"]
    omegas = []
    for line in ir:
        angle = round(line[2], 3)
        if angle not in omegas:
            omegas.append(angle)
    return omegas


def main(args):
    directory = os.path.realpath(args.directory)
    dea = diffraction_experiment_analysis(
        directory=directory,
        name_pattern=args.name_pattern,
    )

    parameters = dea.get_parameters()
    scan_start_angles = get_scan_start_angles(parameters)

    ntrigger = parameters["ntrigger"]
    nimages = parameters["nimages"]
    along_step = parameters["step_size_along_omega"]
    ortho_step = parameters["orthogonal_step_size"]
    seed_positions = np.array(parameters["seed_positions"])
    initial_raster = parameters["initial_raster"]
    reference_position = parameters["reference_position"]
    max_bounding_ray = get_max_bounding_ray(parameters)

    oa = get_opti(directory)
    opti = oa.get_mesh_mm()
    opti.compute_vertex_normals()
    opti.paint_uniform_color(yellow)

    assert len(seed_positions) == ntrigger

    tr = dea.get_tioga_results()
    lines = np.reshape(tr, (ntrigger, nimages))
    lines = lines.T

    along_max = seed_positions[:, -1].max()
    along_min = seed_positions[:, -1].min()
    along_range = along_max - along_min

    along_cells = int(along_range / ortho_step)
    print("along_cells", along_cells)

    profiles = get_profiles(lines, scan_start_angles)

    projections = get_projections_from_profiles(
        profiles,
        nimages,
        seed_positions,
        max_bounding_ray,
        reference_position,
        oa,
        along_cells,
        ortho_step,
    )

    plot_projections(projections, ntrigger, nimages, along_step, ortho_step)
    pylab.show()

    rectified_projections, angles, ortho_cells = get_rectified_projections(
        projections
    )  # , along_cells, ortho_cells)

    reconstruction = get_reconstruction(
        [p > args.min_spots for p in rectified_projections], angles
    )
    print(
        "reconstruction (shape, max, mean)",
        reconstruction.shape,
        reconstruction.max(),
        reconstruction.mean(),
    )

    print("ortho_cells", ortho_cells)

    # _projections.shape (217, 5, 299)
    # reconstruction (shape, max, mean) (217, 598, 598) 5.0 0.4572133
    ortho_range = ortho_cells * ortho_step

    print("along_range, ortho_range", along_range, ortho_range)
    # print('sp.mean - origin', seed_positions.mean(axis=0) - origin_vector)
    along_size = along_range / rectified_projections[0].shape[-1]
    ortho_size = ortho_step
    print("along_size, ortho_size", along_size, ortho_size)
    calibration = np.array([along_size, ortho_size, ortho_size])
    print("calibration", calibration)

    origin_vector = np.array(
        [reference_position[key] for key in ["AlignmentY", "CentringX", "CentringY"]]
    )
    print("origin_vector", origin_vector)
    origin_index = np.array(reconstruction.shape) / 2
    origin_index[0] = reconstruction.shape[0] * (
        np.abs((origin_vector[0] - along_min)) / along_range
    )
    print("origin_index", origin_index)
    # origin_index = origin_index.astype(int)
    # print("origin_index (int)", origin_index)

    volume = get_volume_from_reconstruction(reconstruction, threshold=0.775)
    vadt_px = get_mesh_px(volume, gradient_direction="ascent")
    vadt_px.paint_uniform_color(magenta)
    directions = np.array([1, 1, 1])
    # directions = np.array([1, 1, -1])  # *
    # directions = np.array([ 1, -1,  1])
    # directions = np.array([-1,  1,  1])
    # directions = np.array([ 1, -1, -1]) # *
    # directions = np.array([-1,  1, -1]) # *
    # directions = np.array([-1, -1,  1])
    # directions = np.array([-1, -1, -1])

    vadt_mm = get_mesh_or_pcd_mm(
        vadt_px, calibration, origin_vector, origin_index, directions=directions
    )
    vadt_mm.compute_vertex_normals()
    print(vadt_mm)
    # o3d.visualization.draw_geometries([pcd_px])
    view = {
        "zoom": 1,
        "up": np.array([0, 0, -1]),
        "lookat": np.array([0, 0, -2]),
        "front": np.array(
            [0.51651730889124048, 0.83752848835406313, -0.17820185411804509]
        ),
        "field_of_view": 60.0,
    }
    o3d.visualization.draw_geometries(
        [opti, vadt_mm],
        window_name=f"{directions}",
        # width=480,
        # height=480,
        # left=5,
        # top=5,
        # lookat=view["lookat"],
        # up=view["up"],
        # front=view["front"],
        # zoom=view["zoom"],
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d",
        "--directory",
        # default="/nfs/data4/2024_Run4/com-proxima2a/Commissioning/automated_operation/px2-0021/puck_09_pos_05_a/tomo",
        # default="/home/experiences/proxima2a/com-proxima2a/Documents/Martin/pos_10_a/tomo",
        default="/nfs/data4/2025_Run3/com-proxima2a/Commissioning/mse/px2_0049_pos4b/tomo",
        type=str,
        help="directory",
    )
    parser.add_argument(
        "-n",
        "--name_pattern",
        # default="tomo_a_puck_09_pos_05_a",
        # default="tomo_a_pos_10_a",
        default="vadt",
        type=str,
        help="name_pattern",
    )
    parser.add_argument("-m", "--min_spots", default=7, type=int, help="min_spots")
    parser.add_argument(
        "-t", "--threshold", default=0.125, type=float, help="threshold"
    )
    parser.add_argument("-D", "--display", action="store_true", help="Display")
    parser.add_argument(
        "-M", "--method", default="xds", type=str, help="Analysis method"
    )
    parser.add_argument(
        "-r", "--ratio", default=5, type=int, help="Horizonta/Vertical step size ratio"
    )
    parser.add_argument(
        "-o",
        "--horizontal_beam_size",
        default=0.01,
        type=float,
        help="horizontal beam size",
    )
    parser.add_argument(
        "-R",
        "--detector_row_spacing",
        default=1,
        type=int,
        help="detector vertical pixel size",
    )
    parser.add_argument(
        "-C",
        "--detector_col_spacing",
        default=1,
        type=int,
        help="detector horizontal pixel size",
    )
    parser.add_argument("--min_size", default=10, type=int, help="min_size")

    parser.add_argument("--debug", action="store_true", help="debug")

    args = parser.parse_args()
    print("args", args)

    main(args)
