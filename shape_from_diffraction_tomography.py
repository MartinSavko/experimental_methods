#!/usr/bin/env python
####!/usr/local/conda/envs/murko_3.11/bin/python
# -*- coding: utf-8 -*-
#!/usr/bin/env python

import os
import re
import zmq
import time
import pickle
import copy
import numpy as np
import open3d as o3d
import pylab
import scipy.ndimage as ndi
from skimage.measure import regionprops
from skimage.morphology import remove_small_objects, binary_closing
from scipy.spatial import distance_matrix

from diffraction_tomography import diffraction_tomography
from useful_routines import (
    get_points_in_goniometer_frame,
    get_origin,
    get_voxel_calibration,
    get_distance,
    get_reduced_point,
    principal_axes,
)
from volume_reconstruction_tools import _get_reconstruction

# import seaborn as sns
# sns.set_color_codes()
# from reconstruct import principal_axes

from colors import magenta

def get_calibration(vertical_step_size, horizontal_step_size):
    calibration = np.ones((3,))
    calibration[0] = horizontal_step_size
    calibration[1:] = vertical_step_size
    return calibration


def main():
    import argparse

    parser = argparse.ArgumentParser()

    # parser.add_argument('-d', '--directory', default='/nfs/data2/excenter/2023-04-25T12:19:45.158775', type=str, help='directory')
    parser.add_argument(
        "-d",
        "--directory",
        default="/nfs/data2/excenter/2023-04-15T17:22:01.257118",
        type=str,
        help="directory",
    )
    parser.add_argument(
        "-n", "--name_pattern", default="excenter", type=str, help="name_pattern"
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
    parser.add_argument(
        "-A",
        "--additional_data",
        default="none",
        type=str,
        help="optional additional data",
    )
    parser.add_argument("--debug", action="store_true", help="debug")

    args = parser.parse_args()
    print("args", args)

    dt = diffraction_tomography(
        directory=args.directory,
        name_pattern=args.name_pattern,
        method=args.method,
    )

    # try:
    # da = diffraction_tomography(
    # directory=args.directory,
    # name_pattern=args.additional_data,
    # )
    # da = None
    # except:
    # da = None
    da = None
    parameters = dt.get_parameters()

    # print('parameters', parameters)
    for p in ["scan_start_angles", "ntrigger", "nimages"]:
        if p in parameters:
            print(parameters[p])
        else:
            print("%s is not present in parameters" % p)
    detector_rows = parameters["nimages"]
    detector_cols = int(
        args.horizontal_beam_size / parameters["vertical_step_size"]
    )  # args.ratio

    pcd_filename = os.path.join("%s_%s.pcd" % (dt.get_template(), args.method))
    obj_filename = os.path.join(
        "%s_%s_raddose3d.obj" % (dt.get_template(), args.method)
    )
    img_filename = os.path.join(
        "%s_%s_reconstruction2d.jpg" % (dt.get_template(), args.method)
    )
    txt_filename = os.path.join("%s_%s.txt" % (dt.get_template(), args.method))
    results_filename = os.path.join("%s_%s.results" % (dt.get_template(), args.method))

    # if args.method == 'xds':
    # xds_results = dt.get_xds_results()
    # results = xds_results[:].astype('float') # dozor_results[:]
    # else:
    # dozor_results = dt.get_dozor_results()
    # results = dozor_results[:,1].astype('float')

    results = dt.get_results(method=args.method)

    results[results < args.min_spots] = 0.0
    print("results", results)
    mr = np.mean(results)
    print("mr", mr)
    threshold = args.threshold * mr
    results[results < threshold] = 0.0

    # args.threshold*
    reference_position = dt.get_reference_position()
    result_position = dt.get_result_position()

    if args.display and False:
        pylab.figure(2, figsize=(24, 16))
        pylab.plot(results)
        pylab.show()
        try:
            pylab.figure(1, figsize=(24, 16))
            pylab.grid(0)
            pylab.imshow(parameters["rgbimage"])
        except:
            pass

    # tomo_max = results.max()
    # results /= tomo_max
    raw_projections = []
    for k in range(parameters["ntrigger"]):
        k_start = int(k * parameters["nimages"])
        k_end = int((k + 1) * parameters["nimages"])
        line = results[k_start:k_end][::-1]
        line[line <= line.max() * args.threshold] = 0
        line = binary_closing(line)
        projection = np.zeros((detector_rows, detector_cols)) + np.reshape(
            line, (detector_rows, 1)
        )
        raw_projections.append(projection)

    try:
        angles = parameters["scan_start_angles"]
    except:
        angles = []
        for k in range(parameters["ntrigger"]):
            angles.append(k * 90)

    if da != None:
        irxds = da.get_xds_results()
        dap = da.get_parameters()

        tposition = parameters["reference_position"]
        rposition = dict(
            [
                (motor, dap["reference_position"][motor])
                for motor in [
                    "CentringX",
                    "CentringY",
                    "AlignmentY",
                    "AlignmentZ",
                    "Omega",
                ]
            ]
        )

        if "scan_start_angle" in dap:
            rposition["Omega"] = dap["scan_start_angle"]

        shift = da.goniometer.get_vertical_and_horizontal_shift_between_two_positions(
            tposition, rposition
        )
        vertical_shift = int(shift[0] / dap["vertical_step_size"])
        print("vertical_shift", vertical_shift)
        nimages = dap["nimages"]
        print("nimages", nimages)
        print("projection.shape", projection.shape)
        namp = dt.get_name_pattern()
        print(namp)
        found = re.findall("tomo.*dt_sp_([\d]*)", namp)
        print("found", found)
        k = int(found[0])

        needed_length = len(line)
        print("needed_length", needed_length)
        k_start = k * nimages
        k_end = (k + 1) * nimages
        print("k_start, k_end", k_start, k_end)
        if args.display:
            pylab.title("raw additional_line")
            pylab.plot(irxds[k_start:k_end][::-1])
            pylab.show()

        length = k_end - k_start

        difference = length - needed_length

        print("length", length)
        print("difference", difference)

        if difference:
            shift = difference / 2
            k_start += shift
            # k_start -= vertical_shift
            k_start = int(k_start)
            k_end = k_start + needed_length

        print("k_start, k_end", k_start, k_end)
        line = np.zeros(needed_length)

        line[:] = irxds[k_start:k_end]  # [::-1]
        line[line <= line.max() * args.threshold] = 0
        if args.display:
            pylab.title("additional_line")
            pylab.plot(line)
            pylab.show()
        projection = np.zeros((detector_rows, detector_cols)) + np.reshape(
            line, (detector_rows, 1)
        )
        raw_projections.append(projection)

        angle = rposition["Omega"]

        angles.append(angle)

    angles = np.deg2rad(angles)
    raw_projections = np.array(raw_projections)
    raw_projections /= raw_projections.max()

    print("raw_projections", raw_projections.shape)
    print("angles", angles)

    calibration = get_voxel_calibration(
        parameters["vertical_step_size"], args.horizontal_beam_size / detector_cols
    )
    origin = get_origin(parameters, position_key="reference_position")

    number_of_projections = len(angles)
    projections = np.zeros((detector_cols, number_of_projections, detector_rows))

    for k in range(len(raw_projections)):
        projection = raw_projections[k]
        projection[projection > 0] = 1
        projections[:, k, :] = projection.T

    center_of_mass = ndi.center_of_mass(projections)
    print(
        "projections shape, center_of_mass",
        projections.shape,
        center_of_mass,
        projections.max(),
        projections.mean(),
    )

    # vertical_correction = 0.
    vertical_correction = (
        result_position["AlignmentZ"] - reference_position["AlignmentZ"]
    )
    vertical_correction /= calibration[-1]
    # if not np.isnan(center_of_mass[2]):
    # vertical_correction =  - (center_of_mass[2] - detector_rows/2)
    # else:
    # vertical_correction = 0.
    print("vertical_correction", vertical_correction)

    request = {
        "projections": projections,
        "angles": angles,
        "detector_rows": detector_rows,
        "detector_cols": detector_cols,
        "detector_col_spacing": args.detector_col_spacing,
        "detector_row_spacing": args.detector_row_spacing,
        "vertical_correction": vertical_correction,
    }

    reconstruction = _get_reconstruction(request, port=8900, verbose=True)
    try:
        reconstruction_thresholded = reconstruction > 0.95 * reconstruction.max()
    except:
        print("Could not reconstruct the shape, probably not enough spots found.")
        return reconstruction
    reconstruction_2d = np.mean(reconstruction_thresholded, axis=0) > 0
    sor = remove_small_objects(reconstruction_2d, min_size=args.min_size).astype(
        np.uint8
    )
    if args.display and args.debug:
        pylab.figure(figsize=(24, 16))
        pylab.title("raw reconstruction")
        pylab.imshow(np.mean(reconstruction, axis=0))
        pylab.show()

        # pylab.title('thresholded reconstruction')
        # pylab.imshow(reconstruction_2d)
        # pylab.show()

        # pylab.title('thresholded reconstruction without small objects')
        # pylab.imshow(sor) #.astype(int))
        # pylab.show()

    analysis_results = {}
    selected_props = [
        "centroid",
        "area",
        "orientation",
        "axis_major_length",
        "axis_minor_length",
        "solidity",
        "euler_number",
        "eccentricity",
        "equivalent_diameter_area",
    ]
    try:
        props = regionprops(sor)[0]
        # print('props', props)
        for prop in selected_props:
            print(prop, props[prop])
            analysis_results[prop] = props[prop]
    except IndexError:
        for prop in selected_props:
            analysis_results[prop] = 0.0

    try:
        centroid = np.array(analysis_results["centroid"])[::-1]
    except:
        centroid = np.array([0.0, 0.0])
    orientation = analysis_results["orientation"]
    amaxl = analysis_results["axis_major_length"]
    aminl = analysis_results["axis_minor_length"]

    # R = np.array([[np.cos(orientation), np.sin(orientation)], [np.sin(orientation), np.cos(orientation)]])
    # amaxp = np.dot(R, np.array([-amaxl/2, 0])) + np.array(centroid).T
    # aminp = np.dot(R, np.array([0, aminl/2])) + np.array(centroid).T

    amaxp = (
        np.array([-np.sin(orientation) * amaxl, -np.cos(orientation) * amaxl]) / 2
        + centroid
    )
    aminp = (
        np.array([np.cos(orientation) * aminl, -np.sin(orientation) * aminl]) / 2
        + centroid
    )

    maxl = np.vstack([centroid, amaxp])
    minl = np.vstack([centroid, aminp])

    print(
        "reconstruction",
        reconstruction.shape,
        reconstruction.max(),
        reconstruction.mean(),
    )
    sor_threshold = np.zeros(reconstruction.shape)
    sor_threshold[:, ::] = sor
    print("sor", sor.shape, sor.dtype)

    objectpoints = np.argwhere(sor_threshold)
    analysis_results["volume_voxels"] = len(objectpoints)
    voxel_mm3 = np.prod(calibration)
    analysis_results["voxel_mm^3"] = voxel_mm3
    analysis_results["volume_mm^3"] = len(objectpoints) * voxel_mm3
    analysis_results["axis_major_length_mm"] = analysis_results[
        "axis_major_length"
    ] * np.mean(calibration[1:])
    analysis_results["axis_minor_length_mm"] = analysis_results[
        "axis_minor_length"
    ] * np.mean(calibration[1:])

    print("#objectpoints", len(objectpoints))
    print("objectpoints.shape", objectpoints.shape)
    # print('objectpoints[:10]', objectpoints[:10])
    # >args.threshold*reconstruction.max())

    # center = np.array(reconstruction.shape)/2 #np.array([detector_cols/2, detector_rows/2, detector_rows/2])
    center = np.array([2.0, centroid[1], centroid[0]])
    analysis_results["calibration"] = calibration
    analysis_results["origin"] = origin
    analysis_results["center"] = center
    analysis_results["reference_position"] = reference_position
    analysis_results["result_position"] = result_position

    # center[2] += vertical_correction
    print("center", center)
    print("calibration", calibration)
    print("origin", origin)
    objectpoints_mm = get_points_in_goniometer_frame(
        objectpoints,
        calibration,
        origin[:3],
        center=center,
        directions=np.array([-1, 1, 1]),
    )
    # positive_pixel is negative_movement for CentringX
    # negative_pixel is negative_movement for CentringY
    print("result_position (cx, cy, ay)")
    print(
        result_position["CentringX"],
        result_position["CentringY"],
        result_position["AlignmentY"],
    )
    print("objectpoints_px mean")
    print(np.mean(objectpoints, axis=0))
    print("objectpoints_mm median")
    print(np.median(objectpoints_mm, axis=0))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(objectpoints_mm)
    pcd.estimate_normals()
    o3d.io.write_point_cloud(pcd_filename, pcd)

    try:
        pcd_rd3_points = objectpoints_mm[:, [0, 2, 1]]
        pcd_rd3_points -= pcd_rd3_points.mean(axis=0)
        pcd_rd3_points *= 1000
        pcd_rd3 = o3d.geometry.PointCloud()
        pcd_rd3.points = o3d.utility.Vector3dVector(pcd_rd3_points)
        rd3_mesh, rd3_points = pcd_rd3.compute_convex_hull()
        rd3_mesh.compute_vertex_normals()
        rd3_mesh.compute_triangle_normals()
        o3d.io.write_triangle_mesh(obj_filename, rd3_mesh)

        hull_mesh, hull_points = pcd.compute_convex_hull()
        hull_mesh.compute_triangle_normals()
        hull_mesh.compute_vertex_normals()
        hull_points_mm = objectpoints_mm[hull_points]
        hull_points_px = objectpoints[hull_points]
    except:
        hull_points_mm = objectpoints_mm
        hull_points_px = objectpoints

    try:
        dm = distance_matrix(hull_points_mm, hull_points_mm)
        print("max dm", np.max(dm))

        extreme_points = np.unravel_index(np.argmax(dm), dm.shape)
        print("argmax dm", extreme_points)

        point1_mm = hull_points_mm[extreme_points[0]]
        point2_mm = hull_points_mm[extreme_points[1]]
        point1_px = hull_points_px[extreme_points[0]]
        point2_px = hull_points_px[extreme_points[1]]

        print("objectpoints extreme mm ", point1_mm, point2_mm)
        print("objectpoints extreme px ", point1_px, point2_px)

        ep1 = copy.copy(result_position)
        ep1["CentringX"], ep1["CentringY"], ep1["AlignmentY"] = point1_mm

        ep2 = copy.copy(result_position)
        ep2["CentringX"], ep2["CentringY"], ep2["AlignmentY"] = point2_mm
        analysis_results["extreme_points"] = [ep1, ep2]

        pca = principal_axes(sor, verbose=True)
        pca_center = pca[-1]
        pca_s = pca[-2]
        pca_e = pca[1]
        print("pca_s", pca_s)
        print("pca_e, sqrt(pca_e)", np.round(pca_e, 3), np.round(np.sqrt(pca_e), 3))
    except:
        pass

    try:
        print(
            "ratio of eig1/eig2 %.3f, sqrt(eig1/eig2) %.3f"
            % (pca_e[0] / pca_e[1], np.sqrt(pca_e[0] / pca_e[1]))
        )
        print("ratio of major and minor axes %.3f" % (amaxl / aminl))
        # amaxp = np.array([-np.sin(orientation) * amaxl, -np.cos(orientation) * amaxl])/2 + centroid
        # amaxp_0p95_shift_px = 0.9*np.dot(R , np.array([-amaxl/2, 0]))
        # amaxp_0p95_shift_px = np.dot(pca_s.T, np.array([[amaxl, 0], [0, aminl]]))
    except:
        pass
    # amaxp_0p95_shift_px = 0.9*amaxl*np.array([-np.sin(orientation), -np.cos(orientation)])/2
    # print('amaxp_0p95_shift_px from major', amaxp_0p95_shift_px)

    # amaxp_0p95_a_px = centroid + amaxp_0p95_shift_px
    # amaxp_0p95_b_px = centroid - amaxp_0p95_shift_px

    # print('centroid from major', centroid)
    # print('amaxp_0p95_a_px from major', amaxp_0p95_a_px)
    # print('amaxp_0p95_b_px from maror', amaxp_0p95_b_px)

    try:
        amaxp_0p95_shift_px = 0.45 * pca_s.T[0, :] * amaxl
        print("amaxp_0p95_shift_px from pca", amaxp_0p95_shift_px)

        amaxp_0p95_a_px = (
            pca_center + amaxp_0p95_shift_px
        )  # (pca_e[0]/np.sum(pca_e))*100
        amaxp_0p95_b_px = (
            pca_center - amaxp_0p95_shift_px
        )  # (pca_e[0]/np.sum(pca_e))*100

        print("centroid from pca", pca_center)
        print("amaxp_0p95_a_px from pca", amaxp_0p95_a_px)
        print("amaxp_0p95_b_px from pca", amaxp_0p95_b_px)

        # positive_pixel is negative_movement for CentringX
        # negative_pixel is negative_movement for CentringY

        amaxp_0p95_shift_mm = amaxp_0p95_shift_px * calibration[1:] - 0.005
        print("amaxp_0p95_shift_mm", amaxp_0p95_shift_mm)

        pca_point_a = copy.copy(result_position)
        pca_point_b = copy.copy(result_position)

        pca_point_a["CentringX"] += -amaxp_0p95_shift_mm[0]
        pca_point_a["CentringY"] += amaxp_0p95_shift_mm[1]

        pca_point_b["CentringX"] -= -amaxp_0p95_shift_mm[0]
        pca_point_b["CentringY"] -= amaxp_0p95_shift_mm[1]
        analysis_results["pca_points"] = [pca_point_a, pca_point_b]

        print("extreme_points")
        print(
            [
                get_reduced_point(point, keys=["CentringX", "CentringY"])
                for point in analysis_results["extreme_points"]
            ]
        )
        print(
            "get_distance(extreme_points)",
            get_distance(*analysis_results["extreme_points"]),
        )
        print("pca points")
        print(
            [
                get_reduced_point(point, keys=["CentringX", "CentringY"])
                for point in analysis_results["pca_points"]
            ]
        )
        print("get_distance(pca_points)", get_distance(*analysis_results["pca_points"]))
        print("distances extreme vs pca")
        print(
            get_distance(
                analysis_results["extreme_points"][0], analysis_results["pca_points"][0]
            )
        )
        print(
            get_distance(
                analysis_results["extreme_points"][1], analysis_results["pca_points"][1]
            )
        )
        print(
            get_distance(
                analysis_results["extreme_points"][1], analysis_results["pca_points"][0]
            )
        )
        print(
            get_distance(
                analysis_results["extreme_points"][0], analysis_results["pca_points"][1]
            )
        )
    except:
        pass

    # reconstruction_2d_pca = np.dot(reconstruction_2d, pca_s)

    f = open(results_filename, "wb")
    pickle.dump(analysis_results, f)
    f.close()

    f = open(txt_filename, "w")
    for key in analysis_results:
        # if type(analysis_results[key]) is float:
        v = analysis_results[key]
        # print('%s, %s, type %s' % (key, v, type(v)))
        try:
            if type(v) is int or key == "area":
                ark = "%d" % v
            elif abs(v) > 1e-9:
                ark = "%.9f" % v
            else:
                ark = v
        except:
            ark = v
        f.write("%s: %s\n" % (key, ark))
    f.close()

    pylab.figure(figsize=(24, 16))
    pylab.imshow(sor, label="reconstruction mean")
    pylab.grid(False)
    pylab.title("vertical_correction plus %s" % args.name_pattern)

    try:
        ax = pylab.gca()
        c = pylab.Circle(centroid, radius=3, color="red")
        e1 = pylab.Circle(point1_px[::-1], radius=2)
        e2 = pylab.Circle(point2_px[::-1], radius=2)
        aa = pylab.Circle(amaxp_0p95_a_px[::-1], radius=2, color="cyan")
        ab = pylab.Circle(amaxp_0p95_b_px[::-1], radius=2, color="cyan")
        pca_c = pylab.Circle(pca_center[::-1], radius=2, color="green")

        p1 = (
            pca_center + amaxp_0p95_shift_px
        )  # pca_s[0,:].T*(pca_e[0]/np.sum(pca_e))*100
        p2 = (
            pca_center + 0.45 * pca_s.T[1, :] * aminl
        )  # pca_s[1,:].T*(pca_e[1]/np.sum(pca_e))*100
        pylab.plot([pca_center[1], p1[1]], [pca_center[0], p1[0]], label="eig1")
        pylab.plot([pca_center[1], p2[1]], [pca_center[0], p2[0]], label="eig2")

        ax.add_patch(c)
        ax.add_patch(pca_c)
        ax.add_patch(e1)
        ax.add_patch(e2)
        ax.add_patch(aa)
        ax.add_patch(ab)
        pylab.plot(maxl[:, 0], maxl[:, 1], label="axis_major")
        pylab.plot(minl[:, 0], minl[:, 1], label="axis_minor")
        pylab.legend()
    except:
        pass
    pylab.savefig(img_filename)
    # pylab.figure()
    # pylab.imshow(reconstruction_2d_pca, 'in own coordinates')
    if args.display:
        pylab.show()

    if args.display:
        pcd.paint_uniform_color(magenta)
        o3d.visualization.draw_geometries(
            [pcd], window_name="reconstructed crystal volume"
        )


if __name__ == "__main__":
    main()
