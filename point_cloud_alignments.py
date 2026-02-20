#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

sys.path.insert(0, "/usr/local/experimental_methods")
# import reconstruct
# print('reconstruct.__file__', reconstruct.__file__)
from reconstruct import reconstruct, principal_axes
import open3d as o3d
import numpy as np
import scipy.spatial
import pylab
import pickle

from goniometer import goniometer
from shape_from_history import get_origin, get_calibration, get_kappa_phi
import h5py
import traceback


def get_pcd(alignment):
    pcd_filename = os.path.join(alignment, "foreground_reconstruction.pcd")
    if not os.path.isfile(pcd_filename):
        reconstruct(
            os.path.realpath(alignment),
            generate_report=True,
            display=False,
            save_raw_reconstructions=False,
        )
    pcd = o3d.io.read_point_cloud(pcd_filename)
    return pcd


def get_parameters(alignment):
    parameters_filename = os.path.join(
        alignment, "%s_parameters.pickle" % os.path.basename(alignment)
    )
    parameters = pickle.load(open(parameters_filename, "rb"), encoding="bytes")
    return parameters


def get_T(eigenvectors, center):
    T = eigenvectors.T
    return T


def get_transformed_pcd(pcd, flip=False, hull=False):
    points = np.asarray(pcd.points)
    if hull:
        convex_hull, point_indices = pcd.compute_convex_hull()
        ch_points = points[point_indices]
        transformed_ch_points, T, center = get_transformed_points(ch_points, flip=flip)
        transformed_points = np.dot(points - center, T)
    else:
        transformed_points, T, center = get_transformed_points(points, flip=flip)
    transformed_pcd = get_pcd_from_points(transformed_points)
    return transformed_pcd, T, center


def get_principal_axes(points):
    # inertia, eigenvalues, eigenvectors, center, T, Eor = principal_axes(points, verbose=True)
    # return inertia, eigenvalues, eigenvectors, center, T, Eor
    inertia, eigenvalues, eigenvectors, center = principal_axes(points, verbose=True)
    return inertia, eigenvalues, eigenvectors, center


def get_transformed_points(points, flip=False):
    # inertia, eigenvalues, eigenvectors, center, T, Eor = get_principal_axes(points)
    inertia, eigenvalues, eigenvectors, center = get_principal_axes(points)
    T = eigenvectors[:, :]
    if flip:
        # T[:,0] *= -1
        T[:, 1] *= -1
        T[:, 2] *= -1
    transformed_points = np.dot(points - center, T)
    return transformed_points, T, center


def get_pcd_from_points(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals()
    return pcd


def print_inertia_aspects(points):
    inertia, eigenvalues, eigenvectors, center, Vor, Eor = principal_axes(points)

    print("npoints", len(points))
    print("inertia")
    print(np.round(inertia, 4))
    print("eigenvalues")
    print(np.round(eigenvalues, 4))
    print("eigenvectors")
    print(np.round(eigenvectors, 4))
    print("center", np.round(center, 4))
    print("Vor")
    print(np.round(Vor, 4))
    print("Eor")
    print(np.round(Eor, 4))
    print("Range")
    p_min = np.min(points, 0)
    p_max = np.max(points, 0)
    print("min", p_min)
    print("max", p_max)
    print("range", p_max - p_min)
    print("det(eigenvectors.T)", np.linalg.det(eigenvectors.T))
    print("det(Vor)", np.linalg.det(Vor))
    print()


def get_difference_and_discrepancy(a, b):
    difference = abs(len(a) - len(b))
    max_len = max([len(a), len(b)])
    discrepancy = difference / max_len
    return max_len, difference, discrepancy


def get_the_extreme_reference_point(points, reference, threshold=0.95):
    dm = scipy.spatial.distance_matrix(points, reference)
    print("dm.max() %s" % (dm.max()))
    print("dm.shape", dm.shape)
    extreme_points = points[(dm > threshold * dm.max()).flatten()]
    print("len(extreme_points)", len(extreme_points))
    return np.median(extreme_points, axis=0)


def get_better_aligned_ab(points_a, points_b, margin=0.25):
    a_center = np.mean(points_a, axis=0)
    b_center = np.mean(points_b, axis=0)

    print(
        "a_center %s, b_center %s, shift %s, distance %s"
        % (
            str(a_center),
            str(b_center),
            str(a_center - b_center),
            str(np.linalg.norm(a_center - b_center, ord=2)),
        )
    )

    a_extreme = get_the_extreme_reference_point(points_a, np.expand_dims(a_center, 0))
    b_extreme = get_the_extreme_reference_point(points_b, np.expand_dims(b_center, 0))

    print("a_extreme %s, b_extreme %s" % (str(a_extreme), str(b_extreme)))
    a_span = np.linalg.norm(a_extreme - a_center)
    b_span = np.linalg.norm(b_extreme - b_center)
    if a_span > b_span:
        span = b_span
    else:
        span = a_span
    print("span", span)
    span *= margin

    a_dm = scipy.spatial.distance_matrix(points_a, np.expand_dims(a_extreme, 0))
    b_dm = scipy.spatial.distance_matrix(points_b, np.expand_dims(b_extreme, 0))

    ar_indices = (a_dm <= span).flatten()
    br_indices = (b_dm <= span).flatten()
    print("len(ar_indices)", len(ar_indices))
    print("len(br_indices)", len(br_indices))
    ar = points_a[ar_indices]
    br = points_b[br_indices]

    return ar, br, ar_indices, br_indices


def get_aligned_ab(points_a, points_b, axis=0, margin=0.2):
    a_len = len(points_a)
    a_min = np.min(points_a, 0)
    a_max = np.max(points_a, 0)
    a_range = a_max - a_min

    b_len = len(points_b)
    b_min = np.min(points_b, 0)
    b_max = np.max(points_b, 0)
    b_range = b_max - b_min

    if a_len > b_len:
        range_s = b_range
    else:
        range_s = a_range

    print(
        "original max_points %d, #points difference %d and discrepancy %.4f"
        % get_difference_and_discrepancy(points_a, points_b)
    )
    print("a_range", a_range)
    print("a_min", a_min)
    print("a_max", a_max)
    print("b_min", b_min)
    print("b_max", b_max)
    print("b_range", b_range)
    print("range", range_s)
    print("range_discrepancy", np.abs(a_range - b_range) / range_s)
    allowed_range = range_s * (1.0 - margin)
    print("allowed_range", allowed_range)
    a_limit = a_max - allowed_range
    b_limit = b_max - allowed_range
    print("a_limit %s, b_limit %s" % (a_limit, b_limit))
    ar = points_a[
        np.logical_and(
            np.logical_and(points_a[:, 0] > a_limit[0], points_a[:, 1] > a_limit[1]),
            points_a[:, 2] > a_limit[2],
        )
    ]
    br = points_b[
        np.logical_and(
            np.logical_and(points_b[:, 0] > b_limit[0], points_b[:, 1] > b_limit[1]),
            points_b[:, 2] > b_limit[2],
        )
    ]
    print(
        "after restriction max_points %d, # difference %d and discrepancy %.4f"
        % get_difference_and_discrepancy(ar, br)
    )
    print()

    return ar, br


def get_origin_from_parameters(parameters):
    p = parameters[b"position"]
    o = np.array([p[b"CentringX"], p[b"CentringY"], p[b"AlignmentY"]])
    return o


def get_calibration_from_parameters(parameters):
    calibration = np.ones((3,))
    c = parameters[b"calibration"]
    calibration[1], calibration[2] = c[0], c[0]
    calibration[0] = c[1]
    return calibration


def get_transformed_clouds(pcd_a, pcd_b, pcd_ab, hull=False):
    pcd_at, T_at, center_at = get_transformed_pcd(pcd_a, hull=hull)
    pcd_bt, T_bt, center_bt = get_transformed_pcd(pcd_b, hull=hull)
    pcd_btf, T_btf, center_btf = get_transformed_pcd(pcd_b, hull=hull, flip=True)
    pcd_abt, T_abt, center_abt = get_transformed_pcd(pcd_ab, hull=hull)

    bt_colinear = np.trace(np.dot(T_bt.T, T_abt))
    btf_colinear = np.trace(np.dot(T_btf.T, T_abt))

    if btf_colinear > bt_colinear:
        pcd_bt = pcd_btf
        T_bt = T_btf
        center_bt = center_btf
    return pcd_at, T_at, center_at, pcd_bt, T_bt, center_bt, pcd_abt, T_abt, center_abt


def align(
    alignment_a,
    alignment_b,
    margin=0.25,
    center=np.array([160, 256, 256]),
    directions=np.array([1, 1, -1]),
    order=[1, 2, 0],
    origin_a=None,
    origin_b=None,
    calibration_a=None,
    calibration_b=None,
    kappa_a=None,
    kappa_b=None,
    phi_a=None,
    phi_b=None,
    a_color=[1, 0.706, 0],
    b_color=[0, 0.706, 1],
    ab_color=[1, 0, 0.706],
    cryst_color=[0.706, 0, 1],
    crystal=None,
):
    g = goniometer()

    if origin_a is None:
        parameters_a = get_parameters(alignment_a)
        parameters_b = get_parameters(alignment_b)
    if type(alignment_a) is not o3d.cuda.pybind.geometry.PointCloud:
        pcd_a = get_pcd(alignment_a)
    else:
        pcd_a = alignment_a
    if type(alignment_b) is not o3d.cuda.pybind.geometry.PointCloud:
        pcd_b = get_pcd(alignment_b)
    else:
        pcd_b = alignment_b

    if crystal is not None:
        co = np.asarray(crystal.points)
    a = pcd_a.points
    b = pcd_b.points
    ao = np.asarray(a)
    bo = np.asarray(b)

    if origin_a is None:
        origin_a = get_origin_from_parameters(parameters_a)
        origin_b = get_origin_from_parameters(parameters_b)
        calibration_a = get_calibration_from_parameters(parameters_a)
        calibration_b = get_calibration_from_parameters(parameters_b)

    print("a.shape", ao.shape)
    print("b.shape", bo.shape)

    # ao = ao[ao[:,0]>40]
    # bo = bo[bo[:,0]>40]
    if kappa_a is None:
        center_a = [160] + list(np.mean(ao, axis=0)[1:])
        center_b = [160] + list(np.mean(bo, axis=0)[1:])
        print("center_a", center_a)
        print("center_b", center_b)
        a_mm = g.get_points_in_goniometer_frame(
            ao,
            calibration_a,
            origin_a[:3],
            center=center_a,
            directions=directions,
            order=order,
        )
        b_mm = g.get_points_in_goniometer_frame(
            bo,
            calibration_b,
            origin_b[:3],
            center=center_b,
            directions=directions,
            order=order,
        )
    else:
        a_mm = ao[:] * np.array([-1, 1, 1])
        b_mm = bo[:] * np.array([-1, 1, 1])
        c_mm = co[:] * np.array([-1, 1, 1])
    # a_mm = ((ao-center)*calibration_a*directions)[:, order] + origin_a
    # b_mm = ((bo-center)*calibration_b*directions)[:, order] + origin_b
    amean = a_mm.mean(axis=0)
    bmean = b_mm.mean(axis=0)
    print("a_mm.mean", amean)
    print("b_mm.mean", bmean)
    shift = amean - bmean
    print("shift:", shift)
    print("distance", np.linalg.norm(shift))

    pcd_a_mm = get_pcd_from_points(a_mm)
    pcd_b_mm = get_pcd_from_points(b_mm)
    pcd_cryst_mm = get_pcd_from_points(c_mm)
    o3d.visualization.draw_geometries(
        [
            pcd_a_mm.paint_uniform_color(a_color),
            pcd_b_mm.paint_uniform_color(b_color),
            pcd_cryst_mm.paint_uniform_color(cryst_color),
        ],
        window_name="original",
    )

    if kappa_a is None:
        kappa_a = parameters_a[b"kappa"]
        kappa_b = parameters_b[b"kappa"]
        phi_a = parameters_a[b"phi"]
        phi_b = parameters_b[b"phi"]

    ab_mm = g.get_shift(kappa_a, phi_a, a_mm - amean, kappa_b, phi_b)
    cryst_ab_mm = g.get_shift(kappa_a, phi_a, c_mm - amean, kappa_b, phi_b)
    ab_mm += bmean
    cryst_ab_mm += bmean
    pcd_ab_mm = get_pcd_from_points(ab_mm)
    pcd_cryst_ab_mm = get_pcd_from_points(cryst_ab_mm)

    o3d.visualization.draw_geometries(
        [
            pcd_a_mm.paint_uniform_color(a_color),
            pcd_b_mm.paint_uniform_color(b_color),
            pcd_ab_mm.paint_uniform_color(ab_color),
            pcd_cryst_ab_mm.paint_uniform_color(cryst_color),
        ],
        window_name="orginal with estimated transform",
    )

    abo = np.asarray(pcd_ab_mm.points)
    abomean = abo.mean(axis=0)
    shift = bmean - abomean
    print("calibration shift", shift)
    print("calibration distance", np.linalg.norm(shift))

    o3d.visualization.draw_geometries(
        [
            pcd_b_mm.paint_uniform_color(b_color),
            pcd_ab_mm.paint_uniform_color(ab_color),
        ],
        window_name="estimated transform and the real thing",
    )

    (
        pcd_at,
        T_at,
        center_at,
        pcd_bt,
        T_bt,
        center_bt,
        pcd_abt,
        T_abt,
        center_abt,
    ) = get_transformed_clouds(pcd_a_mm, pcd_b_mm, pcd_ab_mm)

    at = np.asarray(pcd_abt.points)
    bt = np.asarray(pcd_bt.points)
    abt = np.asarray(pcd_abt.points)

    o3d.visualization.draw_geometries(
        [pcd_abt.paint_uniform_color(ab_color), pcd_bt.paint_uniform_color(b_color)],
        window_name="transformed",
    )

    shift = np.mean(at, axis=0) - np.mean(bt, axis=0)
    print("transformed shift", shift)
    print("transformed distance", np.linalg.norm(shift))

    alignment = np.dot(T_bt.T, T_abt)
    print("initial alignment")
    print(alignment)
    alignment_trace = np.trace(alignment)
    print("initial alignment trace", alignment_trace)
    print()

    # ar, br = get_aligned_ab(att, btt, margin=margin)
    # ar, br, ar_indices, br_indices = get_better_aligned_ab(at, bt)

    # pcd_ar = get_pcd_from_points(a_mm[ar_indices])
    # pcd_br = get_pcd_from_points(b_mm[br_indices])
    # pcd_abr = get_pcd_from_points(ab_mm[ar_indices])

    # pcd_art, T_art, center_art, pcd_brt, T_brt, center_brt, pcd_abrt, T_abrt, center_abrt = get_transformed_clouds(pcd_ar, pcd_br, pcd_abr)

    # at = np.asarray(pcd_abrt.points)
    # bt = np.asarray(pcd_brt.points)
    # abt = np.asarray(pcd_abrt.points)

    # o3d.visualization.draw_geometries([pcd_brt.paint_uniform_color(b_color), pcd_abrt.paint_uniform_color(ab_color)], window_name='transformed after trying to address visual field mismatch')
    # shift = np.mean(at, axis=0) - np.mean(abt, axis=0)
    # print('transformed after addressing vfm shift', shift)
    # print('transformed after addressing vfm distance', np.linalg.norm(shift))

    # alignment = np.dot(T_brt.T, T_abrt)
    # print('visual match alignment')
    # print(alignment)
    # alignment_trace = np.trace(alignment)
    # print('visual match alignment trace', alignment_trace)
    # print()

    ##att_all = np.dot(np.dot(ab_mm - center_abrt, T_abrt) - center_abrt, T_abrt)
    ##btt_all = np.dot(np.dot(b_mm - center_brt, T_brt) - center_brt, T_brt)
    # att_all = np.dot(ab_mm - center_abrt, T_abrt)
    # btt_all = np.dot(b_mm - center_brt, T_brt)
    # print('T_abrt')
    # print(T_abrt)
    # print('T_brt')
    # print(T_brt)

    att_all = np.dot(a_mm - center_at, T_at)
    btt_all = np.dot(b_mm - center_bt, T_bt)
    cryst_all = np.dot(c_mm - center_at, T_at)

    shift = np.mean(att_all, axis=0) - np.mean(btt_all, axis=0)
    print("transformed all points shift", shift)
    print("transformed after addressing vfm distance", np.linalg.norm(shift))
    pcd_att_all = get_pcd_from_points(att_all)
    pcd_btt_all = get_pcd_from_points(btt_all)
    pcd_cryst_all = get_pcd_from_points(cryst_all)
    o3d.visualization.draw_geometries(
        [
            pcd_att_all.paint_uniform_color(a_color),
            pcd_btt_all.paint_uniform_color(b_color),
            pcd_cryst_all.paint_uniform_color(cryst_color),
        ],
        window_name="aligned, all points",
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-a",
        "--alignment_a",
        default="/nfs/data2/Martin/Research/murko/minikappa_calibration/kappa_95_phi_50_z1",
        type=str,
        help="alignment a",
    )
    # parser.add_argument('-b', '--alignment_b', default='minikappa_calibration/kappa_35_phi_125_z1', type=str, help='alignment b')
    # parser.add_argument('-a', '--alignment_a', default='minikappa_calibration/kappa_60_phi_335_z1', type=str, help='alignment a')
    parser.add_argument(
        "-b",
        "--alignment_b",
        default="/nfs/data2/Martin/Research/murko/minikappa_calibration/kappa_165_phi_100_z1",
        type=str,
        help="alignment b",
    )

    # parser.add_argument('-c', '--crystal', default='/nfs/data4/2023_Run3/com-proxima2a/2023-06-02/RAW_DATA/Martin/px2-0007/pos9/pos9_p2r_d_xds.pcd', type=str, help='diffracting volume')
    args = parser.parse_args()
    print(args)

    try:
        m_a = h5py.File(args.alignment_a, "r")
        origin_a = get_origin(m_a)[:3]
        calibration_a = get_calibration(m_a)
        pcd_a = o3d.io.read_point_cloud(args.alignment_a.replace(".h5", ".pcd"))
        kappa_a, phi_a = get_kappa_phi(m_a)
        m_b = h5py.File(args.alignment_b, "r")
        origin_b = get_origin(m_b)[:3]
        calibration_b = get_calibration(m_b)
        pcd_b = o3d.io.read_point_cloud(args.alignment_b.replace(".h5", ".pcd"))
        kappa_b, phi_b = get_kappa_phi(m_b)
        print(
            pcd_a,
            pcd_b,
            origin_a,
            origin_b,
            calibration_a,
            calibration_b,
            kappa_a,
            kappa_b,
            phi_a,
            phi_b,
        )
        pcd_cryst = o3d.io.read_point_cloud(args.crystal)
        align(
            pcd_a,
            pcd_b,
            origin_a=origin_a,
            origin_b=origin_b,
            calibration_a=calibration_a,
            calibration_b=calibration_b,
            kappa_a=kappa_a,
            kappa_b=kappa_b,
            phi_a=phi_a,
            phi_b=phi_b,
            crystal=pcd_cryst,
        )
    except:
        traceback.print_exc()
        align(args.alignment_a, args.alignment_b)
