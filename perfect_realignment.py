#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import time
import logging
import pickle
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
import open3d as o3d
from sklearn.cluster import k_means

try:
    from probreg import cpd, filterreg
    from probreg import math_utils as mu
except:
    cpd = None
    filterreg = None
    mu = None
try:
    import point_cloud_utils as pcu
except:
    pcu = None
from pprint import pprint
import copy
import pylab

from useful_routines import (
    get_position_from_vector,
    get_vector_from_position
)
from colors import (
    yellow,
    blue,
    green,
    magenta,
    cyan,
    purpre,
)

max_iteration = 50
threshold = 0.001

def get_surface(
    pcd,
    surface_threshold=14,
    crust_threshold=18,
    include_crust=True,
    compute_normals=True,
):
    distance = np.median(pcd.compute_nearest_neighbor_distance())
    cube = np.sqrt(3 * (distance) ** 2)
    points = np.asarray(pcd.points)
    interior, ipi = pcd.remove_radius_outlier(
        surface_threshold, cube, print_progress=False
    )
    surface_points = np.delete(points, ipi, axis=0)
    surface = get_pcd_from_points(surface_points, normals=False)
    if compute_normals:
        crust_points = 0
        if include_crust:
            cinterior, cipi = pcd.remove_radius_outlier(
                crust_threshold, cube, print_progress=False
            )
            crust_points = np.delete(points, cipi, axis=0)
            interior = cinterior
        normals = get_surface_normals(surface, crust_points, interior)
        surface.normals = o3d.utility.Vector3dVector(normals)
    return surface


def get_critical_points(
    analysis,
    critical_points=[
        "extreme",
        "most_likely_click",
        "start_likely",
        "end_likely",
        "start_possible",
    ],
    keys=["CentringX", "CentringY", "AlignmentY"],
):
    cp = []
    ap = analysis["aligned_positions"]
    for critical_point in critical_points:
        p = get_vector_from_position(ap[critical_point], keys=keys)
        cp.append(p)

    return np.array(cp)


def get_transformed_positions(
    positions, source, target, keys=["CentringX", "CentringY", "AlignmentY"]
):
    r, t_extreme, sO, sC, tO, tC = perfect_realignment(source, target)

    s_points = [get_vector_from_position(position, keys=keys) for position in positions]

    t_points = find_points_from_S_in_T(s_points, sO, sC, tO, tC)

    transformed_positions = [
        get_position_from_vector(t_point, keys=keys) for t_point in t_points
    ]

    return transformed_positions


def get_transformed_points(
    points,
    source,
    target,
):
    r, t_extreme, sO, sC, tO, tC = perfect_realignment(source, target)
    t_points = find_points_from_S_in_T(s_points, sO, sC, tO, tC)

    return t_points


def find_points_from_S_in_T(s_points, sO, sC, tO, tC):
    t_points = s_points - sC
    t_points = np.dot(t_points, sO)
    t_points = np.dot(t_points, np.linalg.inv(tO))
    t_points = t_points + tC
    return t_points


def rotate_points(points, R, center):
    points -= center
    roints = np.dot(points, R)
    roints += center
    return roints


def rotate_pcd(pcd, R, center):
    rpcd = copy.deepcopy(pcd)
    rpcd.rotate(R.T, center)
    return rpcd


def get_pcd_in_eigenspace(pcd):
    rpcd = copy.deepcopy(pcd)
    pO, pC, pW = get_oriented_axes(pcd)
    rpcd.rotate(pO.T, pC)
    return rpcd


def get_pcd_rotated_at_point_along_axis(
    pcd, angle, point=None, axis=np.array([0, 0, 1]), degrees=False
):
    rpcd = copy.deepcopy(pcd)
    r = Rotation.from_rotvec(angle * axis, degrees=degrees)
    if point is None:
        point = rpcd.get_center()
    rpcd.rotate(r.as_matrix(), point)
    return rpcd


def check_hand(S):
    return np.dot(np.cross(S[:, 0], S[:, 1]), S[:, 2])


def understand_rotation(spcd, tpcd, sr, tr, shift=0.001):
    sO, sC, sW = get_oriented_axes(spcd)
    tO, tC, tW = get_oriented_axes(tpcd)

    scp = get_critical_points(sr)
    tcp = get_critical_points(tr)

    scp_pcd = get_pcd_from_points(scp)
    tcp_pcd = get_pcd_from_points(tcp)
    scp_pcd.paint_uniform_color(magenta)
    tcp_pcd.paint_uniform_color(green)

    spcd_rotated = rotate_pcd(spcd, sO, sC)
    tpcd_rotated = rotate_pcd(tpcd, tO, tC)

    scp_pcd_rotated = rotate_pcd(scp_pcd, sO, sC)
    tcp_pcd_rotated = rotate_pcd(tcp_pcd, tO, tC)

    scp_points_rotated = rotate_points(scp, sO, sC) + shift
    tcp_points_rotated = rotate_points(tcp, tO, tC) + shift

    scp_points_rotated_pcd = get_pcd_from_points(scp_points_rotated)
    tcp_points_rotated_pcd = get_pcd_from_points(tcp_points_rotated)

    scp_points_rotated_pcd.paint_uniform_color(unknown1)
    tcp_points_rotated_pcd.paint_uniform_color(unknown2)

    o3d.visualization.draw_geometries(
        [
            spcd_rotated,
            tpcd_rotated,
            scp_pcd_rotated,
            tcp_pcd_rotated,
            scp_points_rotated_pcd,
            tcp_points_rotated_pcd,
        ]
    )


# def get_likely_part(pcd, analysis):
# points = np.asarray(pcd.points)
## start_likely = analysis['aligned_positions']['start_likely']
# cp = get_critical_points(analysis)
# indices = np.argwhere(points[:, 2] <= cp[2][2])  # start_likely['AlignmentY'])
# likely_part = pcd.select_by_index(indices)
# pO, pC, pW = get_oriented_axes(likely_part)

# rpcd = rotate_pcd(pcd, pO, pC)
# rcp = rotate_points(cp, pO, pC)

# rpoints = np.asarray(rpcd.points)
# indices = np.argwhere(rpoints[:, 0] >= rcp[2][0])
# likely_part = pcd.select_by_index(indices)

# return likely_part


def get_likely_part(pcd, analysis, min_number_of_points=100):
    points = np.asarray(pcd.points)
    # start_likely = analysis['aligned_positions']['start_likely']
    cp = get_critical_points(analysis)
    indices = np.argwhere(points[:, 2] <= cp[2][2])  # start_likely['AlignmentY'])
    # likely_part = pcd.select_by_index(indices)
    # pO, pC, pW = get_oriented_axes(likely_part)

    # rpcd = rotate_pcd(pcd, pO, pC)
    # rcp = rotate_points(cp, pO, pC)

    # rpoints = np.asarray(rpcd.points)
    # indices = np.argwhere(rpoints[:, 0] >= rcp[2][0])
    print("indices", indices, len(indices))
    if len(indices) > min_number_of_points:
        likely_part = pcd.select_by_index(indices)
    else:
        likely_part = copy.copy(pcd)
    print("likely_part", likely_part, likely_part.points)

    return likely_part


def get_extreme_from_pcd(pcd, axis=0, percentile=0.99):
    points = np.asarray(pcd.points)
    mobb = pcd.get_minimal_oriented_bounding_box()
    extreme_point = points[np.argmax(points[:, axis])]
    zero = extreme_point[axis] - mobb.extent[axis]
    threshold = zero + percentile * mobb.extent[axis]

    extreme_indices = np.argwhere(points[:, axis] > threshold)
    extreme_points = points[extreme_indices]
    extreme = np.mean(extreme_points, axis=0)
    return extreme[0]


def get_both_extremes_from_pcd(pcd, axis=0, percentile=0.99, eigenbasis=True):
    if eigenbasis:
        O, C, W = get_oriented_axes(pcd)
        pcd = rotate_pcd(pcd, O, C)

    points = np.asarray(pcd.points)
    bb = pcd.get_axis_aligned_bounding_box()
    extent = bb.get_extent()
    max_point = points[np.argmax(points[:, axis])]
    min_point = points[np.argmin(points[:, axis])]
    zero = max_point[axis] - extent[axis]
    threshold_max = zero + percentile * extent[axis]
    threshold_min = zero + (1 - percentile) * extent[axis]

    max_indices = np.argwhere(points[:, axis] > threshold_max)
    max_points = points[max_indices]
    min_indices = np.argwhere(points[:, axis] < threshold_min)
    min_points = points[min_indices]

    maxp = np.mean(max_points, axis=0)[0]
    minp = np.mean(min_points, axis=0)[0]
    if eigenbasis:
        maxp = rotate_points([maxp], np.linalg.inv(O), C)[0]
        minp = rotate_points([minp], np.linalg.inv(O), C)[0]

    return maxp, minp


## minimal bounding boxes for whole source and target
# tsaabb = espcd.get_axis_aligned_bounding_box()
# ttaabb = etpcd.get_axis_aligned_bounding_box()
## maximal acceptable extent
# total_conservative_extent = percentile*min(tsaabb.get_extent()[axis], ttaabb.get_extent()[axis])
# espcd = rotate_pcd(spcd, sO, sC)
# etpcd = rotate_pcd(tpcd, tO, tC)
# espcd.translate(-sC)
# etpcd.translate(-tC)
# rotate to eigenbasis

#####################################
# slr = rotate_pcd(sl, sO, sC)
# tlr = rotate_pcd(tl, tO, tC)
# slr.translate(-sC)
# tlr.translate(-tC)

## determine extreme point
# sle = get_extreme_from_pcd(slr)
# tle = get_extreme_from_pcd(tlr)

# sle += sC
# tle += tC
# seo = rotate_points([sle], np.linalg.inv(sO), sC)
# teo = rotate_points([tle], np.linalg.inv(tO), tC)
# t_extreme = (teo - seo)[0]

## minimal bounding boxes for extent determination
# saabb = slr.get_axis_aligned_bounding_box()
# taabb = tlr.get_axis_aligned_bounding_box()
## maximal acceptable extent
# conservative_extent = percentile*min(saabb.get_extent()[axis], taabb.get_extent()[axis])

## points for convenience
# sp = np.array(spcd.points)
# tp = np.array(tpcd.points)

## distances
# sd = sp - seo
# td = tp - teo

## accepted indices
# sai = np.argwhere(np.linalg.norm(sd, axis=1) <= conservative_extent)
# tai = np.argwhere(np.linalg.norm(td, axis=1) <= conservative_extent)


## matched subsets
# sm = spcd.select_by_index(sai)
# tm = tpcd.select_by_index(tai)
######################################
def get_matched_subsets(sl, tl, spcd, tpcd, sO, sC, tO, tC, axis=0, percentile=0.8):
    # rotate to eigenbasis
    slr = rotate_pcd(sl, sO, sC)
    tlr = rotate_pcd(tl, tO, tC)
    slr.translate(-sC)
    tlr.translate(-tC)

    # determine extreme point
    sle = get_extreme_from_pcd(slr)
    tle = get_extreme_from_pcd(tlr)

    sle += sC
    tle += tC
    seo = rotate_points([sle], np.linalg.inv(sO), sC)
    teo = rotate_points([tle], np.linalg.inv(tO), tC)
    # t_extreme = (teo - seo)[0]

    # minimal bounding boxes for extent determination
    saabb = slr.get_axis_aligned_bounding_box()
    taabb = tlr.get_axis_aligned_bounding_box()
    # maximal acceptable extent
    conservative_extent = percentile * min(
        saabb.get_extent()[axis], taabb.get_extent()[axis]
    )

    # points for convenience
    sp = np.array(spcd.points)
    tp = np.array(tpcd.points)

    # distances
    sd = sp - seo
    td = tp - teo

    # accepted indices
    sai = np.argwhere(np.linalg.norm(sd, axis=1) <= conservative_extent)
    tai = np.argwhere(np.linalg.norm(td, axis=1) <= conservative_extent)

    # matched subsets
    sm = spcd.select_by_index(sai)
    tm = tpcd.select_by_index(tai)

    return sm, tm


def get_rotation_and_translation(
    spcd, tpcd, sr, tr, axis=0, percentile=0.8, verbose=False, display=False
):
    if verbose:
        o3d.visualization.draw_geometries(
            [spcd, tpcd],
            window_name="inputs",
        )

    # get likely parts
    sl = get_likely_part(spcd, sr)
    tl = get_likely_part(tpcd, tr)
    # sl = copy.deepcopy(spcd)
    # tl = copy.deepcopy(tpcd)
    if verbose:
        o3d.visualization.draw_geometries(
            [sl, tl, get_coordinate_frame(sl), get_coordinate_frame(tl)],
            window_name="likely parts",
        )

    # calculate eigenbasis
    sO, sC, sW = get_oriented_axes(sl)
    tO, tC, tW = get_oriented_axes(tl)
    if verbose:
        print(f"check hand sO {check_hand(sO)}")
        print(f"check hand tO {check_hand(tO)}")

        print(f"sO")
        print(f"{sO}")
        print(f"tO")
        print(f"{tO}")

        print(f"sO*tO")
        print(f"{np.dot(sO.T, tO)}")

        print(f"sC {sC}")
        print(f"tC {tC}")

    # weights = np.mean([sW, tW], axis=0)
    # r, e = Rotation.align_vectors(sO, tO, weights=weights)

    sm, tm = get_matched_subsets(
        sl, tl, spcd, tpcd, sO, sC, tO, tC, axis=axis, percentile=percentile
    )

    if verbose:
        o3d.visualization.draw_geometries(
            [sm, tm], window_name="selected matched parts"
        )

    sO, sC, sW = get_oriented_axes(sm)
    tO, tC, tW = get_oriented_axes(tm)
    ctbo_display = verbose and display
    spcdm, tpcdm = get_matched_subsets(
        spcd, tpcd, spcd, tpcd, sO, sC, tO, tC, axis=axis, percentile=percentile
    )
    if verbose:
        o3d.visualization.draw_geometries(
            [spcdm, tpcdm], window_name="selected matched from original pcd"
        )

    tO, rmse = get_consistent_target_basis_orientation(
        spcdm, tpcdm, sO, tO, sC, tC, display=ctbo_display
    )
    if display:
        spcdr = rotate_pcd(spcd, sO, sC)
        spcdr.translate(-sC)
        tpcdr = rotate_pcd(tpcd, tO, tC)
        tpcdr.translate(-tC)
        print(f"alignment errror: {rmse}")
        o3d.visualization.draw_geometries(
            [spcdr, tpcdr], window_name="aligned source and target permutation"
        )

    weights = np.mean([sW, tW], axis=0)
    r, e = Rotation.align_vectors(sO, tO, weights=weights)

    smr = rotate_pcd(sm, sO, sC)
    tmr = rotate_pcd(tm, tO, tC)
    smr.translate(-sC)
    tmr.translate(-tC)

    # determine extreme point again
    sle = get_extreme_from_pcd(smr)
    tle = get_extreme_from_pcd(tmr)
    if verbose:
        o3d.visualization.draw_geometries(
            [
                smr,
                tmr,
                get_coordinate_frame(smr),
                get_coordinate_frame(tmr),
                get_graphical_point(sle),
                get_graphical_point(tle),
            ],
            window_name="matched parts in eigenbases",
        )

    sle += sC
    tle += tC

    seo = rotate_points([sle], np.linalg.inv(sO), sC)
    teo = rotate_points([tle], np.linalg.inv(tO), tC)
    t_extreme = (teo - seo)[0]

    print(f'estimated rotation {r.as_euler("xzy", degrees=True)} error {e}')
    print(f"estimated shift from extreme: {t_extreme}")
    print(f"estimated shift from sC and tC: {tC - sC}")
    if verbose:
        print(f"sC and tC: {sC}, {tC}")
    return r, t_extreme, sO, sC, tO, tC

    # o3d.visualization.draw_geometries([sl_rotated, tl_rotated])


def get_consistent_target_basis_orientation(
    source,
    target,
    sO,
    tO,
    sC,
    tC,
    display=False,
    directions=[(1, 1, 1), (1, 1, -1), (1, -1, -1), (1, -1, 1)],
):
    spcdr = rotate_pcd(source, sO, sC)
    spcdr.translate(-sC)
    target_tree = cKDTree(np.asarray(spcdr.points), leafsize=10)
    best_rmse = np.inf
    best_tO = None
    for d, direction in enumerate(directions):
        tOi = copy.deepcopy(tO)
        for k, m in enumerate(direction):
            tOi[:, k] *= m
        tpcdr = rotate_pcd(target, tOi, tC)
        tpcdr.translate(-tC)
        # if mu is not None:
        rmse = mu.compute_rmse(np.asarray(tpcdr.points), target_tree)

        if display:
            print(f"rmse for direction {d}: {rmse}")
            o3d.visualization.draw_geometries(
                [spcdr, tpcdr],
                window_name="aligned source and target permutation %d" % d,
            )
        if rmse < best_rmse:
            best_rmse = rmse
            best_tO = copy.deepcopy(tOi)

    return best_tO, best_rmse
    # tpcdr = rotate_pcd(tpcd, tO, tC)

    # spcdr.translate(-sC)
    # tpcdr.translate(-tC)

    ##tpcdr.translate(-t_extreme)
    # target_tree = cKDTree(np.asarray(spcdr.points), leafsize=10)
    # rmse_direct = mu.compute_rmse(np.asarray(tpcdr.points), target_tree)
    # print(f'rmse_direct {rmse_direct}')

    # if display:
    # o3d.visualization.draw_geometries([spcdr, tpcdr], window_name='aligned source and target')

    # tOi = copy.deepcopy(tO)
    # tOi[:, 2] *= -1
    # tpcdr_i = rotate_pcd(tpcd, tOi, tC)
    # tpcdr_i.translate(-tC)

    # rmse_inverse = mu.compute_rmse(np.asarray(tpcdr_i.points), target_tree)
    # print(f'rmse_invers {rmse_inverse}')
    # if display:
    # o3d.visualization.draw_geometries([spcdr, tpcdr_i], window_name='aligned source and inverse target')

    # tOi [:, 1] *= -1
    # tpcdr_ii = rotate_pcd(tpcd, tOi, tC)
    # tpcdr_ii.translate(-tC)
    # rmse_inverse = mu.compute_rmse(np.asarray(tpcdr_ii.points), target_tree)
    # print(f'rmse_inver2 {rmse_inverse}')
    # if display:
    # o3d.visualization.draw_geometries([spcdr, tpcdr_ii], window_name='aligned source and iinverse target')

    # tOi [:, 2] *= -1
    # tpcdr_iii = rotate_pcd(tpcd, tOi, tC)
    # tpcdr_iii.translate(-tC)
    # rmse_inverse = mu.compute_rmse(np.asarray(tpcdr_iii.points), target_tree)
    # print(f'rmse_inver3 {rmse_inverse}')
    # if display:
    # o3d.visualization.draw_geometries([spcdr, tpcdr_iii], window_name='aligned source and iiinverse target')


def get_vertex_normals_from_mesh_and_pcd(mesh, pcd, neighbors=27, normalize=True):
    vertices = np.asarray(mesh.vertices)
    points = np.asarray(pcd.points)
    pcd_tree = scipy.spatial.cKDTree(points)
    vertex_normals = []
    for vertex in vertices:
        distances, indices = pcd_tree.query(vertex, k=neighbors)
        closest_points = points[indices]
        normal = vertex - np.mean(closest_points, axis=0)
        vertex_normals.append(normal)
    vertex_normals = np.array(vertex_normals)
    if normalize:
        vertex_normals /= np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    return vertex_normals


def get_triangle_normals_from_vertex_normals(mesh):
    vertex_normals = np.asarray(mesh.vertex_normals)
    triangles = np.asarray(mesh.triangles)
    triangle_normals = []
    for triangle in triangles:
        normal = np.mean(vertex_normals[triangle], axis=0)
        triangle_normals.append(normal)

    return np.array(triangle_normals)


def get_triangle_normals_from_mesh_and_pcd(mesh, pcd, neighbors=27, normalize=True):
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    points = np.asarray(pcd.points)
    pcd_tree = scipy.spatial.cKDTree(points)
    triangle_normals = []
    for face in faces:
        center = points[face].mean(axis=0)
        distances, indices = pcd_tree.query(center, k=neighbors)

        closest_points = points[indices]
        normal = center - np.mean(closest_points, axis=0)
        triangle_normals.append(normal)
    triangle_normals = np.array(triangle_normals)
    if normalize:
        triangle_normals /= np.linalg.norm(triangle_normals, axis=1, keepdims=True)
    return triangle_normals


def get_surface_mesh_from_volume(volume):
    helper_volume = np.zeros(tuple(np.array(volume.shape) + 2))

    helper_volume[1:-1, 1:-1, 1:-1] = volume
    v, f, n, c = marching_cubes(helper_volume, gradient_direction="descent")
    v -= 1

    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(v),
        o3d.utility.Vector3iVector(f),
    )
    # mesh.vertex_normals = o3d.utility.Vector3dVector(n)
    # mesh.compute_triangle_normals()

    return mesh


def get_surface_mesh_from_pcd_mm(pcd_mm, dims=(512, 512, 320)):
    points = np.asarray(pcd_mm.points)
    mini = points.min(axis=0)
    points0 = points - mini
    cal = []
    for k in range(points0.shape[1]):
        c = np.array(list(set(points0[:, k])))
        c.sort()
        step = np.median(c[1:] - c[:-1])
        cal.append(step)

    cal = np.array(cal)

    mi = (points0 / cal).astype(int)
    umi = np.ravel_multi_index(tuple(mi[:, k] for k in range(mi.shape[1])), dims=dims)
    volume = np.zeros(np.prod(dims), np.int8)
    volume[umi] = 1
    volume = np.reshape(volume, dims)

    mesh = get_surface_mesh_from_volume(volume)

    normals = get_normals_from_mesh_and_pcd(mesh, pcd_mm)
    vertices = np.asarray(mesh.vertices)
    vertices *= cal
    vertices += mini
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.vertex_normals = o3d.utility.Vector3dVector(normals)

    return mesh


def get_surface_from_local_density(pcd, surface_threshold=14, crust_threshold=18):
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    distance = np.median(pcd.compute_nearest_neighbor_distance())
    cube = np.sqrt(3 * (distance) ** 2)
    surface_points = []
    crust_points = []
    interior_points = []
    for point in pcd.points:
        nns = pcd_tree.search_radius_vector_3d(point, cube)
        if nns[0] <= surface_threshold:
            surface_points.append(point)
        elif nns[0] <= crust_threshold:
            crust_points.append(point)
        else:
            interior_points.append(point)
    surface = o3d.geometry.PointCloud()
    surface.points = o3d.utility.Vector3dVector(surface_points)
    crust = o3d.geometry.PointCloud()
    crust.points = o3d.utility.Vector3dVector(crust_points)
    interior = o3d.geometry.PointCloud()
    interior.points = o3d.utility.Vector3dVector(interior_points)
    return surface, crust, interior


def get_surface_normals(surface, crust, interior, neighbors=7, normalize=True):
    normals = []
    interior_tree = o3d.geometry.KDTreeFlann(interior)
    for surface_point in surface.points:
        interior_points = []
        l, closest, distance = interior_tree.search_knn_vector_3d(
            surface_point, neighbors
        )
        for c in closest:
            interior_points.append(interior.points[c])
        normal = surface_point - np.median(interior_points, axis=0)
        normals.append(normal)
    normals = np.array(normals)
    if normalize:
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return normals


def get_surface_from_pcd(pcd, normalize=True, threshold=14):
    surface, crust, interior = get_surface_from_local_density(
        pcd, surface_threshold=threshold
    )
    normals = get_surface_normals(surface, crust, interior, normalize=normalize)
    surface.normals = o3d.utility.Vector3dVector(normals)
    return surface


def get_mesh_from_pcd(pcd, threshold=14):
    surface, crust, interior = get_surface_from_local_density(
        pcd, surface_threshold=threshold
    )
    normals = get_surface_normals(surface, crust, interior)
    surface.normals = o3d.utility.Vector3dVector(normals)
    mesh = get_mesh_from_surface(surface)
    return mesh, normals


def get_mesh_from_surface(surface, factor=2**0.5):
    avg = np.median(surface.compute_nearest_neighbor_distance())
    r = factor * avg
    radii = [r, 2 * r, 3 * r]
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        surface, o3d.utility.DoubleVector(radii)
    )
    return mesh


def get_mesh_from_vertices_and_triangles(v, f):
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(f)
    )
    return mesh


def get_watertight_mesh(mesh, accuracy=1000):
    v = np.asarray(mesh.vertices)
    f = np.asarray(mesh.triangles)

    vw, fw = pcu.make_mesh_watertight(v, f, accuracy)
    meshw = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vw), o3d.utility.Vector3iVector(fw)
    )
    meshw.compute_vertex_normals()
    return meshw


def get_euler(R):
    r = Rotation.from_matrix(R)
    angles = r.as_euler("xyz", degrees=True)
    return angles


def get_R_from_euler(alpha=0, beta=0, gamma=0):
    R = Rotation.from_euler("xyz", [alpha, beta, gamma], degrees=True)
    return R.as_matrix()


def estimate_rotation_from_normals(
    source, target, n_decision=7, n_clusters=27, plot=True
):
    sn = np.asarray(source.normals)
    tn = np.asarray(target.normals)

    s_vectors, s_labels, sfit = k_means(sn, n_clusters)
    t_vectors, t_labels, tfit = k_means(tn, n_clusters)

    sh = np.histogram(s_labels, bins=n_clusters)
    th = np.histogram(t_labels, bins=n_clusters)

    ssort = np.argsort(sh[0])
    tsort = np.argsort(th[0])

    s_ordered = s_vectors[ssort[::-1]]
    t_ordered = t_vectors[tsort[::-1]]

    s_counts = sh[0][ssort[::-1]]
    t_counts = th[0][tsort[::-1]]

    weights = np.mean([s_counts, t_counts], axis=0)

    if plot:
        pylab.plot(s_counts, "-o", label="source")
        pylab.plot(t_counts, "-o", label="target")
        pylab.show()

    rotation, rssd = Rotation.align_vectors(
        s_ordered[:n_decision], t_ordered[:n_decision], weights=weights[:n_decision]
    )

    return rotation, rssd


def estimate_rotation_from_inertia(source, target):
    sO, sC, sW = get_oriented_axes(source)
    tO, tC, tW = get_oriented_axes(target)

    weights = np.mean([sW, tW], axis=0)
    rotation, rssd = Rotation.align_vectors(sO, tO, weights=weights)

    return rotation, rssd


def get_line_set(points, edges, colors=[yellow, blue, magenta]):
    points = o3d.utility.Vector3dVector(points)
    edges = o3d.utility.Vector2iVector(edges)
    line_set = o3d.geometry.LineSet(points=points, lines=edges)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


# def get_bbox_frame(bbox):

# points = [center, center + S[:,0]*es[0], center + S[:,1]*es[0], center + S[:,2]*es[0]]
# edges = [[0, 1], [0, 2], [0, 3]]
# colors = [yellow, blue, magenta]
# coordinate_frame = get_line_set(points, edges)

# return coordinate_frame


def get_coordinate_frame(pcd):
    inertia, evals, S, center = principal_axes(np.asarray(pcd.points))
    O = get_oriented_eigenvectors(S)
    mobb = pcd.get_minimal_oriented_bounding_box()
    extent = mobb.extent

    esum = evals.sum()
    esum /= extent.max()
    es = np.log(1 + evals) / esum
    es *= 100
    points = [
        center,
        center + O[:, 0] * es[0],
        center + O[:, 1] * es[1],
        center + O[:, 2] * es[2],
    ]
    edges = [[0, 1], [0, 2], [0, 3]]
    colors = [yellow, blue, magenta]
    coordinate_frame = get_line_set(points, edges, colors)

    return coordinate_frame


def principal_axes(array, verbose=False):
    # https://github.com/pierrepo/principal_axes/blob/master/principal_axes.py
    if array.shape[1] != 3:
        xyz = np.argwhere(array == 1)
    else:
        xyz = array[:, :]

    coord = np.array(xyz, float)
    center = np.mean(coord, 0)
    coord = coord - center
    inertia = np.dot(coord.transpose(), coord)
    e_values, e_vectors = np.linalg.eig(inertia)
    order = np.argsort(e_values)[::-1]
    eigenvalues = np.array(e_values[order])
    eigenvectors = np.array(e_vectors[:, order])
    if verbose:
        print("principal axes")
        print("intertia tensor")
        print(inertia)
        print("eigenvalues")
        print(eigenvalues)
        print("eigenvectors")
        print(eigenvectors)
    return inertia, eigenvalues, eigenvectors, center


def get_oriented_eigenvectors(S, direction=np.array([0, 0, -1])):
    O = copy.copy(S)
    if np.dot(O[:, 0], direction) < 0.0:
        O[:, 0] *= -1
    return O


def get_oriented_axes(pcd, reference=None):
    inertia, evals, S, center = principal_axes(np.asarray(pcd.points))
    O = get_oriented_eigenvectors(S)
    if reference is not None:
        sOtO = np.dot(reference.T, O)
        asOtO = np.abs(sOtO)
        for k in range(3):
            if sOtO[np.argmax(asOtO[:, k]), k] < 0:
                O[:, k] *= -1
    esum = evals.sum()
    weights = evals / esum
    return O, center, weights


def get_graphical_point(point, scale=0.1, color=(0.706, 0, 0.294)):
    s = scale * np.eye(3)
    a = point + s[:, 0]
    b = point + s[:, 1]
    c = point + s[:, 2]
    x = point - s[:, 0]
    y = point - s[:, 1]
    z = point - s[:, 2]
    points = [a, x, b, y, c, z]
    edges = [[0, 1], [2, 3], [4, 5]]

    point = get_line_set(points, edges, colors=[color for k in range(3)])
    return point


def get_transformation(tf):
    scale = tf.scale
    R = tf.rot
    t = tf.t
    logging.info("scale %s" % scale)
    logging.info("translation %s" % t)
    logging.info("rotation")
    logging.info("\n%s" % str(R))
    logging.info("euler %s " % get_euler(R))
    transformation = np.eye(4)
    transformation[:3, :3] = R
    transformation[:3, 3] = t
    logging.info("transformation matrix")
    pprint(np.round(transformation, 4))
    return transformation


def plot_results(
    source,
    target,
    result,
    source_color=yellow,
    target_color=blue,
    transformed_color=green,
):
    source.paint_uniform_color(source_color)
    target.paint_uniform_color(target_color)
    result.paint_uniform_color(transformed_color)

    o3d.visualization.draw_geometries([source, target, result])


def get_pcd_from_points(points, normals=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if normals is not None:
        try:
            pcd.normals = o3d.utility.Vector3dVector(normals)
        except:
            pcd.estimate_normals()
    return pcd


def get_analysis_results(fname):
    return pickle.load(open(fname, "rb"))


def perfect_realignment(source_pcd_name, target_pcd_name, verbose=False, display=False):
    pcd_source = o3d.io.read_point_cloud(source_pcd_name)
    pcd_target = o3d.io.read_point_cloud(target_pcd_name)

    source_analysis = get_analysis_results(
        source_pcd_name.replace("_mm.pcd", "_results.pickle")
    )
    target_analysis = get_analysis_results(
        target_pcd_name.replace("_mm.pcd", "_results.pickle")
    )

    pcd_source.paint_uniform_color(yellow)
    pcd_target.paint_uniform_color(blue)

    result = get_rotation_and_translation(
        pcd_source,
        pcd_target,
        source_analysis,
        target_analysis,
        verbose=verbose,
        display=display,
    )
    return result


def main():
    import argparse

    logging.basicConfig(
        format="%(asctime)s|%(module)s| %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-s",
        "--source",
        default="/nfs/data4/2023_Run5/Commissioning/optical_alignment/mkc3/k_0_p_0_zoom_1_mm_corrected.pcd",
        type=str,
        help="source",
    )
    parser.add_argument(
        "-t",
        "--target",
        default="/nfs/data4/2023_Run5/Commissioning/optical_alignment/mkc3/k_180_p_60_zoom_1_mm_corrected.pcd",
        type=str,
        help="target",
    )
    parser.add_argument("-m", "--method", type=str, default="cpd", help="method")
    parser.add_argument(
        "-w",
        "--watertight",
        action="store_true",
        help="if set will create watertight mesh",
    )
    parser.add_argument(
        "-n",
        "--normals",
        action="store_true",
        help="if set will determine rotation from surface normals distribution",
    )
    parser.add_argument(
        "-S",
        "--surface",
        action="store_true",
        help="if set will work with surface points instead of a complete point cloud",
    )
    parser.add_argument(
        "-D", "--display", action="store_true", help="if set will display the result"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="if set will show additional debug images and messages",
    )
    parser.add_argument(
        "-N", "--npoints", type=int, default=1000, help="Number of sample points"
    )
    args = parser.parse_args()

    pcd_source = o3d.io.read_point_cloud(args.source)
    pcd_target = o3d.io.read_point_cloud(args.target)

    # source_analysis = get_analysis_results(args.source.replace('_mm_corrected.pcd', '_results.pickle'))
    # target_analysis = get_analysis_results(args.target.replace('_mm_corrected.pcd', '_results.pickle'))

    source_analysis = get_analysis_results(
        args.source.replace("_mm.pcd", "_results.pickle")
    )
    target_analysis = get_analysis_results(
        args.target.replace("_mm.pcd", "_results.pickle")
    )

    pcd_source.paint_uniform_color(yellow)
    pcd_target.paint_uniform_color(blue)

    if args.surface:
        source = get_surface(pcd_source)
        target = get_surface(pcd_target)
    else:
        source = copy.deepcopy(pcd_source)
        target = copy.deepcopy(pcd_target)

    _start = time.time()
    result = get_rotation_and_translation(
        source,
        target,
        source_analysis,
        target_analysis,
        verbose=args.verbose,
        display=args.display,
    )

    _end = time.time()

    print("result")
    print(result)
    print(f"rotation and translation determined in {_end-_start:.4f} seconds")

    ##source_mesh, source_normals = get_mesh_from_pcd(pcd_source)
    ##target_mesh, target_normals = get_mesh_from_pcd(pcd_target)
    # r, e = estimate_rotation_from_normals(source, target, n_decision=3)
    # print('rotation N', r.as_euler('xyz', degrees=True))
    # print('error N', e)

    # r, e = estimate_rotation_from_inertia(source, target)
    # print('rotation I', r.as_euler('xyz', degrees=True))
    # print('error I', e)

    # o3d.visualization.draw_geometries([source, target, get_coordinate_frame(source), get_coordinate_frame(target)], line_width=5, point_show_normal=True)

    # o3d.visualization.draw([source, target, get_coordinate_frame(source), get_coordinate_frame(target)], line_width=5)

    sys.exit()

    if args.watertight:
        source_mesh = get_watertight_mesh(source_mesh)
        target_mesh = get_watertight_mesh(target_mesh)

    if args.normals:
        sample_source_normals = source_normals[
            np.random.randint(0, len(source_normals), args.npoints)
        ]
        sample_target_normals = target_normals[
            np.random.randint(0, len(target_normals), args.npoints)
        ]
        source = get_pcd_from_points(sample_source_normals)
        target = get_pcd_from_points(sample_target_normals)
    else:
        source = get_pcd_from_points(np.asarray(source_mesh.vertices))
        target = get_pcd_from_points(np.asarray(target_mesh.vertices))

    if args.method == "cpd":
        res = cpd.registration_cpd(source, target, maxiter=max_iteration, tol=threshold)
    elif args.method == "filterreg":
        res = filterreg.registration_filterreg(
            source, target, sigma2=None, maxiter=max_iteration, tol=threshold
        )
    elif args.method == "icp":
        res = o3d.pipelines.registration.registration_icp(
            source,
            target,
            0.5,
            np.identity(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=max_iteration
            ),
        )
        result = copy.deepcopy(pcd_source)
        result.transform(res.transformation)
        T = copy.deepcopy(res.transformation)
        R = T[:3, :3]
        logging.info("rotation")
        logging.info("\n%s" % str(R))
        logging.info("euler %s " % get_euler(R))
        logging.info("res %s" % str(res))

    if args.method in ["cpd", "filterreg"]:
        transformation = get_transformation(res[0])
        result = copy.deepcopy(pcd_source)
        result.transform(transformation)
    logging.info("rmse: %s " % mu.compute_rmse(np.asarray(result.points), target_tree))
    # result.points = res[0].transform(result.points)
    plot_results(source_mesh, target_mesh, result)


if __name__ == "__main__":
    main()
