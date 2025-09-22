#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import zmq
import pickle
import copy
import numpy as np
import open3d as o3d
from skimage.measure import marching_cubes

from goniometer import get_points_in_goniometer_frame

def get_points_mm(
    points_px,
    calibration,
    origin_vector,
    origin_index,
    directions=np.array([1, 1, 1]),
    order=[0, 1, 2],
):
    
    points_mm = get_points_in_goniometer_frame(
        points_px,
        calibration,
        origin_vector,
        center=origin_index,
        directions=directions,
        order=order,
    )
    points_mm = points_mm[:, [1, 2, 0]]
    
    return points_mm


def get_reconstruction(
    projections,
    angles,
    vertical_correction=0.0,
    horizontal_correction=0.0,
    volume_rows_factor=1,
    volume_cols_factor=1,
):

    detector_rows, detector_cols = projections[0].shape
    number_of_projections = len(projections)

    _projections = np.zeros((detector_rows, number_of_projections, detector_cols))
    
    for k, projection in enumerate(projections):
        _projections[:, k, :] = projection

    print("_projections.shape", _projections.shape)
    
    request = {
        "projections": _projections,
        "angles": np.deg2rad(angles),
        "detector_rows": detector_cols,
        "detector_cols": detector_rows,
        "vertical_correction": vertical_correction,
        "horizontal_correction": horizontal_correction,
        "volume_rows_factor": volume_rows_factor,
        "volume_cols_factor": volume_cols_factor,
    }

    reconstruction = _get_reconstruction(request)
    return reconstruction


def _get_reconstruction(request, port=89001, verbose=False):
    start = time.time()
    context = zmq.Context()
    if verbose:
        print("Connecting to server ...")
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:%d" % port)
    socket.send(pickle.dumps(request))
    reconstruction = pickle.loads(socket.recv())
    if verbose:
        print("Received reconstruction in %.3f seconds" % (time.time() - start))
    return reconstruction


def get_volume_from_reconstruction(reconstruction, threshold=0.95):
    volume = reconstruction > threshold * reconstruction.max()
    return volume


def get_points_from_volume(volume):
    objectpoints = np.argwhere(volume)
    print("objectpoints.shape", objectpoints.shape)
    return objectpoints


def get_surface_mesh_from_volume(volume):
    helper_volume = np.zeros(tuple(np.array(volume.shape) + 2))

    helper_volume[1:-1, 1:-1, 1:-1] = volume
    v, f, n, c = marching_cubes(helper_volume, gradient_direction="descent")
    v -= 1

    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(v),
        o3d.utility.Vector3iVector(f),
    )
    mesh.vertex_normals = o3d.utility.Vector3dVector(n/np.linalg.norm(n))
    return mesh


def get_mesh_px(volume):
    mesh_px = get_surface_mesh_from_volume(volume)
    return mesh_px


def get_points_px(volume):
    points_px = get_points_from_volume(volume)
    return points_px


def get_pcd_px(volume):
    points_px = get_points_px(volume)
    pcd_px = get_pcd(points_px)
    return pcd_px


def get_pcd(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals()
    return pcd

def get_points_from_mesh_or_pcd(mesh_or_pcd):
    points = np.asarray(mesh_or_pcd.vertices if hasattr(mesh_or_pcd, "vertices") else mesh_or_pcd.points)
    return points

def set_points_to_mesh_or_pcd(mesh_or_pcd, points):
    points = o3d.utility.Vector3dVector(points)
    if hasattr(mesh_or_pcd, "vertices"):
        mesh_or_pcd.vertices = points
    else:
        mesh_or_pcd.points = points
    return mesh_or_pcd
        
def get_mesh_or_pcd_mm(
    mesh_or_pcd_px, 
    calibration,
    origin_vector,
    origin_index,
    directions=np.array([1, 1, 1]),
    order=np.array([0, 1, 2]),
):
    mesh_or_pcd_mm = copy.copy(mesh_or_pcd_px)
    points_px = get_points_from_mesh_or_pcd(mesh_or_pcd_px)
    points_mm = get_points_mm(points_px, calibration, origin_vector, origin_index, directions=directions, order=order)
    mesh_or_pcd_mm = set_points_to_mesh_or_pcd(mesh_or_pcd_mm, points_mm)
    return mesh_or_pcd_mm

    
def save_mesh(mesh, filename):
    _start = time.time()
    o3d.io.write_triangle_mesh(filename, mesh)
    print(f"triangle mesh {filename} save took {time.time()-_start: .4f} seconds")


def save_pcd(pcd, filename):
    _start = time.time()
    o3d.io.write_point_cloud(filename, pcd)
    print(f"point cloud {filename} save took {time.time()-_start: .4f} seconds")
