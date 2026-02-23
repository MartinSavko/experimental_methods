#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import traceback
import logging
import time
import re
import numpy as np
import pickle
import h5py
import simplejpeg
from scipy.spatial import distance_matrix
from scipy.constants import c, eV, h, angstrom
from skimage.morphology import convex_hull_image
import datetime
import subprocess
from numbers import Number
import pprint

try:
    import cv2 as cv
except ImportError:
    cv = None
import gzip
import pylab
from math import (
    sin,
    cos,
    atan2,
    radians,
    sqrt,
    ceil,
)

import lmfit
import redis

DEFAULT_BROKER_PORT = 5555
MOTOR_BROKER_PORT = 5557
CAMERA_BROKER_PORT = 5556

match_number_in_spot_file = re.compile(".*([\d]{6}).adx.gz")
match_number_in_cbf = re.compile(".*([\d]{6}).cbf.gz")


run_pattern = "(\/nfs\/data\d\/\d\d\d\d_Run\d).*"

black = (0, 0, 0)
white = (1, 1, 1)
yellow = (1, 0.706, 0)
blue = (0, 0.706, 1)
green = (0.706, 1, 0)
magenta = (0.706, 0, 1)
cyan = (0, 1, 0.706)
purpre = (1, 0, 0.706)

notions = ["background", "foreground", "pin", "stem", "loop", "loop_inside", "crystal"]

colors_for_labels = {
    "crystal": green,
    "loop": purpre,
    "loop_inside": blue,
    "stem": cyan,
    "pin": magenta,
    "foreground": white,
    "background": black,
}

parameters_setup = {
    # center of rotation
    "c": {
        "value": None,
        "vary": True,
        "min": 0.0,
        "max": 1216.0,
        "default": "np.median(clicks)",
    },
    # radius of rotation
    "r": {
        "value": None,
        "vary": True,
        "min": 0.0,
        "max": 2.0 * 1216.0,
        "default": "np.std(clicks) / np.sin(np.pi / 4)",
    },
    # phase of rotation around the center
    "alpha": {
        "value": None,
        "vary": True,
        "min": 0.0,
        "max": 2.0 * np.pi,
        "default": "np.random.random() * 2 * np.pi",
    },
    # phase of rotation of the plane of the refractive planparallel slab
    "beta": {
        "value": None,
        "vary": True,
        "min": 0.0,
        "max": 2.0 * np.pi,
        "default": "np.random.random() * 2 * np.pi",
    },
    # thickness of the planparallel slab
    "thickness": {
        "value": 100.0,
        "vary": True,
        "min": 0.0,
        "max": 1216.0,
        "default": 0.0,
    },
    # immersion depth of the sample within the planparallel slab
    "depth": {
        "value": 0.025,
        "vary": True,
        "min": 0.0,
        "max": 1216.0,
        "default": 0.0,
    },
    # index of refraction of the material of the planparallel slab
    "index_of_refraction": {
        "value": 1.31,
        "vary": False,
        "min": 1.0,
        "max": 2.0,
        "default": 1.31,
    },
}


def get_dirname(path):
    dirname = os.path.dirname(os.path.realpath(path))
    return dirname


def set_mxcube_camera(mxcube_camera="oav"):
    redis.StrictRedis().set("mxcube_camera", mxcube_camera)


def get_mxcube_camera():
    return redis.StrictRedis().get("mxcube_camera").decode()


def save_pickled_file(filename, object_to_pickle, mode="wb"):
    f = open(filename, mode)
    pickle.dump(object_to_pickle, f)
    f.close()


def get_pickled_file(filename, mode="rb"):
    try:
        try:
            pickled_file = pickle.load(open(filename, mode))
        except:
            pickled_file = pickle.load(open(filename, mode), encoding="latin1")
    except IOError:
        pickled_file = None
    return pickled_file


def get_full_run_path(filename, run_pattern=run_pattern):
    full_run_path = re.findall(run_pattern, filename)
    if full_run_path:
        full_run_path = full_run_path[0]
    else:
        full_run_path = ""
    return full_run_path


def adjust_filename_for_ispyb(filename, ispyb_base_path="/nfs/ruche/proxima2a-users"):
    full_run_path = get_full_run_path(filename)
    if full_run_path:
        filename = filename.replace(full_run_path, ispyb_base_path)
    return filename


def adjust_filename_for_archive(filename):
    for directory in ["RAW_DATA", "PROCESSED_DATA"]:
        if directory in filename:
            filename = filename.replace(directory, "ARCHIVE")
    return filename


def adjust_filename(filename, archive, ispyb):
    if archive:
        filename = adjust_filename_for_archive(filename)
    if ispyb:
        filename = adjust_filename_for_ispyb(filename)
    return filename


def _check_image(image):
    try:
        image = simplejpeg.decode_jpeg(image)
    except:
        traceback.print_exc()
    return image


def is_jpeg(image):
    return simplejpeg.is_jpeg(image)


def is_number(variable):
    return isinstance(variable, Number)


def get_camera_history(template):
    complete_h5 = f"{template}_sample_view.h5"
    lean_h5 = f"{template}_sample_view_lean.h5"
    movie = f"{template}_sample_view_movie.mp4"
    if os.path.isfile(complete_h5):
        history = h5py.File(complete_h5, "r")
        timestamps = history["history_timestamps"][()]
        images = history["history_images"][()]
    elif os.path.isfile(lean_h5) and os.path.isfile(movie):
        history = h5py.File(lean_h5, "r")
        timestamps = history["history_timestamps"]
        images = movie2images(movie)
    return timestamps, images


def movie2images(movie="examples/opti/zoom_X_careful_sample_view_movie.mp4"):
    # https://stackoverflow.com/questions/30136257/how-to-get-image-from-video-using-opencv-python
    print(f"getting images from movie {movie}")
    _start = time.time()
    video = cv.VideoCapture(movie)
    images = []
    read, img = video.read()
    while read:
        images.append(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        read, img = video.read()
    _end = time.time()
    print(f"{len(images)} images read from {movie} in {_end - _start:.3f} seconds")
    return images


def images2movie(images, movie="video.avi", frame_rate=20, codec="mp4v"):
    # https://stackoverflow.com/questions/43048725/python-creating-video-from-images-using-opencv
    _start = time.time()
    fourcc = cv.VideoWriter_fourcc(*codec)
    video = cv.VideoWriter(movie, fourcc, frame_rate, images[0].shape[:2][::-1])
    for img in images:
        video.write(img)
    video.release()
    _end = time.time()
    print(f"movie {movie} encoded in {_end - _start:.3f} seconds")


def get_color(colorin):
    if type(colorin) is str:
        color = hex_to_rgb(sns.xkcd_rgb[colorin])
    else:
        color = [int(255 * item) for item in colorin]
    return color


def get_lut(negative=False):
    lut = np.zeros((256, 1, 3))
    for k, notion in enumerate(notions):
        if negative and notion in ["foreground", "not_background"]:
            colorin = colors_for_labels["background"]
        elif negative and notion in ["background"]:
            colorin = colors_for_labels["foreground"]
        else:
            colorin = colors_for_labels[notion]

        color = get_color(colorin)

        print(f"transform {colorin} to {color}")
        lut[k] = color
    for k in range(len(notions), len(lut)):
        if negative:
            lut[k] = (1, 1, 1)
        else:
            lut[k] = (0, 0, 0)
    lut = lut.astype("uint8")
    return lut


def label2rgb(label, lut):
    rgb = cv.LUT(label, lut)
    return rgb


def hex_to_rgb(_hex):
    if _hex.startswith("#"):
        _hex = _hex[1:]
    return tuple(int(_hex[i : i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(_rgb):
    return ("{:02X}" * 3).format(*_rgb)


def get_bbox_from_mask(mask):
    contour = get_mask_boundary(mask)
    bbox = cv.boundingRect(contour)
    return bbox


def get_mask_boundary(mask):
    contours, _ = cv.findContours(
        mask.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE
    )
    shape = contours[0].shape
    db = np.reshape(contours[0], (shape[0], shape[-1]))
    mask_boundary = db
    return mask_boundary


def _get_results(method, args, filename, force=False):
    print(f"_get_results called with {method}")
    # print(f" args: {args}")
    print(f" filename {filename}")
    print(f" force {force}")
    _start = time.time()

    if not force and os.path.isfile(filename) and os.stat(filename).st_size > 0:
        results = pickle.load(open(filename, "rb"))
    else:
        results = method(*args)
        results_file = open(filename, "wb")
        pickle.dump(results, results_file)
        results_file.close()
    _end = time.time()
    print(f"_get_results took {_end - _start:.4f} seconds")
    return results


def get_ordinal_from_spot_file_name(spot_file_name):
    ordinal = -1
    try:
        ordinal = int(match_number_in_spot_file.findall(spot_file_name)[0])
    except:
        pass
    return ordinal


def get_ordinal_from_cbf_file_name(cbf_file_name):
    ordinal = -1
    try:
        ordinal = int(match_number_in_cbf.findall(cbf_file_name)[0])
    except:
        pass
    return ordinal


def get_spots(spots_file):
    return spots


def get_spots_lines(spots_file, mode="rb", encoding="ascii"):
    try:
        spots_lines = (
            gzip.open(spots_file, mode=mode)
            .read()
            .decode(encoding=encoding)
            .split("\n")[:-1]
        )
    except:
        spots_lines = []
    return spots_lines


def get_spots_tioga(spots_file, mode="rb", encoding="ascii"):
    spots_lines = get_spots_lines(spots_file, mode=mode, encoding=encoding)
    spots = [list(map(float, line.split())) for line in spots_lines]
    return spots


def get_spot_xds(filename="SPOT.XDS"):
    spot_xds = np.loadtxt(filename)
    return spot_xds


def get_number_of_spots(spots_file):
    spots_lines = get_spots_lines(spots_file)
    number_of_spots = len(spots_lines)
    return number_of_spots


def get_spots_resolution(spots_mm, wavelength, detector_distance):
    distances = np.linalg.norm(spots_mm[:, :2], axis=1)
    resolutions = get_resolution_from_radial_distance(
        distances, wavelength, detector_distance
    )
    return resolutions


def get_tioga_results(total_number_of_images, spot_file_template):
    print(
        f"get_tioga_results called with {total_number_of_images}, {spot_file_template}"
    )
    tioga_results = np.zeros((total_number_of_images,))
    image_number_range = range(1, total_number_of_images + 1)
    spot_files = [spot_file_template % d for d in image_number_range]
    for sf in spot_files:
        if os.path.isfile(sf):
            nos = get_number_of_spots(sf)
            ordinal = get_ordinal_from_spot_file_name(sf)
            if ordinal != -1:
                tioga_results[ordinal - 1] = nos
    return tioga_results


def get_tioga_spots_filename(spot_file_template):
    tioga_spots_filename = spot_file_template.replace("spot_list", "process").replace(
        "_??????.adx.gz", "_spot.tioga"
    )

    return tioga_spots_filename


def get_tioga_spots(spot_file_template):
    tioga_spots_filename = get_tioga_spots_filename(spot_file_template)
    if not os.path.isfile(tioga_spots_filename):
        os.system(f"cat {spot_file_template} | gunzip - > {tioga_spots_filename}")
    tioga_spots = get_spot_xds(filename=tioga_spots_filename)
    return tioga_spots


colspot_spots = re.compile("[\s]+[\d]+[\s]+[\d]+[\s]+([\d]+)[\s]+[\d]+")


def get_colspot_results(fname="COLSPOT.LP"):
    colspot_lp = open(fname).read()
    colspot_results = np.array(list(map(int, colspot_spots.findall(colspot_lp))))
    return colspot_results


def save_and_plot_tioga_results(
    tioga_results, image_path, csv_path, figsize=(16, 9), grid=True
):
    pylab.figure(1, figsize=figsize)
    pylab.grid(grid)
    ordinals = range(1, len(tioga_results) + 1)
    tog = np.vstack([ordinals, tioga_results]).T
    pylab.plot(tog[:, 0], tog[:, 1], "-o", label="spots")
    pylab.ylim((0, tog[:, 1].max() * 1.05))
    pylab.legend()
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    pylab.savefig(image_path)
    np.savetxt(
        csv_path, tog, delimiter=",", fmt="%7d", header="ordinal, number of spots"
    )


def get_spots_mm(spots, beam_center, pixel_size=0.075):
    if type(spots) is str and os.path.isfile(spots):
        spots = get_spots(spots)

    centered_spots_px = spots[:, :2] - beam_center
    centered_spots_mm = centered_spots_px * pixel_size

    return centered_spots_mm


def get_scattered_rays(spots, beam_center, detector_distance, pixel_size=0.075):
    spots_mm = get_spots_mm(spots, beam_center, pixel_size=pixel_size)

    scattered_rays = np.hstack(
        (
            spots,
            np.ones((spots.shape[0], 1)) * detector_distance,
        )
    )
    scattered_rays /= np.linalg.norm(scattered_rays, axis=1, keepdims=True)

    return scattered_rays


def get_rays_from_all_images(
    total_number_of_images, spot_file_template, beam_center, detector_distance
):
    image_number_range = range(1, total_number_of_images + 1)
    spot_files = [spot_file_template % d for d in image_number_range]

    rays_from_all_images = {}
    for sf in spot_files:
        if os.path.isfile(sf):
            rays = get_scattered_rays(sf, beam_center, detector_distance)
            ordinal = get_ordinal_from_spot_file_name(sf)
            rays_from_all_images[ordinal] = rays

    return rays_from_all_images


def get_polygon_patch(points, color="green", lw=2, fill=False):
    # points = points[:, ::-1]
    patch = pylab.Polygon(
        points,
        color=color,
        lw=lw,
        fill=fill,
    )
    return patch


def get_mask_boundary(mask, approximate=False):
    if approximate:
        flag = cv.CHAIN_APPROX_SIMPLE
    else:
        flag = cv.CHAIN_APPROX_NONE
    contours, _ = cv.findContours(mask.astype(np.uint8), cv.RETR_EXTERNAL, flag)
    if len(contours) > 1:
        contours = list(contours)
        contours.sort(key=lambda x: -len(x))

    if len(contours) >= 1:
        largest = contours[0]

        shape = largest.shape
        mask_boundary = np.reshape(largest, (shape[0], shape[-1]))
    else:
        mask_boundary = None
    return mask_boundary


def normalize(image):
    return (image - image.min()) / (image.max() - image.min())


def get_index_of_max_or_min(image, max_or_min="max"):
    return np.unravel_index(getattr(np, f"arg{max_or_min}")(image), image.shape)


def principal_axes(array, verbose=False):
    # https://github.com/pierrepo/principal_axes/blob/master/principal_axes.py
    _start = time.time()
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
    _end = time.time()
    if verbose:
        print("principal axes")
        print("intertia tensor")
        print(inertia)
        print("eigenvalues")
        print(eigenvalues)
        print("eigenvectors")
        print(eigenvectors)
        print("principal_axes calculated in %.3f seconds" % (_end - _start))
        print()
    return inertia, eigenvalues, eigenvectors, center


def get_notion_string(notion):
    if type(notion) is list:
        notion_string = ",".join(notion)
    else:
        notion_string = notion
    return notion_string


def get_element(puck, sample):
    element = f"{puck:d}_{sample:02d}"
    return element


def get_string_from_timestamp(
    timestamp=None, fmt="%Y%m%d_%H%M%S", method="datetime", modify=True
):
    if timestamp is None:
        timestamp = time.time()
    if method == "datetime":
        timestring = datetime.datetime.fromtimestamp(timestamp).strftime(fmt)
    else:
        timestring = time.ctime(timestamp)
        if modify:
            timestring = timestring.replace(" ", "_").replace(":", "")
    return timestring


def get_time_from_string(timestring, format="%Y-%m-%d %H:%M:%S.%f", method=1):
    if method == 1:
        dt = datetime.datetime.strptime(timestring, format)
        time_from_string = dt.timestamp()
    else:
        micros = float(timestring[timestring.find(".") :])
        time_from_string = time.mktime(time.strptime(timestring, format)) + micros
    return time_from_string


def get_d_min_for_ddv(r_min, wavelength, detector_distance):
    d_min = get_resolution_from_radial_distance(r_min, wavelength, detector_distance)
    return d_min


def get_ddv(spots_mm, r_min, wavelength, detector_distance):
    d_min = get_d_min_for_ddv(r_min, wavelength, detector_distance)
    dm = np.triu(distance_matrix(spots_mm, spots_mm, p=2))
    dm = dm[np.logical_and(dm <= d_min, dm > 0)]
    h = np.histogram(dm, bins=100)
    valu = h[0]
    reso = (h[1][1:] + h[1][:-1]) / 2.0
    valu = np.hstack([[0], valu])
    reso = np.hstack([[0], reso])
    return valu, reso


def get_ddv_as_image(valu, offset=5):
    image = np.zeros((offset + valu.max() + offset, valu.shape[0]))
    for k, v in enumerate(valu):
        image[:v, k] = 1
    image = (image == 0).astype(int)
    return image


def get_baseline(valu, reso):
    image = get_ddv_as_image(valu)
    chi = convex_hull_image(image)
    bi = np.argmax(image, axis=0) == np.argmax(chi, axis=0)
    bvalu = valu[bi]
    breso = reso[bi]
    return bvalu, breso


def get_slope(valu, reso):
    bvalu, breso = get_baseline(valu, reso)
    A = np.expand_dims(breso, 1)
    b = np.expand_dims(bvalu, 1)
    slope, residual, rank, s = np.linalg.lstsq(A, b, rcond=None)
    print(f"slope: {slope}, residual: {residual}, rank: {rank}, s: {s}")
    return np.squeeze(slope)


def get_vertical_and_horizontal_shift_between_two_positions(
    aligned_position, reference_position, epsilon=1.0e-3
):
    shift = {}
    for key in aligned_position:
        if key in reference_position:
            shift[key] = aligned_position[key] - reference_position[key]
    focus, orthogonal_shift = get_focus_and_orthogonal_from_position(shift)
    if abs(shift["AlignmentZ"]) > epsilon:
        orthogonal_shift += shift["AlignmentZ"]
    vertical_shift = shift["AlignmentY"]
    return np.array([vertical_shift, orthogonal_shift])


def get_shift_from_aligned_position_and_reference_position(
    aligned_position,
    reference_position,
    omega=None,
    AlignmentZ_reference=0.0,
    epsilon=1.0e-3,
):
    if omega is None:
        omega = reference_position["Omega"]

    alignmentz_shift = (
        (reference_position["AlignmentZ"] - aligned_position["AlignmentZ"])
        if "AlignmentZ" in reference_position and "AlignmentZ" in aligned_position
        else 0.0
    )
    alignmenty_shift = (
        (aligned_position["AlignmentY"] - reference_position["AlignmentY"])
        if "AlignmentY" in reference_position and "AlignmentY" in aligned_position
        else 0.0
    )
    centringx_shift = (
        (reference_position["CentringX"] - aligned_position["CentringX"])
        if "CentringX" in reference_position and "CentringX" in aligned_position
        else 0.0
    )
    centringy_shift = (
        (aligned_position["CentringY"] - reference_position["CentringY"])
        if "CentringY" in reference_position and "CentringY" in aligned_position
        else 0.0
    )

    along_shift = alignmenty_shift

    focus, orthogonal_shift = get_focus_and_orthogonal(
        centringx_shift,
        centringy_shift,
        omega,
    )

    if abs(alignmentz_shift) > epsilon:
        orthogonal_shift += alignmentz_shift

    return np.array([along_shift, orthogonal_shift])


def get_aligned_position_from_reference_position_and_shift(
    reference_position,
    orthogonal_shift,
    along_shift,
    omega=None,
    AlignmentZ_reference=0.0,  # ALIGNMENTZ_REFERENCE,  # 0.0100,
    epsilon=1e-3,
    debug=False,
):
    if omega is None:
        omega = reference_position["Omega"]

    alignmentz_shift = reference_position["AlignmentZ"] - AlignmentZ_reference
    if abs(alignmentz_shift) < epsilon:
        alignmentz_shift = 0

    if debug:
        logging.info("get_ap_from_ps")
        logging.info(f"p: {reference_position}")
        logging.info(f"h: {orthogonal_shift}")
        logging.info(f"v: {along_shift}")
        logging.info(f"omega: {omega}")

        logging.info(
            f"executing centringx_shift, centringy_shift = get_cx_and_cy(0, {orthogonal_shift}, {omega})"
        )

    orthogonal_shift -= alignmentz_shift
    centringx_shift, centringy_shift = get_cx_and_cy(0, orthogonal_shift, omega)

    if debug:
        logging.info(f"cx_shift: {centringx_shift}")
        logging.info(f"cy_shift: {centringy_shift}")

    aligned_position = copy_position(reference_position)
    aligned_position["AlignmentZ"] -= alignmentz_shift  # ap = rp - s"
    aligned_position["AlignmentY"] += along_shift  # ap = rp + s"
    aligned_position["CentringX"] -= centringx_shift  # ap = rp - s"
    aligned_position["CentringY"] += centringy_shift  # ap = rp + s"

    return aligned_position


def get_rotation_matrix(omega_radians):
    R = np.array(
        [
            [cos(omega_radians), -sin(omega_radians)],
            [sin(omega_radians), cos(omega_radians)],
        ]
    )
    return R


def get_cx_and_cy(focus, orthogonal, omega):
    omega_radians = -radians(omega)
    R = get_rotation_matrix(omega_radians)
    R = np.linalg.pinv(R)
    return np.dot(R, [-focus, orthogonal])


def get_focus_and_orthogonal(cx, cy, omega):
    omega_radians = radians(omega)
    R = get_rotation_matrix(omega_radians)
    return np.dot(R, [-cx, cy])


def get_focus_and_orthogonal_from_position(
    position,
    centringy_direction=-1,
):
    cx = position["CentringX"]
    cy = position["CentringY"] * centringy_direction
    omega = position["Omega"]
    focus, vertical = get_focus_and_orthogonal(cx, cy, omega)
    return focus, vertical


def get_position_dictionary_from_position_tuple(position_tuple, consider=[]):
    position_dictionary = dict(
        [
            (m.split("=")[0], float(m.split("=")[1]))
            for m in position_tuple
            if m.split("=")[1] != "NaN"
            and (consider == [] or m.split("=")[0] in consider)
        ]
    )
    return position_dictionary


def get_voxel_calibration(vertical_step, horizontal_step):
    calibration = np.ones((3,))
    calibration[0] = horizontal_step
    calibration[1:] = vertical_step
    return calibration


def get_origin(parameters, position_key="reference_position"):
    p = parameters[position_key]
    o = np.array([p["CentringX"], p["CentringY"], p["AlignmentY"], p["AlignmentZ"]])
    return o


def get_points_in_goniometer_frame(
    points_px,
    calibration,
    origin,
    center=np.array([160, 256, 256]),
    directions=np.array([-1, -1, 1]),
    order=[1, 2, 0],
):
    points_mm = ((points_px - center) * calibration * directions)[:, order] + origin
    return points_mm


def get_points_in_camera_frame(
    points_mm,
    calibration,
    origin,
    center=np.array([160, 256, 256]),
    directions=np.array([-1, -1, 1]),
    order=[1, 2, 0],
):
    mm = points_mm - origin
    mm = mm[:, order[::-1]]
    mm *= directions
    mm /= calibration
    points_px = mm + center
    return points_px


def add_shift(
    position, shift, keys=["CentringX", "CentringY", "AlignmentY", "AlignmentZ"]
):
    shifted_position = {}
    for k, key in enumerate(keys):
        shifted_position[key] = position[key] + shift[k]
    return shifted_position


def get_shift(
    position, reference, keys=["CentringX", "CentringY", "AlignmentY", "AlignmentZ"]
):
    p = get_vector_from_position(position, keys=keys)
    r = get_vector_from_position(reference, keys=keys)
    shift = p - r
    return shift


def get_shift_between_positions(
    aligned_position,
    reference_position,
    omega=None,
    AlignmentZ_reference=0.0,
    epsilon=1.0e-3,
):
    if omega is None:
        omega = aligned_position["Omega"]

    alignmentz_shift = (
        (reference_position["AlignmentZ"] - aligned_position["AlignmentZ"])
        if "AlignmentZ" in reference_position and "AlignmentZ" in aligned_position
        else 0.0
    )
    alignmenty_shift = (
        (aligned_position["AlignmentY"] - reference_position["AlignmentY"])
        if "AlignmentY" in reference_position and "AlignmentY" in aligned_position
        else 0.0
    )
    centringx_shift = (
        (reference_position["CentringX"] - aligned_position["CentringX"])
        if "CentringX" in reference_position and "CentringX" in aligned_position
        else 0.0
    )
    centringy_shift = (
        (aligned_position["CentringY"] - reference_position["CentringY"])
        if "CentringY" in reference_position and "CentringY" in aligned_position
        else 0.0
    )

    along_shift = alignmenty_shift

    focus, orthogonal_shift = get_focus_and_orthogonal(
        centringx_shift,
        centringy_shift,
        omega,
    )

    if abs(alignmentz_shift) > epsilon:
        orthogonal_shift += alignmentz_shift

    return np.array([along_shift, orthogonal_shift])


def positions_close(
    p1,
    p2,
    keys=[
        "CentringX",
        "CentringY",
        "AlignmentY",
        "AlignmentZ",
        "AlignmentX",
        "Kappa",
        "Phi",
    ],
    atol=1.0e-4,
):
    try:
        v1 = get_vector_from_position(p1, keys=keys)
        v2 = get_vector_from_position(p2, keys=keys)
        allclose = np.allclose(v1, v2, atol=atol)
    except:
        allclose = False
    return allclose


def get_position_from_vector(
    v,
    keys=[
        "CentringX",
        "CentringY",
        "AlignmentY",
        "AlignmentZ",
        "AlignmentX",
        "Kappa",
        "Phi",
    ],
):
    return dict([(key, value) for key, value in zip(keys, v)])


def get_vector_from_position(
    p,
    keys=[
        "CentringX",
        "CentringY",
        "AlignmentY",
        "AlignmentZ",
        "AlignmentX",
        "Kappa",
        "Phi",
    ],
):
    return np.array([p[key] for key in keys if key in p])


def get_distance(p1, p2, keys=["CentringX", "CentringY"]):
    return np.linalg.norm(
        get_vector_from_position(p1, keys=keys)
        - get_vector_from_position(p2, keys=keys)
    )


def get_reduced_point(p, keys=["CentringX", "CentringY"]):
    return dict([(key, value) for key, value in p.items() if key in keys])


def copy_position(p, method=1):
    if method == 1:
        new_position = p.copy()
    else:
        new_position = {}
        for key in p:
            new_position[key] = p[key]
    return new_position


def get_point_between(
    p1, p2, keys=["CentringX", "CentringY", "AlignmentY", "AlignmentZ"]
):
    v1 = get_vector_from_position(p1, keys=keys)
    v2 = get_vector_from_position(p2, keys=keys)
    v = v1 + (v2 - v1) / 2.0
    p = get_position_from_vector(v, keys=keys)
    return p


# minikappa translational offsets
def circle_model(angle, center, amplitude, phase):
    return center + amplitude * np.cos(angle - phase)


def line_and_circle_model(kappa, intercept, growth, amplitude, phase):
    return intercept + kappa * growth + amplitude * np.sin(np.radians(kappa) - phase)


def amplitude_y_model(kappa, amplitude, amplitude_residual, amplitude_residual2):
    return (
        amplitude * np.sin(0.5 * kappa)
        + amplitude_residual * np.sin(kappa)
        + amplitude_residual2 * np.sin(2 * kappa)
    )


def get_alignmentz_offset(kappa, phi):
    return 0


def get_alignmenty_offset(
    kappa,
    phi,
    center_center=-2.2465475,
    center_amplitude=0.3278655,
    center_phase=np.radians(269.3882546),
    amplitude_amplitude=0.47039019,
    amplitude_amplitude_residual=0.01182333,
    amplitude_amplitude_residual2=0.00581796,
    phase_intercept=-4.7510392,
    phase_growth=0.5056157,
    phase_amplitude=-2.6508604,
    phase_phase=np.radians(14.9266433),
):
    center = circle_model(kappa, center_center, center_amplitude, center_phase)
    amplitude = amplitude_y_model(
        kappa,
        amplitude_amplitude,
        amplitude_amplitude_residual,
        amplitude_amplitude_residual2,
    )
    phase = line_and_circle_model(
        kappa, phase_intercept, phase_growth, phase_amplitude, phase_phase
    )
    phase = np.mod(phase, 180)

    alignmenty_offset = circle_model(phi, center, amplitude, np.radians(phase))

    return alignmenty_offset


def amplitude_cx_model(kappa, *params):
    kappa = np.mod(kappa, 2 * np.pi)
    powers = []

    if type(params) == tuple and len(params) < 2:
        params = params[0]
    else:
        params = np.array(params)

    if len(params.shape) > 1:
        params = params[0]

    params = np.array(params)

    params = params[:-2]
    amplitude_residual, phase_residual = params[-2:]

    kappa = np.array(kappa)

    for k in range(len(params)):
        powers.append(kappa**k)

    powers = np.array(powers)

    return np.dot(powers.T, params) + amplitude_residual * np.sin(
        2 * kappa - phase_residual
    )


def amplitude_cx_residual_model(kappa, amplitude, phase):
    return amplitude * np.sin(2 * kappa - phase)


def phase_error_model(kappa, amplitude, phase, frequency):
    return amplitude * np.sin(frequency * np.radians(kappa) - phase)


def get_centringx_offset(
    kappa,
    phi,
    center_center=0.5955864,
    center_amplitude=0.7738802,
    center_phase=np.radians(222.1041400),
    amplitude_p1=0.63682813,
    amplitude_p2=0.02332819,
    amplitude_p3=-0.02999456,
    amplitude_p4=0.00366993,
    amplitude_residual=0.00592784,
    amplitude_phase_residual=1.82492612,
    phase_intercept=25.8526552,
    phase_growth=1.0919045,
    phase_amplitude=-12.4088622,
    phase_phase=np.radians(96.7545812),
    phase_error_amplitude=1.23428124,
    phase_error_phase=0.83821785,
    phase_error_frequency=2.74178863,
    amplitude_error_amplitude=0.00918566,
    amplitude_error_phase=4.33422268,
):
    amplitude_params = [
        amplitude_p1,
        amplitude_p2,
        amplitude_p3,
        amplitude_p4,
        amplitude_residual,
        amplitude_phase_residual,
    ]

    center = circle_model(kappa, center_center, center_amplitude, center_phase)
    amplitude = amplitude_cx_model(kappa, *amplitude_params)
    amplitude_error = amplitude_cx_residual_model(
        kappa, amplitude_error_amplitude, amplitude_error_phase
    )
    amplitude -= amplitude_error
    phase = line_and_circle_model(
        kappa, phase_intercept, phase_growth, phase_amplitude, phase_phase
    )
    phase_error = phase_error_model(
        kappa, phase_error_amplitude, phase_error_phase, phase_error_frequency
    )
    phase -= phase_error
    phase = np.mod(phase, 180)

    centringx_offset = circle_model(phi, center, amplitude, np.radians(phase))

    return centringx_offset


def amplitude_cy_model(
    kappa, center, amplitude, phase, amplitude_residual, phase_residual
):
    return (
        center
        + amplitude * np.sin(kappa - phase)
        + amplitude_residual * np.sin(kappa * 2 - phase_residual)
    )


def get_centringy_offset(
    kappa,
    phi,
    center_center=0.5383092,
    center_amplitude=-0.7701891,
    center_phase=np.radians(137.6146006),
    amplitude_center=0.56306051,
    amplitude_amplitude=-0.06911649,
    amplitude_phase=0.77841959,
    amplitude_amplitude_residual=0.03132799,
    amplitude_phase_residual=-0.12249943,
    phase_intercept=146.9185176,
    phase_growth=0.8985232,
    phase_amplitude=-17.5015172,
    phase_phase=-409.1764969,
    phase_error_amplitude=1.18820494,
    phase_error_phase=4.12663751,
    phase_error_frequency=3.11751387,
):
    center = circle_model(kappa, center_center, center_amplitude, center_phase)  # 3
    amplitude = amplitude_cy_model(
        kappa,
        amplitude_center,
        amplitude_amplitude,
        amplitude_phase,
        amplitude_amplitude_residual,
        amplitude_phase_residual,
    )  # 5
    phase = line_and_circle_model(
        kappa, phase_intercept, phase_growth, phase_amplitude, phase_phase
    )  # 4
    phase_error = phase_error_model(
        kappa, phase_error_amplitude, phase_error_phase, phase_error_frequency
    )  # 3
    phase -= phase_error
    phase = np.mod(phase, 180)
    centringy_offset = circle_model(phi, center, amplitude, np.radians(phase))
    return centringy_offset


def get_move_vector_dictionary_from_fit(
    vertical,
    horizontal,
    orientation="vertical",
    centringx_direction=+1.0,
    centringy_direction=+1.0,
    alignmenty_direction=+1.0,
    alignmentz_direction=-1.0,
):
    if orientation == "vertical":
        along = vertical.params.valuesdict()
        ortho = horizontal.params.valuesdict()
    else:
        along = horizontal.params.valuesdict()
        ortho = vertical.params.valuesdict()

    c, r, alpha = ortho["c"], ortho["r"], ortho["alpha"]
    along_shift = along["c"]

    d_sampx = centringx_direction * r * np.sin(alpha)
    d_sampy = centringy_direction * r * np.cos(alpha)
    d_y = alignmenty_direction * along_shift
    d_z = alignmentz_direction * c

    move_vector_dictionary = {
        "AlignmentZ": d_z,
        "AlignmentY": d_y,
        "CentringX": d_sampx,
        "CentringY": d_sampy,
    }

    return move_vector_dictionary


def get_aligned_position_from_fit_and_reference(
    fit_vertical,
    fit_horizontal,
    reference,
    orientation="vertical",
):
    move_vector_dictionary = get_move_vector_dictionary_from_fit(
        fit_vertical,
        fit_horizontal,
        orientation=orientation,
    )
    aligned_position = {}
    for key in reference:
        aligned_position[key] = reference[key]
        if key in move_vector_dictionary:
            aligned_position[key] += move_vector_dictionary[key]
    return aligned_position


def get_initial_parameters(
    clicks,
    parameters_setup,
):
    initial_parameters = lmfit.Parameters()
    for name in parameters_setup:
        parameter = lmfit.Parameter(name)
        value = parameters_setup[name]["value"]
        default = parameters_setup[name]["default"]
        parameter.set(
            value=value if value is not None else eval(default),
            vary=parameters_setup[name]["vary"],
            min=parameters_setup[name]["min"],
            max=parameters_setup[name]["max"],
        )
        initial_parameters[name] = parameter

    return initial_parameters


def fit_circle(
    radians,
    clicks,
    parameter_names=["c", "r", "alpha"],
    c_value=None,
    c_optimize=True,
    report=True,
    method="nelder",
    parameters_setup=parameters_setup,
):
    parameters_setup["c"]["value"] = c_value
    parameters_setup["c"]["vary"] = c_optimize

    fit = _fit(
        circle_model_residual,
        radians,
        clicks,
        parameters_setup,
        parameter_names,
        method,
        report,
    )

    return fit


def fit_refractive(
    radians,
    clicks,
    parameter_names=["c", "r", "alpha", "beta", "thickness", "depth", "n"],
    c_value=None,
    c_optimize=True,
    thickness_value=None,
    thickness_optimize=True,
    report=True,
    method="nelder",
    parameters_setup=parameters_setup,
):
    parameters_setup["c"]["value"] = c_value
    parameters_setup["c"]["vary"] = c_optimize
    parameters_setup["thickness"]["value"] = thickness_value
    parameters_setup["thickness"]["vary"] = thickness_optimize

    fit = _fit(
        refractive_model_residual,
        radians,
        clicks,
        parameters_setup,
        parameter_names,
        method,
        report,
    )

    return fit


def fit_projection(
    radians,
    clicks,
    parameter_names=["c", "r", "alpha"],
    c_value=None,
    c_optimize=True,
    report=True,
    method="nelder",
    parameters_setup=parameters_setup,
):
    parameters_setup["c"]["value"] = c_value
    parameters_setup["c"]["vary"] = c_optimize

    fit = _fit(
        projection_model_residual,
        radians,
        clicks,
        parameters_setup,
        parameter_names,
        method,
        report,
    )

    return fit


def _fit(residual, angles, clicks, parameters_setup, parameter_names, method, report):
    _parameters_setup = {}

    for pn in parameter_names:
        _parameters_setup[pn] = parameters_setup[pn].copy()

    fit = lmfit.minimize(
        residual,
        get_initial_parameters(clicks, _parameters_setup),
        args=(angles, clicks),
        method=method,
    )

    if report:
        print(lmfit.fit_report(fit))
        print(f"residual {fit.residual.mean()} {fit.residual}")

    return fit


def get_move_vector_dictionary(
    vertical_displacements,
    horizontal_displacements,
    angles,
    calibrations,
    centringx_direction=-1,
    centringy_direction=1.0,
    alignmenty_direction=1.0,  # -1.0,
    alignmentz_direction=-1.0,  # 1.0,
    centring_model="circle",
):
    if centring_model == "refractive":
        initial_parameters = lmfit.Parameters()
        initial_parameters.add_many(
            ("c", 0.0, True, -5e3, +5e3, None, None),
            ("r", 0.0, True, 0.0, 4e3, None, None),
            ("alpha", -np.pi / 3, True, -2 * np.pi, 2 * np.pi, None, None),
            ("front", 0.01, True, 0.0, 1.0, None, None),
            ("back", 0.005, True, 0.0, 1.0, None, None),
            ("n", 1.31, True, 1.29, 1.33, None, None),
            ("beta", 0.0, True, -2 * np.pi, +2 * np.pi, None, None),
        )

        fit_y = lmfit.minimize(
            refractive_model_residual,
            initial_parameters,
            method="nelder",
            args=(angles, vertical_discplacements),
        )
        logging.info(fit_report(fit_y))
        optimal_params = fit_y.params
        v = optimal_params.valuesdict()
        c = v["c"]
        r = v["r"]
        alpha = v["alpha"]
        front = v["front"]
        back = v["back"]
        n = v["n"]
        beta = v["beta"]
        c *= 1.0e-3
        r *= 1.0e-3
        front *= 1.0e-3
        back *= 1.0e-3

    elif centring_model == "circle":
        initial_parameters = [
            np.mean(vertical_discplacements),
            np.std(vertical_discplacements) / np.sin(np.pi / 4),
            np.random.rand() * np.pi,
        ]
        fit_y = minimize(
            circle_model_residual,
            initial_parameters,
            method="nelder-mead",
            args=(angles, vertical_discplacements),
        )

        c, r, alpha = fit_y.x
        c *= 1.0e-3
        r *= 1.0e-3
        v = {"c": c, "r": r, "alpha": alpha}

    horizontal_center = np.mean(horizontal_displacements)

    d_sampx = centringx_direction * r * np.sin(alpha)
    d_sampy = centringy_direction * r * np.cos(alpha)
    d_y = alignmenty_direction * horizontal_center
    d_z = alignmentz_direction * c

    move_vector_dictionary = {
        "AlignmentZ": d_z,
        "AlignmentY": d_y,
        "CentringX": d_sampx,
        "CentringY": d_sampy,
    }

    return move_vector_dictionary


def get_model_parameters(parameters, keys=["c", "r", "alpha"]):
    if type(parameters) is lmfit.parameter.Parameters:
        v = parameters.valuesdict()
        parameters = (v[key] for key in keys)
    return parameters


def circle_model(radians, c, r, alpha):
    return c + r * np.cos(radians - alpha)


def circle_model_residual(varse, radians, data, keys=["c", "r", "alpha"]):
    c, r, alpha = get_model_parameters(varse, keys=keys)
    model = circle_model(radians, c, r, alpha)
    return cost_array(data, model)


def projection_model(radians, c, r, alpha):
    return c + r * np.cos(np.dot(2, radians) - alpha)


def projection_model_residual(varse, radians, data):
    c, r, alpha = get_model_parameters(varse)
    model = projection_model(radians, c, r, alpha)
    return cost_array(data, model)


def incident(t, n):
    return np.arcsin(np.sin(t) / n)


def planparallel_shift(depth, t, n, sense=1):
    i = incident(t, n)
    return -depth * np.sin(sense * t - i) / np.cos(i)


def refractive_shift(t, f, b, n, beta):
    t = t - beta
    dt = np.degrees(t)
    s = np.zeros(dt.shape)
    t_base = t % (2 * np.pi)
    mask = np.where(((t_base < 3 * np.pi / 2) & (t_base >= np.pi / 2)), 1, 0)
    s[mask == 0] = planparallel_shift(f, t_base[mask == 0], n, sense=1)
    s[mask == 1] = planparallel_shift(b, t_base[mask == 1], n, sense=-1)
    return s


def refractive_model(t, c, r, alpha, beta, thickness, depth, n):
    front = depth
    back = thickness - depth
    return circle_model(t, c, r, alpha) - refractive_shift(t, front, back, n, beta)


def refractive_model_residual(
    parameters,
    angles,
    data,
    keys=["c", "r", "alpha", "beta", "thickness", "depth", "n"],
):
    c, r, alpha, beta, thickness, depth, n = get_model_parameters(parameters, keys=keys)
    model = refractive_model(angles, c, r, alpha, beta, thickness, depth, n)
    return cost_array(data, model)


def cost_array(data, model):
    return np.abs(data - model) ** 2


def cost(data, model, factor=1.0, normalize=False):
    if normalize == True:
        factor = 1.0 / (2 * len(model))
    return factor * np.sum(np.sum(np.abs(data - model) ** 2))


def test_tioga_results(force=False):
    from diffraction_experiment_analysis import diffraction_experiment_analysis

    dea = diffraction_experiment_analysis(
        directory="/nfs/data4/2025_Run4/com-proxima2a/Commissioning/automated_operation/PX2_0049/pos7_explore/tomo_range_15keV_15trans_45_range_0",
        name_pattern="vadt_test",
    )
    _start = time.time()
    tr = dea.get_tioga_results(force=force)
    _end = time.time()

    print(tr)
    print(f"tioga results obtained in {_end - _start:.3f} seconds")
    _start = time.time()
    rays = dea.get_rays_from_all_images(force=force)
    _end = time.time()
    # print(rays)
    print(f"rays obtained in {_end - _start:.3f} seconds")


def select_better_model(fit1, fit2):
    if hasattr(fit1, "fun"):
        f1 = fit1.fun
    else:
        f1 = fit1.residual.mean()
    if hasattr(fit2, "fun"):
        f2 = fit2.fun
    else:
        f2 = fit2.residual.mean()
    if f1 <= f2:
        return fit1, 1
    else:
        return fit2, 2


def restart_server(server="mdbroker"):
    a = subprocess.getoutput(f"ps aux | grep {server} | grep -v grep")
    print("a", a)
    if a != "":
        pid = a.split()[1]
        print(f"killing pid {pid}")
        os.kill(int(pid), 15)
        time.sleep(5)
    print("starting server")
    os.system(f"{server} &")


def from_number_sequence_to_character_sequence(number_sequence, separator=";"):
    character_sequence = ""
    number_strings = [str(n) for n in number_sequence]
    return separator.join(number_strings)


def merge_two_overlapping_character_sequences(
    seq1, seq2, alignment_length=250, separator=";"
):
    start = seq1.index(seq2[:alignment_length])
    n_new = seq2.count(separator) - seq2[start:].count(separator)
    merged = seq1[:start] + seq2
    return merged, n_new


def from_character_sequence_to_number_sequence(character_sequence, separator=";"):
    return list(map(float, character_sequence.split(";")))


def merge_two_overlapping_number_sequences(r1, r2, alignment_length=250, separator=";"):
    c1 = from_number_sequence_to_character_sequence(r1)
    c2 = from_number_sequence_to_character_sequence(r2)
    c, start = merge_two_overlapping_character_sequences(c1, c2, alignment_length)
    r = from_character_sequence_to_number_sequence(c)
    return r, start


def find_overlap(r1, r2, alignment_length=250, separator=";"):
    c1 = from_number_sequence_to_character_sequence(r1)
    c2 = from_number_sequence_to_character_sequence(r2)
    start = c1.index(c2[:alignment_length])
    start = c2.count(separator) - c2[start:].count(separator)
    return start


def merge_two_overlapping_buffers(seq1, seq2, alignment_length=250):
    try:
        start = seq1.index(seq2[:alignment_length])
    except ValueError:
        start = -1
    merged = seq1[:start] + seq2
    n_new = int((len(merged) - len(seq1)) / 8)
    return merged, n_new


def position_valid(position):
    invalid = position in [None, np.nan, np.inf, -np.inf] or (
        not position > 0 and not position <= 0
    )
    return not invalid


def get_duration(times):
    # if type(times[0]) is np.ndarray:
    # times = times[0]
    times, mt, mc = demulti(times)
    duration = times[-1] - times[0]
    return duration


def demulti(history_times):
    if type(history_times) is not np.ndarray:
        timestamps = np.array(history_times)
    multistamp = False
    multichann = False
    tshape = timestamps.shape
    if len(tshape) > 1:
        if tshape[0] > tshape[1]:
            multistamp = True
            timestamps = timestamps[:, 0]
        else:
            multichann = True
            timestamps = timestamps[0]
    return timestamps, multistamp, multichann


def make_sense_of_request(request, parent, service_name=None, serialize=True):
    if service_name is None:
        service_name = getattr(parent, "service_name_str")
    logging.info(f"make_sense_of_request (service {service_name})")
    _start = time.time()

    request = pickle.loads(request)
    logging.info("request decoded %s" % request)
    value = None
    try:
        method_name = request["method"]
        method = getattr(parent, method_name)
        params = request["params"]
        args = ()
        kwargs = {}
        if type(params) is dict:
            if "args" in request["params"]:
                args = params["args"]
            if "kwargs" in request["params"]:
                kwargs = params["kwargs"]
        elif params is not None:
            args = (params,)
        value = method(*args, **kwargs)
    except:
        method_name = ""
        logging.exception("%s" % traceback.format_exc())

    if serialize:
        value = pickle.dumps(value)
    logging.info("requests processed in %.7f seconds" % (time.time() - _start))
    return method_name, value


def get_energy_from_wavelength(wavelength):
    """energy in eV, wavelength in angstrom"""
    return (h * c) / (eV * angstrom) / wavelength


def get_wavelength_from_energy(energy):
    """energy in eV, wavelength in angstrom"""
    return (h * c) / (eV * angstrom) / energy


def get_theta_from_energy(energy):
    wavelength = get_wavelength_from_energy(energy)
    theta = get_theta_from_wavelength(wavelength)
    return theta


def get_energy_from_theta(theta):
    wavelength = get_wavelength_from_theta(theta)
    energy = get_energy_from_wavelength(wavelength)
    return energy


def get_wavelength_from_theta(theta, d=3.1347507142511746):
    """wavelength in angstrom, Si 111"""
    wavelength = 2 * d * np.sin(np.radians(theta))
    return wavelength


def get_theta_from_wavelength(wavelength, d=3.1347507142511746):
    theta = np.degrees(np.arcsin(wavelength / (2 * d)))
    return theta


def get_resolution_from_radial_distance(radial_distance, wavelength, detector_distance):
    tans = radial_distance / detector_distance
    twotheta = np.arctan(tans)
    theta = twotheta / 2
    resolution = wavelength / (2 * np.sin(theta))
    return resolution


def get_resolution_from_detector_distance(
    detector_distance, wavelength, radial_distance
):
    resolution = get_resolution_from_radial_distance(
        radial_distance, wavelength, detector_distance
    )
    return resolution


def get_theta_from_resolution_and_wavelength(resolution, wavelength):
    theta = np.arcsin(wavelength / (2 * resolution))
    return theta


def get_radial_distance_from_resolution(resolution, wavelength, detector_distance):
    theta = get_theta_from_resolution_and_wavelength(resolution, wavelength)
    twotheta = 2 * theta
    tans = np.tan(twotheta)
    radial_distance = detector_distance * tans
    # radial_distance = detector_distance * np.tan(2 * np.arcsin(wavelength / (2 * resolution)))
    return radial_distance


def get_detector_distance_from_resolution(resolution, wavelength, radial_distance):
    theta = get_theta_from_resolution_and_wavelength(resolution, wavelength)
    twotheta = 2 * theta
    tans = np.tan(twotheta)
    detector_distance = radial_distance / tans

    return detector_distance


def get_camera_id(camera, name_modifier):
    if name_modifier:
        camera_id = f"{camera:s}_{name_modifier:s}"
    else:
        camera_id = f"{camera:s}"
    return camera_id


def get_camera_services(start=True, restart=False, stop=False, port=CAMERA_BROKER_PORT):
    from oav_camera import oav_camera
    from axis_stream import axis_camera

    services = {}
    # oav
    services["oav"] = oav_camera(
        service="oav_camera", codec="h264", server=False, port=CAMERA_BROKER_PORT
    )

    # axis cameras
    for kam in ["1", "6", "8", "13", "14_quad", "14_1", "14_2", "14_3", "14_4"]:
        codec = "hevc"
        service = f"cam{kam}"

        print(f"kam {kam}, {service}")
        if "_" in kam:
            cam, name_modifier = service.split("_")
        else:
            cam, name_modifier = service, None

        if kam in ["1", "6", "8"]:
            codec = "h264"

        print(f"initializing {service}")

        services[service] = axis_camera(
            cam,
            name_modifier=name_modifier,
            codec=codec,
            service=service,
            server=False,
            port=port,
        )
        services[service].set_codec(codec=codec)

    start_stop_restart(services, port, start, stop, restart)

    return services


def get_motor_services(start=True, restart=False, stop=False, port=MOTOR_BROKER_PORT):
    from speaking_motor import tango_motor, monochromator_rx_motor, undulator

    services = {}
    services["mono_rx"] = monochromator_rx_motor(port=port)
    services["undulator"] = undulator(port=port)
    # tango motors
    for service, device_name in [
        ("slits1_west", "i11-ma-c02/ex/fent_h.1-mt_i"),
        ("slits1_east", "i11-ma-c02/ex/fent_h.1-mt_o"),
        ("slits1_south", "i11-ma-c02/ex/fent_v.1-mt_d"),
        ("slits1_north", "i11-ma-c02/ex/fent_v.1-mt_u"),
        ("slits2_west", "i11-ma-c04/ex/fent_h.2-mt_i"),
        ("slits2_east", "i11-ma-c04/ex/fent_h.2-mt_o"),
        ("slits2_south", "i11-ma-c04/ex/fent_v.2-mt_d"),
        ("slits2_north", "i11-ma-c04/ex/fent_v.2-mt_u"),
        ("slits3_tz", "i11-ma-c05/ex/fent_v.3-mt_tz"),
        ("slits3_tx", "i11-ma-c05/ex/fent_h.3-mt_tx"),
        ("slits5_tz", "i11-ma-c06/ex/fent_v.5-mt_tz"),
        ("slits5_tx", "i11-ma-c06/ex/fent_h.5-mt_tx"),
        ("slits6_tz", "i11-ma-c06/ex/fent_v.6-mt_tz"),
        ("slits6_tx", "i11-ma-c06/ex/fent_h.6-mt_tx"),
        ("vfm_pitch", "i11-ma-c05/op/mir.2-mt_rx"),
        ("hfm_pitch", "i11-ma-c05/op/mir.3-mt_rz"),
        ("vfm_trans", "i11-ma-c05/op/mir.2-mt_tz"),
        ("hfm_trans", "i11-ma-c05/op/mir.3-mt_tx"),
        ("tab2_tx1", "i11-ma-c05/ex/tab.2-mt_tx.1"),
        ("tab2_tx2", "i11-ma-c05/ex/tab.2-mt_tx.2"),
        ("tab2_tz1", "i11-ma-c05/ex/tab.2-mt_tz.1"),
        ("tab2_tz2", "i11-ma-c05/ex/tab.2-mt_tz.2"),
        ("tab2_tz3", "i11-ma-c05/ex/tab.2-mt_tz.3"),
        ("mono_rx_fine", "i11-ma-c03/op/mono1-mt_rx_fine"),
        ("tdl_x", "tdl-i11-ma/vi/mtx.1"),
        ("tdl_z", "tdl-i11-ma/vi/mtz.1"),
        ("shutter_x", "i11-ma-c06/ex/shutter-mt_tx"),
        ("shutter_z", "i11-ma-c06/ex/shutter-mt_tz"),
    ]:
        services[service] = tango_motor(
            device_name=device_name,
            server=False,
            service=service,
            sleeptime=1.0,
            port=port,
        )

    start_stop_restart(services, port, start, stop, restart)

    return services


def get_sai_services(start=True, restart=False, stop=False, port=CAMERA_BROKER_PORT):
    from speaking_sai import sai

    services = {}

    for service, device_name, number_of_channels in [
        ("sai1", "i11-ma-c00/ca/sai.1", 4),
        ("sipin", "i11-ma-c00/ca/sai.2", 1),
        ("sai3", "i11-ma-c00/ca/sai.3", 4),
        ("sai4", "i11-ma-c00/ca/sai.4", 4),
        ("sai5", "i11-ma-c00/ca/sai.5", 4),
    ]:
        services[service] = sai(
            device_name=device_name,
            service=service,
            server=False,
            number_of_channels=number_of_channels,
            port=port,
        )

    start_stop_restart(services, port, start, stop, restart)

    return services


def get_singleton_services(
    start=True, restart=False, stop=False, port=DEFAULT_BROKER_PORT
):
    from speaking_goniometer import speaking_goniometer
    from transmission import transmission
    from beam_position_controller import speaking_bpc

    services = {}
    # singletons
    services["speaking_goniometer"] = speaking_goniometer(
        service="speaking_goniometer", server=False, port=port
    )
    services["transmission"] = transmission(port=port, server=False)
    services["vbpc"] = speaking_bpc(
        monitor="cam", actuator="vertical_trans", server=False, port=port
    )
    services["hbpc"] = speaking_bpc(
        monitor="cam", actuator="horizontal_trans", server=False, port=port
    )

    start_stop_restart(services, port, start, stop, restart)

    return services


def get_services(start=True, stop=False, restart=False, port=None):
    services = {}

    # singletons
    services.update(get_singleton_services(start=False))
    # cameras
    services.update(get_camera_services(start=False))
    # sais
    services.update(get_sai_services(start=False))
    # motors
    services.update(get_motor_services(start=False))

    start_stop_restart(services, port, start, stop, restart)

    return services


def handle_brokers(start=True, stop=False, restart=False):
    for port in [DEFAULT_BROKER_PORT, CAMERA_BROKER_PORT, MOTOR_BROKER_PORT]:
        os.system(f"mdbroker.py -p {port} &")


def start_services(services, port=None):
    pprint.pprint(f"services to start {services}")
    for service in services:
        if not services[service].service_already_registered():
            print(f"launching {service}")
            cml = services[service].get_command_line(port=port)
            print(f"cml {cml}")
            os.system(f"{cml} &")
        else:
            print(f"service {service} already registered")


def stop_services(services, port=None):
    pprint.pprint(f"services to stop {services}")
    for service in services:
        if services[service].service_already_registered():
            print(f"stopping {service}")
            pid = services[service].get_pid()
            print(f"pid {pid}")
            if is_number(pid):
                os.kill(pid, 15)
            # services[service].kill()
        else:
            print(f"service {service} not running")


def start_stop_restart(services, port, start, stop, restart):
    if restart or stop:
        stop_services(services, port=port)
    elif not stop and (start or restart):
        start_services(services, port=port)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-s",
        "--services",
        default="all",
        type=str,
        help="services to launch",
    )

    parser.add_argument("-v", "--verbose", action="store_true", help="verbose")
    parser.add_argument("-R", "--restart", action="store_true", help="verbose")
    parser.add_argument("-S", "--start", action="store_true", help="verbose")
    parser.add_argument("-P", "--stop", action="store_true", help="verbose")
    args = parser.parse_args()
    print(args)

    if args.services == "brokers":
        handle_brokers()
    elif args.services == "services":
        services = get_services(start=args.start, stop=args.stop, restart=args.restart)
    elif args.services == "cameras":
        services = get_camera_services(
            start=args.start, stop=args.stop, restart=args.restart
        )
    elif args.services == "motors":
        services = get_motor_services(
            start=args.start, stop=args.stop, restart=args.restart
        )
    elif args.services == "sais":
        services = get_sai_services(
            start=args.start, stop=args.stop, restart=args.restart
        )
    elif args.services == "singletons":
        services = get_singleton_services(
            start=args.start, stop=args.stop, restart=args.restart
        )
    elif args.services == "all":
        print("handle brokers ...")
        handle_brokers()
        time.sleep(1)
        print("handle services ...")
        services = get_services(start=True, stop=False, restart=False)


if __name__ == "__main__":
    main()
    # test_tioga_results()
