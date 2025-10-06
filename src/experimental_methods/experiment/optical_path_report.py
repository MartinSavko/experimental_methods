#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import pickle
import copy
import traceback
import logging
import os

try:
    import skimage
    from skimage import img_as_float
    from skimage.io import imread
    from skimage.filters import (
        threshold_otsu,
        threshold_triangle,
        threshold_local,
        sobel,
        rank,
        gaussian,
    )
    from skimage import exposure

    from skimage.measure import regionprops, label
    from skimage.morphology import (
        closing,
        square,
        opening,
        disk,
        remove_small_objects,
        dilation,
    )
except ImportError:
    pass

from matplotlib.patches import Rectangle, Circle
import pylab
import numpy as np
import math

try:
    from scipy.misc import imsave
except ImportError:
    from skimage.io import imsave

    # from imageio import imwrite as imsave

try:
    import peakutils
    from scipy.ndimage.filters import gaussian_filter1d
    from scipy.ndimage import center_of_mass

    from scipy.signal import medfilt

    from scipy.optimize import minimize
    from scipy.optimize import leastsq
    from scipy.optimize import curve_fit
except ImportError:
    pass

import re
import sys
import time
import h5py
from multiprocessing import Process, Queue

calibrations = {
    1: np.array([0.00160829, 0.001612]),
    2: np.array([0.00129349, 0.0012945]),
    3: np.array([0.00098891, 0.00098577]),
    4: np.array([0.00075432, 0.00075136]),
    5: np.array([0.00057437, 0.00057291]),
    6: np.array([0.00043897, 0.00043801]),
    7: np.array([0.00033421, 0.00033406]),
    8: np.array([0.00025234, 0.00025507]),
    9: np.array([0.00019332, 0.00019494]),
    10: np.array([0.00015812, 0.00015698]),
}


def create_mosaic(images):
    y = math.sqrt(len(images))
    x = math.ceil(len(images) / y)
    y = math.ceil(len(images) / x)
    rows, cols = map(int, (y, x))

    ims = images[0].shape
    ish = np.array([1, ims[0], ims[1]])
    print("ims", ims)
    print("ish", ish)
    mosaic_shape = np.array([rows, cols]) * ish[1:]
    print("mosaic_shape", mosaic_shape)
    print("len(ims)", len(ims))
    if len(ims) == 3:
        mosaic_shape = tuple(mosaic_shape) + (ims[2],)
    print("mosaic_shape", mosaic_shape)
    mosaic = np.zeros(mosaic_shape)
    print("mosaic.shape", mosaic.shape)

    if rows * cols > len(images):
        empty = np.zeros((1,) + ims)
        for k in range(rows * cols - len(images)):
            images = np.vstack([images, empty])

    for k, image in enumerate(images):
        c, r = divmod(k, rows)
        ro = r * ims[0]
        co = c * ims[1]
        if len(mosaic_shape) == 3:
            mosaic[ro : ro + ims[0], co : co + ims[1]] = image[:] / 255.0
        else:
            mosaic[ro : ro + ims[0], co : co + ims[1]] = image[:]
    return mosaic


def load_from_file(template):
    images = np.array(
        [
            img_as_float(imread(img, as_gray=False))
            for img in glob.glob(template)
            if "report" not in img
        ]
    )
    omegas = np.array(
        [
            float(re.findall(".*_([\d\.]).png", img)[0])
            for img in glob.glob(template)
            if "report" not in img
        ]
    )
    try:
        calibration = pickle.load(open(template.replace("*.png", "parameters.pickle")))[
            "calibration"
        ]
    except IOError:
        calibration = np.mean(np.array([0.0016026, 0.0016051]))
    return images, omegas, calibration


def load_from_directory(directory, suffix=".jpg", default_zoom=1):
    image_names = glob.glob(os.path.join(directory, "*%s" % suffix))
    images = np.array([img_as_float(imread(img, as_gray=False)) for img in image_names])
    omegas = np.array(
        [
            float(re.findall(".*omega_([\d\.]*)[_\.].*%s" % suffix, img)[0])
            for img in image_names
        ]
    )
    try:
        zoom = int(re.findall(".*zoom_([\d]*).*%s" % suffix, image_names[0]))
    except:
        zoom = default_zoom
    calibration = calibrations[zoom]
    return images, omegas, calibration


def load_from_pickle(pickled_series, nimages=24):
    series = pickle.load(open(pickled_series))
    print("len(series)", len(series))
    # print('series[0]', series[0])
    if type(series) == dict:
        print("exiting")
        sys.exit()
    images = np.array([img_as_float(img[-1]) for img in series])
    omegas = np.array([img[1] for img in series])
    omin, omax = omegas.min(), omegas.max()
    selected_omegas = np.arange(omin, omax, (omax - omin) / nimages)
    selected_images = []
    corresponding_omegas = []
    for so in selected_omegas:
        selected = np.argmin(np.abs(omegas - so))
        selected_images.append(images[selected])
        corresponding_omegas.append(omegas[selected])
    try:
        if "_images.pickle" in pickled_series:
            calibration = pickle.load(
                open(pickled_series.replace("_images.pickle", "_parameters.pickle"))
            )["calibration"]
        else:
            calibration = pickle.load(
                open(
                    pickled_series.replace(".pck", "_parameters.pck").replace(
                        ".pickle", "_parameters.pickle"
                    )
                )
            )["calibration"]
    except:
        print(traceback.print_exc())
        calibration = np.array([0.0016026, 0.0016051])
    return np.array(selected_images), np.array(corresponding_omegas), calibration


def vertical_variance(images):
    return np.var(images, axis=0).mean(axis=0)


def horizontal_variance(images):
    return np.var(images, axis=0).mean(axis=1)


def get_right_left_boundary(mean_variance_curve, threshold=0.005, medfilt_kernel=5):
    gradient = np.abs(medfilt(np.gradient(mean_variance_curve), medfilt_kernel))
    tmvc = mean_variance_curve > mean_variance_curve.mean() * threshold
    return np.argwhere(tmvc)[-1][0], np.argwhere(tmvc)[0][0], gradient


def get_loop_rectangle(
    binary_image,
    xmin,
    xmax,
    ymin,
    ymax,
    pin_xmin=None,
    pin_xmax=None,
    threshold=0.01,
    rightmost_pixels=10,
    angle=None,
    output_queue=None,
):
    # print('get_loop_rectangle', xmin, xmax, ymin, ymax, pin_xmin, pin_xmax, threshold, rightmost_pixels, angle)

    bi = copy.deepcopy(binary_image > 0).astype(np.uint8)

    try:
        loop_segment = bi[ymin : ymax + 1, xmin : xmax + 1]
    except:
        loop_segment = bi[:, xmin : xmax + 1]

    x_extent = xmax - xmin
    try:
        lss = loop_segment.sum(axis=1)
        relative_y_indices = np.argwhere(lss != 0)
        ymin = ymin + relative_y_indices.min()
        ymax = ymin + relative_y_indices.max()
    except:
        com = np.array(center_of_mass(loop_segment))
        try:
            y_center, x_center = map(int, com)
            ymin = y_center - x_extent / 2
            ymax = y_center + x_extent / 2
        except:
            if output_queue != None and angle != None:
                xmin, ymin, xmax, ymax, centroid, rightmost_point = (
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    (np.nan, np.nan),
                    (np.nan, np.nan),
                )
                output_queue.put(
                    [angle, xmin, ymin, xmax, ymax, centroid, rightmost_point, bi]
                )
            return

    loop_segment = bi[ymin : ymax + 1, xmin : xmax + 1]
    try:
        lsm = loop_segment.sum(axis=0)
        x_zeros = np.argwhere(lsm != 0)
        relative_xmax = x_zeros.max()
        xmax = xmin + relative_xmax
    except:
        pass

    loop_segment = bi[ymin : ymax + 1, xmin : xmax + 1]
    try:
        lsm = loop_segment.mean(axis=1)
        y_nonzeros = np.argwhere(lsm > 0)
        relative_ymin = y_nonzeros.min()
        relative_ymax = y_nonzeros.max()
        ymin = ymin + relative_ymin
        ymax = ymin + relative_ymax
    except:
        xmin, ymin, xmax, ymax, centroid, rightmost_point = (
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            (np.nan, np.nan),
            (np.nan, np.nan),
        )
        output_queue.put([angle, xmin, ymin, xmax, ymax, centroid, rightmost_point, bi])
        return np.nan, np.nan, np.nan, np.nan, (np.nan, np.nan), (np.nan, np.nan), bi

    loop_segment = bi[ymin : ymax + 1, xmin : xmax + 1]

    loop_segment[loop_segment == 1] = 2

    for c in range(loop_segment.shape[1]):
        column = loop_segment[:, c]
        nonzero = np.argwhere(column == 2)
        if len(nonzero) > 1:
            start = min(nonzero)[0]
            end = max(nonzero)[0]
            loop_segment[:, c][start:end] = 2

    for r in range(loop_segment.shape[0]):
        row = loop_segment[r, :]
        nonzero = np.argwhere(row == 2)
        if len(nonzero) > 1:
            start = min(nonzero)[0]
            end = max(nonzero)[0]
            loop_segment[r, :][start:end] = 2

    centroid = np.array([ymin, xmin]) + np.array(center_of_mass(loop_segment))

    rightmost_point = np.array(
        [ymin, xmin + loop_segment.shape[1] - rightmost_pixels]
    ) + np.array(center_of_mass(loop_segment[:, -rightmost_pixels:]))

    bi[ymin : ymax + 1, xmin : xmax + 1] = loop_segment

    loop_segment_props = regionprops(loop_segment)[0]

    if pin_xmin != None and pin_xmax != None:
        pin_segment = bi[:, pin_xmin:pin_xmax]
        pin_ymean = pin_segment.mean(axis=1)
        pin_x_extent = pin_xmax - pin_xmin

        com = np.array(center_of_mass(pin_segment))

        try:
            pin_y_center, pin_x_center = map(int, com)
        except ValueError:
            pin_y_center, pin_x_center = (
                ymin + (ymax - ymin) / 2,
                pin_xmin + pin_x_extent,
            )

        relative_y_indices = np.argwhere(pin_ymean > threshold * pin_ymean.mean())
        if len(relative_y_indices) > 0:
            pin_ymin = relative_y_indices.min()
            pin_ymax = relative_y_indices.max() + 1
        else:
            pin_ymin = pin_y_center - pin_x_extent / 2
            pin_ymax = pin_y_center + pin_x_extent / 2

        pin_segment = bi[pin_ymin : pin_ymax + 1, pin_xmin : pin_xmax + 1]

        pin_centroid = np.array([pin_ymin, pin_xmin]) + np.array(
            center_of_mass(pin_segment)
        )

        pin_segment[pin_segment == 1] = 3

        for c in range(pin_segment.shape[1]):
            column = pin_segment[:, c]
            nonzero = np.argwhere(column == 3)
            if len(nonzero) > 1:
                start = min(nonzero)[0]
                end = max(nonzero)[0]
                pin_segment[:, c][start:end] = 3

        for r in range(pin_segment.shape[0]):
            row = pin_segment[r, :]
            nonzero = np.argwhere(row == 3)
            if len(nonzero) > 1:
                start = min(nonzero)[0]
                end = max(nonzero)[0]
                pin_segment[r, :][start:end] = 3

        bi[pin_ymin : pin_ymax + 1, pin_xmin : pin_xmax + 1] = pin_segment

        psm = pin_segment.sum(axis=0)
        stem = np.argwhere(psm < 5 * threshold * psm.max()).flatten()
        pin = np.argwhere(psm >= 5 * threshold * psm.max()).flatten()

        for c in stem:
            if c > pin.max():
                column = pin_segment[:, c]
                column[column == 3] = 1
                pin_segment[:, c] = column
        try:
            pin_segment_props = regionprops(pin_segment)[0]
        except IndexError:
            print("IndexError probably no pixels in the area where pin is expected")
        except ValueError:
            print("ValueError probably no pixels in the area where pin is expected")
        except TypeError:
            print("TypeError")
            print("pin_segment", pin_segment)

    if output_queue != None and angle != None:
        output_queue.put([angle, xmin, ymin, xmax, ymax, centroid, rightmost_point, bi])
    return


def normalize(array, return_norm=False):
    norm = np.linalg.norm(array)
    result = array / norm
    if return_norm:
        result = result, norm
    return result


def get_gaussian_fit(start, stop, peak, search_line):
    a, b, c = None, None, None
    try:
        a, b, c = peakutils.gaussian_fit(
            np.arange(start, stop), search_line[start:stop], center_only=False
        )
        if abs(start + (stop - start) / 2 - b) > (stop - start):
            c = (stop - peak) / 2
            print("gaussian fit not good")
            print("c", c)
    except:
        print("gaussian fit failed")
        print(traceback.print_exc())
        if stop != None and peak != None:
            c = stop - peak
        print("c", c)
    return a, b, c


def residual_leastsq(varse, angles, data):
    c, r, alpha = varse
    model = circle_model(angles, c, r, alpha)
    return np.abs(data - model)


def circle_model(angles, c, r, alpha):
    return c + r * np.cos(angles - alpha)


def circle_model_residual(varse, angles, data):
    c, r, alpha = varse
    model = circle_model(angles, c, r, alpha)
    return 1.0 / (2 * len(model)) * np.sum(np.sum(np.abs(data - model) ** 2))


def projection_model(angles, c, r, alpha):
    return c + r * np.cos(2 * angles - alpha)


def projection_model_residual(varse, angles, data):
    c, r, alpha = varse
    model = projection_model(angles, c, r, alpha)
    return 1.0 / (2 * len(model)) * np.sum(np.sum(np.abs(data - model) ** 2))


def circle_projection_model(angles, c, r, alpha, k=1):
    possible_k = np.array([1, 2])
    k = possible_k[np.argmin(np.abs(k - possible_k))]
    return c + r * np.cos(k * angles - alpha)


def circle_projection_model_residual(varse, angles, data):
    c, r, alpha, k = varse
    model = circle_projection_model(angles, c, r, alpha, k)
    return 1.0 / (2 * len(model)) * np.sum(np.sum(np.abs(data - model) ** 2))


def select_better_model(fit1, fit2):
    if fit1.fun <= fit2.fun:
        return fit1, 1
    else:
        return fit2, 2


def remove_small_objects_wrapper(img, thresh, min_size, angle=None, output_queue=None):
    segmented = remove_small_objects(img >= thresh, min_size=min_size)

    if output_queue != None and angle != None:
        output_queue.put([angle, segmented])
    else:
        return segmented


def remove_noise_and_smooth(image, sigma=7):
    return ndi.gaussian_filter(ndi.median_filter(image, sigma), sigma)


def get_threshold(ndif, min_size=200, out=150):
    threshold = threshold_otsu(ndif)
    diff = ndif > threshold
    n_threshold = np.sum(diff)
    n_se = np.sum(diff[-out:, -out:])
    while n_se and n_threshold > min_size:
        threshold *= 1.1
        diff = ndif > threshold
        n_threshold = np.sum(diff)
        n_se = np.sum(diff[-out:, -out:])
    return threshold


def _generate_report(
    original_images,
    loop_rectangles,
    labeled_images,
    right_loop_boundary,
    left_loop_boundary,
    right_pin_boundary,
    left_pin_boundary,
    search_line,
    fits,
    calibration,
    gve,
    destination="",
    template="optical_path_analysis",
    display=True,
):
    report_generation_start = time.time()
    images_mosaic = create_mosaic(original_images)
    edges_mosaic = create_mosaic(np.array(labeled_images))

    rightmost = fits["rightmost"]
    centroids = fits["centroids"]
    widths = fits["widths"]
    heights = fits["heights"]
    areas = fits["areas"]

    angles = rightmost[:, 0]
    rightmost_vertical = rightmost[:, 1]
    rightmost_horizontal = rightmost[:, 2]
    centroids_vertical = centroids[:, 0]
    centroids_horizontal = centroids[:, 1]

    fig, axes = pylab.subplots(2, 3, figsize=(20, 12))
    ax = axes.flatten()
    ax[0].imshow(images_mosaic, cmap="gray")
    number_of_images = len(labeled_images)
    y = math.sqrt(number_of_images)
    x = math.ceil(number_of_images / y)
    y = math.ceil(number_of_images / x)
    rows, cols = map(int, (y, x))
    ims = original_images[0].shape
    ish = np.array([1, original_images[0].shape[0], original_images[0].shape[1]])
    for k, (image, (xmin, ymin, xmax, ymax, centroid, rightmost_point)) in enumerate(
        zip(original_images, loop_rectangles)
    ):
        c, r = divmod(k, rows)
        ro = r * ims[0]
        co = c * ims[1]
        re = Rectangle(
            (co + xmin, ro + ymin), xmax - xmin, ymax - ymin, fill=False, color="green"
        )
        ci = Circle((co + centroid[-1], ro + centroid[0]), color="red")
        rp = Circle((co + rightmost_point[-1], ro + rightmost_point[0]), color="blue")
        ax[0].add_patch(re)
        ax[0].add_patch(ci)
        ax[0].add_patch(rp)

    ax[3].imshow(edges_mosaic)
    ax[4].plot(normalize(search_line), "c-", label="search line")
    ax[4].plot(normalize(gve), "m-", label="vertical gradient")
    ax[4].vlines(
        right_loop_boundary,
        normalize(search_line).min(),
        normalize(search_line).max(),
        color="k",
        lw=3,
        label="right loop boundary",
    )
    ax[4].vlines(
        left_loop_boundary,
        normalize(search_line).min(),
        normalize(search_line).max(),
        color="k",
        lw=3,
        label="left loop boundary",
    )
    ax[4].vlines(
        right_pin_boundary,
        normalize(search_line).min(),
        normalize(search_line).max(),
        color="y",
        lw=3,
        label="right pin boundary",
    )
    ax[4].vlines(
        left_pin_boundary,
        normalize(search_line).min(),
        normalize(search_line).max(),
        color="y",
        lw=3,
        label="left pin boundary",
    )
    ax[4].legend(loc="upper right")

    k_height = fits["height"][1]
    c_height, r_height, alpha_height = fits["height"][0].x
    test_angles = np.radians(np.linspace(0, 360, 1000))
    angle_height_extreme = alpha_height / k_height

    if k_height == 1:
        test_heights = circle_model(test_angles, c_height, r_height, alpha_height)
        special_height = circle_model(
            angle_height_extreme, c_height, r_height, alpha_height
        )
    else:
        test_heights = projection_model(test_angles, c_height, r_height, alpha_height)
        special_height = projection_model(
            angle_height_extreme, c_height, r_height, alpha_height
        )
    if abs(special_height - test_heights.max()) < abs(
        special_height - test_heights.min()
    ):
        angle_height_max = np.degrees(angle_height_extreme)
    else:
        angle_height_max = np.degrees(angle_height_extreme) + 90.0

    angle_height_min = angle_height_max + 90.0

    ax[1].plot(np.degrees(angles), np.sqrt(areas), "o", color="green", label="area")
    ax[1].plot(
        np.degrees(angles),
        np.sqrt(circle_projection_model(angles, *fits["area"][0].x, k=fits["area"][1])),
        "--",
        color="green",
        label="fit area",
    )
    ax[1].plot(np.degrees(angles), widths, "o", color="blue", label="width")
    ax[1].plot(
        np.degrees(angles),
        circle_projection_model(angles, *fits["width"][0].x, k=fits["width"][1]),
        "--",
        color="blue",
        label="fit width",
    )
    ax[1].plot(np.degrees(angles), heights, "o", color="red", label="height")
    ax[1].plot(
        np.degrees(angles),
        circle_projection_model(angles, *fits["height"][0].x, k=k_height),
        "--",
        color="red",
        label="fit height",
    )
    ax[1].vlines(
        angle_height_max,
        heights.min(),
        heights.max(),
        color="magenta",
        label="max height orientation",
    )
    ax[1].vlines(
        angle_height_min,
        heights.min(),
        heights.max(),
        color="orange",
        label="min height orientation",
    )
    ax[1].legend(loc="upper right")

    ax[2].plot(
        np.degrees(angles),
        (rightmost_vertical - fits["rightmost_vertical"][0].x[0]) * calibration[1],
        "o",
        color="blue",
        label="rightmost",
    )
    ax[2].plot(
        np.degrees(angles),
        (
            circle_projection_model(
                angles,
                *fits["rightmost_vertical"][0].x,
                k=fits["rightmost_vertical"][1]
            )
            - fits["rightmost_vertical"][0].x[0]
        )
        * calibration[1],
        "--",
        color="blue",
        label="fit rightmost",
    )
    ax[2].set_title("vertical movements")
    ax[2].legend(loc="upper right")

    ax[5].plot(
        np.degrees(angles),
        (rightmost_horizontal - fits["rightmost_horizontal"][0].x[0]) * calibration[0],
        "o",
        color="blue",
        label="rightmost",
    )
    ax[5].plot(
        np.degrees(angles),
        (
            circle_projection_model(
                angles,
                *fits["rightmost_horizontal"][0].x,
                k=fits["rightmost_horizontal"][1]
            )
            - fits["rightmost_horizontal"][0].x[0]
        )
        * calibration[0],
        "--",
        color="blue",
        label="fit rightmost",
    )
    ax[5].set_title("horizontal movements")
    ax[5].legend(loc="upper right")

    if template == "":
        filename = os.path.join(destination, "report.png")
    else:
        filename = os.path.join(destination, "%s_report.png" % template)

    pylab.suptitle("%s" % template)
    pylab.savefig(filename)
    report_generation_end = time.time()
    print(
        "Report generation took %.2f "
        % (report_generation_end - report_generation_start)
    )

    if display == True:
        pylab.show()


def get_fits(omegas, loop_rectangles):
    initial_parameters = [512.0, 100.0, 0.0]

    rightmost = np.array(
        [
            (np.radians(omega), lr[-1][0], lr[-1][1])
            for omega, lr in zip(omegas, loop_rectangles)
            if not np.isnan(lr[-1][0])
        ]
    )

    angles = rightmost[:, 0]
    rightmost_vertical = rightmost[:, 1]
    rightmost_horizontal = rightmost[:, 2]

    centroids = np.array([lr[-2] for lr in loop_rectangles if not np.isnan(lr[-1][0])])
    centroids_vertical = centroids[:, 0]
    centroids_horizontal = centroids[:, 1]

    widths = np.array(
        [lr[2] - lr[0] for lr in loop_rectangles if not np.isnan(lr[-1][0])]
    )
    heights = np.array(
        [lr[3] - lr[1] for lr in loop_rectangles if not np.isnan(lr[-1][0])]
    )
    areas = widths * heights

    fit_start = time.time()
    # circle
    fit_width_circle = minimize(
        circle_model_residual,
        initial_parameters,
        method="nelder-mead",
        args=(angles, widths),
    )
    fit_height_circle = minimize(
        circle_model_residual,
        initial_parameters,
        method="nelder-mead",
        args=(angles, heights),
    )
    fit_area_circle = minimize(
        circle_model_residual,
        initial_parameters,
        method="nelder-mead",
        args=(angles, areas),
    )
    fit_rightmost_vertical_circle = minimize(
        circle_model_residual,
        initial_parameters,
        method="nelder-mead",
        args=(angles, rightmost_vertical),
    )
    fit_rightmost_horizontal_circle = minimize(
        circle_model_residual,
        initial_parameters,
        method="nelder-mead",
        args=(angles, rightmost_horizontal),
    )
    fit_centroid_vertical_circle = minimize(
        circle_model_residual,
        initial_parameters,
        method="nelder-mead",
        args=(angles, centroids_vertical),
    )
    fit_centroid_horizontal_circle = minimize(
        circle_model_residual,
        initial_parameters,
        method="nelder-mead",
        args=(angles, centroids_horizontal),
    )

    # projection
    fit_width_projection = minimize(
        projection_model_residual,
        initial_parameters,
        method="nelder-mead",
        args=(angles, widths),
    )
    fit_height_projection = minimize(
        projection_model_residual,
        initial_parameters,
        method="nelder-mead",
        args=(angles, heights),
    )
    fit_area_projection = minimize(
        projection_model_residual,
        initial_parameters,
        method="nelder-mead",
        args=(angles, areas),
    )
    fit_rightmost_vertical_projection = minimize(
        projection_model_residual,
        initial_parameters,
        method="nelder-mead",
        args=(angles, rightmost_vertical),
    )
    fit_rightmost_horizontal_projection = minimize(
        projection_model_residual,
        initial_parameters,
        method="nelder-mead",
        args=(angles, rightmost_horizontal),
    )
    fit_centroid_vertical_projection = minimize(
        projection_model_residual,
        initial_parameters,
        method="nelder-mead",
        args=(angles, centroids_vertical),
    )
    fit_centroid_horizontal_projection = minimize(
        projection_model_residual,
        initial_parameters,
        method="nelder-mead",
        args=(angles, centroids_horizontal),
    )
    fit_end = time.time()

    print("Fit took %.2f" % (fit_end - fit_start))
    # selection of best model
    selection_start = time.time()

    fit_width, k_width = select_better_model(fit_width_circle, fit_width_projection)
    fit_height, k_height = select_better_model(fit_height_circle, fit_height_projection)
    fit_area, k_area = select_better_model(fit_area_circle, fit_area_projection)
    fit_rightmost_vertical, k_rightmost_vertical = select_better_model(
        fit_rightmost_vertical_circle, fit_rightmost_vertical_projection
    )
    fit_rightmost_horizontal, k_rightmost_horizontal = select_better_model(
        fit_rightmost_horizontal_circle, fit_rightmost_horizontal_projection
    )
    fit_centroid_vertical, k_centroid_vertical = select_better_model(
        fit_centroid_vertical_circle, fit_centroid_vertical_projection
    )
    fit_centroid_horizontal, k_centroid_horizontal = select_better_model(
        fit_centroid_horizontal_circle, fit_centroid_horizontal_projection
    )

    selection_end = time.time()
    print("Model selection took %.2f" % (selection_end - selection_start,))

    fits = {
        "rightmost_vertical": (fit_rightmost_vertical, k_rightmost_vertical),
        "rightmost_horizontal": (fit_rightmost_horizontal, k_rightmost_horizontal),
        "centroid_vertical": (fit_centroid_vertical, k_centroid_vertical),
        "centroid_horizontal": (fit_centroid_horizontal, k_centroid_horizontal),
        "height": (fit_height, k_height),
        "width": (fit_width, k_width),
        "area": (fit_area, k_area),
        "centroids": centroids,
        "rightmost": rightmost,
        "widths": widths,
        "heights": heights,
        "areas": areas,
    }

    return fits


def segment(
    omegas,
    edges,
    right_loop_boundary,
    left_loop_boundary,
    right_pin_boundary,
    left_pin_boundary,
    ymax_orig,
    ymin_orig,
):
    loop_rectangles = []
    labeled_images = []

    output_queue = Queue()
    start_segmenting_time = time.time()

    jobs = []
    for angle, edge in zip(omegas, edges):
        p = Process(
            target=get_loop_rectangle,
            args=(
                edge,
                left_loop_boundary,
                right_loop_boundary,
                ymin_orig,
                ymax_orig,
            ),
            kwargs={
                "pin_xmin": left_pin_boundary,
                "pin_xmax": right_pin_boundary,
                "angle": angle,
                "output_queue": output_queue,
            },
        )
        jobs.append(p)
    for p in jobs:
        p.start()
    print("start finished")
    for p in jobs:
        p.join(timeout=0.01)
    end_segmenting_time = time.time()
    print("join finished")
    print(
        "Time to segment all %d images using multiprocessing %6.2f"
        % (len(jobs), end_segmenting_time - start_segmenting_time)
    )

    omegas = []
    results = []
    for p in jobs:
        results.append(output_queue.get())
    results.sort(key=lambda x: x[0])
    for r in results:
        angle, xmin, ymin, xmax, ymax, centroid, rightmost_point, labeled_image = r
        omegas.append(angle)
        loop_rectangles.append([xmin, ymin, xmax, ymax, centroid, rightmost_point])
        # labeled_images = np.hstack((labeled_images, labeled_image)) if labeled_images != np.array([]) else labeled_image
        labeled_images.append(labeled_image)

    labeled_images = np.array(labeled_images)
    return omegas, loop_rectangles, labeled_images


def subtract_background(img, normalized_background, background_norm):
    return np.abs(normalize(img) - normalized_background) * background_norm


def get_difference_images(
    images, omegas, background_image=None, dark=False, min_size=100, k=5
):
    _start = time.time()
    if background_image is None:
        background_image = np.median(images, axis=0)
    print("initial background check took %.4f" % (time.time() - _start))

    _s2 = time.time()
    if dark == False:
        # nimages, height, width = images.shape
        # images_reshaped = images.reshape((nimages, height*width))
        # normalized_background, background_norm = normalize(background_image.reshape((height*width,)), return_norm=True)
        # difference_images = np.apply_along_axis(subtract_background, 1, images_reshaped, normalized_background, background_norm)
        # difference_images = difference_images.reshape((nimages, height, width))
        normalized_background, background_norm = normalize(
            background_image, return_norm=True
        )
        difference_images = [
            subtract_background(img, normalized_background, background_norm)
            for img in images
        ]
    else:
        difference_images = copy.copy(images)
    _e2 = time.time()
    print("Background subtraction took %.4f" % (_e2 - _s2))

    _s3 = time.time()
    for dif in difference_images:
        try:
            boundary = get_threshold(dif)
        except:
            boundary = 5 * dif.mean()

        dif[dif < boundary] = 0.0
        remove_small_objects(dif != 0, min_size=min_size, in_place=True)
    _e3 = time.time()
    print("threshold and small object removal took %.4f" % (_e3 - _s3))

    _s4 = time.time()
    differences = np.array([img.flatten().mean() for img in difference_images])
    differences_mean = differences.mean()

    _e4 = time.time()
    print("differences_mean took %.4f" % (_e4 - _s4))

    _s5 = time.time()
    imagesomegas = []

    for difference, img, diff, omega in zip(
        differences, images, difference_images, omegas
    ):
        if (
            difference < k * differences_mean
            and len(np.argwhere(diff != 0)) > k * min_size
        ):
            imagesomegas.append((img, diff, omega))

    imagesomegas.sort(key=lambda x: x[2])

    images = [io[0] for io in imagesomegas]
    edges = [io[1] for io in imagesomegas]
    omegas = [io[2] for io in imagesomegas]
    _e5 = time.time()
    print("imagesomegas took %.4f" % (_e5 - _s5))

    _s6 = time.time()
    edges = np.array(edges)
    _e6 = time.time()
    print("edges took %.4f" % (_e6 - _s6))
    _end = time.time()
    print("get_difference_images took %.4f" % (_end - _start))
    return edges, images, omegas


# def get_edges(difference_images, images, omegas, min_size=100, k=5):
# _start = time.time()
# differences = np.array([img.flatten().mean() for img in difference_images])

# differences_median = np.median(differences)
# differences_sigma = np.std(differences)

# imagesomegas = []

# for difference, img, diff, omega in zip(differences, images, difference_images, omegas):
# if difference < 2*differences.mean() and len(np.argwhere(diff!=0)) > k*min_size:
# imagesomegas.append((img, diff, omega))

# imagesomegas.sort(key=lambda x: x[2])

# images = [io[1] for io in imagesomegas]
# omegas = [io[2] for io in imagesomegas]

# thresholds = np.array([get_threshold(img) for img in images])

# edges = np.array([remove_small_objects(img >= thresh, min_size=min_size) for img, thresh in zip(images, thresholds)])
# _end = time.time()
# print('get_edges took %.4f' % (_end-_start))
# return edges, images, omegas


def get_search_lines_peaks(edges, calibration, number_of_views, smoothing_factor=0.050):
    vs_start_time = time.time()

    vse = edges.sum(axis=0).mean(axis=0)
    hse = edges.sum(axis=0).mean(axis=1)
    if number_of_views > 1:
        hve = horizontal_variance(edges)
        vve = vertical_variance(edges)

    vs_end_time = time.time()
    print()
    print("Variances and sum calculation took %6.2f s" % (vs_end_time - vs_start_time,))

    search_start_time = time.time()

    search_line_hse = gaussian_filter1d(hse, smoothing_factor / calibration[0])
    search_line_vse = gaussian_filter1d(vse, smoothing_factor / calibration[0])
    peaks_vse = peakutils.indexes(
        search_line_vse, thres=0.01, min_dist=smoothing_factor / calibration[0]
    )
    print("peaks_vse", peaks_vse)

    if number_of_views > 1:
        search_line_hve = gaussian_filter1d(hve, smoothing_factor / calibration[0])
        search_line_vve = gaussian_filter1d(vve, smoothing_factor / calibration[0])
        peaks_vve = peakutils.indexes(
            search_line_vve, thres=0.01, min_dist=smoothing_factor / calibration[0]
        )
    else:
        search_line_hve, search_line_vve = [], []
        peaks_vve = []

    print("peaks_vve", peaks_vve)

    search_end_time = time.time()

    print("Search took %6.2f s" % (search_end_time - search_start_time,))
    print("get_search_lines_peaks took %6.2f" % (search_end_time - vs_start_time))
    return (
        peaks_vse,
        peaks_vve,
        search_line_hse,
        search_line_vse,
        search_line_hve,
        search_line_vve,
    )


def get_pin_and_loop_boundaries(
    peaks_vse,
    peaks_vve,
    search_line_hse,
    search_line_vse,
    search_line_hve,
    search_line_vve,
    calibration,
    smoothing_factor=0.05,
):
    _start = time.time()
    peak_interpret_start = time.time()

    if type(peaks_vse) != int and type(peaks_vve) != int:
        if len(peaks_vve) > len(peaks_vse):
            peaks = peaks_vve
            search_line = search_line_vve
        else:
            peaks = peaks_vse
            search_line = search_line_vse
    elif type(peaks_vve) == int:
        peaks = [peaks_vve]
    elif type(peaks_vse) == int:
        peaks = [peaks_vse]
    else:
        peaks = None

    print("peaks", peaks)

    right_boundary, left_boundary, gve = get_right_left_boundary(search_line)
    print("right_boundary", right_boundary)
    print("left_boundary", left_boundary)
    print(
        "right_boundary - left_boundary [mm]",
        (right_boundary - left_boundary) * calibration[0],
    )
    ymax_orig, ymin_orig, ghe = get_right_left_boundary(search_line_hse)
    print("ymin", ymin_orig)
    print("ymax", ymax_orig)

    if type(peaks) != type(None) and len(peaks) > 1:
        pin_peak = peaks[0]
        loop_peak = peaks[-1]
    elif type(peaks) != type(None) and len(peaks) == 1:
        loop_peak = min(
            [
                np.mean((right_boundary, left_boundary)),
                int(right_boundary - 2 * smoothing_factor / calibration[0]),
            ]
        )
        print("peaks[0] - loop_peak", loop_peak - peaks[0])
        if abs(loop_peak - peaks[0]) > 2 * smoothing_factor / calibration[0]:
            pin_peak = peaks[0]
        else:
            pin_peak = None
    else:
        if right_boundary - peaks > 2 * smoothing_factor / calibration[0]:
            pin_peak = peaks
            loop_peak = right_boundary - smoothing_factor / calibration[0]
        else:
            loop_peak = peaks
            pin_peak = None

    print("loop_peak", loop_peak)
    print("pin_peak", pin_peak)

    left_loop_boundary = int(loop_peak - (right_boundary - loop_peak) / 2)
    right_loop_boundary = int(right_boundary)

    print("initial left_loop_boundary", left_loop_boundary)
    print("initial right_loop_boundary", right_loop_boundary)

    if pin_peak != None:
        left_pin_boundary = left_boundary
        right_pin_boundary = pin_peak + (pin_peak - left_pin_boundary)
    else:
        left_pin_boundary = np.nan
        right_pin_boundary = np.nan

    print("initial left_pin_boundary", left_pin_boundary)
    print("initial right_pin_boundary", right_pin_boundary)

    # try to get loop and pin positions from gaussian fit of the right most area of the image
    a_loop, b_loop, c_loop = get_gaussian_fit(
        left_loop_boundary, right_loop_boundary, loop_peak, search_line
    )
    print("loop gaussian fit", a_loop, b_loop, c_loop)

    if left_pin_boundary != None and right_pin_boundary != None:
        a_pin, b_pin, c_pin = get_gaussian_fit(
            left_pin_boundary,
            right_pin_boundary + (left_pin_boundary - right_pin_boundary) / 2,
            pin_peak,
            search_line,
        )
        print("pin gaussian fit", a_pin, b_pin, c_pin)

    try:
        left_loop_boundary = max(
            [loop_peak - (right_loop_boundary - loop_peak), int(loop_peak - 1 * c_loop)]
        )
    except:
        pass

    try:
        right_pin_boundary = max([right_pin_boundary, int(pin_peak + 1.5 * c_pin)])
    except:
        pass

    left_pin_boundary = max([left_boundary, left_pin_boundary])
    left_loop_boundary = max([left_pin_boundary, left_loop_boundary])
    right_pin_boundary = min([left_loop_boundary, right_pin_boundary])

    peak_interpret_end = time.time()
    print(
        "Peak interpretation took %6.2f s "
        % (peak_interpret_end - peak_interpret_start)
    )

    print("left_loop_boundary", left_loop_boundary)
    print("right_loop_boundary", right_loop_boundary)

    print("left_pin_boundary", left_pin_boundary)
    print("right_pin_boundary", right_pin_boundary)
    _end = time.time()
    print("get_pin_and_loop_boundaries took %.4f" % (_end - _start))
    print()
    return (
        right_loop_boundary,
        left_loop_boundary,
        right_pin_boundary,
        left_pin_boundary,
        search_line,
        ymax_orig,
        ymin_orig,
        gve,
    )


def optical_path_analysis(
    images,
    omegas,
    calibration,
    background_image=None,
    display=False,
    smoothing_factor=0.050,
    min_size=100,
    threshold_method="otsu",
    template="optical_alignment",
    generate_report=False,
    dark=False,
    save_images=False,
    destination="shapes_of_light",
    save_results=True,
    generate_color_and_gray=True,
):
    _start = time.time()

    number_of_views = len(images)

    if images[0].shape[-1] == 3:
        images_rgb = images[::]
        images = images.mean(axis=3)
    else:
        images_rgb = None

    background_start = time.time()

    edges, images, omegas = get_difference_images(
        images, omegas, background_image=background_image, dark=dark, min_size=min_size
    )

    # edges, images, omegas = get_edges(difference_images, images, omegas, min_size=min_size)

    background_end = time.time()
    print("Background treatment took %6.2f s" % (background_end - background_start))

    if len(images) == 0:
        print("optical_path_analysis: Sample does not seem to be visible on the image")
        return -1

    print(
        "optical_path_analysis: There seems to be a sample visible in the image -- will try to locate precisely!"
    )

    (
        peaks_vse,
        peaks_vve,
        search_line_hse,
        search_line_vse,
        search_line_hve,
        search_line_vve,
    ) = get_search_lines_peaks(
        edges, calibration, number_of_views, smoothing_factor=smoothing_factor
    )

    (
        right_loop_boundary,
        left_loop_boundary,
        right_pin_boundary,
        left_pin_boundary,
        search_line,
        ymax_orig,
        ymin_orig,
        gve,
    ) = get_pin_and_loop_boundaries(
        peaks_vse,
        peaks_vve,
        search_line_hse,
        search_line_vse,
        search_line_hve,
        search_line_vve,
        calibration,
        smoothing_factor=smoothing_factor,
    )

    omegas, loop_rectangles, labeled_images = segment(
        omegas,
        edges,
        right_loop_boundary,
        left_loop_boundary,
        right_pin_boundary,
        left_pin_boundary,
        ymax_orig,
        ymin_orig,
    )

    fits = get_fits(omegas, loop_rectangles)
    _end = time.time()
    print("Analysis took %.4f" % (_end - _start))

    if save_results:
        save_fits(fits, destination, os.path.basename(template))

    if save_images:
        save_all_images(
            images,
            edges,
            labeled_images,
            omegas,
            os.path.basename(template),
            destination,
            images_rgb,
            generate_color_and_gray=generate_color_and_gray,
        )

    if generate_report:
        _generate_report(
            images,
            loop_rectangles,
            labeled_images,
            right_loop_boundary,
            left_loop_boundary,
            right_pin_boundary,
            left_pin_boundary,
            search_line,
            fits,
            calibration,
            gve,
            destination=os.path.join(destination, os.path.basename(template)),
            template="",
            display=display,
        )

    return fits


def save_individual_images(images, angles, prefix, template, suffix, destination):
    _s = time.time()
    filename_template = prefix + "_%.2f." + suffix
    destination = os.path.join(destination, template)
    if not os.path.isdir(destination):
        os.makedirs(destination)
    for img, angle in zip(images, angles):
        filename = os.path.join(destination, filename_template % angle)
        print(filename, img.shape)
        imsave(filename, img)
    _e = time.time()
    print("Saving %s took %.4f" % (prefix, _e - _s))


def save_all_images(
    images,
    edges,
    labeled_images,
    omegas,
    template,
    destination,
    images_rgb,
    generate_color_and_gray=True,
):
    _s = time.time()
    if generate_color_and_gray:
        ims, prefix, suffix = (images, "gray", "jpg")
        save_individual_images(
            ims, omegas, prefix, os.path.basename(template), suffix, destination
        )
        if images_rgb is not None:
            ims, prefix, suffix = (images_rgb, "color", "jpg")
            save_individual_images(
                ims, omegas, prefix, os.path.basename(template), suffix, destination
            )
    ims, prefix, suffix = (labeled_images.astype(np.int8), "labeled", "png")
    save_individual_images(
        ims, omegas, prefix, os.path.basename(template), suffix, destination
    )
    binary_images = labeled_images.astype(np.int8)
    binary_images[binary_images != 0] = 1

    images = np.array(images)
    masked_background_images = images[::]
    masked_background_images[labeled_images == 0] = 0

    to_save = [
        (edges, "difference", "jpg"),
        (binary_images, "binary", "png"),
        (masked_background_images, "foreground", "jpg"),
    ]
    for item in to_save:
        ims, prefix, suffix = item
        save_individual_images(
            ims, omegas, prefix, os.path.basename(template), suffix, destination
        )
    _e = time.time()
    print("Saving all images took %.4f" % (_e - _s))


def save_fits(fits, destination, name_pattern):
    _s = time.time()
    fits_filename = os.path.join(destination, name_pattern, "fits.pickle")
    directory = os.path.dirname(fits_filename)
    if not os.path.isdir(directory):
        os.makedirs(directory)
    else:
        print("in save fits directory %s already exists" % directory)
    f = open(fits_filename, "w")
    pickle.dump(fits, f)
    f.close()
    _e = time.time()
    print("Saving fits took %.4f" % (_e - _s))


def main():
    import optparse

    parser = optparse.OptionParser()

    # parser.add_option('-i', '--images', type=str, default='/927bis/ccd/gitRepos/Sample_in_refractive_media/pcks/images_ahmed_7_just_mounted_zoom1.pickle', help='path to the file to the pickled images or one of the images of a series')
    parser.add_option(
        "-i",
        "--images",
        type=str,
        default="/home/smartin/ResearchDevelopmentEducation/learning_dataset/184451_Thu_Oct__3_23:37:20_2019",
        help="path to the file to the pickled images or one of the images of a series",
    )
    parser.add_option(
        "-d",
        "--destination",
        type=str,
        default="/home/smartin/ResearchDevelopmentEducation/learning_dataset_targets",
        help="where to save the outputs and intermediates for future learning",
    )
    parser.add_option("-D", "--display", action="store_true", help="Show the plot")
    parser.add_option(
        "-s",
        "--smoothing_factor",
        type=float,
        default=0.05,
        help="Smoothing factor for the search line",
    )
    parser.add_option(
        "-m",
        "--min_size",
        type=float,
        default=100,
        help="Minimum object size in loop segment",
    )
    parser.add_option(
        "-t", "--threshold_method", type=str, default="otsu", help="Threshold method"
    )
    parser.add_option(
        "-g", "--generate_report", action="store_true", help="Generate report"
    )
    parser.add_option("-S", "--save_images", action="store_true", help="Save images")
    parser.add_option("-R", "--save_results", action="store_true", help="Save fits")
    parser.add_option("-f", "--force", action="store_true", help="Overwrite if exists")

    options, args = parser.parse_args()
    print("options")
    print(options)

    name_pattern = (
        options.images.replace("*.png", "")
        .replace(".pck", "")
        .replace(".pickle", "")
        .replace(".h5", "")
    )

    if (
        os.path.isdir(os.path.join(options.destination, os.path.basename(name_pattern)))
        and options.force != True
    ):
        sys.exit("already touched, moving on ...")
    else:
        print(os.path.join(options.destination, os.path.basename(name_pattern)))
        print("This was not done yet, getting on it ...")

    if os.stat(options.images).st_size < 10e3 and options.images[-3:] == ".h5":
        sys.exit("size of history file is too small, moving on ...")

    background_image = None
    generate_color_and_gray = True
    if "png" in options.images:
        images, omegas, calibration = load_from_file(options.images)
        generate_color_and_gray = False
    elif os.path.isdir(options.images):
        images_rgb, omegas, calibration = load_from_directory(options.images)
        generate_color_and_gray = False
    elif ".pickle" in options.images or ".pck" in options.images:
        images_rgb, omegas, calibration = load_from_pickle(options.images)
    elif ".h5" in options.images and "history" in options.images:
        from camera import camera

        cam = camera()
        m = h5py.File(options.images, "r")
        images_rgb = m["history_images"][()]
        # images = images_rgb.mean(axis=3)
        hsv = m["history_state_vectors"][()]
        omegas = hsv[:, 0]
        if sys.version_info < (3, 0):
            clicks = pickle.load(
                open(options.images.replace("_history.h5", "_clicks.pickle"))
            )
            calibration = clicks["calibrations"][0][0]
            mean_calibration = np.mean(clicks["calibrations"])
        else:
            clicks = pickle.load(
                open(options.images.replace("_history.h5", "_clicks.pickle"), "rb"),
                encoding="bytes",
            )
            calibration = clicks[b"calibrations"][0][0]
            mean_calibration = np.mean(clicks[b"calibrations"])
        background_image = cam.get_default_background(
            zoom=cam.get_zoom_from_calibration(mean_calibration)
        ).mean(axis=2)

    if options.generate_report != True:
        generate_report = False
    else:
        generate_report = True

    if options.save_images != True:
        save_images = False
    else:
        save_images = True

    if options.save_results != True:
        save_results = False
    else:
        save_results = True

    if not os.path.isdir(options.destination):
        os.makedirs(options.destination)

    optical_path_analysis(
        images_rgb,
        omegas,
        calibration,
        background_image=background_image,
        display=options.display,
        smoothing_factor=options.smoothing_factor,
        min_size=options.min_size,
        threshold_method=options.threshold_method,
        template=name_pattern,
        generate_report=generate_report,
        save_images=save_images,
        destination=options.destination,
        save_results=save_results,
        generate_color_and_gray=generate_color_and_gray,
    )


if __name__ == "__main__":
    main()
    # pass
