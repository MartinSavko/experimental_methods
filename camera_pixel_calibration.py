#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import pylab

from skimage.feature import match_template
from random import random, choice

from oav_camera import oav_camera as camera
from goniometer import goniometer

from useful_routines import (
    get_image_shift_from_com,
    get_string_from_timestamp,
    save_pickled_file,
    get_cx_and_cy,
)

g = goniometer()
cam = camera()

def find_match(init_image, new_image, template):
    mt = match_template(new_image.mean(axis=2), template)
    mt = mt.reshape(mt.shape[:2])
    _skimage = np.unravel_index(mt.argmax(), mt.shape)
    _cv = get_image_shift_from_com(init_image, new_image)
    return _skimage, _cv


def main():
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-n",
        "--name_pattern",
        default="date",
        type=str,
        help="distinguishing name for the result files",
    )
    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        default="/nfs/data4/2026_Run2/com-proxima2a/Commissioning/camera_pixel_calibration/2024-03-29",
        help="directory",
    )

    parser.add_argument(
        "-z", "--zoom", default=1, type=int, help="Zoom (default=%default)"
    )
    parser.add_argument(
        "-N", "--number_of_points", default=40, type=int, help="Number of points"
    )
    parser.add_argument("-c", "--crop", default=64, type=int, help="crop size")

    parser.add_argument("-m", "--method", default="skimage", type=str, help="method")

    parser.add_argument("-s", "--stage", default="centring", type=str, help="stage for horizontal movements")
    
    args = parser.parse_args()

    zoom = args.zoom
    number_of_points = args.number_of_points
    if args.name_pattern == "date":
        name_pattern = get_string_from_timestamp()
    else:
        name_pattern = args.name_pattern

    g.set_zoom(zoom)
    time.sleep(0.25)
    
    init_image = cam.get_image(color=True)

    shape = np.array(init_image.shape[:2])
    center = shape / 2

    crop = args.crop
    a = int(center[0] - crop)
    b = int(center[0] + crop)
    c = int(center[1] - crop)
    d = int(center[1] + crop)

    template = init_image.mean(axis=2)[a:b, c:d]

    available_range = shape * cam.get_calibration() * 0.15

    reference_position = g.get_aligned_position()

    h = reference_position["AlignmentZ"]
    v = reference_position["AlignmentY"]
    
    _skimage, _cv = find_match(init_image, init_image, template)
    
    results = [[0.0, 0.0, _skimage[0], _skimage[1], _cv[0], _cv[1]]]

    for k in range(number_of_points):
        print(f"point {k+1:d}")
        v = random() * available_range[0] * choice([-1, 1])
        h = random() * available_range[1] * choice([-1, 1])

        new_position = dict(
            [(key, reference_position[key]) for key in reference_position]
        )
        # new_position['AlignmentZ'] += v
        # new_position['AlignmentY'] += h

        if args.stage == "centring":
            cx, cy = get_cx_and_cy(0., h, g.get_omega_position())
            new_position["CentringX"] += cx
            new_position["CentringY"] += cy
        else:
            new_position["AlignmentZ"] += h
            
        new_position["AlignmentY"] += v

        g.set_position(new_position, wait=True)
        time.sleep(0.15)
        new_image = cam.get_image(color=True)
        # if args.method == "skimage":
        _skimage, _cv = find_match(init_image, new_image, template)

        result = [
            v,
            h,
            _skimage[0],
            _skimage[1],
            _cv[0],
            _cv[1],
        ]
        print(f"shifts {result}")
        results.append(result)

    results = np.array(results)
    results_dictionary = {"results": results}
    print(f"results\n{results}")
    results_from_origin = results - results[0, :]
    rfo = results_from_origin[1:, :]

    print(f"rfo\n{rfo}")
    
    results_dictionary["results_from_origin"] = results_from_origin
    results_dictionary["rfo"] = rfo

    horizontal_calibration_skimage = np.abs(rfo[:, 1]) / np.abs(rfo[:, 3])
    vertical_calibration_skimage = np.abs(rfo[:, 0]) / np.abs(rfo[:, 2])

    horizontal_calibration_cv = np.abs(rfo[:, 1]) / np.abs(rfo[:, -1])
    vertical_calibration_cv = np.abs(rfo[:, 0]) / np.abs(rfo[:, -2])

    results_dictionary[
        "horizontal_calibration_skimage"
    ] = horizontal_calibration_skimage
    results_dictionary["vertical_calibration_skimage"] = vertical_calibration_skimage

    results_dictionary["horizontal_calibration_cv"] = horizontal_calibration_cv
    results_dictionary["vertical_calibration_cv"] = vertical_calibration_cv

    print(f"vertical_calibration; current {cam.get_calibration()[0]}")
    print("skimage")
    print(vertical_calibration_skimage)
    print(
        "vertical mean %.10f, median %.10f"
        % (vertical_calibration_skimage.mean(), np.median(vertical_calibration_skimage))
    )

    print("cv")
    print(vertical_calibration_cv)
    print(
        "vertical mean %.10f, median %.10f"
        % (vertical_calibration_cv.mean(), np.median(vertical_calibration_cv))
    )

    print(f"horizontal_calibration; current {cam.get_calibration()[1]}")
    print("skimage")
    print(horizontal_calibration_skimage)
    print(
        "horizontal mean %.10f, median %.10f"
        % (
            horizontal_calibration_skimage.mean(),
            np.median(horizontal_calibration_skimage),
        )
    )

    print("cv")
    print(horizontal_calibration_cv)
    print(
        "horizontal mean %.10f, median %.10f"
        % (
            horizontal_calibration_cv.mean(),
            np.median(horizontal_calibration_cv),
        )
    )

    g.set_position(reference_position)

    fname = os.path.join(args.directory, f"{args.name_pattern}_{zoom:d}.pickle")
    save_pickled_file(fname, results_dictionary)

    pylab.figure(figsize=(16, 9))
    pylab.title(f"Camara pixel size calibration at zoom {zoom}")
    pylab.plot(vertical_calibration_skimage, label="vertical calibration skimage")
    pylab.plot(vertical_calibration_cv, label="vertical calibration cv")
    pylab.plot(horizontal_calibration_skimage, label="horizontal_calibration skimage")
    pylab.plot(horizontal_calibration_cv, label="horizontal_calibration cv")
    pylab.grid(True)
    pylab.xlabel("point")
    pylab.ylabel("mm/pixel")
    pylab.legend()
    figname = os.path.join(args.directory, f"{args.name_pattern}_{zoom:d}.png")
    pylab.savefig(figname)
    pylab.show()


# old zoom 10 calibration X: 1.5746e-4, Y: 1.6108e-4
def get_current_calibrations(sleep_time=0.5):
    calibrations = {}
    for z in range(10, 0, -1):
        g.set_zoom(z, wait=True)
        time.sleep(sleep_time)
        calibrations[z] = np.array([g.md.CoaxCamScaleY, g.md.CoaxCamScaleX])

    return calibrations


def test():
    pass


if __name__ == "__main__":
    main()
