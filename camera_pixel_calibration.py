#!/usr/bin/env python
# -*- coding: utf-8 -*-

from skimage.feature import match_template
from random import random, choice

from oav_camera import oav_camera as camera
from goniometer import goniometer

import pickle
import numpy as np
import pylab
import time

g = goniometer()
cam = camera()


def main():
    import optparse

    parser = optparse.OptionParser()
    parser.add_option(
        "-z", "--zoom", default=1, type=int, help="Zoom (default=%default)"
    )
    parser.add_option(
        "-N", "--number_of_points", default=40, type=int, help="Number of points"
    )
    parser.add_option("-c", "--crop", default=64, type=int, help="crop size")
    parser.add_option(
        "-n",
        "--name_pattern",
        default="date",
        type="str",
        help="distinguishing name for the result files",
    )
    options, args = parser.parse_args()

    zoom = options.zoom
    number_of_points = options.number_of_points
    if options.name_pattern == "date":
        name_pattern = time.strftime("%Y-%M-%d")
    else:
        name_pattern = options.name_pattern

    g.set_zoom(zoom)

    init_image = cam.get_image(color=False)

    shape = np.array(init_image.shape[:2])
    center = shape / 2

    crop = options.crop
    a = int(center[0] - crop)
    b = int(center[0] + crop)
    c = int(center[1] - crop)
    d = int(center[1] + crop)

    template = init_image[a:b, c:d]

    available_range = shape * cam.get_calibration() * 0.15

    reference_position = g.get_aligned_position()

    h = reference_position["AlignmentZ"]
    v = reference_position["AlignmentY"]

    mt = match_template(init_image, template)
    mt = mt.reshape(mt.shape[:2])
    position_max = np.unravel_index(mt.argmax(), mt.shape)

    results = [[0.0, 0.0, position_max[0], position_max[1]]]

    for k in range(number_of_points):
        print("point %d," % k)
        v = random() * available_range[0] * choice([-1, 1])
        h = random() * available_range[1] * choice([-1, 1])

        new_position = dict(
            [(key, reference_position[key]) for key in reference_position]
        )
        # new_position['AlignmentZ'] += v
        # new_position['AlignmentY'] += h

        new_position["AlignmentZ"] += h
        new_position["AlignmentY"] += v

        g.set_position(new_position, wait=True)

        mt = match_template(cam.get_image(color=False), template)
        mt = mt.reshape(mt.shape[:2])
        position_max = np.unravel_index(mt.argmax(), mt.shape)
        result = [v, h, position_max[0], position_max[1]]
        results.append(result)

    results = np.array(results)
    results_dictionary = {"results": results}

    results_from_origin = results - results[0, :]
    rfo = results_from_origin[1:, :]

    results_dictionary["results_from_origin"] = results_from_origin
    results_dictionary["rfo"] = rfo

    horizontal_calibration = np.abs(rfo[:, 1]) / np.abs(rfo[:, 3])
    vertical_calibration = np.abs(rfo[:, 0]) / np.abs(rfo[:, 2])

    results_dictionary["horizontal_calibration"] = horizontal_calibration
    results_dictionary["vertical_calibration"] = vertical_calibration

    print("vertical_calibration")
    print(vertical_calibration)
    print(
        "vertical mean %.10f, median %.10f"
        % (vertical_calibration.mean(), np.median(vertical_calibration))
    )
    print("current vertical camera calibration", cam.get_calibration()[0])
    print("horizontal_calibration")
    print(horizontal_calibration)
    print(
        "horizontal mean %.10f, median %.10f"
        % (horizontal_calibration.mean(), np.median(horizontal_calibration))
    )
    print("current horizontal camera calibration", cam.get_calibration()[1])
    print()

    g.set_position(reference_position)

    f = open(
        "camera_calibration_results_zoom_%d_%s.pickle" % (zoom, name_pattern), "wb"
    )
    pickle.dump(results_dictionary, f)
    f.close()

    pylab.plot(vertical_calibration, label="vertical calibration")
    pylab.plot(horizontal_calibration, label="horizontal_calibration")
    pylab.grid(True)
    pylab.xlabel("point")
    pylab.ylabel("mm/pixel")
    pylab.legend()
    pylab.savefig("camera_calibration_results_zoom_%d_%s.png" % (zoom, name_pattern))
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
