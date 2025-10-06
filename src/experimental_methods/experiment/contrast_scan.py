#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import pylab
from oav_camera import oav_camera as camera
from goniometer import goniometer
import numpy as np


def main(halfrange=0.1, nsteps=25):
    cam = camera()
    g = goniometer()

    x = np.linspace(-halfrange, halfrange, nsteps)
    # x = x[::-1]
    contrasts = []

    for p in x:
        g.set_position({"AlignmentX": p}, wait=True)
        time.sleep(0.01)
        img = cam.get_image()
        contrasts.append(cam.get_contrast(image=img[300:-300, 400:-400]))

    pylab.plot(x, contrasts)
    pylab.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--halfrange", default=0.1, type=float, help="half range")
    parser.add_argument("-n", "--nsteps", default=25, type=int, help="number of steps")
    args = parser.parse_args()
    main(args.halfrange)
