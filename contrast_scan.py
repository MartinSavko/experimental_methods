#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import pylab
from oav_camera import oav_camera as camera
from goniometer import goniometer
import numpy as np
from useful_routines import get_string_from_timestamp

def main(halfrange=0.1, nsteps=25, crop=128):
    cam = camera()
    g = goniometer()

    center = np.array(cam.get_image_dimensions()) // 2
    x = np.linspace(-halfrange, halfrange, nsteps)
    # x = x[::-1]
    pylab.figure()
    contrasts = []
    omega_start = g.get_omega_position()
    for orientation in [0, 180, 270, 90]:
        contrast_curve = []
        for p in x:
            g.set_position({"AlignmentX": p, "Omega": omega_start + orientation}, wait=True)
            time.sleep(0.01)
            img = cam.get_image()
            contrast_curve.append(cam.get_contrast(image=img[center[0]-crop: center[0] + crop, center[1]-crop: center[1] + crop]))
        contrasts.append(contrast_curve)
        pylab.plot(x, contrast_curve, 'o', label=f"{orientation:.1f}")
    contrasts = np.array(contrasts)
    cmean = contrasts.mean(axis=0)
    pylab.plot(x, cmean, label="mean")
    pylab.legend()
    pylab.xlabel("AlignmentX [mm]")
    pylab.ylabel("Contrast")
    pylab.savefig(f"contrast_scan_{get_string_from_timestamp()}.png")
    print(f"contrasts max at {x[np.argmax(cmean)]:.4f}")
    
    pylab.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,)
    parser.add_argument("-r", "--halfrange", default=0.05, type=float, help="half range")
    parser.add_argument("-n", "--nsteps", default=25, type=int, help="number of steps")
    args = parser.parse_args()
    main(args.halfrange)
