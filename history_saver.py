#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import h5py
import time
import os
import numpy as np
from oav_camera import oav_camera

try:
    import simplejpeg
except ImportError:
    import complexjpeg as simplejpeg


def get_jpegs_from_arrays(images):
    jpegs = []
    for img in images:
        jpeg = simplejpeg.encode_jpeg(img)
        jpeg = np.frombuffer(jpeg, dtype="uint8")
        jpegs.append(jpeg)
    return jpegs


def main():
    import optparse

    parser = optparse.OptionParser()

    parser.add_option("-d", "--directory", type=str, help="directory")
    parser.add_option("-n", "--name_pattern", type=str, help="filename template")
    parser.add_option("-s", "--start", type=float, help="start")
    parser.add_option("-e", "--end", type=float, help="end")
    parser.add_option("-S", "--suffix", default="history", type=str, help="suffix")

    options, args = parser.parse_args()
    print("options", options)
    print("args", args)
    filename = "%s_%s.h5" % (
        os.path.join(options.directory, options.name_pattern),
        options.suffix,
    )


if __name__ == "__main__":
    main()
