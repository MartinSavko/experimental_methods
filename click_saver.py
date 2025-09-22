#!/usr/bin/env python

# from camera import camera
from oav_camera import oav_camera
import h5py
import time
import os
import numpy as np

try:
    import simplejpeg
except ImportError:
    import complexjpeg as simplejpeg


def main():
    import optparse

    parser = optparse.OptionParser()

    parser.add_option("-d", "--directory", type=str, help="directory")
    parser.add_option("-n", "--name_pattern", type=str, help="filename template")
    parser.add_option("-y", "--click_y", type=float, help="click y")
    parser.add_option("-x", "--click_x", type=float, help="click x")
    parser.add_option(
        "-t",
        "--timestamp",
        default=None,
        type=float,
        help="if specified will look for image in history corresponding to the timestamp instead of grabing a new one",
    )

    options, args = parser.parse_args()

    cam = oav_camera()
    s = time.time()
    click = np.array([options.click_y, options.click_x])
    if options.timestamp != None:
        image = cam.get_image_corresponding_to_timestamp(options.timestamp)
    else:
        image = cam.get_image()
    zoom = cam.get_zoom()
    calibration = cam.get_calibration()

    e = time.time()

    print("image read in %.3f seconds" % (e - s))

    if not os.path.isdir(options.directory):
        os.makedirs(options.directory)

    s = time.time()
    image_filename = "%s_zoom_%d_y_%d_x_%d.jpg" % (
        os.path.join(options.directory, options.name_pattern),
        int(zoom),
        int(click[0]),
        int(click[1]),
    )
    cam.save_image(image_filename)

    double_click_file = h5py.File(
        "%s.h5" % os.path.join(options.directory, options.name_pattern), "w"
    )

    # dt = h5py.special_dtype(vlen=np.dtype('uint8'))

    jpeg = simplejpeg.encode_jpeg(image)
    jpeg = np.frombuffer(jpeg, dtype="uint8")

    double_click_file.create_dataset("image", data=jpeg)

    # double_click_file.create_dataset('image',
    # data=image,
    # compression='gzip',
    # dtype=np.uint8)

    double_click_file.create_dataset("zoom", data=int(zoom))

    double_click_file.create_dataset("click", data=click)

    double_click_file.create_dataset("calibration", data=calibration)

    double_click_file.close()

    e = time.time()

    print("click written in %.3f seconds" % (e - s))


if __name__ == "__main__":
    main()
