#!/usr/bin/env python

# from camera import camera
from oav_camera import oav_camera
from goniometer import goniometer
import h5py
import time
import os
import numpy as np

try:
    import simplejpeg
except ImportError:
    import complexjpeg as simplejpeg

from useful_routines import get_vector_from_position

def main(
    motor_names=[
        "AlignmentX",
        "AlignmentY",
        "AlignmentZ",
        "CentringX",
        "CentringY",
        "Kappa",
        "Phi",
        "Omega",
    ]
):
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )


    parser.add_argument("-d", "--directory", type=str, help="directory")
    parser.add_argument("-n", "--name_pattern", type=str, help="filename template")
    parser.add_argument("-y", "--click_y", type=float, help="click y")
    parser.add_argument("-x", "--click_x", type=float, help="click x")
    parser.add_argument("-o", "--omega", default=None, type=float, help="omega")
    parser.add_argument(
        "-t",
        "--timestamp",
        default=None,
        type=float,
        help="if specified will look for image in history corresponding to the timestamp instead of grabing a new one",
    )

    args = parser.parse_args()

    s = time.time()

    cam = oav_camera()
    g = goniometer()

    click = np.array([args.click_y, args.click_x]).astype(int)
    if args.timestamp != None:
        image = cam.get_image_corresponding_to_timestamp(args.timestamp)
    else:
        image = cam.get_image()
        
    zoom = cam.get_zoom()
    calibration = cam.get_calibration()
    position = g.get_aligned_position(
        motor_names=motor_names
    )
    if args.omega is not None:
        position["Omega"] = args.omega
        
    e = time.time()

    print("image read in %.3f seconds" % (e - s))

    if not os.path.isdir(args.directory):
        os.makedirs(args.directory)

    s = time.time()
    fname_base = os.path.join(args.directory, args.name_pattern)
    #image_filename = f'{fname_base}_ax_{position["AlignmentX"]:.3f}_ay_{position["AlignmentY"]:.3f}_az_{position["AlignmentZ"]:.3f}_cx_{position["CentringX"]:.3f}_cy_{position["CentringY"]:.3f}_y_{click[0]:d}_x_{click[1]:d}.jpg'

    image_filename = f'{fname_base}_y_{click[0]:d}_x_{click[1]:d}.jpg'
    
    # image_filename = "%s_zoom_%d_y_%d_x_%d.jpg" % (
    # fname_base,
    # ay,
    # az,
    # cx,
    # cy,
    # o,
    # k,
    # p,
    # int(zoom),
    # int(click[0]),
    # int(click[1]),
    # )

    cam.save_image(image_filename)

    double_click_file = h5py.File(
        "%s.h5" % os.path.join(args.directory, args.name_pattern), "w"
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
    
    double_click_file.create_dataset("position", data=get_vector_from_position(position, keys=motor_names))

    double_click_file.close()

    e = time.time()

    print("click written in %.3f seconds" % (e - s))


if __name__ == "__main__":
    main()
