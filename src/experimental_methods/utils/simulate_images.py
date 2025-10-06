#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import itertools
import pylab
import gzip
import skimage
import time
import simplejpeg

detector_families = {
    "pilatus": {
        "nmodules": {
            "12M": (5, 24),
            "6M": (5, 12),
            "2M": (3, 8),
            "1M": (2, 5),
            "300K-W": (3, 1),
            "300K": (1, 3),
            "200K": (1, 2),
            "100K": (1, 1),
        },
        "module": {
            "size": (487, 195),
            "gap": (7, 17),
            "pixel_size": (0.172e-03, 0.172e-03),
            "nchips": (8, 2),
        },
        "chip": {
            "size": (60, 97),
            "gap": (1, 1),
        },
        "sizes": {},  # will be populated with correct sizes
    },
    "eiger": {
        "nmodules": {
            "1M": (1, 2),
            "4M": (2, 4),
            "9M": (3, 6),
            "16M": (4, 8),
        },
        "module": {
            "size": (1030, 514),
            "gap": (10, 37),
            "pixel_size": (0.075e-03, 0.075e-03),
            "nchips": (4, 2),
        },
        "chip": {
            "size": (256, 256),
            "gap": (2, 2),
        },
        "sizes": {},  # will be populated with correct sizes
    },
}


def get_blank_image(family="eiger", model="9M"):
    _start = time.time()
    gw, gh = detector_families[family]["module"]["gap"]
    mw, mh = detector_families[family]["module"]["size"]
    cols, rows = detector_families[family]["nmodules"][model]
    shape = (mh * rows + (rows - 1) * gh, mw * cols + (cols - 1) * gw)
    print("%s %s shape %s" % (family, model, shape))
    blank_image = np.ones(shape, dtype=np.uint8) * 255
    gap_mask = np.zeros(shape, dtype=np.uint8)
    for i in range(1, rows):
        start = i * mh + (i - 1) * gh + 1
        gap_mask[start : start + gh, :] = 1
    for i in range(1, cols):
        start = i * mw + (i - 1) * gw + 1
        gap_mask[:, start : start + gw] = 1

    blank_image[gap_mask == 1] = 3 * 64
    print("blank_image generation took %.4f seconds" % (time.time() - _start))
    return blank_image


def get_spots(spot_file, mode="r"):
    _start = time.time()
    spots = gzip.open(spot_file, mode=mode).read().decode().split("\n")
    print("spots read in %.4f seconds" % (time.time() - _start))
    return spots[:-1]


def draw_image(
    spot_file,
    blank_image=None,
    family="eiger",
    model="9M",
    model_spot=skimage.morphology.disk(3),
):
    _start = time.time()
    if blank_image is None:
        image = get_blank_image(family=family, model=model)
    else:
        image = blank_image[:]
    spots = get_spots(spot_file)
    model_spot *= 0
    spot_sizev, spot_sizeh = model_spot.shape
    spot_sizev = int(spot_sizev / 2)
    spot_sizeh = int(spot_sizeh / 2)
    k = 0
    _start_ds = time.time()
    for spot in spots:
        x, y, n, i = map(float, spot.split(" "))
        v = int(y)
        h = int(x)
        image[
            v - spot_sizev : v + spot_sizev + 1, h - spot_sizeh : h + spot_sizeh + 1
        ] = model_spot
        k += 1
    print("all %d spots added in %.4f seconds" % (k, time.time() - _start_ds))
    print(
        "drawing of image with %d spots took %.4f seconds" % (k, time.time() - _start)
    )
    _start = time.time()
    f = open("/tmp/img2.jpg", "wb")
    f.write(simplejpeg.encode_jpeg(np.expand_dims(image, 2), colorspace="gray"))
    f.close()
    print("jpeg encoded and saved in %.4f seconds" % (time.time() - _start))
    # pylab.imshow(image, cmap='gist_yarg')
    ##pylab.show()


if __name__ == "__main__":
    ##for model in ['1M', '4M', '9M', '16M']:
    # get_blank_image(model=model)
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        default="/nfs/data4/2023_Run5/20201290/2023-11-04/RAW_DATA/krey/160/4",
        help="directory",
    )
    parser.add_argument(
        "-n", "--name_pattern", type=str, default="data_2_1", help="name_pattern"
    )
    parser.add_argument(
        "-i", "--image_number", type=int, default=3600, help="image number"
    )
    parser.add_argument("-s", "--spot_size", type=int, default=5, help="spot size")
    parser.add_argument(
        "-p",
        "--spot_file_path_pattern",
        type=str,
        default="{directory:s}/spot_list/{name_pattern:s}_{image_number:06d}.adx.gz",
        help="spot file path pattern",
    )
    # '{directory:s}/{name_pattern:s}_cbf/spot_list/{name_pattern:s}_{image_number:06d}.adx.gz'
    args = parser.parse_args()

    print(args)
    dargs = dict(args._get_kwargs())
    dargs["model_spot"] = skimage.morphology.disk(args.spot_size)
    spot_file = args.spot_file_path_pattern.format(**dargs)

    blank_image = get_blank_image(family="eiger", model="9M")
    draw_image(spot_file, blank_image=blank_image)
