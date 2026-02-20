#!/usr/bin/env python

import os
import time
import pickle
import pylab
import skimage
import cv2 as cv
import numpy as np
import seaborn as sns

# https://opencv.org/blog/image-annotation-using-opencv/
# https://learnopencv.com/applycolormap-for-pseudocoloring-in-opencv-c-python/
# https://stackoverflow.com/questions/42406338/why-cv2-imwrite-changes-the-color-of-pics
# https://www.30secondsofcode.org/python/s/hex-to-rgb/

# assert rgb_to_hex((255, 165, 1)) # 'FFA501'
from useful_routines import (
    black,
    white,
    yellow,
    blue,
    green,
    magenta,
    cyan,
    purpre,
    notions,
    colors_for_labels,
    movie2images,
    images2movie,
    get_color,
    get_lut,
    label2rgb,
    hex_to_rgb,
    rgb_to_hex,
    get_bbox_from_mask,
)


def omalovanka(
    hierarchical_mask,
    save=False,
    destination="examples/opti/detections",
    prefix="hm",
    suffix="png",
    lut=None,
    negative=False,
):
    gre = cv.cvtColor(hierarchical_mask, cv.COLOR_GRAY2RGB)
    if lut is None:
        lut = get_lut(negative=negative)
    rgb = cv.LUT(gre, lut)
    if save:
        fname = os.path.join(destination, f"{prefix}.{suffix}")
        cv.imwrite(fname, cv.cvtColor(rgb, cv.COLOR_RGB2BGR))
    return rgb


def get_pts(d, key="aoi_bbox", shape=None):
    if len(key) == 2:
        mask = d[key[0]][key[1]]
        x, y, w, h = get_bbox_from_mask(mask)
        pt1 = y, x
        pt2 = y + h, x + w
    else:
        if type(d[key]) is tuple:
            present, r, c, h, w = d[key][:5]
        elif type(d[key]) is dict:
            present, r, c, h, w = [d[key][w] for w in ["present", "r", "c", "h", "w"]]

        print("present, r, c, h, w", present, int(r), int(c), h, w)

        pt1 = r - h / 2, c - w / 2
        pt2 = r + h / 2, c + w / 2

    pt1 = np.array(pt1)
    pt2 = np.array(pt2)

    if key in ["aoi_bbox"]:
        shape = np.array(d["prediction_shape"])
        pt1 *= shape
        pt2 *= shape

    return pt1.astype("int")[::-1], pt2.astype("int")[::-1]


def explore(
    descriptions,
    k=17,
    every=False,
    save=False,
    destination="examples/opti/detections",
    prefix="hm",
    suffix="png",
    figsize=(9, 6),
    key="hierarchical_mask",
    negative=False,
):
    descriptions = _check_descriptions(descriptions)
    if not os.path.isdir(destination):
        os.makedirs(destination)

    lut = get_lut(negative=negative)
    rcolor = get_color(green)
    if every:
        images = []
        for k, d in enumerate(descriptions):
            hierarchical_mask = d[key]
            rgb = omalovanka(
                hierarchical_mask,
                save=True,
                destination=destination,
                prefix=f"{prefix}_{k:03d}",
                suffix=suffix,
                lut=lut,
            )
            pt1, pt2 = get_pts(d, key=["loop", "notion_mask"])

            cv.rectangle(rgb, pt1, pt2, rcolor)
            rgb = cv.cvtColor(rgb, cv.COLOR_RGB2BGR)

            images.append(rgb)
        images2movie(
            images, movie=os.path.join(destination, f"{prefix}.avi"), codec="mp4v"
        )
    else:
        hierarchical_mask = descriptions[k][key]
        rgb = omalovanka(
            hierarchical_mask,
            save=save,
            destination=destination,
            prefix=prefix,
            suffix=suffix,
            lut=lut,
        )
        pylab.figure(1, figsize)
        pylab.imshow(rgb)
        pylab.show()


def get_monotonous(angles):
    bangles = np.insert(angles, 0, angles[0] + (angles[0] - angles[1]))
    aangles = np.append(angles, angles[-1] + (angles[-1] - angles[-2]))

    ab = angles - bangles[:-1]
    aa = aangles[1:] - angles
    indices = np.logical_or(ab >= 0, aa >= 0)
    print("indices")
    print(indices)

    monotonous = angles[indices]

    return monotonous, np.squeeze(np.argwhere(indices))


def _check_descriptions(descriptions):
    if type(descriptions) is str and os.path.isfile(descriptions):
        descriptions = pickle.load(open(descriptions, "rb"))
    return descriptions


def _get_aspects(descriptions, what="angle"):
    aspect = [item[what] for item in descriptions if what in item]
    return aspect


def plot_bounding_boxes(descriptions):
    descriptions = _check_descriptions(descriptions)

    angles = np.array(_get_aspects(descriptions, what="angle"))
    crystal = _get_aspects(descriptions, what="crystal")
    loop = _get_aspects(descriptions, "loop")
    aoi = np.array(_get_aspects(descriptions, "aoi_bbox_mm"))
    foreground = _get_aspects(descriptions, "foreground")

    angles, indices = get_monotonous(angles)
    angles[angles < angles[0]] += 360
    aoi = np.array([aoi[i] for i in indices])
    print("aoi.shape", aoi.shape)
    loop = np.array(
        [[loop[i][key] for key in ["present", "r", "c", "h", "w"]] for i in indices]
    )
    crystal = np.array(
        [[crystal[i][key] for key in ["present", "r", "c", "h", "w"]] for i in indices]
    )
    foreground = np.array(
        [
            [foreground[i][key] for key in ["present", "r", "c", "h", "w"]]
            for i in indices
        ]
    )
    fig, axes = pylab.subplots(1, 2)
    axes[0].plot(angles, crystal[:, 0], "o", label="crystal seen")
    axes[0].plot(angles, crystal[:, 1], "o", label="crystal r")
    axes[0].plot(angles, crystal[:, 2], "o", label="crystal c")
    axes[0].legend()
    # axes[1].plot(angles, loop[:, -2], 'o', label="height loop")
    # axes[1].plot(angles, loop[:, -1], 'o', label="width loop")
    axes[1].plot(angles, aoi[:, -2], "o", label="height aoi")
    axes[1].plot(angles, aoi[:, -1], "o", label="width aoi")
    axes[1].legend()
    pylab.show()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-d",
        "--descriptions",
        default="./examples/opti/zoom_X_careful_descriptions.pickle",
        type=str,
        help="descriptions",
    )
    parser.add_argument("-k", "--index", default=17, type=int, help="index")
    parser.add_argument("--every", action="store_true", help="all")
    parser.add_argument("-S", "--save", action="store_true", help="save")
    parser.add_argument(
        "-D",
        "--destination",
        default="examples/opti/detections",
        type=str,
        help="destination",
    )
    parser.add_argument("-p", "--prefix", default="hm", type=str, help="prefix")
    parser.add_argument("-s", "--suffix", default="png", type=str, help="suffix")
    parser.add_argument("--negative", action="store_true", help="negative")
    args = parser.parse_args()
    print(args)

    explore(
        args.descriptions,
        k=args.index,
        every=args.every,
        save=args.save,
        destination=args.destination,
        prefix=args.prefix,
        suffix=args.suffix,
        negative=args.negative,
    )

    # plot_bounding_boxes(args.descriptions)


if __name__ == "__main__":
    main()
