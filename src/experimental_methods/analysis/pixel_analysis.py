#!/usr/bin/env python
# -*- coding: utf-8 -*-

import h5py
import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy.signal import medfilt
from matplotlib.patches import Circle, Rectangle
from matplotlib.collections import PatchCollection
import pickle


def image_variance(img, window=3):
    wmean, wsqrmean = (
        cv2.boxFilter(x, -1, (window, window), borderType=cv2.BORDER_REFLECT)
        for x in (img, img * img)
    )
    return wsqrmean - wmean * wmean


def get_median_filtered_image(img, window=3):
    return cv2.medianBlur(img, window)


def get_dead(pixel_mask):
    return np.argwhere(pixel_mask == 2**1)


def get_cold(pixel_mask):
    return np.argwhere(pixel_mask == 2**2)


def get_hot(pixel_mask):
    return np.argwhere(pixel_mask == 2**3)


def get_noisy(pixel_mask):
    return np.argwhere(pixel_mask == 2**4)


def get_new_dead(image):
    return np.argwhere(image == 0)


def get_new_cold(image, median_filtered_image, threshold=1):
    return np.argwhere((median_filtered_image - image) > threshold)


def get_new_hot(image, median_filtered_image, threshold=1):
    return np.argwhere((median_filtered_image - image) > threshold)


def get_new_noisy(image):
    pass


def main():
    import optparse

    parser = optparse.OptionParser()

    parser.add_option(
        "-f",
        "--filename",
        default="sum_master.h5",
        type=str,
        help="eiger image master file",
    )
    parser.add_option(
        "-t", "--threshold", default=0.3, type=float, help="eiger image master file"
    )
    options, args = parser.parse_args()

    m = h5py.File(options.filename)

    print(list(m["/entry/data"].keys()))

    image = m["/entry/data/data_000001"].value[0]

    pixel_mask = m["/entry/instrument/detector/detectorSpecific/pixel_mask"].value

    img_float = image.astype(np.float32)

    background = 255 * np.ones(pixel_mask.shape)
    ib = background.astype(np.uint8)
    ib[pixel_mask == 1] = 255
    ib[pixel_mask == 2**1] = 32 - 1
    ib[pixel_mask == 2**2] = 64 - 1
    ib[pixel_mask == 2**3] = 128 - 1
    ib[pixel_mask == 2**4] = 256 - 1

    print("ib.max", ib.max())
    print("ib.min", ib.min())

    plt.imshow(ib, cmap="gray")

    dead = get_dead(pixel_mask)
    cold = get_cold(pixel_mask)
    hot = get_hot(pixel_mask)
    noisy = get_noisy(pixel_mask)


center_y = m["/entry/instrument/detector/beam_center_y"][()]
center_x = m["/entry/instrument/detector/beam_center_x"][()]

cy = int(center_y)
cx = int(center_x)

not_truested[cy - 50 : cy + 50, 0 : cx + 50] = 1

ax = plt.gca()

legend_patches = []

legend = {}

for k, p in enumerate(dead):
    if not_truested[p[0], p[1]] != 0:
        continue
    atch = ax.add_patch(
        Rectangle(p[::-1] - np.array([0.5, 0.5]), 1, 1, color="red", ec="red", fc="red")
    )
    if k == 0:
        legend["dead"] = atch
for k, p in enumerate(cold):
    if not_truested[p[0], p[1]] != 0:
        continue
    atch = ax.add_patch(
        Rectangle(
            p[::-1] - np.array([0.5, 0.5]),
            1,
            1,
            color="orange",
            ec="orange",
            fc="orange",
        )
    )
    if k == 0:
        legend["cold"] = atch
for k, p in enumerate(hot):
    if not_truested[p[0], p[1]] != 0:
        continue
    atch = ax.add_patch(
        Rectangle(
            p[::-1] - np.array([0.5, 0.5]), 1, 1, color="pink", ec="pink", fc="pink"
        )
    )
    if k == 0:
        legend["hot"] = atch
for k, p in enumerate(noisy):
    if not_truested[p[0], p[1]] != 0:
        continue
    atch = ax.add_patch(
        Rectangle(
            p[::-1] - np.array([0.5, 0.5]), 1, 1, color="green", ec="green", fc="green"
        )
    )
    if k == 0:
        legend["noisy"] = atch

    print("ax", ax)
    accounted_for = []
    for c in [dead, cold, hot, noisy]:
        accounted_for += list(c)

    print("dead", len(dead))
    print("cold", len(cold))
    print("hot", len(hot))
    print("noisy", len(noisy))
    new_dead = get_new_dead(image)
    print("new dead", len(new_dead))

    accounted_for += list(new_dead)

    img_medfilt = cv2.medianBlur(img_float, 3)

    reference = img_float[:]

    accounted_for = [tuple(item) for item in accounted_for]

    new_hott = [
        tuple(item)
        for item in list(
            np.argwhere((img_float - img_medfilt) / (img_float + 1) > options.threshold)
        )
    ]

    new_coldd = [
        tuple(item)
        for item in list(
            np.argwhere((img_medfilt - img_float) / (img_float + 1) > options.threshold)
        )
    ]

    new_hot = [nh for nh in new_hott if nh not in accounted_for]
    new_cold = [nc for nc in new_coldd if nc not in accounted_for]

    f = open("pixels_to_mask.pickle", "w")
    pickle.dump({"new_dead": new_dead, "new_cold": new_hot, "new_hot": new_hot}, f)
    f.close()

    print("new hot", len(new_hot))

    print("new cold", len(new_cold))

    for nh in new_hot:
        accounted_for.append(nh)
    for nc in new_cold:
        accounted_for.append(nc)

    pm = np.zeros(pixel_mask.shape)

    for k, p in enumerate(new_dead):
        atch = ax.add_patch(
            Rectangle(p[::-1] - np.array([0.5, 0.5]), 1, 1, color="green")
        )  # , ec='green', fc='green'))
        if k == 0:
            legend["new_dead"] = atch
    for k, p in enumerate(new_cold):
        atch = ax.add_patch(
            Rectangle(p[::-1] - np.array([0.5, 0.5]), 1, 1, color="magenta")
        )  # , ec='green', fc='green'))
        if k == 0:
            legend["new_cold"] = atch
    for k, p in enumerate(new_hot):
        atch = ax.add_patch(
            Rectangle(p[::-1] - np.array([0.5, 0.5]), 1, 1, color="cyan")
        )  # , ec='green', fc='green'))
        if k == 0:
            legend["new_hot"] = atch

    for index in accounted_for:
        pm[index] = 1

    print("Number of bad pixels overall", np.sum(pm))
    import itertools

    modules = itertools.product(list(range(12)), list(range(3)))
    mh = 257  # 514
    mw = 1030
    gh = 37
    gw = 10
    for module in modules:
        print(module)
        i, k = module

        v_start = i * mh + int(i / 2) * gh

        v_end = (i + 1) * mh + int(i / 2) * gh

        h_start = k * (mw + gw)
        h_end = (k + 1) * mw + k * gw

        submodule = pm[v_start:v_end, h_start:h_end]
        print("v_start", v_start)
        print("h_start", h_start)
        print("v_end", v_end)
        print("h_end", h_end)
        ax.add_patch(
            Rectangle(
                [h_start - 0.5, v_start - 0.5],
                mw,
                mh,
                alpha=0.5,
                fc="none",
                ec="magenta",
            )
        )
        if module == (5, 0):
            print(module)
            acht = np.argwhere(submodule != 0)
            print(len(acht))
            print(np.array([v_start, h_start]) + acht)

        ax.text(
            h_start + 10,
            v_start + 40,
            "Bad pixels in module (%d, %d): %d"
            % (k, i, submodule[submodule != 0].sum()),
        )
        print(
            "Number of bad pixels in module (%d, %d)" % (k, i),
            submodule[submodule != 0].sum(),
        )

    ax.set_title("Bad pixels of Eiger 9M of Proxima 2A")
    # legend_keys = ['dead', 'cold', 'hot', 'noisy', 'new_dead', 'new_cold', 'new_hot']
    legend_keys = ["new_dead", "new_cold", "new_hot"]
    plt.legend(
        [legend[key] for key in legend_keys],
        legend_keys,
        numpoints=0.1,
        bbox_to_anchor=(1, 1),
        fancybox=True,
        shadow=True,
    )
    plt.show()


if __name__ == "__main__":
    main()
