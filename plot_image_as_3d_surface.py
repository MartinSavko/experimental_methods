#!/usr/bin/env python
# -*- coding: utf-8 -*-
# https://stackoverflow.com/questions/31805560/how-to-create-surface-plot-from-greyscale-image-with-matplotlib
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.patches
import imageio
import skimage
from gaussfitter import gaussfit, twodgaussian
import cv2 as cv
from useful_routines import get_index_of_max_or_min

# bzoom camera pixel size at zoom 7 np.array([0.000113, 0.000113]


def get_img(image_name):
    img = imageio.imread(image_name).astype(float)
    img /= 255.0
    if len(img.shape) == 3:
        img = img.mean(axis=2)
    median = np.median(img)
    print("median", median)
    img[img < 1.2 * median] = 0
    img[img >= 1.2 * median] -= 1.2 * median
    img[img < 0] = 0
    mask = skimage.morphology.remove_small_objects(img > 0)
    img[mask == 0] = 0
    return img


def get_image_at_resolution(
    image_name,
    beamfit=None,
    resolution=0.002,
    calibration=np.array([0.000113, 0.000113]),
    save=True,
):
    img = get_img(image_name)
    if beamfit is None:
        beamfit = gaussfit(img, return_all=0)
        beamfit[2] = img.shape[0] // 2
        beamfit[3] = img.shape[1] // 2

    xx, yy = np.mgrid[0 : img.shape[0], 0 : img.shape[1]]
    tdg = twodgaussian(beamfit, 0, 1, 1)
    bimg = tdg(xx, yy)
    size_at_resolution = np.round(np.array(img.shape) * calibration / resolution)
    print(f"size_at_resolution {size_at_resolution}")
    image_at_resolution = cv.resize(bimg, size_at_resolution[::-1].astype(int))

    if save:
        image_to_save = (
            255
            * (image_at_resolution - image_at_resolution.min())
            / (image_at_resolution.max() - image_at_resolution.min())
        )
        print(
            f"max: {image_to_save.max()}, {get_index_of_max_or_min(image_to_save)}, center: {size_at_resolution/2}"
        )
        imageio.imsave(
            image_name.replace(".png", f"_{resolution}.png"),
            image_to_save.astype(np.uint8),
        )

    return image_at_resolution


def plot_image_as_3d_surface(
    image_name,
    what="surface",
    rstride=10,
    cstride=10,
    return_all=0,
    horizontal_pixel_size=0.000113,  # 0.00014931,
    vertical_pixel_size=0.000113,  # 0.00015034,
):
    img = get_img(image_name)

    xx, yy = np.mgrid[0 : img.shape[0], 0 : img.shape[1]]

    fig = plt.figure(figsize=(16, 9))

    fig.suptitle(os.path.basename(image_name))
    ax = fig.add_subplot(2, 3, 1, projection=None)
    ax.imshow(img)
    ax.set_title("beam")

    ax = fig.add_subplot(2, 3, 4, projection="3d")
    if what == "surface":
        # ax = fig.gca(projection='3d')
        # ax.plot_surface(xx, yy, img,rstride=1, cstride=1, cmap=plt.cm.gray,
        # linewidth=0)

        surf = ax.plot_surface(
            xx,
            yy,
            img,
            rstride=rstride,
            cstride=cstride,
            cmap=cm.coolwarm,
            linewidth=0,
            antialiased=False,
        )
        ax.set_title("3d view of the beam")
        # ax.zaxis.set_major_locator(LinearLocator(10))
        # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        fig.colorbar(surf, shrink=0.5, aspect=5)

    elif what == "wireframe":
        ax.plot_wireframe(xx, yy, img, rstride=rstride, cstride=cstride)

    calibration = np.array([horizontal_pixel_size, vertical_pixel_size])

    print("Moments")
    # https://dsp.stackexchange.com/questions/48717/fitting-a-gaussian-image-using-opencv
    # https://www.physicsforums.com/threads/rotate-2d-gaussian-given-parameters-a-b-and-c.997100/
    # https://math.stackexchange.com/questions/3438407/derivation-of-2d-binormal-bivariate-gaussian-general-equation

    M = cv.moments(img)
    print(M)
    print()
    scale = M["m00"]
    mcenter = np.array([M["m10"], M["m01"]])
    mcenter_scaled = mcenter / scale

    sigma2 = np.array([[M["mu20"], M["mu11"]], [M["mu11"], M["mu02"]]])

    sigma_scaled = np.sqrt(sigma2 / scale)

    print("scale", scale)
    print("center from moments", mcenter_scaled)
    print("sigma from moments")
    print(sigma_scaled)
    fwhm_px = 2 * np.sqrt(2 * np.log(2)) * sigma_scaled.diagonal()
    print("fwhm_px from moments", fwhm_px)
    fwhm_mm = fwhm_px * calibration
    print("fwhm_mm from moments", fwhm_mm)
    print()

    print("Gaussian fit parameters:")
    if return_all:
        p, cov, infodict, errmsg = gaussfit(img, return_all=1)
        print("cov", cov)
        print("infodict", infodict)
        print("errmsg", errmsg)
    else:
        p = gaussfit(img, return_all=0)

    print("p", p)
    height, amplitude, center_v, center_h, width_v, width_h, rotation = p
    print("height", height)
    print("amplitude", amplitude)
    print("center", center_h, center_v)
    print("width [px]", width_h, width_v)
    width_h_mm = width_h * horizontal_pixel_size
    width_v_mm = width_v * vertical_pixel_size
    print("width [mm]", width_h_mm, width_v_mm)
    print("rotation", rotation)
    sigma = np.array([[width_v, 0], [0, width_h]])
    print("sigma", sigma)
    fwhm_px = 2 * np.sqrt(2 * np.log(2)) * sigma.diagonal()
    print("fwhm_px", fwhm_px)

    fwhm_mm = fwhm_px * calibration
    print("fwhm_mm", fwhm_mm)
    print()

    bimgs = get_image_at_resolution(image_name)
    beamfit = twodgaussian(p, 0, 1, 1)
    beam_model = beamfit(xx, yy)

    ax = fig.add_subplot(2, 3, 3)
    ax.set_title("model beam")
    ax.imshow(beam_model)
    patch = matplotlib.patches.Ellipse(
        (center_h, center_v),
        fwhm_px[1],
        fwhm_px[0],
        angle=rotation,
        color="red",
        fill=False,
        lw=2,
    )
    ax.add_patch(patch)

    ax = fig.add_subplot(2, 3, 6, projection="3d")
    surf = ax.plot_surface(
        xx,
        yy,
        beam_model,
        rstride=rstride,
        cstride=cstride,
        cmap=cm.coolwarm,
        linewidth=0,
        antialiased=False,
    )
    fig.colorbar(surf, shrink=0.5, aspect=5)

    ax = fig.add_subplot(2, 3, 2)
    ax.set_title("difference between beam and the model")
    ax.imshow(beam_model - img)

    ax = fig.add_subplot(2, 3, 5, projection="3d")
    surf = ax.plot_surface(
        xx,
        yy,
        beam_model - img,
        rstride=rstride,
        cstride=cstride,
        cmap=cm.coolwarm,
        linewidth=0,
        antialiased=False,
    )
    ax.set_title("3d view of the beam and the model")
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-i",
        "--image",
        default="/home/experiences/projects/99240201/beam_align/425221_Tue_Oct_22_14:00:43_2024_2.png",
        type=str,
        help="image",
    )

    parser.add_argument("-w", "--what", default="surface", type=str, help="what")

    parser.add_argument("-r", "--rstride", default=50, type=int, help="rstride")

    parser.add_argument("-c", "--cstride", default=50, type=int, help="cstride")

    parser.add_argument(
        "-V",
        "--vertical_pixel_size",
        default=0.000_150_34,
        type=float,
        help="vertical_pixel_size",
    )

    parser.add_argument(
        "-H",
        "--horizontal_pixel_size",
        default=0.000_149_31,
        type=float,
        help="horizontal_pixel_size",
    )

    args = parser.parse_args()

    plot_image_as_3d_surface(
        args.image, what=args.what, rstride=args.rstride, cstride=args.cstride
    )
