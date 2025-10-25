#!/usr/bin/env python

import os
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
black = (0, 0, 0)
white = (1, 1, 1)
yellow = (1, 0.706, 0)
blue = (0, 0.706, 1)
green = (0.706, 1, 0)
magenta = (0.706, 0, 1)
cyan = (0, 1, 0.706)
purpre = (1, 0, 0.706)

notions = ["background", "foreground", "pin", "stem", "loop", "loop_inside", "crystal"]

colors_for_labels = {
    "crystal": green,
    "loop": purpre,
    "loop_inside": blue,
    "stem": cyan,
    "pin": magenta,
    "foreground": white,
    "background": black,
}


def get_lut(negative=False):
    lut = np.zeros((256, 1, 3))
    for k, notion in enumerate(notions):
        if negative and notion in ["foreground", "not_background"]:
            colorin = colors_for_labels["background"]
        elif negative and notion in ["background"]:
            colorin = colors_for_labels["foreground"]
        else:
            colorin = colors_for_labels[notion]
        
        if type(colorin) is str:
            color = hex_to_rgb(sns.xkcd_rgb[colorin])
        else:
            color = [int(255 * item) for item in colorin]
        print(f"transform {colorin} to {color}")
        lut[k] = color
    for k in range(len(notions), len(lut)):
        if negative:
            lut[k] = (1, 1, 1)
        else:
            lut[k] = (0, 0, 0)
    lut = lut.astype("uint8")
    return lut


def label2rgb(label, lut):
    rgb = cv.LUT(label, lut)
    return rgb


def hex_to_rgb(_hex):
    if _hex.startswith("#"):
        _hex = _hex[1:]
    return tuple(int(_hex[i : i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(_rgb):
  return ('{:02X}' * 3).format(*_rgb)

#assert rgb_to_hex((255, 165, 1)) # 'FFA501'

def omalovanka(
    hierarchical_mask, save=False, destination="detections", prefix="hm", suffix="png", lut=None, negative=False,
):
    gre = cv.cvtColor(hierarchical_mask, cv.COLOR_GRAY2RGB)
    if lut is None:
        lut = get_lut(negative=negative)
    rgb = cv.LUT(gre, lut)
    if save:
        fname = os.path.join(destination, f"{prefix}.{suffix}")
        cv.imwrite(fname, cv.cvtColor(rgb, cv.COLOR_RGB2BGR))
    return rgb


def explore(
    descriptions,
    k=17,
    every=False,
    save=False,
    destination="detections",
    prefix="hm",
    suffix="png",
    figsize=(9, 6),
    key="hierarchical_mask",
    negative=False,
):

    descriptions = _check_descriptions(descriptions)
        
    lut = get_lut(negative=negative)
    if every:
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
    aangles = np.append(angles, angles[-1] + ( angles[-1]-angles[-2]))
    
    ab = angles - bangles[:-1]
    aa = aangles[1:] - angles
    indices = np.logical_or(ab>=0, aa>=0)
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
    angles[angles<angles[0]] += 360
    aoi = np.array([aoi[i] for i in indices])
    print("aoi.shape", aoi.shape)
    loop = np.array(
        [
            [
                loop[i][key] for key in ["present", "r", "c", "h", "w"]
            ] for i in indices
        ]
    )
    crystal = np.array(
        [
            [
                crystal[i][key] for key in ["present", "r", "c", "h", "w"]
            ] for i in indices
        ]
    )
    foreground = np.array(
        [
            [
                foreground[i][key] for key in ["present", "r", "c", "h", "w"]
            ] for i in indices
        ]
    )
    fig, axes = pylab.subplots(1, 2)
    axes[0].plot(angles, crystal[:, 0], 'o', label="crystal seen")
    axes[1].plot(angles, loop[:, -2], 'o', label="height loop")
    axes[1].plot(angles, loop[:, -1], 'o', label="width loop")
    axes[1].plot(angles, aoi[:, -2], 'o', label="height aoi")
    axes[1].plot(angles, aoi[:, -1], 'o', label="width aoi")
    
    pylab.legend()
    pylab.show()
    
    
def main():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d",
        "--descriptions",
        default="/home/smartin/ResearchDevelopmentEducation/experimental_methods_gh/examples/opti/zoom_X_careful_descriptions.pickle",
        type=str,
        help="descriptions",
    )
    parser.add_argument("-k", "--index", default=17, type=int, help="index")
    parser.add_argument("--every", action="store_true", help="all")
    parser.add_argument("-S", "--save", action="store_true", help="save")
    parser.add_argument(
        "-D", "--destination", default="detections", type=str, help="destination"
    )
    parser.add_argument("-p", "--prefix", default="hm", type=str, help="prefix")
    parser.add_argument("-s", "--suffix", default="png", type=str, help="suffix")
    parser.add_argument("--negative", action="store_true", help="negative")
    args = parser.parse_args()
    print(args)

    #explore(
        #args.descriptions,
        #k=args.index,
        #every=args.every,
        #save=args.save,
        #destination=args.destination,
        #prefix=args.prefix,
        #suffix=args.suffix,
        #negative=args.negative,
    #)
    
    plot_bounding_boxes(args.descriptions)
    
    
    
if __name__ == "__main__":
    main()
