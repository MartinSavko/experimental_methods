#!/usr/bin/env python

import sys
import os
import pylab
import numpy as np
import h5py
import pickle
from skimage.feature import match_template
import time
from scipy.ndimage import center_of_mass
import math
import copy
from camera import camera


def get_minimum_angle_difference(delta):
    return (delta + 180.0) % 360.0 - 180.0


def plot_clicks(imgs, clicks, options):
    nrows = 3
    ncols = len(imgs)

    fig1, axes1 = pylab.subplots(nrows=nrows, ncols=ncols, figsize=(30, 16), num=0)

    fig1.suptitle(options.name_pattern, fontsize=24)

    for a, av, ah, img, omega, hc, vc in zip(
        axes1[0],
        axes1[1],
        axes1[2],
        imgs,
        clicks["omegas"],
        clicks["horizontal_clicks"],
        clicks["vertical_clicks"],
    ):
        a.imshow(img)
        a.plot(hc, vc, "rx", lw=1, markersize=15)
        a.set_title("Omega: %.2f" % omega)
        a.set_axis_off()

        gim = img.mean(axis=2)
        av.plot(gim[:, hc - options.size : hc + options.size].mean(axis=1), "r")
        av.vlines(vc, 0, 255, "k")
        av.set_title("vertical slice at the coordinate of the click")

        ah.plot(gim[vc - options.size : vc + options.size, :].mean(axis=0), "g")
        ah.vlines(hc, 0, 255, "k")
        ah.set_title("horizontal slice at the coordinate of the click")

    pylab.savefig(
        "%s_manual.png"
        % os.path.join(options.destination_directory, options.name_pattern)
    )


def plot_clicks_and_estimates(
    imgs,
    clicks,
    homegas,
    himgs,
    size,
    clicks_maximum_attempted_prediction_distance,
    threshold,
):
    l = 0
    ncols = 4

    manually_annotated_omegas = []
    manually_annotated_verticals = []
    manually_annotated_horizontal = []

    auto_annotated_omegas = []
    auto_annotated_verticals = []
    auto_annotated_horizontal = []

    for img, omega, hc, vc in zip(
        imgs, clicks["omegas"], clicks["horizontal_clicks"], clicks["vertical_clicks"]
    ):
        manually_annotated_omegas.append(omega)
        manually_annotated_verticals.append(vc)
        manually_annotated_horizontal.append(hc)

        l += 1
        gim = img.mean(axis=2)
        temp = gim[vc - 5 * size : vc + 5 * size, hc - size : hc + size]
        deltas = np.abs(get_minimum_angle_difference(omega - homegas))

        selected_indices = np.argwhere(
            deltas < clicks_maximum_attempted_prediction_distance
        )
        print("selected_indices and omegas")
        # print(omegas[selected_indices][:,0])
        # print(selected_indices[:,0])
        print(np.vstack((homegas[selected_indices][:, 0], selected_indices[:, 0])).T)
        matches = []
        for i in selected_indices:
            # himg = himgs[i]
            # himg = np.reshape(himg, himg.shape[1:])
            himg = get_image_from_tensor(himgs[i])
            himg = himg.mean(axis=2)
            search_part = himg[:, hc - 3 * size : hc + 3 * size]
            print("search_part.shape", search_part.shape)
            # _start = time.time()
            # mt = match_template(himg, temp, pad_input=True)
            # max_indices = list(np.unravel_index(mt.argmax(), mt.shape))
            # _end = time.time()
            # print('full search took %.2f' % (_end-_start))
            # print('max_indices full search', max_indices)
            _start = time.time()
            mt = match_template(search_part, temp, pad_input=True)
            _end = time.time()
            print("partial search took %.2f" % (_end - _start))
            maximum_correlation = mt.max()
            if maximum_correlation > threshold:
                mt[mt <= threshold] = 0
                max_indices = list(center_of_mass(mt))
                # max_indices = list(np.unravel_index(mt.argmax(), mt.shape))
                max_indices[1] += hc - 3 * size
                print("max_indices", max_indices)
                matches.append(max_indices + [maximum_correlation])
                auto_annotated_omegas.append(homegas[i])
                auto_annotated_verticals.append(max_indices[0])
                auto_annotated_horizontal.append(max_indices[1])

        nrows, remainder = divmod(len(matches) + 1, ncols)
        if remainder:
            nrows += 1

        fig, axes = pylab.subplots(num=l, ncols=ncols, nrows=nrows)
        fig.suptitle("prediction based on click at angle %.2f" % omega)

        a = axes.flatten()

        a[0].imshow(img)
        a[0].set_title("Omega: %.2f" % omega)
        a[0].text(0.4, 0.8, "click: %d, %d" % (hc, vc), transform=a[0].transAxes)
        a[0].plot(hc, vc, "bx", lw=1, markersize=12)
        a[0].set_axis_off()

        for i in range(len(matches)):
            k = i + 1
            si = selected_indices[i]
            himg = get_image_from_tensor(himgs[si])
            a[k].imshow(himg)
            a[k].set_title("Omega: %.2f" % homegas[si])
            hc, vc, corr = matches[i][1], matches[i][0], matches[i][2]
            a[k].plot(hc, vc, "bx", lw=1, markersize=12)
            a[k].text(
                0.4,
                0.8,
                "predicted click: %d, %d" % (hc, vc),
                color="black",
                transform=a[k].transAxes,
            )
            a[k].text(
                0.4,
                0.7,
                "correlation: %.2f" % corr,
                color="green",
                transform=a[k].transAxes,
            )
            a[k].set_axis_off()

    pylab.figure(l + 1)
    pylab.plot(
        manually_annotated_omegas, manually_annotated_verticals, "ro", label="manual"
    )
    pylab.plot(auto_annotated_omegas, auto_annotated_verticals, "go", label="auto")


def select_unseen(seen, unseen, seen_by_eye_omegas):
    # closest_delta = -1
    # selected_unseen = -1
    # closest_seen = -1
    # closest_seen_by_eye = -1
    # distance_to_closest_seen_by_eye = -1
    closest_delta = np.inf
    selected_unseen = None
    closest_seen = None
    closest_seen_by_eye = None
    distance_to_closest_seen_by_eye = None
    for l, v in enumerate(unseen):
        distances_to_seen_by_eye = np.abs(
            get_minimum_angle_difference(seen_by_eye_omegas - v)
        )
        index = np.argmin(distances_to_seen_by_eye)

        for k, t in enumerate(seen):
            correlation = t[3]
            distance = (
                np.abs(get_minimum_angle_difference(t[0] - v))
                + distances_to_seen_by_eye[index]
                + 10 * (1.0 - correlation)
            )
            if distance < closest_delta:
                closest_delta = distance
                selected_unseen = l
                closest_seen = k
                dtsby = distances_to_seen_by_eye
                closest_seen_by_eye = seen_by_eye_omegas[index]
                distance_to_closest_seen_by_eye = distances_to_seen_by_eye[index]

    return (
        selected_unseen,
        seen[closest_seen],
        closest_seen_by_eye,
        closest_delta,
        distance_to_closest_seen_by_eye,
    )


def main():
    import optparse

    parser = optparse.OptionParser()

    parser.add_option(
        "-n",
        "--name_pattern",
        type=str,
        default="100161_Thu_Oct__3_17:57:40_2019",
        help="Template of the files with alignment data",
    )
    parser.add_option(
        "-d",
        "--directory",
        type=str,
        default="/nfs/data3/Martin/Research/manual_optical_alignment",
    )
    parser.add_option(
        "-e",
        "--destination_directory",
        type=str,
        default="/nfs/data3/Martin/Research/manual_optical_alignment_analysis",
    )
    parser.add_option(
        "-s",
        "--size",
        type=int,
        default=25,
        help="how many pixels around the click do we consider",
    )
    parser.add_option(
        "-t",
        "--threshold",
        type=float,
        default=0.8,
        help="Crosscorelation threshold when searching for corresponding points at the next image",
    )
    parser.add_option(
        "-r",
        "--clicks_maximum_attempted_prediction_distance",
        type=float,
        default=10.0,
        help="how far to try to predict",
    )
    parser.add_option(
        "-c",
        "--clicks",
        type=str,
        default="all",
        help="which manual clicks to consider",
    )
    parser.add_option("-D", "--display", action="store_true", help="display plots")
    options, args = parser.parse_args()

    if not os.path.isdir(options.destination_directory):
        os.makedirs(options.destination_directory)

    cam = camera()

    filename_template = os.path.join(options.directory, options.name_pattern)

    clicks = pickle.load(open("%s_clicks.pickle" % filename_template))
    print("clicks[omegas]", clicks["omegas"])
    try:
        images = h5py.File("%s_images.h5" % filename_template, "r")
        imgs = images["images"].value
    except:
        images = pickle.load(open("%s_images.pickle" % filename_template, "r"))
        imgs = np.array(images)

    plot_clicks(imgs, clicks, options)

    try:
        history = h5py.File("%s_history.h5" % filename_template, "r")
    except:
        print("history not available")
        sys.exit()

    hsv = history["history_state_vectors"].value
    himgs = history["history_images"].value
    print("himgs.shape", himgs.shape)

    homegas = hsv[:, 0]

    estimates = []

    seen_by_eye = list(
        zip(
            clicks["omegas"],
            clicks["vertical_clicks"],
            clicks["horizontal_clicks"],
            [1.0] * len(clicks["omegas"]),
            imgs,
        )
    )

    if options.clicks == "all":
        pass
    else:
        selected_clicks = eval(options.clicks)
        seen_by_eye = [seen_by_eye[s] for s in selected_clicks]

    seen_by_eye_omegas = np.array([item[0] for item in seen_by_eye])
    seen = copy.copy(seen_by_eye)

    unseen = list(homegas[:])
    himgs = list(himgs)

    size = options.size
    threshold = options.threshold

    _start = time.time()
    o = 0
    while unseen:
        o += 1
        (
            selected_unseen,
            closest_seen,
            closest_seen_by_eye,
            closest_delta,
            distance_to_closest_seen_by_eye,
        ) = select_unseen(seen, unseen, seen_by_eye_omegas)

        candidate = unseen.pop(selected_unseen)
        himg_color = himgs.pop(selected_unseen)

        print(
            "no. candidate(selected_unseen), closest_seen, closest_seen_by_eye, closest_delta, distance_to_closest_seen_by_eye"
        )
        print(
            "%d. %.2f(%d), %.2f, %.2f, %.4f, %.4f"
            % (
                o,
                candidate,
                selected_unseen,
                closest_seen[0],
                closest_seen_by_eye,
                closest_delta,
                distance_to_closest_seen_by_eye,
            )
        )

        himg = himg_color.mean(axis=2)
        reference = closest_seen

        gim = reference[-1].mean(axis=2)
        vc = int(reference[1])
        hc = int(reference[2])

        temp = gim[vc - 5 * size : vc + 5 * size, hc - 2 * size : hc + 2 * size]
        search_part = himg[:, hc - int(2.5 * size) : hc + int(2.5 * size)]

        try:
            mt = match_template(search_part, temp, pad_input=True)

            maximum_correlation = mt.max()
            print("maximum_correlation", maximum_correlation)
            print()

            if maximum_correlation > threshold:
                # mt[mt<=threshold] = 0
                # max_indices = list(center_of_mass(mt))
                max_indices = list(np.unravel_index(mt.argmax(), mt.shape))
                max_indices[1] += hc - int(2.5 * size)
                seen.append(
                    [
                        candidate,
                        max_indices[0],
                        max_indices[1],
                        maximum_correlation,
                        himg_color,
                    ]
                )
        except IndexError:
            print("search unsuccesful")

    _end = time.time()
    print("automated annotation took %.2f seconds" % (_end - _start))

    zoom = cam.get_zoom_from_calibration(np.median(clicks["calibrations"]))

    pylab.figure(1, figsize=(16, 9))
    pylab.title("Summary %s, zoom %d" % (options.name_pattern, zoom), fontsize=24)
    pylab.xlabel("Omega [degrees]", fontsize=18)
    pylab.ylabel("Click [pixels]", fontsize=18)
    pylab.plot(
        [item[0] for item in seen],
        [item[1] for item in seen],
        "go",
        label="auto vertical",
    )
    pylab.plot(
        [item for item in clicks["omegas"]],
        [item for item in clicks["vertical_clicks"]],
        "o",
        color="red",
        label="manual vertical",
    )
    pylab.plot(
        [item[0] for item in seen],
        [item[2] for item in seen],
        "bo",
        label="auto horizontal",
    )
    pylab.plot(
        [item for item in clicks["omegas"]],
        [item for item in clicks["horizontal_clicks"]],
        "o",
        color="orange",
        label="manual horizontal",
    )
    pylab.legend()

    pylab.savefig(
        "%s_summary.png"
        % os.path.join(options.destination_directory, options.name_pattern)
    )

    y = math.sqrt(len(seen))
    x = math.ceil(len(seen) / y)
    y = math.ceil(len(seen) / x)

    rows, cols = list(map(int, (y, x)))

    fig, axes = pylab.subplots(ncols=cols, nrows=rows, num=2, figsize=(25, 16))
    # fig.suptitle('Virtual clicks %s, zoom %d' % (options.name_pattern, zoom), fontsize=24)

    a = axes.flatten()
    seen.sort(key=lambda x: x[0])

    for k, t in enumerate(seen):
        a[k].imshow(t[-1])
        a[k].set_title("Omega: %.2f" % t[0])
        hc, vc, corr = t[2], t[1], t[3]
        if corr == 1.0:
            color = "red"
            designation = "manual"
        else:
            color = "blue"
            designation = "predicted"

        a[k].plot(hc, vc, "x", color=color, lw=1, markersize=12)
        a[k].text(
            0.01,
            0.9,
            "%s click: %d, %d" % (designation, hc, vc),
            color="black",
            transform=a[k].transAxes,
        )
        a[k].text(
            0.01,
            0.8,
            "correlation: %.2f" % corr,
            color="black",
            transform=a[k].transAxes,
        )
        a[k].set_axis_off()

    fig.tight_layout()
    pylab.savefig(
        "%s_clicks.png"
        % os.path.join(options.destination_directory, options.name_pattern)
    )

    if options.display == True:
        pylab.show()


def get_image_from_tensor(tensor):
    shape = tensor.shape[1:]
    image = np.reshape(tensor, shape)
    return image


if __name__ == "__main__":
    main()
