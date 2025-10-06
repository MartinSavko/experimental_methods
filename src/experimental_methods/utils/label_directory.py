#!/usr/bin/env python

import random
import re
import os
import glob
import numpy as np


def get_minimum_angle_difference(delta):
    return (delta + 180.0) % 360.0 - 180.0


def get_max_distance_index(selected_indices, angles):
    distances = []
    print("selected_indices", selected_indices)
    print("selected angles", angles[selected_indices])
    for s in selected_indices:
        delta = angles[s] - angles
        minimum_angle_difference = get_minimum_angle_difference(delta)
        distances.append(minimum_angle_difference)
    distances = np.array(np.abs(distances))
    print("distances.shape", distances.shape)
    print("distances", distances)
    # norm_total = np.linalg.norm(distances, ord=2, axis=0)
    norm_total = np.min(distances, axis=0)
    print("norm_total", norm_total)
    return np.argmax(norm_total)


def main():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        default="/home/smartin/ResearchDevelopmentEducation/learning_dataset/184451_Thu_Oct__3_23:37:20_2019",
        help="directory to label",
    )
    parser.add_argument(
        "-N", "--images_to_label", type=int, default=7, help="Number of images to label"
    )

    args = parser.parse_args()

    print(args)

    image_names = glob.glob(os.path.join(args.directory, "*.jpg"))
    label_records = glob.glob(os.path.join(args.directory, "*.json"))
    already_labeled_images = [item.replace(".json", ".jpg") for item in label_records]

    possible_angles = [
        re.findall(
            ".*_omega_([\d\.]*).*|color_zoom_[\d]*_([\d\.]*).jpg|color_([\d\.]*).jpg",
            img,
        )[0]
        for img in image_names
    ]
    print("possible_angles", possible_angles)
    angles = []
    for pa in possible_angles:
        if pa[0] != "":
            angles.append(float(pa[0]))
        elif pa[1] != "":
            angles.append(float(pa[1]))
        elif pa[2] != "":
            angles.append(float(pa[2]))

    angles = np.array(angles)
    selected_indices = [image_names.index(item) for item in already_labeled_images]
    if len(selected_indices) == 0:
        selected_indices.append(random.choice(range(len(angles))))

    while len(selected_indices) < args.images_to_label and len(selected_indices) < len(
        image_names
    ):
        selected_indices.append(get_max_distance_index(selected_indices, angles))

    selected_images = [image_names[k] for k in selected_indices]
    print("selected_indices", selected_indices)
    print("selected angles", angles[selected_indices])
    print("selected images", selected_images)

    os.system("eog %s &" % selected_images[0])
    for img in selected_images:
        if os.path.isfile(img.replace(".jpg", ".json")):
            continue
        line = (
            "/usr/bin/labelme --autosave --labels  crystal,loop_inside,loop,stem,pin,capillary,support_type,ice,dust,not_background,background,cd_stem,cd_loop,user_click %s"
            % img
        )
        os.system(line)


if __name__ == "__main__":
    main()
