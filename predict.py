#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Martin Savko (martin.savko@synchrotron-soleil.fr)

import os
import zmq
import time
import pickle
import numpy as np
import skimage

import scipy.ndimage as ndi

from skimage.morphology import remove_small_objects
from skimage.measure import regionprops


def get_predictions(request_arguments, port=8099, verbose=False):
    start = time.time()
    context = zmq.Context()
    if verbose:
        print("Connecting to server ...")
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:%d" % port)
    socket.send(pickle.dumps(request_arguments))
    predictions = pickle.loads(socket.recv())
    if verbose:
        print("Received predictions in %.4f seconds" % (time.time() - start))
    return predictions


def get_notion_prediction(
    predictions,
    notion,
    k=0,
    notion_indices={
        "crystal": 0,
        "loop_inside": 1,
        "loop": 2,
        "stem": 3,
        "pin": 4,
        "foreground": 5,
    },
    threshold=0.5,
    min_size=32,
):
    present, r, c, h, w, r_max, c_max, area, notion_prediction = [np.nan] * 9

    if type(notion) is list:
        notion_prediction = np.zeros(predictions[0].shape[1:3], dtype=bool)
        for n in notion:
            index = notion_indices[n]
            noti_pred = predictions[index][k, :, :, 0] > threshold
            noti_pred = remove_small_objects(noti_pred, min_size=min_size)
            notion_prediction = np.logical_or(notion_prediction, noti_pred)

    elif type(notion) is str:
        index = notion_indices[notion]

        notion_prediction = predictions[index][k, :, :, 0] > threshold
        notion_prediction = remove_small_objects(notion_prediction, min_size=min_size)

    if np.any(notion_prediction):
        labeled_image = notion_prediction.astype("uint8")
        properties = regionprops(labeled_image)[0]
        if properties.area_convex > min_size:
            present = 1
            area = properties.area_convex
        else:
            present = 0
        bbox = properties.bbox
        h = bbox[2] - bbox[0]
        w = bbox[3] - bbox[1]
        r, c = properties.centroid
        c_max = bbox[3]
        r_max = ndi.center_of_mass(labeled_image[:, c_max - 5 : c_max])[0]
        if notion == "foreground" or type(notion) is list:
            notion_prediction[
                bbox[0] : bbox[2], bbox[1] : bbox[3]
            ] = properties.image_filled
        else:
            notion_prediction[
                bbox[0] : bbox[2], bbox[1] : bbox[3]
            ] = properties.image_convex
    return present, r, c, h, w, r_max, c_max, area, notion_prediction


def get_most_likely_click(predictions, min_size=32):
    _start = time.time()
    for notion in ["crystal", "loop"]:
        notion_prediction = get_notion_prediction(
            predictions, notion, min_size=min_size
        )
        if notion_prediction[0] == 1:
            most_likely_click = notion_prediction[1], notion_prediction[2]
            print("click determined in %.4f seconds" % (time.time() - _start))
            return most_likely_click
    foreground = get_notion_prediction(predictions, "foreground", min_size=min_size)
    if foreground[0] == 1:
        most_likely_click = foreground[5], foreground[6]
    else:
        most_likely_click = -1, -1
    print("click determined in %.4f seconds" % (time.time() - _start))
    return most_likely_click


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-t",
        "--to_predict",
        type=str,
        default="learning_dataset/100161_Sat_Jun__1_15:43:20_2019",
        help="to_predict",
    )
    parser.add_argument(
        "-H", "--prediction_heigth", default=256, type=int, help="prediction_heigth"
    )
    parser.add_argument(
        "-W", "--prediction_width", default=320, type=int, help="prediction_width"
    )
    parser.add_argument(
        "-m", "--min_size", default=32, type=int, help="minimum object size"
    )
    parser.add_argument("-s", "--save", default=1, type=int, help="save")
    parser.add_argument("-p", "--prefix", type=str, default="test", help="prefix")
    parser.add_argument("-v", "--verbose", type=int, default=1, help="verbose")
    args = parser.parse_args()

    print("args", args)

    model_img_size = (args.prediction_heigth, args.prediction_width)

    request_arguments = {}
    to_predict = None
    if os.path.isdir(args.to_predict):
        to_predict = args.to_predict
    elif os.path.isfile(args.to_predict):
        if args.to_predict.endswith(".jpg") or args.to_predict.endswith(".jpeg"):
            to_predict = [open(args.to_predict, "rb").read()]
        if args.to_predict.endswith(".tif") or args.to_predict.endswith(".tiff"):
            to_predict = [os.path.realpath(args.to_predict)]
        else:
            to_predict = [os.path.realpath(args.to_predict)]

    request_arguments["to_predict"] = to_predict
    request_arguments["model_img_size"] = model_img_size
    request_arguments["save"] = bool(args.save)
    request_arguments["prefix"] = args.prefix
    predictions = get_predictions(request_arguments, verbose=bool(args.verbose))
    most_likely_click = get_most_likely_click(predictions)
    print("most_likely_click", most_likely_click)
    print()
