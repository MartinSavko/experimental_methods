#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import time
import redis
import logging

from axis_stream import axis_camera
from oav_camera import oav_camera
from speaking_goniometer import speaking_goniometer
from useful_routines import get_string_from_timestamp, CAMERA_BROKER_PORT, DEFAULT_BROKER_PORT


class cameraman:
    def __init__(self):
        self.cameras = {}
        for kam in ["1", "6", "8", "13", "14_quad", "14_1", "14_2", "14_3", "14_4"]:
            codec = "hevc"
            service = f"cam{kam}"

            if "_" in kam:
                cam, name_modifier = service.split("_")
            else:
                cam, name_modifier = service, None
                
            if kam in ["1", "6", "8"]:
                codec = "h264"

            self.cameras[service] = axis_camera(
                cam, name_modifier=name_modifier, codec=codec, service=service, port=CAMERA_BROKER_PORT,
            )
            self.cameras[service].set_codec(codec=codec)

        self.cameras["sample_view"] = oav_camera(service="oav_camera", codec="h264", port=CAMERA_BROKER_PORT)
        self.cameras["goniometer"] = speaking_goniometer(service="speaking_goniometer", port=DEFAULT_BROKER_PORT)

    def save_history(self, filename_template, start, end, local=False, cameras=[]):
        if cameras == []:
            cameras = list(self.cameras.keys())

        # print("self", self)
        # print(self.cameras)
        for cam in cameras:
            filename = f"{filename_template}_{cam}.h5"
            logging.info(f"saving history {filename}")
            # if cam == "sample_view":
            # self.cameras[cam]._save_history(filename, start, end, None)
            if local:
                self.cameras[cam].save_history_local(filename, start, end)
            else:
                self.cameras[cam].save_history(filename, start, end)


camm = cameraman()
redis = redis.StrictRedis()


def record_video(func):
    def record(*args, **kwargs):
        print("args", args)
        redis.set("mounting", 1)
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        trajectory = "trajectory"
        element = "unknown"
        try:
            element = args[0].get_element()
            trajectory = func.__name__
        except:
            pass
        timestring = get_string_from_timestamp(start)
        name_pattern = f"{trajectory}_{os.getuid()}_element_{element}_{timestring}"
        directory = f"{os.getenv('HOME')}/manual_optical_alignment"
        filename_template = os.path.join(directory, name_pattern)
        camm.save_history(filename_template, start, end)
        redis.set("mounting", 0)
        return result

    return record
