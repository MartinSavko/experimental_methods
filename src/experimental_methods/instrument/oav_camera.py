#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import time
import traceback

from typing import Optional

try:
    from pymba import Vimba, Frame
except:
    print("could not import pymba, please check")
    Vimba = None
    Frame = None

import redis
import numpy as np
import simplejpeg

from experimental_methods.utils.speech import defer
from experimental_methods.instrument.zmq_camera import zmq_camera
# from speaking_goniometer import speaking_goniometer
from experimental_methods.instrument.goniometer import goniometer


class oav_camera(zmq_camera):
    def __init__(
        self,
        port=5555,
        history_size_target=45000,
        debug_frequency=100,
        framerate_window=25,
        codec="h264",
        mode="redis_bzoom",  # pymba, vimba, redis_local, redis_bzoom
        service="oav_camera",
        sleeptime=1e-3,
        verbose=None,
        server=None,
    ):
        self.mode = mode

        zmq_camera.__init__(
            self,
            port=port,
            history_size_target=history_size_target,
            debug_frequency=debug_frequency,
            sleeptime=sleeptime,
            framerate_window=framerate_window,
            codec=codec,
            service=service,
            verbose=verbose,
            server=server,
        )

        self.magnifications = np.array(
            [
                1.0,
                1.19760479,
                1.53453078,
                1.99980002,
                8.84994911,
                11.4187839,
                17.69911504,
            ]
        )
        self.calibrations = {
            1: np.array([0.002, 0.002]),
            2: np.array([0.00167, 0.00167]),
            3: np.array([0.00130333, 0.00130333]),
            4: np.array([0.0010001, 0.0010001]),
            5: np.array([0.00022599, 0.00022599]),
            6: np.array([0.00017515, 0.00017515]),
            7: np.array([0.000113, 0.000113]),
        }

        
        self.redis = None
        self.redis_local = None
        
        getattr(self, f"initialize_{self.mode}")()
        try:
            self.goniometer = goniometer()
        except:
            self.goniometer = None
        
        
    def handle_frame(self, frame: Frame, delay: Optional[int] = 1) -> None:
        self.frame0 = frame

    def initialize_pymba(self, camera_id="DEV_000F315E0B4F"):
        print("initialize pymba")
        self.initialize_redis_local()
        self.camera_id = camera_id
        vimba = Vimba()
        vimba.startup()
        system = vimba.system()

        if system.GeVTLIsPresent:
            system.run_feature_command("GeVDiscoveryAllOnce")
            time.sleep(3)

        camera_ids = vimba.camera_ids()
        print("camera_ids %s" % camera_ids)

        self.camera = vimba.camera(self.camera_id)
        self.camera.open()
        self.camera.PixelFormat = "RGB8Packed"
        self.camera.arm("Continuous", self.handle_frame)
        self.camera.start_frame_acquisition()

    def initialize_vimba(self):
        pass

    def initialize_redis_local(self, host="172.19.10.125"):
        self.redis_local = redis.StrictRedis(host=host)

    def initialize_redis_bzoom(self):
        self.initialize_redis_local()
        self.redis = redis.StrictRedis(host="172.19.10.181")
        self.x_pixels_in_detector = int(self.redis.get("image_width"))
        self.y_pixels_in_detector = int(self.redis.get("image_height"))
        self.expected_length = self.y_pixels_in_detector * self.x_pixels_in_detector * 3

    @defer
    def get_shape(self):
        if self.mode == "redis_bzoom":
            shape = (self.y_pixels_in_detector, self.x_pixels_in_detector)
        return shape

    def initialize(self):
        super().initialize()

    def get_last_image_data(self):
        last_image_data = None
        if self.mode == "redis_local" and self.redis_local is not None:
            last_image_data = self.redis_local.get(
                f"last_image_data_{self.service_name}"
            )
        elif self.mode == "redis_bzoom" and self.redis is not None:
            image_data = self.redis.get("bzoom:RAW")
            raw = np.frombuffer(image_data[-self.expected_length :], dtype=np.uint8)
            img = np.reshape(
                raw, (self.y_pixels_in_detector, self.x_pixels_in_detector, 3)
            )
            last_image_data = self.encode_jpeg(img)
        elif self.mode == "pymba":
            if hasattr(self, "frame0"):
                try:
                    img = self.frame0.buffer_data_numpy()
                    last_image_data = simplejpeg.encode_jpeg(img)
                except:
                    time.sleep(0.01)

        return last_image_data

    @defer
    def get_value_id(self):
        value_id = -1
        try:
            if self.mode == "pymba":
                value_id = self.frame0.data.frameID
            elif self.mode == "redis_local":
                value_id = int(self.redis.get(f"last_image_id_{self.service_name}"))
            elif self.mode == "redis_bzoom":
                if self.get_zoom() >= 5:
                    value_id_key = "acA2440-x30::video_last_image_counter"
                else:
                    value_id_key = "acA2500-x5::video_last_image_counter"
                value_id = int(self.redis.get(value_id_key))
        except:
            print("could not get current frame id, please check")

        return value_id

    def acquire(self):
        value_id = self.get_value_id()
        if value_id != self.value_id:
            self.value_id = value_id
            self.timestamp = time.time()
            self.value = self.get_last_image_data()
            self.redis_local.set(f"last_image_data_{self.service_name}", self.value)
            self.redis_local.set(f"last_image_id_{self.service_name}", self.value_id)
        super().acquire()

    def get_calibration(self, zoom=None):
        try:
            if zoom is None:
                cx = self.goniometer.md.coaxcamscalex
                cy = self.goniometer.md.coaxcamscaley
                calibration = np.array([cx, cy])
            else:
                # zoom = self.get_zoom()
                calibration = self.calibrations[zoom]
        except:
            print("failed reading calibration")
            traceback.print_exc()
            calibration = np.array([0.0019, 0.0019])
        return calibration

    def get_horizontal_calibration(self):
        return self.get_calibration()[1]

    def get_vertical_calibration(self):
        return self.get_calibration()[0]

    @defer
    def get_beam_position(self):
        return np.array(self.get_shape()) / 2

    def get_beam_position_vertical(self):
        try:
            p = self.get_beam_position()[0]
        except:
            p = self.get_image().shape[0] / 2
        return p

    def get_beam_position_horizontal(self):
        try:
            p = self.get_beam_position()[1]
        except:
            p = self.get_image().shape[1] / 2
        return p

    def get_horizontal_calibration(self):
        return self.get_calibration()[1]

    def get_vertical_calibration(self):
        return self.get_calibration()[0]

    def get_zoom_from_calibration(self, calibration):
        a = list([(key, value[0]) for key, value in list(self.calibrations.items())])
        a.sort(key=lambda x: x[0])
        a = np.array(a)
        return list(range(1, 11))[np.argmin(np.abs(calibration - a[:, 1]))]

    # @defer
    def get_zoom(self):
        zoom = -1
        if self.redis is not None:
            try:
                # zoom = self.goniometer.md.coaxialcamerazoomvalue
                zoom = int(self.redis.get("video_zoom_idx"))
            except:
                print("could not read zoom, please check")
        return zoom

    @defer
    def set_gain(self):
        if not (gain >= 0 and gain <= 24):
            print("specified gain value out of the supported range (0, 24)")
            return -1
        if self.mode == "pymba":
            self.camera.GainRaw = int(gain)
        elif self.mode == "redis_bzoom":
            self.redis.set("video_gain", gain)

    @defer
    def get_gain(self):
        gain = -1
        if self.mode == "pymba":
            gain = self.camera.GainRaw
        elif self.mode == "redis_bzoom":
            gain = float(self.redis.get("video_gain"))
        return gain

    @defer
    def set_exposure(self, exposure=0.05):
        if type(exposure) != float:
            try:
                exposure = float(exposure)
            except:
                print(f"exposure {exposure}, {type(exposure)} type not supported")
                return -1
        if not (exposure >= 3.0e-6 and exposure < 3):
            print("specified exposure time is out of the supported range (3e-6, 3)")
            return -1
        if self.mode == "pymba":
            self.camera.ExposureTimeAbs = exposure * 1.0e6
        elif self.mode == "redis_bzoom":
            self.redis.set("camera_exposure_time", exposure)

    @defer
    def get_exposure(self):
        if self.mode == "pymba":
            exposure = self.camera.ExposureTimeAbs / 1.0e6
        elif self.mode == "redis_bzoom":
            exposure = float(self.redis.get("camera_exposure_time"))
        return exposure

    def set_exposure_time(self, exposure_time):
        self.set_exposure(exposure_time)

    def get_exposure_time(self):
        return self.get_exposure()


def main():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--mode", default="redis_bzoom", type=str, help="mode")
    parser.add_argument(
        "-k", "--debug_frequency", default=100, type=int, help="debug frame"
    )
    parser.add_argument(
        "-s",
        "--service",
        type=str,
        default="oav_camera",
        help="debug string add to the outputs",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose")
    parser.add_argument("-o", "--codec", type=str, default="h264", help="video codec")
    args = parser.parse_args()
    print(args)

    cam = oav_camera(
        mode=args.mode,
        service=args.service,
        debug_frequency=args.debug_frequency,
        codec=args.codec,
        verbose=False,
    )
    cam.verbose = args.verbose
    cam.set_server(True)
    cam.serve()

    sys.exit(0)


if __name__ == "__main__":
    main()
