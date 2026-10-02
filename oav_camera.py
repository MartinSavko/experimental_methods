#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import base64
import json
import sys
import time
import traceback
import re
import threading

from datetime import datetime
from typing import Optional

try:
    from pymba import Vimba, Frame
except:
    print("could not import pymba, please check")
    Vimba = None
    Frame = None

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#https://stackoverflow.com/questions/65298241/what-does-this-tensorflow-message-mean-any-side-effect-was-the-installation-su
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
#https://stackoverflow.com/questions/78780089/how-do-i-get-rid-of-the-annoying-terminal-warning-when-using-gemini-api
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
try:
    import tensorflow as tf
    from tensorflow import keras
except:
    tf = None
    keras = None


import numpy as np
import simplejpeg
from speech import defer
from zmq_camera import zmq_camera

# from speaking_goniometer import speaking_goniometer
from goniometer import goniometer
from useful_routines import (
    CAMERA_BROKER_PORT,
    get_redis_connection,
    get_mxcube_camera,
    set_mxcube_camera,
    zmurko,
)


class oav_camera(zmq_camera):
    def __init__(
        self,
        port=CAMERA_BROKER_PORT,
        history_size_target=25000,
        debug_frequency=100,
        framerate_window=25,
        codec="h264",
        mode="redis_bzoom",  # pymba, vimba, redis_local, redis_bzoom
        service="oav_camera",
        sleeptime=5e-3,
        verbose=None,
        server=None,
        mxcube_publish=False,
        mxcube_channel="mxcubeweb",
        murko_repository="/nfs/data2/Martin/Research/murko_gh2",
        model_name="/nfs/data2/Martin/Research/murko_gh2/results/fcdn103_p2_fixed_128x128_fs_7_b_24_no_transform_with_arthur_validated.keras",
        serve_murko=False,
        model_img_size=None,
        murko_output="hierarchy_detailed_hierarchy",
        murko_redis_key="value_murko",
    ):
        self.mode = mode
        self.verbose = verbose
        if self.verbose:
            print(
                f"Starting oav...\nPublishing ---> {mxcube_publish} on redis channel {mxcube_channel}"
            )
        self.mxcube_publish = mxcube_publish
        self.mxcube_channel = mxcube_channel

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
            1: np.array([0.00181047, 0.00181047]),
            2: np.array([0.00011300, 0.00011300]),
            3: np.array([0.00118266, 0.00118266]),
            4: np.array([0.00090940, 0.00090940]),
            5: np.array([0.00030851, 0.00030851]),
            6: np.array([0.00017891, 0.00017891]),
            7: np.array([0.00011794, 0.00011794]),
        }

        self.redis = None

        getattr(self, f"initialize_{self.mode}")()
        try:
            self.goniometer = goniometer()
        except:
            self.goniometer = None

        self._value_id = -1

        self.model_name = model_name
        self.serve_murko = serve_murko
        self.murko_repository = murko_repository
        self.murko_output = murko_output
        self.luts_key = murko_output.replace("_hierarchy", "")
        self.set_murko_output(murko_output)
        self.murko_redis_key = murko_redis_key

        # try:
        # if self.server and self.serve_murko:
        ##import tensorflow as tf
        ##self.tf = tf
        ##from tensorflow import keras
        ##self.keras = keras
        # for gpu in tf.config.list_physical_devices("GPU"):
        # print("setting memory_growth on", gpu)
        # tf.config.experimental.set_memory_growth(gpu, True)

        # sys.path.insert(0, murko_repository)
        # from murko import (
        # WSConv2D,
        # WSSeparableConv2D,
        # )
        # self.custom_objects = {
        # "WSConv2D": WSConv2D,
        # "WSSeparableConv2D": WSSeparableConv2D,
        # }
        # from utils import guess_model_img_size, label2rgb
        # self.guess_model_img_size = guess_model_img_size
        # self.label2rgb = label2rgb
        # from config import luts
        # self.luts = luts
        # from sample import get_resized_image
        # self.get_resized_image = get_resized_image
        # self.load_model(self.model_name)

        # else:
        # sys.path.insert(0, murko_repository)
        # from utils import label2rgb, get_resized_image
        # self.label2rgb = label2rgb
        # self.get_resized_image = get_resized_image
        # from config import luts
        # self.luts = luts
        # self.murko_available = True
        # except:
        self.murko_available = False

    def get_predictions(self, image=None, host="localhost", port=89012):
        if image is not None:
            to_predict = image
        else:
            to_predict = self.get_image()

        request_arguments = {
            "to_predict": [to_predict],
            "hierarchy_output_name": "hierarchy_detailed_hierarchy",
        }

        predictions = get_predictions(request_arguments, host=host, port=port)

        return predictions

    def zmurko(
        self,
        image=None,
        batch_size=1,
        preserve_shape=True,
        threshold=0.5,
        store_in_redis=True,
    ):
        print("in zmurko")
        if not self.murko_available:
            return

        if image is None:
            image = self.get_image()

        murko_jpeg = zmurko(
            image,
            batch_size,
            preserve_shape,
            threshold,
            store_in_redis,
            self.redis_local,
        )
        # self.murko_thread = threading.Thread(
        # target=zmurko,
        # args=(image, batch_size, preserve_shape, threshold, store_in_redis),
        # )
        # self.murko_thread.daemon = False
        # self.murko_thread.start()
        return murko_jpeg

    @defer
    def set_serve_zmurko(self, serve_zmurko=True):
        self.serve_zmurko = serve_zmurko

    @defer
    def set_murko_output(self, murko_output):
        self.murko_output = murko_output
        self.luts_key = murko_output.replace("_hierarchy", "")

    @defer
    def load_model(self, model_name, warmup=True):
        _start = time.time()
        print(f"loading the model {model_name}")
        self.model_img_size = self.guess_model_img_size(model_name)
        self.model = keras.models.load_model(
            model_name,
            custom_objects=self.custom_objects,
        )
        _load_end = time.time()
        if warmup:
            print(f"warming up the system")
            shape = (1,) + self.model_img_size + (3,)
            self.model.predict(np.zeros(shape), batch_size=1)
        _warmup_end = time.time()
        self.set_murko_pick_index()
        print(
            f"load and warmup of the model took {time.time()-_start:.3f} seconds ({_load_end-_start:.3f} + {_warmup_end-_load_end:.3f})"
        )

    @defer
    def set_murko_pick_index(self):
        self.murko_pick_index = self.model.output_names.index(self.murko_output)

    @defer
    def _zmurko(self, image=None, batch_size=1, preserve_shape=True, threshold=0.5):
        if get_mxcube_camera() != "murko":
            return

        _start = time.time()
        if image is None:
            image = self.get_image()

        original_shape = image.shape[:2]
        _start_r = time.time()
        imr = self.get_resized_image(image, self.model_img_size)
        _end_r = time.time()
        print(
            f"resize from {original_shape} to {self.model_img_size} took {_end_r-_start_r:.3f} seconds"
        )
        imr_e = np.expand_dims(imr, 0)
        _start_pred = time.time()
        self.murko_prediction = self.model.predict(
            imr_e, batch_size=batch_size, verbose=0
        )
        _end_pred = time.time()
        print(f"prediction took {_end_pred-_start_pred:.3f} seconds")

        _start_omalovanka = time.time()
        p = self.murko_prediction[self.murko_pick_index]
        if len(p.shape) == 4:
            p = p[0]
        output = None
        if "hierarchy" in self.murko_output:
            label = np.argmax(p, axis=2).astype("uint8")
        elif (
            "distance_transform" in self.murko_output or "encoder" in self.murko_output
        ):
            output = p
        elif "binary_segment" in self.murko_output:
            label = (p > threshold).astype("uint8")
        if output is None:
            output = self.label2rgb(label, self.luts[self.luts_key])
        _end_omalovanka = time.time()
        print(
            f"coloring the result image took {_end_omalovanka-_start_omalovanka:.3f} seconds"
        )

        if preserve_shape:
            _start_ps = time.time()
            output = self.get_resized_image(output, original_shape)
            _end_ps = time.time()
            print(
                f"resize from {self.model_img_size} to {original_shape} took {_end_ps-_start_ps:.3f} seconds"
            )

        self.last_murko_image = output
        self.redis_local.set(self.murko_redis_key, simplejpeg.encode_jpeg(output))
        print(f"zmurko took {time.time()-_start:.3f} seconds")

    @defer
    def get_last_murko_image(self):
        return self.last_murko_image

    @defer
    def get_last_murko_prediction(self):
        return self.murko_prediction

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

    def initialize_redis_bzoom(self):
        self.initialize_redis_local()
        self.bzoom_value_id_key = "acA2500-x5::video_last_image_counter"
        self.redis = get_redis_connection("172.19.10.181")
        self.x_pixels_in_detector = int(self.redis.get("image_width"))
        self.y_pixels_in_detector = int(self.redis.get("image_height"))
        self.expected_length = self.y_pixels_in_detector * self.x_pixels_in_detector * 3
        # Read once: video-streamer fixes ffmpeg's source size at startup, so
        # md_camera.yaml width/height must agree with what is printed here.
        if self.verbose:
            print(
                f"bzoom frame size {self.x_pixels_in_detector}x{self.y_pixels_in_detector}"
            )

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
            last_image_data = self.redis_local.get(self.value_key)
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
                value_id = int(self.redis.get(self.value_id_key))
            elif self.mode == "redis_bzoom":
                value_id = int(self.redis.get(self.bzoom_value_id_key))
        except:
            print("could not get current frame id, please check")

        return value_id

    def age_limit(self, age_limit=0.01):
        return time.time() - self.timestamp > age_limit

    def publish_mxcubeweb(self, jpeg):
        """Feed MXCuBE's live video and snapshot paths from the current frame.

        Both consumers key off the same name (`mxcubeweb` by default), which is
        legal because Redis pub/sub channels and the keyspace are separate
        namespaces -- and it is what mxcubecore's RedisMpegVideo already assumes,
        since it passes its `redis_key` both as the streamer's `-irc` channel and
        to `lrange`.

          * PUBLISH -> video-streamer's RedisCamera, which decodes `data` with
            cv2.imdecode and pipes raw RGB into ffmpeg.
          * LPUSH/LTRIM -> RedisMpegVideo.get_last_image, which does
            `lrange(key, 0, 0)`; depth 1 is all it ever reads.
        """
        if not jpeg:
            return
        frame = {
            "data": base64.b64encode(jpeg).decode("utf-8"),
            # RedisCamera._set_size reads _height = size[0], _width = size[1],
            # so this is (height, width) -- not the (width, height) that
            # video-streamer's own LimaCamera publishes.
            "size": [self.y_pixels_in_detector, self.x_pixels_in_detector],
            "time": datetime.now().strftime("%H:%M:%S.%f"),
            "frame_number": self.value_id,
        }
        try:
            self.redis_local.publish(self.mxcube_channel, json.dumps(frame))
            self.redis_local.lpush(self.mxcube_channel, jpeg)
            self.redis_local.ltrim(self.mxcube_channel, 0, 0)
        except:
            print("could not publish frame to mxcube, please check")
            traceback.print_exc()

    def acquire(self):
        value_id = self.get_value_id()
        if self._value_id != value_id or self.age_limit():
            self.value_id += 1
            self._value_id = value_id
            self.timestamp = time.time()
            self.value = self.get_last_image_data()
            self.redis_local.set(self.value_key, self.value)
            self.redis_local.set(self.value_id_key, self.value_id)

            if self.mxcube_publish:
                self.publish_mxcubeweb(self.value)
            if self.serve_murko:
                self.zmurko()

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
            calibration = self.calibrations[1]
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

    def get_command_line(self, port=None):
        if port is None:
            port = self.port
        return f"oav_camera.py -p {port}"


def main():
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

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
    parser.add_argument(
        "-p", "--port", default=CAMERA_BROKER_PORT, type=int, help="port"
    )
    parser.add_argument(
        "-M",
        "--mxcube",
        action="store_true",
        help="also publish every frame to the redis channel/list mxcube reads",
    )
    parser.add_argument(
        "-c",
        "--mxcube_channel",
        type=str,
        default="mxcubeweb",
        help="redis channel and list name mxcube reads (redis_key in md_camera.yaml)",
    )
    parser.add_argument(
        "--serve_murko",
        action="store_true",
        help="serve murko",
    )
    args = parser.parse_args()
    print(args)

    cam = oav_camera(
        mode=args.mode,
        port=args.port,
        service=args.service,
        debug_frequency=args.debug_frequency,
        codec=args.codec,
        verbose=False,
        server=None,
        mxcube_publish=args.mxcube,
        mxcube_channel=args.mxcube_channel,
        serve_murko=args.serve_murko,
    )
    print("we are here, about to start serving")
    cam.verbose = args.verbose
    # if not cam.server:
    # cam.set_server(True)
    print("starting the server thread")
    # cam.start_serve()
    cam.serve()
    print("Done serving and exiting ...\nBye!")

    sys.exit(0)


if __name__ == "__main__":
    main()
