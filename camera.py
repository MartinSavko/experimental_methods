#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import logging
import gevent
import traceback
import struct
import redis
import pickle

import numpy as np

try:
    from pymba import Vimba, Frame
except:
    Vimba, Frame = None, None
    traceback.print_exc()

from typing import Optional
from skimage.io import imsave
from scipy.ndimage import center_of_mass

try:
    if sys.version_info.major == 3:
        import tango
    else:
        import PyTango as tango
except ImportError:
    print("camera could not import tango")

from goniometer import goniometer

try:
    import simplejpeg
except ImportError:
    import complexjpeg as simplejpeg

from predict import get_predictions, get_most_likely_click

# MD2
# calibrations for mako done on 2022-03-21 bis
# camear pixel calibration mm/pix [vertical, horizotal]
# calibrations = \
# {1: np.array([0.00133385, 0.00135215]),
# 2: np.array([0.00117585, 0.00116689]),
# 3: np.array([0.00088749, 0.00089089]),
# 4: np.array([0.00066537, 0.00067563]),
# 5: np.array([0.00051219, 0.00051137]),
# 6: np.array([0.00039121, 0.00039123]),
# 7: np.array([0.00029608, 0.00030089]),
# 8: np.array([0.00022955, 0.00023025]),
# 9: np.array([0.00016862, 0.00017721]),
# 10: np.array([0.00015034, 0.00014931])}

# MD3 2025-01-15
calibrations = {
    1: np.array([0.002, 0.002]),
    2: np.array([0.00167, 0.00167]),
    3: np.array([0.00130333, 0.00130333]),
    4: np.array([0.0010001, 0.0010001]),
    5: np.array([0.00022599, 0.00022599]),
    6: np.array([0.00017515, 0.00017515]),
    7: np.array([0.000113, 0.000113]),
}


class camera(object):
    def __init__(
        self,
        camera_type="prosilica",
        y_pixels_in_detector=1024,  # 1200, #1024,
        x_pixels_in_detector=1216,  # 1600, #1360,
        channels=3,
        default_exposure_time=0.05,
        default_gain=0.0,
        pixel_format="RGB8Packed",
        tango_address="i11-ma-cx1/ex/imag.1",
        tango_beamposition_address="i11-ma-cx1/ex/md3-beamposition",
        use_redis=True,
        use_jpeg=True,
        redis_host="172.19.10.181",
        history_size_threshold=10000,
        state_difference_threshold=0.005,
        publish_in_arinax_format=True,
    ):
        self.last_zoom = 1
        self.y_pixels_in_detector = y_pixels_in_detector
        self.x_pixels_in_detector = x_pixels_in_detector
        self.channels = channels
        self.default_exposure_time = default_exposure_time
        self.current_exposure_time = None
        self.default_gain = default_gain
        self.current_gain = None
        self.pixel_format = pixel_format
        self.goniometer = goniometer()
        self.use_redis = use_redis
        self.redis_host = redis_host
        if self.use_redis == True:
            self.camera = None
            self.redis = redis.StrictRedis(self.redis_host)
        else:
            self.camera = tango.DeviceProxy(tango_address)
            self.redis = None
        self.use_jpeg = use_jpeg
        try:
            self.tango_beamposition = tango.DeviceProxy(tango_beamposition_address)
        except:
            self.tango_beamposition = None
        self.camera_type = camera_type
        self.shape = (y_pixels_in_detector, x_pixels_in_detector, channels)
        self.history_size_threshold = history_size_threshold
        self.state_difference_threshold = state_difference_threshold
        self.publish_in_arinax_format = publish_in_arinax_format
        self.arinax_key = "bzoom:RAW"
        self.arinax_redis_format_key = "arinax_redis_format"
        self.arinax_redis_format_default = 6
        self.arinax_redis_formats = [
            "Y8",
            "Y16",
            "Y32",
            "RGB555",
            "RGB565",
            "RGB24",
            "RGB32",
            "BGR24",
            "BGR32",
            "BAYER_RG8",
            "BAYER_GR8",
            "BAYER_RG16",
            "BAYER_BG8",
            "BAYER_GB8",
            "BAYER_BG16",
            "I420",
            "YUV411",
            "YUV422",
            "YUV444",
        ]

        # After changing zoom 2019-03-07
        # self.focus_offsets = \
        # {1: 0.010,
        # 2: 0.010,
        # 3: 0.017,
        # 4: 0.022,
        # 5: 0.027,
        # 6: 0.021,
        # 7: 0.023,
        # 8: 0.018,
        # 9: 0.022,
        # 10: 0.022}
        # After changing camera 2024-06-04
        # self.focus_offsets = \
        # {1: 0.088,
        # 2: 0.088,
        # 3: 0.088,
        # 4: 0.088,
        # 5: 0.088,
        # 6: 0.088,
        # 7: 0.088,
        # 8: 0.088,
        # 9: 0.088,
        # 10: 0.088}
        # Smarmagnet adjustments 2024-09-20 contrast_scan.py
        self.exposure = {
            1: [1877, 2092, 2006, 1961, 2028, 2060, 2025, 1995, 1978],
            2: [1877, 2092, 2202, 2105, 1996, 1976, 2150, 1961],
            3: [2211, 2092, 2432, 2246, 2093, 2182, 2105],
            4: [2495, 2681, 2690, 2677, 2484, 2465, 2469],
            5: [3166, 3501, 3364, 3684, 3475, 3296],
            6: [4741, 5249, 5285, 5067, 4600, 5026],
            7: [8305, 8611, 8110, 8100, 7956, 8406, 8166, 7728, 7384, 7604],
            8: [
                12770,
                14212,
                13271,
                13826,
                13804,
                12178,
                12999,
                12286,
                14308,
                14222,
                12889,
            ],
            9: [
                22601,
                23286,
                22300,
                22247,
                23834,
                22959,
                22962,
                23503,
                22994,
                21344,
                23071,
            ],
            10: [
                29766,
                30395,
                28728,
                29080,
                30675,
                30605,
                28007,
                30624,
                27996,
                30612,
                28081,
            ],
        }

        self.exposure_median = dict(
            [(key, np.median(self.exposure[key]) / 1e6) for key in self.exposure]
        )

        # MD2
        # self.focus_offsets = \
        # {1: 0.160,
        # 2: 0.150,
        # 3: 0.104,
        # 4: 0.090,
        # 5: 0.106,
        # 6: 0.112,
        # 7: 0.112,
        # 8: 0.112,
        # 9: 0.112,
        # 10: 0.112}
        # MD3
        self.focus_offsets = {
            1: 0.0,
            2: 0.0,
            3: 0.0,
            4: 0.0,
            5: 0.0,
            6: 0.0,
            7: 0.0,
        }

        # MD2
        # self.zoom_motor_positions = \
        # {1: 33500.0, #34500.0,
        # 2: 31165.0,
        # 3: 27185.0,
        # 4: 23205.0,
        # 5: 19225.0,
        # 6: 15245.0,
        # 7: 11265.0,
        # 8: 7285.0,
        # 9: 3305.0,
        # 10: 100.0 }
        # MD3
        self.zoom_motor_positions = {
            1: 33500.0,  # 34500.0,
            2: 31165.0,
            3: 27185.0,
            4: 23205.0,
            5: 19225.0,
            6: 15245.0,
            7: 11265.0,
        }

        # prosilica
        # self.backlight = \
        # {1: 9.5,
        # 2: 10.0,
        # 3: 11.0,
        # 4: 13.0,
        # 5: 15.0,
        # 6: 21.0,
        # 7: 29.0,
        # 8: 41.0,
        # 9: 50.0,
        # 10: 61.0}
        # mako
        self.backlight = {
            1: 10.0,
            2: 10.0,
            3: 11.0,
            4: 13.0,
            5: 14.0,
            6: 20.0,
            7: 28.0,
            8: 40.0,
            9: 50.0,
            10: 53.0,
        }

        # prosilica
        # self.frontlight = \
        # {1: 10.0,
        # 2: 10.0,
        # 3: 11.0,
        # 4: 13.0,
        # 5: 15.0,
        # 6: 21.0,
        # 7: 29.0,
        # 8: 41.0,
        # 9: 50.0,
        # 10: 61.0}
        # mako
        self.frontlight = {
            1: 15.0,
            2: 17.0,
            3: 18.0,
            4: 19.0,
            5: 20.0,
            6: 23.0,
            7: 29.0,
            8: 32.0,
            9: 35.0,
            10: 39.0,
        }

        self.gain = {
            1: 0.0,
            2: 0.0,
            3: 0.0,
            4: 0.0,
            5: 0.0,
            6: 0.0,
            7: 0.0,
            8: 0.0,
            9: 0.0,
            10: 0.0,
        }

        self.calibrations = calibrations

        nzooms = 7
        self.magnifications = np.array(
            [
                np.mean(self.calibrations[1] / self.calibrations[k])
                for k in range(1, nzooms + 1)
            ]
        )
        self.master = False

    def get_point(self):
        return self.get_image()

    def get_image(self, color=True):
        if color:
            return self.get_rgbimage()
        else:
            return self.get_bwimage()

    def get_image_id(self):
        if self.use_redis:
            image_id = self.redis.get("last_image_id")
        else:
            image_id = self.camera.imagecounter
        return int(image_id)

    def get_last_image_data(self):
        last_image_data = self.redis.get(self.arinax_key)
        return last_image_data

    def get_rgbimage(self, image_data=None):
        if self.use_redis:
            if image_data == None:
                image_data = self.get_last_image_data()
            if simplejpeg.is_jpeg(image_data):
                rgbimage = simplejpeg.decode_jpeg(image_data)
            else:
                rgbimage = np.ndarray(
                    buffer=image_data,
                    dtype=np.uint8,
                    shape=(self.x_pixels_in_detector, self.y_pixels_in_detector, 3),
                )
        else:
            rgbimage = self.camera.rgbimage.reshape((self.shape[0], self.shape[1], 3))
        return rgbimage

    def set_arinax_redis_format(self, arinax_redis_format=6):
        self.redis.set(self.arinax_redis_format_key, arinax_redis_format)
        self.arinax_redis_format = arinax_redis_format

    def get_arinax_redis_format(self):
        try:
            arinax_redis_format = int(self.redis.get(self.arinax_redis_format_key))
        except:
            traceback.print_exc()
            arinax_redis_format = self.arinax_redis_format_default
        return arinax_redis_format

    def get_header(
        self,
        struct_format=">IHHqiiHHHH",
        arinax_redis_format=None,
        endianness=0,
        image_id=None,
    ):
        old_header = "hiihhq"
        if image_id is None:
            image_id = self.get_image_id()
        if arinax_redis_format is None:
            arinax_redis_format = self.get_arinax_redis_format()
        header = struct.pack(
            struct_format,
            arinax_redis_format,
            self.shape[1],
            self.shape[0],
            endianness,
            struct.calcsize(struct_format),
            image_id,
            0,
            0,
            0,
            0,
        )
        return header

    def get_struct_rgbimage(self, header=None, image_data=None):
        if header is None:
            header = self.get_header()
        if image_data is None:
            image_data = self.camera.rgbimage
        struct_rgbimage = header + image_data
        return struct_rgbimage

    def get_bwimage(self, image_data=None):
        rgbimage = self.get_rgbimage(image_data=image_data)
        return rgbimage.mean(axis=2)

    def clear_history(self):
        for item in [
            "history_image_timestamp",
            "history_state_vector",
            "history_image_data",
        ]:
            self.redis.ltrim(item, 0, -2)

    def get_history(self, start, end):
        self.redis.set("can_clear_history", 0)
        try:
            timestamps = self.get_timestamps()

            mask = np.logical_and(timestamps >= start, timestamps <= end)

            interesting_stamps = np.array(
                [
                    float(self.redis.lindex("history_image_timestamp", int(i)))
                    for i in np.argwhere(mask)
                ]
            )

            interesting_images = np.array(
                [
                    self.get_rgbimage(
                        image_data=self.redis.lindex("history_image_data", int(i))
                    )
                    for i in np.argwhere(mask)
                ]
            )

            interesting_state_vectors = np.array(
                [
                    self.get_state_vector_with_float_values_from_state_vector_as_single_string(
                        self.redis.lindex("history_state_vector", int(i)).decode()
                    )
                    for i in np.argwhere(mask)
                ]
            )
        except:
            print(traceback.print_exc())
            interesting_stamps = np.array([])
            interesting_images = np.array([])
            interesting_state_vectors = np.array([])

        self.redis.set("can_clear_history", 1)
        return interesting_stamps, interesting_images, interesting_state_vectors

    def get_timestamps(self):
        timestamps = np.array(
            [
                float(self.redis.lindex("history_image_timestamp", i))
                for i in range(self.redis.llen("history_image_timestamp"))
            ]
        )
        return timestamps

    def get_image_corresponding_to_timestamp(self, timestamp):
        self.redis.set("can_clear_history", 0)
        try:
            timestamps = self.get_timestamps()

            timestamps_before = timestamps[timestamps <= timestamp]

            closest = np.argmin(np.abs(timestamps_before - timestamp))

            corresponding_image = self.get_rgbimage(
                image_data=self.redis.lindex("history_image_data", int(closest))
            )

            # corresponding_state_vector =  self.get_state_vector_with_float_values_from_state_vector_as_single_string(self.redis.lindex('history_state_vector', int(closest)))

        except:
            corresponding_image = self.get_rgbimage()
            # corresponding_state_vector = None

        self.redis.set("can_clear_history", 1)

        return corresponding_image

    def save_image(self, imagename, image=None, color=True):
        if image is None:
            image_id, image = self.get_image_id(), self.get_image(color=color)
        else:
            image_id = -1
        if not os.path.isdir(os.path.dirname(imagename)):
            try:
                os.makedirs(os.path.dirname(imagename))
            except OSError:
                print("Could not create the destination directory")
        imsave(imagename, image)
        return imagename, image, image_id

    def get_zoom_from_calibration(self, calibration):
        a = list([(key, value[0]) for key, value in list(self.calibrations.items())])
        a.sort(key=lambda x: x[0])
        a = np.array(a)
        return list(range(1, 11))[np.argmin(np.abs(calibration - a[:, 1]))]

    def get_zoom(self, zoomposition=None):
        zoom = self.goniometer.md.coaxialcamerazoomvalue

        # a = list(self.zoom_motor_positions.items())
        # a.sort(key=lambda x: x[0])
        # a = np.array(a)
        # try:
        # if zoomposition is None:
        # zoomposition = self.goniometer.get_zoom_position()
        # zoom = list(range(1, 11))[np.argmin(np.abs(zoomposition-a[:,1]))]
        # self.last_zoom = zoom
        # except:
        # print(traceback.print_exc())
        # zoom = self.last_zoom
        return zoom

    def set_zoom(self, value, wait=True, adjust_zoom=True, light_factor=1.0):
        if value is not None:
            value = int(value)
            ##self.set_gain(self.gain[value])
            # if adjust_zoom == True:
            # self.goniometer.set_position({'Zoom': self.zoom_motor_positions[value], 'AlignmentX': self.focus_offsets[value]}, wait=wait)
            # else:
            # self.goniometer.set_position({'Zoom': self.zoom_motor_positions[value]}, wait=wait)

            # self.goniometer.wait()

            # try:
            # self.set_exposure_time(self.exposure_median[value]*light_factor)
            ##self.set_backlightlevel(self.backlight[value] * light_factor)
            ##self.goniometer.md.backlightlevel =
            # except:
            # pass

            self.goniometer.md.coaxialcamerazoomvalue = value

    def get_calibration(self, zoomposition=None):
        if zoomposition is None:
            calibration = np.array(
                [self.get_vertical_calibration(), self.get_horizontal_calibration()]
            )
        else:
            calibration = self.calibrations[self.get_zoom(zoomposition)]
        return calibration

    def get_vertical_calibration(self):
        return self.goniometer.md.coaxcamscaley

    def get_horizontal_calibration(self):
        return self.goniometer.md.coaxcamscalex

    def set_exposure(self, exposure=None):
        print(f"about to set exposure {exposure}")
        if exposure is None:
            return
        if not (exposure >= 3.0e-6 and exposure < 3):
            print("specified exposure time is out of the supported range (3e-6, 3)")
            return -1
        if not self.use_redis:
            self.camera.exposure = exposure
        if self.master:
            print(f"writing exposure to camera {int(exposure*1e6)}")
            self.camera.ExposureTimeAbs = int(exposure * 1.0e6)
            print(f"exposure written to camera {self.camera.ExposureTimeAbs}")
        print(f"writing exposure to redis {exposure}")
        self.redis.set("camera_exposure_time", exposure)

    def get_exposure(self, verbose=False):
        if not self.use_redis:
            exposure = self.camera.exposure
        if self.master:
            try:
                exposure = self.camera.ExposureTimeAbs / 1.0e6
                if verbose:
                    print("exposure from camera %s" % exposure)
                self.exposure = exposure
            except:
                exposure = self.exposure
        else:
            exposure = float(self.redis.get("camera_exposure_time"))
        return exposure

    def set_exposure_time(self, exposure_time):
        self.set_exposure(exposure_time)

    def get_exposure_time(self):
        return self.get_exposure()

    def get_gain(self):
        if not self.use_redis:
            gain = self.camera.gain
        elif self.master:
            gain = self.camera.GainRaw
        else:
            gain = float(self.redis.get("camera_gain"))
        return gain

    def set_gain(self, gain):
        if not (gain >= 0 and gain <= 24):
            print("specified gain value out of the supported range (0, 24)")
            return -1
        if not self.use_redis:
            self.camera.gain = gain
        elif self.master:
            self.camera.GainRaw = int(gain)
        self.redis.set("camera_gain", gain)
        self.current_gain = gain

    def get_beam_position(self, beam_position=[512.0, 612.0]):
        return np.array(beam_position)

    def get_beam_position_vertical(self):
        return self.get_beam_position[0]
        # return self.tango_beamposition.read_attribute('Zoom%d_Z' % self.get_zoom()).value

    def get_beam_position_horizontal(self):
        return self.get_beam_position[1]
        # return self.tango_beamposition.read_attribute('Zoom%d_X' % self.get_zoom()).value

    def set_frontlightlevel(self, frontlightlevel):
        self.goniometer.md.frontlightlevel = frontlightlevel

    def get_frontlightlevel(self):
        frontlightlevel = self.goniometer.md.frontlightlevel
        return frontlightlevel

    def set_backlightlevel(self, backlightlevel):
        self.goniometer.md.backlightlevel = backlightlevel

    def get_backlightlevel(self):
        backlightlevel = self.goniometer.md.backlightlevel
        return backlightlevel

    def get_width(self):
        return self.x_pixels_in_detector

    def get_height(self):
        return self.y_pixels_in_detector

    def get_image_dimensions(self):
        return [self.get_width(), self.get_height()]

    def get_shape(self):
        return np.array(self.shape)

    def _get_com(self, image):
        com = np.array(center_of_mass(image)) / np.array(image.shape)
        return com

    def _get_fwhm(self, image):
        sigma_y = np.std(image.sum(axis=0)) / image.shape[0] ** 2
        sigma_x = np.std(image.sum(axis=1)) / image.shape[1] ** 2

        return 2 * np.sqrt(2 * np.log(2)) * np.array([sigma_y, sigma_x])

    def get_com_fwhm(self, image=None, color=False, threshold=0.1):
        if image is None:
            image = self.get_image(color=color)

        image[image < image.max() * threshold] = 0

        com = self._get_com(image)
        fwhm = self._get_fwhm(image)

        return com, fwhm

    def get_state_vector_with_string_values(self):
        gain = self.get_gain()
        exposure_time = self.get_exposure_time()
        return self.goniometer.get_state_vector() + [
            "%.2f" % gain,
            "%.3f" % exposure_time,
        ]

    def get_state_vector_with_float_values(self, state_vector_with_string_values=None):
        if state_vector_with_string_values is None:
            state_vector_with_string_values = self.get_state_vector_with_string_values()
        return np.array(list(map(float, state_vector_with_string_values)))

    def get_state_vector_as_single_string(self, state_vector_with_string_values=None):
        if state_vector_with_string_values is None:
            state_vector_with_string_values = self.get_state_vector_with_string_values()
        try:
            state_vector_as_single_string = ",".join(state_vector_with_string_values)
        except:
            traceback.print_exc()
            print("state_vector_with_string_values", state_vector_with_string_values)
            state_vector_as_single_string = ""
        return state_vector_as_single_string

    def get_state_vector_with_string_values_from_state_vector_as_single_string(
        self, state_vector_as_single_string
    ):
        return state_vector_as_single_string.split(",")

    def get_state_vector_with_float_values_from_state_vector_as_single_string(
        self, state_vector_as_single_string
    ):
        state_vector_with_string_values = (
            self.get_state_vector_with_string_values_from_state_vector_as_single_string(
                state_vector_as_single_string
            )
        )
        return self.get_state_vector_with_float_values(state_vector_with_string_values)

    def get_last_saved_state_vector_string(self):
        return self.redis.lindex(
            "history_state_vector", self.redis.llen("history_state_vector") - 1
        )

    def get_minimum_angle_difference(self, delta):
        return (delta + 180.0) % 360.0 - 180.0

    def state_vectors_are_different(self, v1, v2):
        delta = v1 - v2
        delta[0] = self.get_minimum_angle_difference(delta[0])
        delta[2] = self.get_minimum_angle_difference(delta[2])
        return np.linalg.norm(delta) > self.state_difference_threshold

    def get_default_background(self, zoom=None):
        if zoom is None:
            background = self.get_rgbimage(
                image_data=self.redis.get(
                    "background_image_data_zoom_%d" % self.get_zoom()
                )
            )
        else:
            background = self.get_rgbimage(
                image_data=self.redis.get("background_image_data_zoom_%d" % zoom)
            )
        return background

    def set_default_background(self):
        self.redis.set(
            "background_image_data_zoom_%d" % self.get_zoom(),
            self.redis.get(self.arinax_key),
        )

    def handle_frame(self, frame: Frame, delay: Optional[int] = 1) -> None:
        self.frame0 = frame

    def run_camera(self, epsilon=1.0e-3):
        self.master = True

        vimba = Vimba()
        system = vimba.system()
        vimba.startup()

        if system.GeVTLIsPresent:
            system.run_feature_command("GeVDiscoveryAllOnce")
            gevent.sleep(3)

        camera_ids = vimba.camera_ids()
        print("camera_ids %s" % camera_ids)
        # self.camera = vimba.camera('DEV_000F315CD6B8') #old px2 makko 20224-04-24
        self.camera = vimba.camera("DEV_000F315E0B4F")
        self.camera.open()
        self.camera.PixelFormat = self.pixel_format

        # self.set_exposure(self.default_exposure_time)
        self.set_gain(self.default_gain)

        self.current_gain = self.get_gain()
        self.current_exposure_time = self.get_exposure_time()

        self.camera.arm("Continuous", self.handle_frame)
        self.camera.start_frame_acquisition()

        k = 0
        last_image_id = 0
        _start = time.time()
        while self.master:
            if hasattr(self, "frame0") and self.frame0.data.frameID != last_image_id:
                k += 1
                try:
                    img = self.frame0.buffer_data_numpy()
                except:
                    gevent.sleep(0.01)
                    continue
                if self.use_jpeg:
                    last_image_data = simplejpeg.encode_jpeg(img)
                else:
                    last_image_data = img.ravel().tostring()
                last_image_timestamp = str(time.time())
                last_image_id = self.frame0.data.frameID
                last_image_frame_timestamp = str(self.frame0._vmb_frame.timestamp)
                self.redis.set(self.arinax_key, last_image_data)
                self.redis.set("last_image_timestamp", last_image_timestamp)
                self.redis.set("last_image_id", last_image_id)
                self.redis.set("last_image_frame_timestamp", last_image_frame_timestamp)

                if self.publish_in_arinax_format:
                    try:
                        header = self.get_header(image_id=int(last_image_id))
                        # image_data = self.frame0.buffer_data()  # self.frame0.data.buffer
                        image_data = img.tobytes()  # tostring()
                        # struct_rgbimage = self.get_struct_rgbimage(header=header, image_data=image_data)
                        struct_rgbimage = header + image_data
                        self.redis.publish(self.arinax_key, struct_rgbimage)
                    except:
                        print("header type", type(header))
                        print("data type", type(image_data))
                        traceback.print_exc()

                current_state_vector_with_string_values = (
                    self.get_state_vector_with_string_values()
                )
                current_state_vector_with_float_values = (
                    self.get_state_vector_with_float_values(
                        current_state_vector_with_string_values
                    )
                )
                current_state_vector_as_single_string = (
                    self.get_state_vector_as_single_string(
                        current_state_vector_with_string_values
                    )
                )

                try:
                    last_saved_state_vector_string = (
                        self.get_last_saved_state_vector_string()
                    )
                    last_saved_state_vector_with_float_values = self.get_state_vector_with_float_values_from_state_vector_as_single_string(
                        last_saved_state_vector_string
                    )
                except:
                    last_saved_state_vector_with_float_values = None

                if (
                    last_saved_state_vector_with_float_values is None
                    or self.state_vectors_are_different(
                        current_state_vector_with_float_values,
                        last_saved_state_vector_with_float_values,
                    )
                ):
                    self.redis.rpush("history_image_data", last_image_data)
                    self.redis.rpush("history_image_timestamp", last_image_timestamp)
                    self.redis.rpush(
                        "history_state_vector", current_state_vector_as_single_string
                    )

                current_history_size = self.redis.llen("history_image_timestamp")
                if (
                    current_history_size > self.history_size_threshold * 1.2
                    and self.redis.get("can_clear_history") == "1"
                ) or current_history_size >= 2 * self.history_size_threshold:
                    for item in [
                        "history_image_data",
                        "history_image_timestamp",
                        "history_state_vector",
                    ]:
                        self.redis.ltrim(
                            item, self.history_size_threshold, self.redis.llen(item)
                        )

                requested_gain = float(self.redis.get("camera_gain"))
                if requested_gain != self.current_gain:
                    self.set_gain(requested_gain)
                requested_exposure_time = float(self.redis.get("camera_exposure_time"))
                current_exposure_time = self.get_exposure_time()
                if abs(requested_exposure_time - current_exposure_time) > epsilon:
                    print(
                        f"current exp time {current_exposure_time}, requested {requested_exposure_time}, diff {requested_exposure_time-current_exposure_time}"
                    )
                    self.set_exposure(requested_exposure_time)
                    print()
            if k % 100 == 0 and k != 0:
                print("length of history %d" % self.redis.llen("history_image_data"))
                print(f"frame id {last_image_id}")

            gevent.sleep(0.01)

        # self.camera.run_feature_command("AcquisitionStop")
        self.camera.stop_frame_acquistion()
        self.close_camera()

    def close_camera(self):
        self.master = False

        with Vimba() as vimba:
            self.camera.flush_capture_queue()
            self.camera.end_capture()
            self.camera.revoke_all_frames()
            vimba.shutdown()

    def start_camera(self):
        return

    def get_filtered_image(self, color=False, threshold=0.95):
        img = self.get_image(color=color)
        threshold = max(128, img.max() * threshold)
        img[img < threshold] = 0
        return img

    def get_integral_of_bright_spots(self, threshold=0.95):
        img = self.get_filtered_image(color=False)
        iobs = img.sum()
        return iobs

    def align_from_single_image(
        self,
        generate_report=False,
        display=False,
        turn=True,
        dark=False,
        predict_img_size=(256, 320),
    ):
        logging.getLogger("HWR").info("camera align_from_single_image")
        _start = time.time()

        reference_position = self.goniometer.get_aligned_position()
        calibration = self.get_calibration()
        zoom = self.get_zoom()
        center = self.get_beam_position()

        logging.getLogger("HWR").info(
            "align_from_single_image: about to acquire an image and start the analysis"
        )
        name_pattern = "autocenter_%s_%s" % (
            os.getuid(),
            time.asctime().replace(" ", "_"),
        )
        logging.getLogger("HWR").info(
            "align_from_single_image: saving the image %s" % name_pattern
        )
        if dark == True:
            name_pattern = "%s_dark_failed.jpg" % name_pattern
        else:
            name_pattern = "%s_bright_failed.jpg" % name_pattern
        directory = "%s/manual_optical_alignment" % os.getenv("HOME")

        imagename, sample_image, image_id = self.save_image(
            os.path.join(directory, name_pattern), color=True
        )

        scale = np.array(sample_image.shape[:2]) / np.array(predict_img_size)
        request_arguments = {}
        request_arguments["to_predict"] = sample_image
        request_arguments["model_img_size"] = predict_img_size
        request_arguments["save"] = False
        request_arguments["prefix"] = "name_pattern"
        predictions = get_predictions(request_arguments)
        try:
            most_likely_click = get_most_likely_click(predictions)
        except:
            most_likely_click = -1
            print(traceback.print_exc())

        _end = time.time()
        sign = -1.0
        step = 0.25
        if most_likely_click == -1:
            logging.getLogger("HWR").info(
                "align_from_single_image: nothing found (sample not visible?)"
            )
            reference_position["Omega"] += 90.0
            reference_position["AlignmentY"] += -1 * sign * step
            self.goniometer.set_position(reference_position)
            return

        logging.getLogger("HWR").info(
            "align_from_single_image: analysis took %.2f seconds" % (_end - _start)
        )
        centroid = np.array(most_likely_click) * scale
        y, x = centroid.astype("int")
        os.rename(
            imagename,
            imagename.replace("_failed.jpg", "_zoom_%d_y_%d_x_%d.jpg" % (zoom, y, x)),
        )

        logging.getLogger("HWR").info("predicted click (y, x): (%d, %d)" % (y, x))

        vector = (centroid - center) * calibration
        logging.getLogger("HWR").info("estimated shift %s" % str(vector))
        aligned_position = (
            self.goniometer.get_aligned_position_from_reference_position_and_shift(
                reference_position, vector[1], vector[0]
            )
        )
        if turn == True:
            aligned_position["Omega"] += 90.0
        self.goniometer.set_position(aligned_position)
        self.goniometer.save_position()
        _end = time.time()
        logging.getLogger("HWR").info(
            "align_from_single_image: analysis + movement took %.2f seconds"
            % (_end - _start)
        )
        # return aligned_position

    def get_contrast(self, image=None, method="RMS", roi=None):
        if image is None:
            image = self.get_image(color=False)
        elif len(image.shape) == 3:
            image = image.mean(axis=2)

        # if roi != None:
        # image =
        # image = image.astype(np.float)
        Imean = image.mean()
        if method == "Michelson":
            Imax = image.max()
            Imin = image.min()
            contrast = (Imax - Imin) / (Imax + Imin)
        elif method == "Weber":
            background = self.get_default_background()
            Ib = background.mean()
            contrast = (Imean - Ib) / Ib
        elif method == "RMS":
            contrast = np.sqrt(np.mean((image - Imean) ** 2))

        return contrast

    def save_reference(self, ref, directory):
        img = self.save_image(os.path.join(directory, "%s.jpg" % ref), color=True)
        pos = self.goniometer.get_aligned_position()
        f = open(os.path.join(directory, "%s.pickle" % ref), "wb")
        pickle.dump(pos, f)
        f.close()


if __name__ == "__main__":
    cam = camera()
    cam.run_camera()
