#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import pickle
import h5py
import numpy as np
import logging
import simplejpeg
import traceback
import time
import threading
from speech import speech, defer
from imageio import imsave
from useful_routines import get_dirname

class zmq_camera(speech):
    service = None
    server = None
    last_reported_value_id = None
    last_handled_value_id = None

    def __init__(
        self,
        port=5555,
        history_size_target=10000,
        debug_frequency=100,
        sleeptime=1.0e-3,
        framerate_window=25,
        codec="hevc",
        service=None,
        verbose=None,
        server=None,
        default_save_destination="/nfs/data4/movies",
    ):
        self.codec = codec
        self.verbose = verbose
        self.service = service

        speech.__init__(
            self,
            port=port,
            service=service,
            verbose=verbose,
            server=server,
            history_size_target=history_size_target,
            debug_frequency=debug_frequency,
            sleeptime=sleeptime,
            framerate_window=framerate_window,
        )
        
    @defer
    def set_codec(self, codec="hevc"):
        self.codec = codec

    def get_jpeg(self):
        return self.get_value()

    @defer
    def get_last_image(self, color=True):
        image = simplejpeg.decode_jpeg(self.get_jpeg())
        if color is False and len(image.shape) == 3:
            image = image.mean(axis=2)
        return image

    @defer
    def encode_jpeg(self, img):
        return simplejpeg.encode_jpeg(img)

    @defer
    def get_image(self, color=True):
        last_image = self.get_last_image()
        if not color:
            last_image = last_image.mean(axis=2)
        return last_image

    @defer
    def get_rgbimage(self):
        rgbimage = self.get_image(color=True)
        return rgbimage

    @defer
    def get_bwimage(self):
        bwimage = self.get_image(color=False)
        return bwimage

    @defer
    def get_contrast(self, image=None, method="RMS", roi=None):
        if image is None:
            image = self.get_last_image(color=False)
        elif len(image.shape) == 3:
            image = image.mean(axis=2)

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

    def save_image(self, imagename, image=None, color=True):
        if image is None:
            image_id, image = self.get_value_id(), self.get_image(color=color)
        else:
            image_id = -1

        dirname = get_dirname(imagename)
        if not os.path.isdir(dirname):
            try:
                os.makedirs(dirname)
            except OSError:
                print("Could not create the destination directory")
        try:
            imsave(imagename, image)
        except:
            print("Could not save image, please check")
            traceback.print_exc()

        return imagename, image, image_id

    def _save_history(self, filename, start, end, last_n):
        try:
            times, values = self.get_history(start=start, end=end, last_n=last_n)
        except:
            return

        if not os.path.isdir(os.path.dirname(filename)):
            try:
                os.makedirs(os.path.dirname(filename))
            except:
                traceback.print_exc()

        if not os.access(os.path.dirname(filename), os.W_OK):
            filename = os.path.join(
                self.default_save_destination, os.path.basename(filename)
            )

        if not os.access(os.path.dirname(filename), os.W_OK):
            filename = os.path.join(
                self.default_save_destination, os.path.basename(filename)
            )

        if filename.endswith("pickle"):
            f = open(filename, "wb")
            pickle.dump({"values": values, "timestamps": list(times)}, f)
            f.close()
        elif filename.endswith("json"):
            f = open(filename, "wb")
            json.dump(
                {"values": b"".join(values), "timestamps": ",".join(list(times))}, f
            )
            f.close()
        else:
            dt = h5py.special_dtype(vlen=np.dtype("uint8"))
            history_file = h5py.File(filename, "w")
            history_file.create_dataset(
                "history_images",
                data=[np.frombuffer(jpeg, dtype="uint8") for jpeg in values],
                dtype=dt,
            )
            history_file.create_dataset("history_timestamps", data=times)
            history_file.close()

        self.can_clear_history = True
        os.system(f"movie_from_history.py -H {filename} -c {self.codec} -r -o &")
        # logging.info(f"save took {time.time() - start:.4f} seconds (service {self.service_name.decode()})")

    @defer
    def get_filtered_image(self, color=False, threshold=0.95):
        img = self.get_image(color=color)
        threshold = max(128, img.max() * threshold)
        img[img < threshold] = 0
        return img

    @defer
    def get_integral_of_bright_spots(self, threshold=0.95):
        img = self.get_filtered_image(color=False)
        iobs = img.sum()
        return iobs

    def get_image_dimensions(self):
        try:
            image_dimensions = self.get_image().shape[:2]
        except:
            image_dimensions = 1024, 1216
        return list(image_dimensions)

    def get_image_corresponding_to_timestamp(self, timestamp):
        return self.get_value_corresponding_to_timestamp(timestamp)
    
    def get_sing_value(self):
        return self.value
    
