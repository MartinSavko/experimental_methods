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
from monitor import monitor
from imageio import imsave

class zmq_camera(speech):
    service = None
    server = None
    last_reported_frame_id = None
    last_handled_frame_id = None

    def __init__(
        self,
        port=5555,
        history_size_target=10000,
        debug_frequency=100,
        framerate_window=25,
        codec="hevc",
        service=None,
        verbose=None,
        server=None,
        default_save_destination="/nfs/data4/movies",
    ):
        self.debug_frequency = debug_frequency
        self.framerate_window = framerate_window
        self.codec = codec
        self.verbose = verbose
        self.service = service
        self.history_size_target = history_size_target
        self.can_clear_history = True
        self.history_jpegs = []
        self.history_times = []
        self.last_sung_id = -1
        self.sung = 0
        self.default_save_destination = default_save_destination

        speech.__init__(
            self, port=port, service=service, verbose=verbose, server=server
        )

    def initialize(self):
        self.frame_id = 0
        self.last_reported_frame_id = -1
        self.last_handled_frame_id = -1

    def acquire(self):
        if self.frame_id and self.frame_id != self.last_handled_frame_id:
            self.history_jpegs.append(self.jpeg)
            self.history_times.append(self.timestamp)
            self.last_handled_frame_id = self.frame_id

    def sing(self):
        if self.frame_id != self.last_sung_id and self.frame_id > 0:
            try:
                self.singer.send_multipart(
                    [
                        self.service_name,
                        self.jpeg,
                        b"%f" % self.timestamp,
                        b"%d" % self.frame_id,
                    ]
                )
                self.last_sung_id = self.frame_id
                self.sung += 1
            except:
                logging.info("Could not sing, please check")

#Traceback (most recent call last):
  #File "/usr/local/lib/python3.8/dist-packages/zmq/sugar/socket.py", line 660, in send_multipart
    #memoryview(msg)
#TypeError: memoryview: a bytes-like object is required, not 'int'

#During handling of the above exception, another exception occurred:

#Traceback (most recent call last):
  #File "./oav_camera.py", line 291, in <module>
    #main()
  #File "./oav_camera.py", line 286, in main
    #cam.serve()
  #File "/usr/local/experimental_methods/./zmq_camera.py", line 113, in serve
    #self.sing()
  #File "/usr/local/experimental_methods/./zmq_camera.py", line 66, in sing
    #self.singer.send_multipart(
  #File "/usr/local/lib/python3.8/dist-packages/zmq/sugar/socket.py", line 665, in send_multipart
    #raise TypeError(
#TypeError: Frame 1 (-1) does not support the buffer interface.
#Segmentation fault (core dumped)

    @defer
    def inform(self):
        if (
            self.frame_id
            and self.frame_id != self.last_reported_frame_id
            and self.frame_id % self.debug_frequency == 0
        ):
            logging.info(f"length of the last frame {len(self.jpeg)}")
            logging.info(
                f"frame_id {self.frame_id}, frame_rate {self.get_framerate():.2f} Hz (computed over last {self.framerate_window} images)"
            )
            logging.info(f"history images {len(self.history_jpegs)}")
            logging.info(f"history duration {self.get_history_duration():.2f}")
            if hasattr(self, "bytess"):
                logging.info(f"len(self.bytess) {len(self.bytess)}")
            logging.info(f"self.sung {self.sung}\n")
            self.last_reported_frame_id = self.frame_id

    def check_history(self, factor=1.5):
        if (
            len(self.history_jpegs) > self.history_size_target * factor
            and self.can_clear_history
        ) or len(self.history_jpegs) > 10 * factor * self.history_size_target:
            if self.verbose:
                logging.info(f"cleaning history {len(self.history_jpegs)}")
            del self.history_jpegs[: -self.history_size_target]
            del self.history_times[: -self.history_size_target]
            if self.verbose:
                logging.info(f"history cleared {len(self.history_jpegs)}")

    def serve(self, sleeptime=1e-6):
        self.initialize()

        while self.server:
            self.acquire()  # will acquire a new image if there is one

            self.sing()

            if self.verbose:
                self.inform()

            self.check_history()

            time.sleep(sleeptime)

    @defer
    def set_codec(self, codec="hevc"):
        self.codec = codec

    @defer
    def get_framerate(self):
        if not self.history_times:
            return -1
        elif len(self.history_times) > self.framerate_window + 1:
            ht = np.array(self.history_times[-self.framerate_window - 1 :])
        else:
            ht = np.array(self.history_times)
        htd = ht[1:] - ht[:-1]
        median_frame_duration = np.median(htd)
        median_frame_rate = 1.0 / median_frame_duration
        return median_frame_rate

    @defer
    def get_history_duration(self):
        return self.history_times[-1] - self.history_times[0]

    
    @defer
    def set_server(self, value=True):
        self.server = value

    @defer
    def get_singer_port(self):
        return self.singer_port

    def get_server(self):
        return self.server

    def start_serve(self):
        self.serve_thread = threading.Thread(target=self.serve)
        self.serve_thread.daemon = True
        self.serve_event = threading.Event()
        self.set_server(True)
        self.serve_thread.start()

    @defer
    def get_jpeg(self):
        return self.jpeg
    
    @defer
    def get_timestamp(self):
        return self.timestamp

    @defer
    def get_last_frame_id(self):
        return self.frame_id
    
    @defer
    def get_last_image(self):
        return simplejpeg.decode_jpeg(self.get_jpeg())
    
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
    def get_contrast(self, image=None, method='RMS', roi=None):
        if image is None:
            image = self.get_last_image(color=False)
        elif len(image.shape) == 3:
            image = image.mean(axis=2)
        
        Imean = image.mean()
        if method == 'Michelson':
            Imax = image.max()
            Imin = image.min()
            contrast = (Imax - Imin)/(Imax + Imin)
        elif method == 'Weber':
            background = self.get_default_background()
            Ib = background.mean()
            contrast = (Imean-Ib)/Ib
        elif method == 'RMS':
            contrast = np.sqrt(np.mean((image - Imean)**2))
        
        return contrast 
    
    def save_image(self, imagename, image=None, color=True):
        if image is None:
            image_id, image = self.get_last_frame_id(), self.get_image(color=color)
        else:
            image_id = -1
        if not os.path.isdir(os.path.dirname(imagename)):
            try:
                os.makedirs(os.path.dirname(imagename))
            except OSError:
                print('Could not create the destination directory')
        imsave(imagename, image)
        return imagename, image, image_id
    
    
    @defer
    def get_history(self, start=-np.inf, end=np.inf, last_n=None):
        self.can_clear_history = False

        timestamps = np.array(self.history_times)
        if last_n is not None:
            mi, ma = -last_n, len(timestamps) + 1
        else:
            indices = np.argwhere(
                np.logical_and(timestamps >= start, timestamps <= end)
            )
            try:
                mi, ma = indices.min(), indices.max() + 1
            except ValueError:
                logging.info("It seems camera stopped before the requested start")
                if self.server:
                    logging.info("attempt to reinitialize ...")
                    self.initialize()
                    return
        times = timestamps[mi:ma]
        jpegs = self.history_jpegs[mi:ma]

        self.can_clear_history = True
        return times, jpegs


    @defer
    def save_history(self, filename, start=-np.inf, end=np.inf, last_n=None):
        _start = time.time()
        self.save_history_thread = threading.Thread(
            target=self._save_history,
            args=(filename, start, end, last_n),
            # kwargs={"start": start, "end": end, "last_n": last_n},
        )
        self.save_history_thread.daemon = False
        self.save_history_thread.start()
        logging.info(f"save_history thread took {time.time() - _start:.4f} seconds")

    def _save_history(self, filename, start, end, last_n):
        times, jpegs = self.get_history(start=start, end=end, last_n=last_n)
        logging.info(f"len(jpegs) {len(jpegs)}, type(jpegs) {type(jpegs)}")
        duration = times[-1] - times[0]
        logging.info(f"duration {duration:.2f} seconds")
        logging.info(f"framerate {len(times)/duration:.2f}")

        if not os.access(os.path.dirname(filename), os.W_OK):
            filename = os.path.join(
                self.default_save_destination, os.path.basename(filename)
            )

        if filename.endswith("pickle"):
            f = open(filename, "wb")
            pickle.dump({"jpegs": jpegs, "timestamps": list(times)}, f)
            f.close()
        elif filename.endswith("json"):
            f = open(filename, "wb")
            json.dump(
                {"jpegs": b"".join(jpegs), "timestamps": ",".join(list(times))}, f
            )
            f.close()
        else:
            dt = h5py.special_dtype(vlen=np.dtype("uint8"))
            history_file = h5py.File(filename, "w")
            logging.info("history_file opened")
            history_file.create_dataset(
                "history_images",
                data=[np.frombuffer(jpeg, dtype="uint8") for jpeg in jpegs],
                dtype=dt,
            )
            logging.info("jpegs written")
            history_file.create_dataset("history_timestamps", data=times)
            history_file.close()

        self.can_clear_history = True
        os.system(f"movie_from_history.py -H {filename} -c {self.codec} -o &")
        logging.info(f"save_history work took {time.time() - start:.4f} seconds\n")
    
    @defer
    def get_image_corresponding_to_timestamp(self, timestamp):
        self.redis.set("can_clear_history", 0)
        try:
            timestamps = self.history_times
            timestamps_before = timestamps[timestamps <= timestamp]
            closest = np.argmin(np.abs(timestamps_before - timestamp))
            corresponding_image = self.history_images[int(closest)]

        except:
            corresponding_image = self.get_rgbimage()
        self.redis.set("can_clear_history", 1)

        return corresponding_image

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
        return list(self.get_image().shape[:2])
