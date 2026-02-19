#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import time
import urllib.request

from zmq_camera import zmq_camera


class axis_camera(zmq_camera):
    def __init__(
        self,
        camera,  # cam14
        name_modifier=None,  # 1
        chunk=2**13,
        port=5555,
        history_size_target=36000,
        debug_frequency=100,
        framerate_window=25,
        codec="hevc",
        service=None,
        verbose=None,
        server=None,
    ):
        self.camera = camera
        self.name_modifier = name_modifier
        self.chunk = chunk

        self.camera_id = f"{self.camera:s}"
        if self.name_modifier:
            self.camera_id = f"{self.camera:s}_{self.name_modifier}"

        self.url = f"http://{self.camera:s}/mjpg/video.mjpg"
        if self.name_modifier:
            self.url = self.url.replace(
                "mjpg/video.mjpg", f"mjpg/{self.name_modifier}/video.mjpg"
            )
        # print(f"self.url is {self.url}")

        zmq_camera.__init__(
            self,
            port=port,
            history_size_target=history_size_target,
            debug_frequency=debug_frequency,
            framerate_window=framerate_window,
            codec=codec,
            service=service,
            verbose=verbose,
            server=server,
        )

    def initialize(self, bytess=bytes()):
        if "http_proxy" in os.environ:
            del os.environ["http_proxy"]
        self.stream = urllib.request.urlopen(self.url)
        self.bytess = bytess

        super().initialize()

    def acquire(self):
        try:
            self.bytess += self.stream.read(self.chunk)
            b = self.bytess.find(b"\xff\xd9")
            a = self.bytess.find(b"\xff\xd8")
            if a != -1 and b != -1:
                self.value_id += 1
                self.timestamp = time.time()
                self.value = self.bytess[a : b + 2]
                self.bytess = self.bytess[b + 2 :]

            super().acquire()
        except:
            self.initialize(bytess=self.bytess)

    def get_command_line(self):
        command_line = f"axis_stream.py -c {self.camera} -s {self.service} -o {self.codec} -C {int(self.chunk**0.5):d}"
        if self.name_modifier is not None:
            command_line += f" -m {self.modifier}"
        return command_line
    

def decode_jpeg(jpg, doer="simplejpeg"):
    if doer == "simplejpeg":
        img = simplejpeg.decode_jpeg(jpg)
    else:
        img = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
    return img


def save_history(filename, jpegs, timestamps=[], state_vectors=[]):
    _start = time.time()
    dt = h5py.special_dtype(vlen=np.dtype("uint8"))

    logging.info(f"len(jpegs) {len(jpegs)}, type(jpegs) {type(jpegs)}")

    history_file = h5py.File(filename, "w")

    history_file.create_dataset("history_images", data=jpegs, dtype=dt)

    if state_vectors != []:
        history_file.create_dataset("history_state_vectors", data=state_vectors)

    if timestamps != []:
        duration = timestamps[-1] - timestamps[0]
        logging.info(f"duration {duration:.2f} seconds")
        logging.info(f"framerate {len(timestamps)/duration:.2f}")
        history_file.create_dataset("history_timestamps", data=timestamps)

    history_file.close()
    logging.info(f"save_history function took {time.time() - _start:.2f} seconds")


def main(camera="cam14", modifier=None, k=20, record=False, display=False, duration=-1):
    logging.info(f"accessing mjpg stream at {camera}")

    camera_id = f"{camera:s}"
    if modifier:
        camera_id = f"{camera:s}_{modifier}"
    url = f"http://{camera:s}/mjpg/video.mjpg"
    if modifier:
        url = url.replace("mjpg/video.mjpg", f"mjpg/{modifier}/video.mjpg")

    history = []
    timestamps = []
    value_ids = []
    stream = urllib.request.urlopen(url)
    bytess = b""
    header = b""
    value_id = 0
    last = time.time()

    # f = False
    # if record:
    # fname = f"{camera}.mjpg"
    # if modifier:
    # fname = f"{camera}_{modifier}.mjpg"
    # f = open(fname, "wb")

    start = time.time()
    print(
        f"duration {duration}, record {record}, {(duration<0 or (duration>0 and duration > time.time() - start))}"
    )
    l = 0
    while True and (duration < 0 or (duration > 0 and duration > time.time() - start)):
        l += 1
        bytess += stream.read(1024)
        a = bytess.find(b"\xff\xd8")
        b = bytess.find(b"\xff\xd9")
        if a != -1 and b != -1:
            header = bytess[:a]
            timestamp = time.time()
            jpg = bytess[a : b + 2]

            if record:
                # f.write(jpg)
                history.append(np.frombuffer(jpg, dtype="uint8"))
                timestamps.append(timestamp)
                value_ids.append(value_id)

            bytess = bytess[b + 2 :]
            value_id += 1

            if display:
                try:
                    i = decode_jpeg(jpg, doer="simplejpeg")
                    cv2.imshow("i", i)
                except:
                    logging.exception(traceback.format_exc())

                if cv2.waitKey(1) == 27:
                    exit(0)

            if value_id % k == 0:
                now = time.time()
                logging.info(f"cycle {l}")
                logging.info(f"length of the last frame {len(jpg)}")
                logging.info(f"value_id {value_id}, frame_rate {k/(now-last):.2f} Hz")
                logging.info(f"header {len(header)}, {header}")
                last = now
        time.sleep(0.00001)
    if record:
        save_history(
            f"{camera_id}_{duration}_seconds_{time.time():.2f}.h5",
            history,
            timestamps=timestamps,
        )


def main():
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,)

    parser.add_argument("-c", "--camera", default="cam14", type=str, help="camera")
    parser.add_argument(
        "-m",
        "--modifier",
        default=None,
        type=str,
        help="specific camera in case of multi camera system",
    )
    parser.add_argument(
        "-k", "--debug_frequency", default=100, type=int, help="debug frame"
    )
    parser.add_argument(
        "-s", "--service", type=str, default="a", help="debug string add to the outputs"
    )
    parser.add_argument(
        "-C",
        "--chunk",
        type=int,
        default=16,
        help="log2 of byte stream chunk to read at a time",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose")
    parser.add_argument("-o", "--codec", type=str, default="h264", help="video codec")
    args = parser.parse_args()
    print(args)

    cam = axis_camera(
        args.camera,
        name_modifier=args.modifier,
        service=args.service,
        debug_frequency=args.debug_frequency,
        chunk=2**args.chunk,
        codec=args.codec,
        verbose=False,
    )
    cam.verbose = args.verbose
    cam.set_server(True)
    cam.serve()

    sys.exit(0)


if __name__ == "__main__":
    main()
