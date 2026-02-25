#!/usr/bin/env python

import os
import sys
import h5py
import logging
import pickle
import redis
import time
import threading
import traceback
import numpy as np
import zmq

sys.path.insert(0, "./")

import MDP
from mdworker import MajorDomoWorker
from mdclient2 import MajorDomoClient

from useful_routines import (
    get_duration,
    demulti,
    make_sense_of_request,
    handle_request,
    is_jpeg,
    is_number,
)


def defer(func):
    def consider(*args, **kwargs):
        arg0 = args[0]
        args = args[1:]
        # print({func.__name__: {"args": args, "kwargs": kwargs}})
        # print('arg0', arg0)
        if getattr(arg0, "server"):
            try:
                considered = func(arg0, *args, **kwargs)
            except:
                traceback.print_exc()
                considered = -1
        else:
            params = {}
            if args:
                params["args"] = args
            if kwargs:
                params["kwargs"] = kwargs
            if not params:
                params = None
            # print('params', params)
            # request = {func.__name__: params}
            request = {"method": func.__name__, "params": params}
            considered = getattr(arg0, "talk")(request)
        return considered

    return consider


class speech:
    server = None
    service = None
    listen_thread = None
    giver = None
    talker = None
    singer = None
    verbose = False
    singer_port = None
    hear_hear = None
    value = None
    value_id = None
    last_sung_id = None
    sung = None
    timestamp = None
    history_values = None
    history_times = None
    last_handled_value_id = None
    last_reported_value_id = None
    redis_local = None

    def __init__(
        self,
        port=5555,
        service=None,
        verbose=True,
        server=None,
        ctx=None,
        history_size_target=10000,
        debug_frequency=100,
        sleeptime=1.0,
        framerate_window=25,
        default_save_destination="/nfs/data4/movies",
        debug=False,
    ):
        logging.basicConfig(
            format="%(asctime)s |%(module)s |%(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            level=logging.INFO,
        )

        self.port = port
        self.broker_address = "tcp://localhost:%s" % port
        self.service = service
        self.verbose = verbose
        self.hear_hear = False
        if self.service is None:
            self.service = self.__class__.__name__

        self.service_name = ("%s" % self.service).encode()
        self.service_name_str = self.service_name.decode()

        self.value_key = f"value_{self.service_name_str}"
        self.value_id_key = f"value_id_{self.service_name_str}"
        self.value_timestamp_key = f"value_timestamp_{self.service_name_str}"

        self.initialize_redis_local()

        self.debug_frequency = debug_frequency
        self.sleeptime = sleeptime
        self.framerate_window = framerate_window
        self.value_id = 0
        self.last_sung_id = 0
        self.sung = 0
        self.timestamp = time.time()
        self.history_values = []
        self.history_times = []
        self.acquisition_timestamps = []
        self.history_size_target = history_size_target
        self.can_clear_history = True
        self.default_save_destination = default_save_destination

        self.ctx = (
            zmq.Context()
        )  # ms 2024-09-18 is that what made zmq issue not appear during the UDC tests in July?

        # if ctx is None:
        # self.ctx = zmq.Context()
        # else:
        # self.ctx = ctx

        self.talker = MajorDomoClient(
            self.broker_address,
            ctx=self.ctx,
            name=self.service_name_str,
        )

        if debug:
            print("service", self.service_name)
            print("server", server)
            print("registered ?", self.service_already_registered())
        if server is None:
            if not self.service_already_registered():
                self.set_up_listen_thread()
                self.start_listen()
                self.server = True
                self.singer = self.ctx.socket(zmq.PUB)
                self.singer_port = self.singer.bind_to_random_port("tcp://*")
                self.singer.setsockopt(zmq.SNDHWM, 1)

                # https://stackoverflow.com/questions/58663965/pyzmq-req-socket-hang-on-context-term
                # self.singer.setsockopt(zmq.IMMEDIATE, 1)
                logging.info(f"singer_port {self.singer_port}")
                logging.info(f"serving {self.service_name_str}")
            else:
                self.server = False
                logging.debug("not serving")
        else:
            self.server = server

        if debug:
            print("self.server", self.server)

    @defer
    def any_method(self, request):
        method_name, value = handle_request(request, self)
        return method_name, value

    @defer
    def get_history_size_target(self):
        return self.history_size_target

    @defer
    def set_history_size_target(self, history_size_target):
        self.history_size_target = history_size_target

    @defer
    def set_sleeptime(self, sleeptime):
        self.sleeptime = sleeptime

    @defer
    def get_sleeptime(self):
        return self.sleeptime

    @defer
    def get_singer_port(self):
        return self.singer_port

    @defer
    def inform(self, force=False):
        message = ""
        if (
            self.value_id
            and self.value_id != self.last_reported_value_id
            and self.value_id % self.debug_frequency == 0
        ) or force:
            message = ""

            m1 = f"{self.service_name_str} size of the last value in bytes {sys.getsizeof(self.value)}"
            m2 = f"{self.service_name_str} value_id {self.value_id}, rate {self.get_rate():.2f} Hz (computed over last {self.framerate_window} images)"
            m3 = f"{self.service_name_str} history values {len(self.history_values)}"
            m4 = f"{self.service_name_str} history duration {self.get_history_duration():.2f}"
            m5 = (
                f"len(self.bytess) {len(self.bytess)}"
                if hasattr(self, "bytess")
                else ""
            )
            m6 = f"{self.service_name_str} sung {self.sung:d}\n"

            message = "\n".join([m1, m2, m3, m4, m5, m6])
            logging.info(message)
            self.last_reported_value_id = self.value_id
        return message


    def initialize_redis_local(self, host="localhost"):
        self.redis_local = redis.StrictRedis(host=host)

    def destroy(self):
        self.ctx.destroy()

    def service_already_registered(self):
        self.talker.send(b"mmi.service", self.service_name)
        reply = self.talker.recv()
        if reply is None:
            ret = True
        else:
            ret = reply[0] == b"200"
        return ret

    def introspect(self, method="get_registered_services", params=None):
        return self.talk(
            {"method": method, "params": params}, service_name=b"introspect.service"
        )

    def make_sense_of_request(self, request):
        method_name, value = make_sense_of_request(request[0], self)
        return value

    def get_sing_value(self):
        return b"%f" % self.value

    def get_sing_timestamp(self):
        return b"%f" % self.timestamp

    def get_sing_value_id(self):
        return b"%d" % self.value_id

    def _sing(self):
        self.singer.send_multipart(
            [
                self.service_name,
                self.get_sing_value(),
                self.get_sing_timestamp(),
                self.get_sing_value_id(),
            ]
        )
        self.last_sung_id = self.value_id
        self.sung += 1

    def sing(self):
        if self.value_id != self.last_sung_id and self.value_id > 0:
            try:
                self._sing()
            except:
                logging.info("Could not sing, please check")
                logging.exception(traceback.format_exc())

    def set_up_listen_singing_thread(self):
        self.listen_singing_event = threading.Event()
        self.listen_singing_thread = threading.Thread(target=self.listen_singing)
        self.listen_singing_thread.daemon = True

    def listen_singing_start(self):
        self.set_up_listen_singing_thread()
        self.listen_singing_thread.start()
        self.hear_hear = True

    def listen_singing_stop(self):
        self.listen_singing_event.set()
        self.hear_hear = False

    def listen_singing(self):
        self.song_listener = self.ctx.socket(zmq.SUB)
        self.song_listener.connect("tcp://localhost:%d" % self.get_singer_port())
        self.song_listener.setsockopt(zmq.SUBSCRIBE, b"%s" % self.service_name)

        self.id_message = -1
        while not self.listen_singing_event.is_set():
            message = self.song_listener.recv_multipart()
            self.id_message += 1
            try:
                self.value = float(message[1].decode())
            except:
                self.value = self.get_value()

            if self.value is None:
                self.value = self.get_value()
            timestamp = message[2]
            value_id = message[3]
            print(
                f"received message {self.id_message}: value: {self.value}, timestamp: {timestamp}, value_id: {value_id}"
            )

    def set_up_listen_thread(self):
        self.listen_thread = threading.Thread(target=self.listen)
        self.listen_thread.daemon = True

    def start_listen(self):
        if self.giver is None:
            self.giver = MajorDomoWorker(
                self.broker_address,
                self.service_name,
                verbose=self.verbose,
                ctx=self.ctx,
                name=self.service_name_str,
            )
        self.set_up_listen_thread()
        self.listen_event = threading.Event()
        self.listen_thread.start()
        self.server = True

    def stop_listen(self):
        logging.info("set listen_event")
        self.listen_event.set()
        self.server = False

    def listen(self):
        reply = None
        if self.listen_event is None:
            self.listen_event = threading.Event()
        while not self.listen_event.is_set():
            logging.info("listening")
            request = self.giver.recv(reply)
            reply = self.make_sense_of_request(request)
            if not isinstance(reply, list):
                reply = [reply]
        reply = [pickle.dumps("stopping to listen")]
        reply = [self.giver.reply_to, b""] + reply
        self.giver.send_to_broker(MDP.W_REPLY, msg=reply)
        logging.info("stop listen")

    def talk(self, request, service_name=None):
        if service_name is None:
            service_name = self.service_name
        encoded_request = pickle.dumps(request)
        self.talker.send(service_name, encoded_request)
        reply = self.talker.recv()
        logging.debug("reply %s" % reply)
        decoded_reply = None
        if reply is not None:
            decoded_reply = pickle.loads(reply[0])
        return decoded_reply

    def too_long(self, array=None, factor=1.5):
        if array is None:
            array = self.history_values
        flag = (
            len(array) > self.history_size_target * factor and self.can_clear_history
        ) or len(array) > 10 * factor * self.history_size_target
        return flag

    def check_history(self):
        if self.too_long():
            if self.verbose:
                logging.info(
                    f"{self.service_name_str} cleaning history {len(self.history_values)}"
                )
            del self.history_values[: -self.history_size_target]
            del self.history_times[: -self.history_size_target]
            if self.verbose:
                logging.info(
                    f"{self.service_name_str} history cleared {len(self.history_values)}"
                )

        if self.too_long(self.acquisition_timestamps):
            del self.acquisition_timestamps[: -self.history_size_target]

    def serve(self):
        self.initialize()

        while self.server:
            self.acquire()  # will acquire a new image if there is one

            self.sing()

            if self.verbose:
                self.inform()

            self.check_history()

            time.sleep(self.sleeptime)

    @defer
    def get_pid(self):
        return os.getpid()

    @defer
    def kill(self, signal=15):
        os.kill(self.get_pid(), signal)

    @defer
    def get_rate(self):
        if not self.acquisition_timestamps:
            return -1
        elif len(self.acquisition_timestamps) > self.framerate_window + 1:
            ht = np.array(self.acquisition_timestamps[-self.framerate_window - 1 :])
        else:
            ht = np.array(self.acquisition_timestamps)
        htd = ht[1:] - ht[:-1]
        median_frame_duration = np.median(htd)
        median_frame_rate = 1.0 / median_frame_duration
        return median_frame_rate

    @defer
    def get_history_duration(self):
        return get_duration(self.history_times)

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
    def get_history(self, start=-np.inf, end=np.inf, last_n=None):
        self.can_clear_history = False
        print(f"start {start}, end {end}, last_n {last_n}")

        times = self.get_history_times()
        values = self.get_history_values()

        timestamps, multistamp, multichannel = demulti(times)

        mi, ma = None, None
        if last_n is None and len(timestamps):
            if start < timestamps[-1]:
                indices = np.argwhere(
                    np.logical_and(start <= timestamps, timestamps <= end)
                )
                print(f"satisfying indices len(indices) {len(indices)}")
                try:
                    mi, ma = indices.min(), indices.max() + 1
                except ValueError:
                    logging.info("It seems monitor stopped before the requested start")
                    if self.server:
                        logging.info("attempt to reinitialize ...")
                        self.initialize()
                        return
            elif end < timestamps[0]:
                if multistamp:
                    start = [start, times[0][1:]]
                    end = [end, times[0][1:]]

                times = np.array([start, end])
                values = np.array([values[0], values[0]])

            elif start >= timestamps[-1]:
                if multistamp:
                    start = times[-1]
                    end = [end, times[-1][-1]]

                times = np.array([start, end])
                values = np.array([values[-1], values[-1]])

        elif last_n is not None:
            mi, ma = -last_n, len(timestamps) + 1

        if mi is not None and ma is not None:
            print(f"mi, ma {mi}, {ma}")
            if multichannel:
                times = [channel[mi:ma] for channel in times]
                values = [channel[mi:ma] for channel in values]
            else:
                times = times[mi:ma]
                values = values[mi:ma]
        else:
            times = [start, end]
            values = [self.get_value(), self.get_value()]
        self.can_clear_history = True
        return times, values

    @defer
    def get_acquisition_timestamps(self):
        return self.acquisition_timestamps

    @defer
    def get_history_times(self):
        return self.history_times

    @defer
    def get_history_values(self):
        history_values = -1
        try:
            if (
                type(self.history_values) is list
                and len(self.history_values)
                and (
                    is_number(self.history_values[0])
                    or (
                        type(self.history_values[0]) is bytes
                        and is_jpeg(self.history_values[0])
                    )
                    or (
                        type(self.history_values[0]) is dict
                    )
                )
            ):
                history_values = self.history_values
            elif type(self.history_values) is bytes:
                history_values = np.frombuffer(self.history_values)
            elif type(self.history_values) is list:
                logging.info(f"get_history_values: case 1")
                history_values = []
                for history in self.history_values:
                    if type(history) is bytes:
                        logging.info(f"get_history_values: case 1a")
                        history_values.append(np.frombuffer(history))
                    else:
                        message = f"W: problem in get_history_values, unhandled case, please check"
                        history_values = message
                        logging.info(message)
            else:
                message = f"W: problem in get_history_values, unhandled case, please check"
                history_values = message
                logging.info(message)
        except:
            logging.info(traceback.format_exc())
            
        if history_values == -1:
            history_values = self.history_values

        return history_values
    
    @defer
    def _get_history_values(self):
        return self.history_values.copy()
    
    @defer
    def save_history(self, filename, start=-np.inf, end=np.inf, last_n=None):
        self.save_history_thread = threading.Thread(
            target=self._save_history,
            args=(filename, start, end, last_n),
        )
        self.save_history_thread.daemon = False
        self.save_history_thread.start()

    def save_history_local(self, filename, start=-np.inf, end=np.inf, last_n=None):
        self.save_history_local_thread = threading.Thread(
            target=self._save_history,
            args=(filename, start, end, last_n),
        )
        self.save_history_local_thread.daemon = False
        self.save_history_local_thread.start()

    def _save_history(self, filename, start, end, last_n, sleeptime=1, timeout=7):
        self.can_clear_history = False
        history_read = False
        _start = time.time()
        k = 0
        times, values = None, None
        while not history_read and time.time() - _start <= timeout:
            try:
                k += 1
                times, values = self.get_history(start=start, end=end, last_n=last_n)
                history_read = True
            except:
                logging.info(f"Could not read history from {self.service_name_str}, please check")
                traceback.print_exc()
                time.sleep(sleeptime)

        if not os.path.isdir(os.path.dirname(filename)):
            try:
                os.makedirs(os.path.dirname(filename))
            except:
                traceback.print_exc()

        if times is not None and values is not None:
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
                history_file = h5py.File(filename, "w")
                history_file.create_dataset(
                    "values",
                    data=self.get_values_as_array(values),
                )
                history_file.create_dataset("timestamps", data=times)
                history_file.close()
        else:
            print(f"Could not read history {history_read}")

        self.can_clear_history = True

    @defer
    def get_value_corresponding_to_timestamp(self, timestamp):
        self.can_clear_history = False
        try:
            timestamps = self.history_times
            timestamps_before = timestamps[timestamps <= timestamp]
            closest = np.argmin(np.abs(timestamps_before - timestamp))
            corresponding_image = self.history_values[int(closest)]
        except:
            corresponding_image = self.get_rgbimage()
        self.can_clear_history = True
        return corresponding_image

    @defer
    def get_timestamp(self):
        return self.timestamp

    @defer
    def get_value_id(self):
        return self.value_id

    @defer
    def get_value(self):
        return self.value

    def get_last_value(self):
        return self.get_value()

    def get_last_value_id(self):
        return self.get_value_id()

    def initialize(self):
        self.value_id = 0
        self.last_reported_value_id = -1
        self.last_handled_value_id = -1

    def get_values_as_array(self, values):
        array = np.array(values)
        return array

    def update_history(self):
        self.history_values.append(self.get_value())
        self.history_times.append(self.get_timestamp())
        self.acquisition_timestamps.append(self.get_timestamp())

    def acquire(self):
        if self.value_id and self.value_id != self.last_handled_value_id:
            self.update_history()
            self.last_handled_value_id = self.value_id
            self.redis_local.set(self.value_key, self.get_value())
            self.redis_local.set(self.value_id_key, self.get_value_id())
            self.redis_local.set(self.value_timestamp_key, self.get_timestamp())

    def get_command_line(self):
        return ""


###################### notes ######################

# Traceback (most recent call last):
# File "/usr/local/lib/python3.8/dist-packages/zmq/sugar/socket.py", line 660, in send_multipart
# memoryview(msg)
# TypeError: memoryview: a bytes-like object is required, not 'int'

# During handling of the above exception, another exception occurred:

# Traceback (most recent call last):
# File "./oav_camera.py", line 291, in <module>
# main()
# File "./oav_camera.py", line 286, in main
# cam.serve()
# File "/usr/local/experimental_methods/./zmq_camera.py", line 113, in serve
# self.sing()
# File "/usr/local/experimental_methods/./zmq_camera.py", line 66, in sing
# self.singer.send_multipart(
# File "/usr/local/lib/python3.8/dist-packages/zmq/sugar/socket.py", line 665, in send_multipart
# raise TypeError(
# TypeError: Frame 1 (-1) does not support the buffer interface.
# Segmentation fault (core dumped)
