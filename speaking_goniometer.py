#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import time
import logging
import pprint
import threading
import numpy as np
import pickle
import h5py
import json
import copy

sys.path.insert(0, "/usr/local/experimental_methods/embl")
from StandardClient import PROTOCOL
from MDClient import MDClient

from speech import speech, defer

#SERVER_ADDRESS = "172.19.10.119"
SERVER_ADDRESS = "172.19.10.181"
SERVER_PORT = 9001
SERVER_PROTOCOL = PROTOCOL.STREAM
TIMEOUT = 3
RETRIES = 1

logging.basicConfig(format="%(asctime)s|%(module)s %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S",
                            level=logging.INFO)

class speaking_goniometer(MDClient, speech):
    def __init__(
        self,
        port=5555,
        history_size_target=10000,
        debug_frequency=100,
        service=None,
        verbose=None,
        server=None,
        default_save_destination="/nfs/data4/histories",
    ):
        self.verbose = verbose
        self.service = service
        self.history_size_target = history_size_target
        self.debug_frequency = debug_frequency
        self.can_clear_history = True
        self.position = {}
        self.history_events = []
        self.history_vectors = []
        self.history_times = []
        self.sung = 0
        self.frame_id = 0
        self.last_reported_frame_id = -1
        self.last_handled_frame_id = -1
        self.default_save_destination = default_save_destination

        MDClient.__init__(
            self, SERVER_ADDRESS, SERVER_PORT, SERVER_PROTOCOL, TIMEOUT, RETRIES
        )
        speech.__init__(
            self, port=port, service=service, verbose=verbose, server=server
        )
        
        if self.server:
            sys.path.insert(0, "/usr/local/experimental_methods/arinax")
            from TestMD import get_server
            self.md = get_server()
        else:
            self.md = None
    
    def onEvent(self, name, value, timestamp):
        self.history_events.append((time.time(), name, value, timestamp))
        if name == self.STATE_EVENT:
            self.onStateEvent(value)
        elif name == self.STATUS_EVENT:
            self.onStatusEvent(value)
        elif name == self.MOTOR_STATES_EVENT:
            array = self.parseArray(value)
            states_dict = self.createDictFromStringList(array)
            self.onMotorStatesEvent(states_dict)
        elif name == self.MOTOR_POSITIONS_EVENT:
            array = self.parseArray(value)
            states_dict = self.createDictFromStringList(array)
            self.onMotorPositionsEvent(states_dict)
        elif name.endswith(self.STATE_EVENT):
            device = name[: (len(name) - len(self.STATE_EVENT))]
            self.onDeviceStateEvent(device, value)
        elif name.endswith(self.VALUE_EVENT):
            device = name[: (len(name) - len(self.VALUE_EVENT))]
            self.onDeviceValueEvent(device, value)
        elif name.endswith(self.POSITION_EVENT):
            device = name[: (len(name) - len(self.POSITION_EVENT))]
            self.onDevicePositionEvent(device, value, timestamp)

    def onStateEvent(self, state):
        pass

    def onStatusEvent(self, status):
        pass

    def onMotorStatesEvent(self, states_dict):
        pprint.pprint(positions_dict)

    def onMotorPositionsEvent(self, positions_dict):
        pprint.pprint(positions_dict)

    def onDeviceValueEvent(self, device, value):
        pass
        # print('onDeviceValueEvent', device, value)

    def onDeviceStateEvent(self, device, state):
        pass

    def onDevicePositionEvent(
        self,
        device,
        position,
        timestamp,
        ignore=[
            "CentringTableFocus",
            "CentringTableVertical",
            "CentringTableOrthogonal",
            "CentringTable",
            "AlignmentTable",
            "Capillary",
            "Scintillator",
            "Aperture",
        ],
    ):
        if device in ignore:
            return

        self.frame_id += 1
        self.position[device] = position
        self.timestamp = time.time()
        self.history_vectors.append(copy.copy(self.position))
        self.history_times.append([self.timestamp, timestamp])

        # print(f"onDevicePositionEvent: {device} = {position}, {timestamp}, {self.timestamp}, {self.timestamp-timestamp/1000.}")

        self.inform()
        self.check_history()

        message = [
            self.service_name,
            bytes(device, encoding="utf-8"),
            bytes(position, encoding="utf-8"),
            b"%f" % self.timestamp,
            b"%d" % timestamp,
        ]
        self.singer.send_multipart(message)
        self.sung += 1
        logging.info(f'self.sung {self.sung} {self.singer_port}')

    def check_history(self, factor=1.5):
        if (
            len(self.history_vectors) > self.history_size_target * factor
            and self.can_clear_history
        ) or len(self.history_vectors) > 10 * factor * self.history_size_target:
            if self.verbose:
                logging.info(f"cleaning history {len(self.history_vectors)}")
            del self.history_vectors[: -self.history_size_target]
            del self.history_times[: -self.history_size_target]
            if self.verbose:
                logging.info(f"history cleared {len(self.history_vectors)}")

    @defer
    def read_attribute(self, attribute):
        return self.md[attribute]
    
    @defer
    def write_attribute(self, attribute, value):
        self.md[attribute] = value
    
    @defer
    def command(self, command_name, args=(), kwargs={}):
        print(f"command {command_name}, {args}, {kwargs}")
        return getattr(self.md, command_name)(*args, **kwargs)
    
    @defer
    def inform(self, force=False):
        if (
            self.frame_id
            and self.frame_id != self.last_reported_frame_id
            and self.frame_id % self.debug_frequency == 0
        ) or force:
            logging.info(f"current position {self.position}")
            logging.info(f"last timestamp {self.timestamp}")
            logging.info(f"history vectors {len(self.history_vectors)}")
            logging.info(f"history duration {self.get_history_duration():.2f}")
            if hasattr(self, "bytess"):
                logging.info(f"len(self.bytess) {len(self.bytess)}")
            logging.info(f"self.sung {self.sung}\n")
            self.last_reported_frame_id = self.frame_id

    @defer
    def get_history_duration(self):
        return self.history_times[-1][0] - self.history_times[0][0]

    @defer
    def get_history_events(self):
        return self.history_events
    
    @defer
    def get_history(self, start=-np.inf, end=np.inf, last_n=None):
        #return self.history_times, self.history_vectors

        self.can_clear_history = False

        timestamps = np.array(self.history_times)[:, 0]
        
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
        times = self.history_times[mi:ma]
        vectors = self.history_vectors[mi:ma]

        self.can_clear_history = True
        return times, vectors
    
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
        _start = time.time()
        
        times, vectors = self.get_history(start=start, end=end)
        
        times = np.array(times)
        times = times[:, 0]
        device_times = times[:, 1]

        logging.info(f"len(vectors) {len(vectors)}, type(vectors) {type(vectors)}")
        duration = times[-1] - times[0]
        logging.info(f"duration {duration:.2f} seconds")
        logging.info(f"framerate {len(times)/duration:.2f}")

        if not os.access(os.path.dirname(filename), os.W_OK):
            filename = os.path.join(
                self.default_save_destination, os.path.basename(filename)
            )

        if filename.endswith("pickle"):
            f = open(filename, "wb")
            pickle.dump({"vectors": vectors, "timestamps": times, "device_times": device_times}, f)
            f.close()
        elif filename.endswith("json"):
            f = open(filename, "wb")
            json.dump(
                {"vectors": b"".join(vectors), "timestamps": ",".join(list(times)), "device_times": ",".join(list(device_times))}, f
            )
            f.close()
        elif filename.endswith("npy"):
            tvectors = [ 
                [
                    float(ts),
                ]
                + self.get_array_from_dictionary(position)
                for ts, dt, position in zip(times, device_times, vectors)
            ]
            np.save(filename, np.array(tvectors))
        else:
            dt = h5py.special_dtype(vlen=np.dtype("uint8"))
            history_file = h5py.File(filename, "w")
            logging.info("history_file opened")
            history_file.create_dataset(
                "history_images",
                data=[np.frombuffer(jpeg, dtype="uint8") for jpeg in vectors],
                dtype=dt,
            )
            logging.info("vectors written")
            history_file.create_dataset("history_timestamps", data=timestamps)
            history_file.create_dataset("history_device_times", data=device_times)
            history_file.close()

        logging.info(f"save_history work took {time.time() - _start:.4f} seconds\n")

    def get_array_from_dictionary(
        self,
        vector,
        keys=[
            "Omega",
            "AlignmentX",
            "AlignmentY",
            "AlignmentZ",
            "CentringX",
            "CentringY",
            "Kappa",
            "Phi",
            "CapillaryVertical",
            "CapillaryHorizontal",
            "ApertureVertical",
            "Zoom",
        ],
    ):
        return [float(vector[key]) for key in vector if key in vector]


if __name__ == "__main__":
    import gevent

    print("--------------   starting  ------------------")
    md = speaking_goniometer()
    print("--------------   started  ------------------")
    methods = md.get_method_list()
    print("--------------   Listing methods  ------------------")
    print("Methods:")
    for method in methods:
        print(method)

    gevent.sleep(3)
    methods = md.get_method_list()
    print("--------------   Listing methods  ------------------")
    print("Methods:")
    for method in methods:
        print(method)
    print("--------------   Listing properties  ------------------")
    properties = md.getPropertyList()
    print("Properties:")
    for property in properties:
        print(property)
    print(property)

    # Example recovering conection after a MD restart
    # It is not needed to call connect explicitly. Connectiong is set with any command/attributr access.
    # Connection may be explicitly restored though to for receiving events
    if not md.isConnected():
        md.connect()
