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

#sys.path.insert(0, "./embl")
from experimental_methods.instrument.embl.StandardClient import PROTOCOL
from experimental_methods.instrument.embl.MDClient import MDClient

from experimental_methods.utils.speech import speech, defer
from experimental_methods.utils.useful_routines import get_position_dictionary_from_position_tuple

# SERVER_ADDRESS = "172.19.10.119"
SERVER_ADDRESS = "172.19.10.181"
SERVER_PORT = 9001
SERVER_PROTOCOL = PROTOCOL.STREAM
TIMEOUT = 3
RETRIES = 1

logging.basicConfig(
    format="%(asctime)s|%(module)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


class speaking_goniometer(MDClient, speech):
    def __init__(
        self,
        port=5555,
        history_size_target=150000,
        debug_frequency=100,
        framerate_window=25,
        service=None,
        verbose=None,
        server=None,
        default_save_destination="/nfs/data4/histories",
        motors_to_watch=[
            "AlignmentX",
            "AlignmentY",
            "AlignmentZ",
            "CentringX",
            "CentringY",
            "Kappa",
            "Phi",
            "Chi",
            "Omega",
        ],
    ):
        self.verbose = verbose
        self.service = service
        self.position = {}
        self.history_events = []
        self.history_values = []
        self.history_times = []
        self.sung = 0
        self.value_id = 0
        self.last_reported_value_id = -1
        self.last_handled_value_id = -1
        self.default_save_destination = default_save_destination
        self.motors_to_watch = motors_to_watch

        MDClient.__init__(
            self,
            SERVER_ADDRESS,
            SERVER_PORT,
            SERVER_PROTOCOL,
            TIMEOUT,
            RETRIES,
        )

        speech.__init__(
            self,
            port=port,
            service=service,
            verbose=verbose,
            server=server,
            history_size_target=history_size_target,
            debug_frequency=debug_frequency,
            framerate_window=framerate_window,
        )

        if self.server:
            sys.path.insert(0, "/usr/local/experimental_methods/arinax")
            from TestMD import get_server

            self.md = get_server()
            self.position = self.get_position_dictionary(
                motors_to_watch=motors_to_watch
            )
            self.value = copy.copy(self.position)
            self.timestamp = time.time()
            self.history_events.append(None)
            self.history_times.append([self.timestamp, np.nan])
            self.history_values.append(self.value)
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
    ):
        # print(device, position, timestamp)
        if device not in self.position:
            # print("ignore")
            return

        self.position[device] = float(position)
        self.timestamp = time.time()
        self.value_id += 1
        self.value = copy.copy(self.position)
        self.history_values.append(self.value)
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
        # logging.info(f'self.sung {self.sung} {self.singer_port}')

    def check_history(self, factor=1.5):
        if (
            len(self.history_values) > self.history_size_target * factor
            and self.can_clear_history
        ) or len(self.history_values) > 10 * factor * self.history_size_target:
            if self.verbose:
                logging.info(f"cleaning history {len(self.history_values)}")
            del self.history_values[: -self.history_size_target]
            del self.history_times[: -self.history_size_target]
            if self.verbose:
                logging.info(f"history cleared {len(self.history_values)}")

    @defer
    def get_position(self):
        return self.position

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
            self.value_id
            and self.value_id != self.last_reported_value_id
            and self.value_id % self.debug_frequency == 0
        ) or force:
            logging.info(f"current position {self.position}")
            logging.info(f"last timestamp {self.timestamp}")
            logging.info(f"history vectors {len(self.history_values)}")
            logging.info(f"history duration {self.get_history_duration():.2f}")
            if hasattr(self, "bytess"):
                logging.info(f"len(self.bytess) {len(self.bytess)}")
            logging.info(f"self.sung {self.sung}\n")
            self.last_reported_value_id = self.value_id

    def _get_duration(self, times):
        return times[-1][0] - times[0][0]

    @defer
    def get_history_events(self):
        return self.history_events

    def get_array_from_dictionary(
        self,
        position_dictionary,
        motors_to_watch=None,
    ):
        if motors_to_watch is None:
            if self.motors_to_watch:
                motors_to_watch = self.motors_to_watch
            else:
                motors_to_watch = list(position_dictionary.keys())

        array = []
        for motor_name in motors_to_watch:
            if motor_name in position_dictionary:
                array.append(position_dictionary[motor_name])
            else:
                array.append(0.0)

        array = np.array(array)
        return array

    def get_values_as_array(self, values):
        array = np.array([self.get_array_from_dictionary(value) for value in values])
        return array

    @defer
    def get_position_dictionary(self, position_tuple=None, motors_to_watch=[]):
        if position_tuple is None:
            position_tuple = self.md["MotorPositions"]
        position_dictionary = get_position_dictionary_from_position_tuple(
            position_tuple, motors_to_watch
        )
        return position_dictionary


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

#
# 2025-06-24 20:14:28,728 self.sung 177299

# Traceback (most recent call last):
# File "/usr/local/experimental_methods/speaking_goniometer.py", line 267, in <module>
# methods = md.get_method_list()
# File "/usr/local/experimental_methods/embl/ExporterClient.py", line 46, in get_method_list
# ret = self.sendReceive(cmd)
# File "/usr/local/experimental_methods/embl/StandardClient.py", line 251, in sendReceive
# return self.__sendReceiveStream__(cmd)
# File "/usr/local/experimental_methods/embl/StandardClient.py", line 238, in __sendReceiveStream__
# raise SocketError("Socket error:" + str(self.error))
# StandardClient.SocketError: Socket error:Disconnected
