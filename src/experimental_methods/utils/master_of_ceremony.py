#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Object provides access to the beamline functionalities. Can execute experiments. Has past and future. Can talk and introspect.

"""

import logging
import threading
import time
from beam_align import beam_align
from speech import speech
import uuid


class master_of_ceremony(speech):
    def __init__(
        self, startup_services=["beam_align"], port=5555, service=None, verbose=False
    ):
        self.verbose = verbose
        self.port = port
        self.service = service

        speech.__init__(self, port=port, service=service, verbose=verbose)

        self.services = []
        self.past = []
        self.present = {}
        self.future = []

        for service_name in startup_services:
            self.add_service(service_name)

        self.start_monitor_future()
        self.start_handle_present()

    def add_service(self, service_name):
        self.services.append(service_name)
        ii = f'{service_name:s}(name_pattern="{service_name:s}", directory="~/{service_name:s}")'
        logging.info(ii)
        setattr(self.__class__, service_name, eval(ii))

    def ask(self, service={}):
        logging.info(f"service {service} requested")
        assigned_id = uuid.uuid4()
        self.future.append((assigned_id, service))
        return assigned_id

    def start_monitor_future(self):
        self.monitor_future_thread = threading.Thread(target=self.monitor_future)
        self.monitor_future_thread.daemon = True
        self.monitor_future_event = threading.Event()
        self.monitor_future_thread.start()

    def stop_monitor_future(self):
        self.monitor_future_event.set()

    def monitor_future(self, sleeptime=0.05):
        while not self.monitor_future_event.is_set():
            if len(self.future):
                uuid, service = self.future.pop()

                logging.info(f"service {service} about to be rendered")

                service_name = service["name"]
                parameters = service["parameters"]
                for parameter in parameters:
                    setattr(
                        getattr(self, service_name), parameter, parameters[parameter]
                    )
                experiment = {}
                experiment["object"] = copy.deepcopy(getattr(self, service_name))
                experiment["service_name"] = service_name
                experiment["parameters"] = parameters

                self.present[uuid] = experiment

            time.sleep(sleeptime)

    def start_handle_present(self):
        self.handle_present_thread = threading.Thread(target=self.handle_present)
        self.handle_present_thread.daemon = True
        self.handle_present_event = threading.Event()
        self.handle_present_thread.start()

    def stop_handle_present(self):
        self.handle_present_event.set()

    def handle_present(self):
        while not self.handle_present_event.is_set():
            for uuid in self.present:
                _start = time.time()
                experiment = self.present[uuid]
                service_name = experiment["service_name"]
                parameters = experiment["parameters"]
                getattr(experiment["object"], "execute")()
                _end = time.time()
                logging.info(
                    f"service {service_name} rendered in {_end-_start:.4f} seconds"
                )
                self.past.append((uuid, self.present[uuid]))
                del self.present[uuid]

    def get_past(self):
        return self.past

    def get_present(self):
        return self.present

    def get_future(self):
        return self.future

    def get_progress(self, uuid):
        return getattr(self.present[uuid], "get_progress")()

    def get_service_catalog(self):
        return self.services


if __name__ == "__main__":
    moc = master_of_ceremony()
