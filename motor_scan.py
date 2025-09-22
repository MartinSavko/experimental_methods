#!/usr/bin/env python
# -*- coding: utf-8 -*-

import traceback
import time
import os

from experiment import experiment
from motor import tango_motor
from instrument import instrument
from monitor import Si_PIN_diode, xbpm
import pylab


class motor_scan(experiment):
    def __init__(
        self,
        name_pattern,
        directory,
        motor,
        start,
        end,
        speed,
        monitors=[],
        display=None,
        analysis=None,
    ):
        experiment.__init__(
            self, name_pattern=name_pattern, directory=directory, analysis=analysis
        )

        if type(motor) == str:
            try:
                self.motor = tango_motor(motor)
            except:
                print(traceback.print_exc())
        else:
            self.motor = motor
        self.start = start
        self.end = end
        self.speed = speed
        self.instrument = instrument()
        self.monitors = monitors
        self.display = display

    def set_up_monitors(self):
        return self.monitors

    def prepare(self):
        self.check_directory(self.directory)
        self.instrument.set_reference()
        self.motor.set_position(start, wait=True)
        self.motor.set_speed(self.speed)
        for monitor in self.monitors:
            monitor.start()
        self.observations = []

    def get_point(self):
        chronos = time.time() - self.scan_start_time
        position = self.motor.get_position()
        intensities = [monitor.get_point() for monitor in self.monitors]
        return [position, chronos] + intensities

    def run(self):
        self.scan_start_time = time.time()

        self.motor.set_position(end, wait=False)

        while self.motor.get_state() != "STANDBY":
            self.observations.append(self.get_point())
