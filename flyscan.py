#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import tango
import traceback
import logging


class flyscan:
    def __init__(self):
        self.frontend = tango.DeviceProxy("flyscan/core/frontend.1")
        self.recording = tango.DeviceProxy("flyscan/core/recording-manager.1")

    def get_result_file_location(self):
        filename = self.recording.filename
        projectpath = self.recording.projectpath
        realpath = projectpath.replace("ruche-proxima2a", "ruche")
        subdirectory = self.recording.datasubdirectory
        result_file_location = os.path.join(realpath, subdirectory, filename)
        return result_file_location

    def set_default_parameters(self):
        self.frontend.configName = "energy"
        # for attribute in ['fluo', 'xbpm1', 'cvd1', 'psd5', 'psd6']:
        # setattr(self.frontend, '%sEnabled' % attribute, True)
        setattr(self.frontend, "energy_step_size", "%.4f" % (0.0005))
        setattr(self.frontend, "gate_mode", "True")
        setattr(self.frontend, "clock_mode", "position")
        setattr(self.frontend, "panda_first_pulse_delay", "%.4f" % (0.001))
        setattr(self.frontend, "start_to_shutter_pos_delay", "%.4f" % (0.0025))

    def set_integration_time(self, integration_time):
        setattr(self.frontend, "integration_time", "%.4f" % (integration_time * 1e3))

    def set_energy_start(self, energy_start):
        setattr(self.frontend, "energy_start", "%.3f" % (energy_start * 1.0e-3))

    def set_energy_end(self, energy_end):
        setattr(self.frontend, "energy_end", "%.3f" % (energy_end * 1.0e-3))

    def run(self):
        self.frontend.Run()
