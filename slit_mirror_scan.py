#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Slits scan. Execute scan on a pair of slits.

"""
import gevent

import PyTango

import traceback
import logging
import time
import itertools
import os
import pickle
import numpy as np
import pylab
from scipy.constants import eV, h, c, angstrom, kilo, degree

from xray_experiment import xray_experiment
from motor import tango_motor
from monitor import xray_camera
from slit_scan import slit_scan


class adaptive_mirror(object):
    mirror_address = {"vfm": "i11-ma-c05/op/mir2-vfm", "hfm": "i11-ma-c05/op/mir3-hfm"}

    def __init__(self, mirror="vfm", check_time=1.0):
        self.mirror = mirror
        self.check_time = check_time
        self.mirror_device = PyTango.DeviceProxy(self.mirror_address[self.mirror])
        channel_base = self.mirror_address[mirror].replace(mirror, "ch.%02d")
        self.channels = [PyTango.DeviceProxy(channel_base % k) for k in range(12)]
        self.pitch = tango_motor(
            self.mirror_address[mirror]
            .replace("mir2-vfm", "mir.2-mt_rx")
            .replace("mir3-hfm", "mir.3-mt_rz")
        )
        self.translation = tango_motor(
            self.mirror_address[mirror]
            .replace("mir2-vfm", "mir.2-mt_tz")
            .replace("mir3-hfm", "mir.3-mt_tx")
        )

    def get_channel_values(self):
        return [getattr(c, "voltage") for c in self.channels]

    def get_channel_target_values(self):
        return [getattr(c, "targetVoltage") for c in self.channels]

    def set_channel_target_values(self, channel_values, number_of_channels=12):
        if len(channel_values) == 0:
            print("not modifying target values as none specified")
        elif len(channel_values) != number_of_channels:
            print(
                "not modifying target values as the value vector length is not consistent with number of channels (%d)"
                % number_of_channels
            )
        else:
            for k, c in enumerate(channel_values):
                if c not in [None, np.nan]:
                    setattr(self.channels[k], "targetVoltage", c)
                else:
                    print(
                        "not modifying channel %d, current value %.1f"
                        % (k, getattr(c, "voltage"))
                    )
        print("current target voltages: %s" % self.get_channel_target_values())

    def set_voltages(self, channel_values):
        self.set_channel_target_values(channel_values)
        self.mirror_device.SetChannelsTargetVoltage()
        self.wait()

    def wait(self):
        print("Setting %s mirror voltages" % self.mirror)
        print("Please wait for %s mirror tensions to settle ..." % self.mirror)
        while self.mirror_device.State().name != "STANDBY":
            gevent.sleep(self.check_time)
            print("wait ", end=" ")
        print()
        print("done!")
        print("%s mirror tensions converged" % self.mirror)

    def get_pitch_position(self):
        return self.pitch.get_position()

    def get_translation_position(self):
        return self.translation.get_position()

    def set_pitch_position(self, position):
        self.pitch.set_position(position)

    def set_translation_position(self, position):
        self.translation.set_position(position)


class mirror_scan(slit_scan):
    mirrors = {"vfm": "i11-ma-c05/op/mir2-vfm", "hfm": "i11-ma-c05/op/mir3-hfm"}

    specific_parameter_fields = [
        {"name": "mirror", "type": "str", "description": "Target mirror"},
        {"name": "channel_values", "type": "list", "description": "Mirror tensions"},
    ]

    def __init__(
        self,
        name_pattern,
        directory,
        mirror="vfm",
        channel_values=[],
        start_position=1.0,
        end_position=-1.0,
        scan_speed=None,
        darkcurrent_time=5.0,
        photon_energy=None,
        diagnostic=True,
        analysis=None,
        conclusion=None,
        simulation=None,
        display=False,
        extract=False,
    ):
        if hasattr(self, "parameter_fields"):
            self.parameter_fields += mirror_scan.specific_parameter_fields
        else:
            self.parameter_fields = mirror_scan.specific_parameter_fields[:]

        self.default_experiment_name = f"Slits {slits:d} mirror scan scan between {start_position:.1f} and {end_position:.1f} mm"

        slit_scan.__init__(
            self,
            name_pattern,
            directory,
            slits=3,
            start_position=start_position,
            end_position=end_position,
            scan_speed=scan_speed,
            darkcurrent_time=darkcurrent_time,
            photon_energy=photon_energy,
            diagnostic=diagnostic,
            analysis=analysis,
            conclusion=conclusion,
            simulation=simulation,
            display=display,
            extract=extract,
        )

        self.xray_camera = xray_camera()

        del self.monitors_dictionary["calibrated_diode"]
        self.monitors.pop()
        self.monitor_names.pop()

        self.monitors_dictionary["xray_camera"] = self.xray_camera
        self.monitor_names += ["xray_camera"]
        self.monitors += [self.xray_camera]
