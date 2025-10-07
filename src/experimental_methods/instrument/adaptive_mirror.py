#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import numpy as np

try:
    import urllib2
except ImportError:
    import urllib.request as urllib2
try:
    import tango
except ImportError:
    import PyTango as tango

from experimental_methods.instrument.motor import tango_motor


class adaptive_mirror(object):
    mirror_address = {"vfm": "i11-ma-c05/op/mir2-vfm", "hfm": "i11-ma-c05/op/mir3-hfm"}
    reset_page = "http://bimorph:8080/cgi-bin/rackconfig_user/channeltest"

    def __init__(self, mirror="vfm", check_time=1.0):
        self.mirror = mirror
        self.check_time = check_time
        self.mirror_device = tango.DeviceProxy(self.mirror_address[self.mirror])
        channel_base = self.mirror_address[mirror].replace(mirror, "ch.%02d")
        self.channels = [tango.DeviceProxy(channel_base % k) for k in range(12)]
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
                    time.sleep(1)
                else:
                    print(
                        "not modifying channel %+d, current value %.1f"
                        % (k, getattr(c, "voltage"))
                    )
        print("current target voltages: %s" % self.get_channel_target_values())

    def set_voltages(self, channel_values):
        current_channel_values = self.get_channel_values()
        _start = time.time()
        if np.allclose(channel_values, current_channel_values):
            print("current channel values", current_channel_values)
            print("values requested are identical, moving on ...")

        else:
            print("Setting %s mirror voltages" % self.mirror)
            print("Please wait for %s mirror tensions to settle ..." % self.mirror)
            self.set_channel_target_values(channel_values)
            self.mirror_device.SetChannelsTargetVoltage()
            while not np.allclose(channel_values, self.get_channel_values()):
                time.sleep(self.check_time)
            self.wait()
        _end = time.time()
        print()
        print("done!")
        print("%s mirror tensions converged" % self.mirror)

        print("set_voltages() took %.2f" % (_end - _start))

    def wait(self):
        while self.mirror_device.State().name != "STANDBY":
            time.sleep(self.check_time)

    def get_pitch_position(self):
        return self.pitch.get_position()

    def get_translation_position(self):
        return self.translation.get_position()

    def set_pitch_position(self, position):
        self.pitch.set_position(position)

    def set_translation_position(self, position):
        self.translation.set_position(position)

    def reload_firmware(self, key="cG93ZXJ1c2VyOnBvd2VydXNlcg=="):
        req = urllib2.Request(self.reset_page)
        req.add_header("Authorization", "Basic %s" % key)
        handle = urllib2.urlopen(req)
        return handle

    def get_position(self):
        position = {
            "pitch": self.get_pitch_position(),
            "translation": self.get_translation_position(),
        }
        return position
