#!/usr/bin/env python
# -*- coding: utf-8 -*-

from experimental_methods.instrument.monitor import tango_monitor


class experimental_table(tango_monitor):
    def __init__(
        self,
        device_name="i11-ma-c05/ex/tab.2",
        name="position_monitor",
        attributes=["pitch", "roll", "yaw", "zC", "xC", "state", "status"],
    ):
        super().__init__(device_name=device_name, name=name, attributes=attributes)

    def get_pitch(self):
        return self.get_attribute("pitch")

    def get_roll(self):
        return self.get_attribute("roll")

    def get_yaw(self):
        return self.get_attribute("yaw")

    def get_xC(self):
        return self.get_attribute("xC")

    def get_zC(self):
        return self.get_attribute("zC")


def main():
    tab2 = experimental_table()


if __name__ == "__main__":
    main()
