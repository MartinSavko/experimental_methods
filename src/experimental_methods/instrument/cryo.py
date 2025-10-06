#!/usr/bin/env python
# -*- coding: utf-8 -*-

from monitor import monitor
from tango import DeviceProxy


class cryostream(monitor):
    def __init__(
        self,
        device_name="i11-ma-cx1/ex/cryostream800",
        name="cryo_monitor",
        attributes=[
            "evapHeat",
            "evapShift",
            "gasHeat",
            "suctHeat",
            "rampRate",
            "backPressure",
            "evapTemp",
            "gasFlow",
            "sampleTemp",
            "setPoint",
            "suctTemp",
            "targetTemp",
            "tempError",
        ],
    ):
        super().__init__(name=name)
        self.device = DeviceProxy(device_name)
        self.attributes = attributes

    def get_point(self):
        return [self.get_attribute(attribute) for attribute in self.attributes]

    def set_attributes(
        self,
        attributes=[
            "evapHeat",
            "evapShift",
            "gasHeat",
            "suctHeat",
            "rampRate",
            "backPressure",
            "evapTemp",
            "gasFlow",
            "sampleTemp",
            "setPoint",
            "suctTemp",
            "targetTemp",
            "tempError",
        ],
    ):
        self.attributes = attributes

    def get_attributes(self):
        return self.attributes

    def get_attribute(self, attribute):
        return self.device.read_attribute(attribute).value

    def get_position(self, attributes=None):
        if attributes is None:
            attributes = self.attributes
        return dict(
            [(attribute, self.get_attribute(attribute)) for attribute in attributes]
        )
