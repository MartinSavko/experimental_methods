#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .monitor import tango_monitor

class cryostream(tango_monitor):
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
            "state",
            "status",
        ],
    ):
        super().__init__(device_name=device_name, name=name, attributes=attributes)


def main():
    cryo = cryostream()


if __name__ == "__main__":
    main()
