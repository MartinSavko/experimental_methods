#!/usr/bin/env python
# -*- coding: utf-8 -*-
from .motor import tango_motor

try:
    import tango
except ImportError:
    import PyTango as tango


class detector_beamstop:
    def __init__(self):
        self.mt_x = tango_motor("i11-ma-cx1/ex/bst.3-mt_tx")
        self.mt_z = tango_motor("i11-ma-cx1/ex/bst.3-mt_tz")
        self.auto = tango.DeviceProxy("i11-ma-cx1/ex/bst3positionauto")

    def set_z(self, position):
        # pass
        self.mt_z.set_position(position)

    def set_x(self, position):
        # pass
        self.mt_x.set_position(position)

    def get_z(self):
        return self.mt_z.get_position()

    def get_x(self):
        return self.mt_x.get_position()

    def enable_tracking(self):
        self.auto.enableTracking = True

    def disable_tracking(self):
        self.auto.enableTracking = False
