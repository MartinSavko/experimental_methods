#!/usr/bin/env python
# -*- coding: utf-8 -*-
from motor import tango_motor

class detector_beamstop:
    def __init__(self):
        self.mt_x = tango_motor('i11-ma-cx1/ex/bst.3-mt_tx')
        self.mt_z = tango_motor('i11-ma-cx1/ex/bst.3-mt_tz')
    def set_z(self, position):
        self.mt_z.set_position(position)
    def set_x(self, position):
        self.mt_z.set_position(position)
    def get_z(self):
        return self.mt_z.get_position()
    def get_x(self):
        return self.mt_x.get_position()
