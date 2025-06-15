#/usr/bin/env python
# -*- coding: utf-8 -*-

import tango

class plate_screener:
    
    def __init__(self):
        self.rx = tango.DeviceProxy('i11-ma-cx1/ex/plate-mt_rx')
        self.tx = tango.DeviceProxy('i11-ma-cx1/ex/plate-mt_tx')
        self.tz = tango.DeviceProxy('i11-ma-cx1/ex/plate-mt_tz')
        self.ts = tango.DeviceProxy('i11-ma-cx1/ex/plate-mt_ts')
        self.sampx = tango.DeviceProxy('i11-ma-cx1/ex/plate_ech_mt_ts')
        self.sampy = tango.DeviceProxy('i11-ma-cx1/ex/plate_ech_mt_tz')
        
        self.axes = ['rx', 'tx', 'tz', 'ts', 'sampx', 'sampy']
        
    def get_position(self):
        return dict([(axis, getattr(self, axis).position) for axis in self.axes])
    
    def set_position(self, position):
        for axis in position:
            motor = getattr(self, axis)
            setattr(motor, 'position', position[axis])
            
    def off(self):
        for axis in self.axes:
            getattr(self, axis).off()

    def get_state(self):
        for axis in self.axes:
            print('axis %s: %s' % (axis, str(getattr(self, axis).state())))
