# -*- coding: utf-8 -*-
import PyTango
import traceback
from md2_mockup import md2_mockup

class fast_shutter(object):
    def __init__(self):
        try:
            self.md2 = PyTango.DeviceProxy('i11-ma-cx1/ex/md2')
            self.motor_x = PyTango.DeviceProxy('i11-ma-c06/ex/shutter-mt_tx')
            self.motor_z = PyTango.DeviceProxy('i11-ma-c06/ex/shutter-mt_tz')
        except:
            print traceback.print_exc()
            self.md2 = md2_mockup()

    def enable(self):
        self.md2.FastShutterIsEnabled = True

    def disable(self):
        self.md2.FastShutterIsEnabled = False

    def open(self):
        self.md2.FastShutterIsOpen = True

    def close(self):
        self.md2.FastShutterIsOpen = False

    def get_x(self):
        return self.motor_x.position
        
    def set_x(self):
        self.motor_x.position = position
    
    def get_z(self):
        return self.motor_z.position
        
    def set_z(self):
        self.motor_z.position = position
        
    def state(self):
        return self.md2.fastshutterisopen