#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import PyTango
import traceback
from md2_mockup import md2_mockup
from monitor import monitor
from motor import tango_motor
import numpy as np
import gevent

class fast_shutter(monitor):
    def __init__(self):
        try:
            self.device = PyTango.DeviceProxy('i11-ma-cx1/ex/md2')
            self.motor_x = tango_motor('i11-ma-c06/ex/shutter-mt_tx')
            self.motor_z = tango_motor('i11-ma-c06/ex/shutter-mt_tz')
        except:
            print traceback.print_exc()
            self.device = md2_mockup()
        
        monitor.__init__(self)
    
    def get_alignment_actuators(self):
        return self.motor_x, self.motor_z
        
    def enable(self):
        self.device.FastShutterIsEnabled = True

    def disable(self):
        self.device.FastShutterIsEnabled = False

    def isopen(self):
        return self.device.FastShutterIsOpen
        
    def open(self, tries=7, sleep_time=0.05):
        k = 0
        success = False
        while self.device.FastShutterIsOpen==False and k<tries:
            k+=1
            try:
                self.device.FastShutterIsOpen = True
                print 'fast shutter opened on %d try' % k
                success = True
            except:
                gevent.sleep(sleep_time)
        if success:
            return True
        else:
            return False
            
    def close(self):
        self.device.FastShutterIsOpen = False

    def get_x(self):
        return self.motor_x.get_position()
        
    def set_x(self, position):
        self.motor_x.set_position(position)
    
    def get_z(self):
        return self.motor_z.get_position()
        
    def set_z(self, position):
        self.motor_z.set_position(position)
        
    def state(self):
        return self.device.fastshutterisopen
    
    def get_point(self):
        return self.device.FastShutterIsOpen
    
    def get_name(self):
        return 'fast_shutter'
    