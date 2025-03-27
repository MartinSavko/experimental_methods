#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import logging
try:
    import tango
except ImportError:
    import PyTango as tango
import traceback
from md_mockup import md_mockup
from monitor import monitor
from goniometer import goniometer
from motor import tango_motor
import numpy as np
import gevent
import time

log = logging.getLogger('experiment')
#stream_handler = logging.StreamHandler(sys.stdout)
#stream_formatter = logging.Formatter('%(asctime)s |%(module)s |%(levelname)-5s | %(message)s')
#stream_handler.setFormatter(stream_formatter)
#log.addHandler(stream_handler)
log.setLevel(logging.INFO)

symbols = ['-','\\','|','/']

class fast_shutter(monitor):
    def __init__(self, panda=False, name='fast_shutter'):
        self.name = name
        self.panda = panda
        try:
            self.device = tango.DeviceProxy('i11-ma-cx1/ex/md3')
            self.pandabox_dataviewer = tango.DeviceProxy('flyscan/clock/pandabox-dataviewer')
            self.motor_x = tango_motor('i11-ma-c06/ex/shutter-mt_tx')
            self.motor_z = tango_motor('i11-ma-c06/ex/shutter-mt_tz')
        except:
            logging.error(traceback.print_exc())
            self.device = md_mockup()
        
        monitor.__init__(self)
    
        self.goniometer = goniometer()
        
    def get_alignment_actuators(self):
        return self.motor_x, self.motor_z
        
    def enable(self):
        if not self.device.FastShutterIsEnabled:
            self.device.FastShutterIsEnabled = True

    def disable(self):
        if self.device.FastShutterIsEnabled:
            self.device.FastShutterIsEnabled = False

    def isopen(self):
        if self.panda:
            return self.pandabox_dataviewer.ttlout2 == 'ZERO\n'
        return self.device.FastShutterIsOpen

    def isclosed(self):
        return not self.isopen()
    
    def open(self, tries=7, sleep_time=0.05):
        log.debug('in fast_shutter open %s' % self.isopen())
        k = 0
        success = False
        
        while self.isopen()==False and k<tries:
            logging.debug('in while')
            k+=1
            self.goniometer.wait()
            try:
                if self.panda:
                    self.pandabox_dataviewer.ttlout2_set_low()
                else:
                    self.device.FastShutterIsOpen = True
                log.debug('fast shutter opened on %d try' % k)
                success = True
            except:
                logging.error(traceback.print_exc())
                gevent.sleep(sleep_time)
        if success:
            return True
        else:
            return False
            
    def close(self):
        if self.panda:
            self.pandabox_dataviewer.ttlout2_set_hight()
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
        return self.name
    
    def expose(self, t):
        start = time.time()
        it = 0
        self.open()
        while time.time() - start < t:
            sys.stdout.write(f'exposing {(100*(time.time()-start)/t):.2f}% {symbols[ it%len(symbols) ]}\r')
            sys.stdout.flush()
            it += 1
            time.sleep(0.1)
        self.close()
        sys.stdout.write("Done!" + it*" ")
