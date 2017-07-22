#!/usr/bin/env python
# -*- coding: utf-8 -*-

import PyTango
import logging
import math
import time

class transmission_mockup:
    def __init__(self):
        self.transmission = 1
    def get_transmission(self):
        return self.transmission
    def set_transmission(self, transmission):
        self.transmission = transmission
    
class transmission:
    def __init__(self, test=False):
        self.horizontal_gap = PyTango.DeviceProxy('i11-ma-c02/ex/fent_h.1')
        self.vertical_gap = PyTango.DeviceProxy('i11-ma-c02/ex/fent_v.1')
        self.fp_parser = PyTango.DeviceProxy('i11-ma-c00/ex/fp_parser')
        self.Const = PyTango.DeviceProxy('i11-ma-c00/ex/fpconstparser')
        self.test = test

    def get_transmission(self):
        return self.fp_parser.TrueTrans_FP

    def set_transmission(self, transmission, wait=True):
        if self.test: return 
        try:
            truevalue = (2.0 - math.sqrt(4 - 0.04 * transmission)) / 0.02
        except ValueError:
            logging.debug('ValueError with %s' % transmission)
            truevalue = x/2.
        newGapFP_H = math.sqrt(
            (truevalue / 100.0) * self.Const.FP_Area_FWHM / self.Const.Ratio_FP_Gap)
        newGapFP_V = newGapFP_H * self.Const.Ratio_FP_Gap

        self.horizontal_gap.gap = newGapFP_H
        self.vertical_gap.gap = newGapFP_V
        
        if wait == True:
            self.wait(self.horizontal_gap)
        if wait == True:
            self.wait(self.vertical_gap)
        
    def wait(self, device, sleeptime=0.1):
        while device.state().name not in ['STANDBY', 'ALARM']:
            time.sleep(sleeptime)
            
def test():
    t = transmission()
    import sys
    print 'current transmission', t.get_transmission()
    print 'setting transmission %s' % sys.argv[1], t.set_transmission(float(sys.argv[1]))
    time.sleep(1)
    print 'current transmission', t.get_transmission()
    
if __name__ == '__main__':
    test()
    