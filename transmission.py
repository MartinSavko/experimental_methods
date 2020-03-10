#!/usr/bin/env python
# -*- coding: utf-8 -*-

from slits import slits1, slits2
import pickle
from scipy.interpolate import interp1d
import numpy as np
import PyTango
import math
import logging
import time

class transmission_mockup:
    def __init__(self):
        self.transmission = 1
    def get_transmission(self):
        return self.transmission
    def set_transmission(self, transmission):
        self.transmission = transmission
    

class transmission:

    def __init__(self, mode='horizontal', 
                 slits1_reference_scan='/usr/local/experimental_methods/s1d_results.pickle',
                 slits2_reference_scan='/usr/local/experimental_methods/s2d_results.pickle', 
                 reference_gap=4., 
                 reference_position=0.,
                 percent_factor=100.):
        
        self.mode = mode
        self.reference_gap = reference_gap
        self.reference_position = reference_position
        self.s1 = slits1()
        self.s2 = slits2()
        self.percent_factor = percent_factor
        
        slits1_scan_results = pickle.load(open(slits1_reference_scan))
        slits2_scan_results = pickle.load(open(slits2_reference_scan))
        
        left = np.vstack([slits1_scan_results['i11-ma-c02/ex/fent_h.1-mt_i']['analysis']['position'], slits1_scan_results['i11-ma-c02/ex/fent_h.1-mt_i']['analysis']['transmission']])
        
        self.left_t_as_f_of_p = interp1d(left[0, :], left[1, :] - 0.5, bounds_error=False, fill_value='extrapolate')
        self.left_p_as_f_of_t = interp1d(left[1, :] - 0.5, left[0, :], bounds_error=False, fill_value='extrapolate')
        right = np.vstack([slits1_scan_results['i11-ma-c02/ex/fent_h.1-mt_o']['analysis']['position'], slits1_scan_results['i11-ma-c02/ex/fent_h.1-mt_o']['analysis']['transmission']])
        
        self.right_t_as_f_of_p = interp1d(right[0, :], right[1, :] - 0.5, bounds_error=False, fill_value='extrapolate')
        self.right_p_as_f_of_t = interp1d(right[1, :] - 0.5, right[0, :], bounds_error=False, fill_value='extrapolate')
        
    def get_transmission(self):
        
        l_t = self.left_t_as_f_of_p(self.s1.i.get_position())
        r_t = self.right_t_as_f_of_p(self.s1.o.get_position())
        
        return min([1, l_t + r_t]) * self.percent_factor
                                    
        
    def set_transmission(self, transmission, wait=True):
        transmission /= self.percent_factor
        self.s1.set_vertical_gap(self.reference_gap)
        self.s1.set_vertical_position(self.reference_position)
        self.s1.set_horizontal_position(self.reference_position)
        
        l_p = self.left_p_as_f_of_t(np.min([0.5, transmission/2.]))
        r_p = self.right_p_as_f_of_t(np.min([0.5, transmission/2.]))
        
        gap = l_p + r_p
        
        self.s1.set_horizontal_gap(gap)
        
    
class old_transmission:
 
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
    