#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import PyTango
import traceback

import numpy as np

class machine_status:
    def __init__(self,
                 device_name='ans/ca/machinestatus'):
        self.device = PyTango.DeviceProxy(device_name)
        
    def get_current(self):
        return self.device.current
    
    def get_historized_current(self, lapse=None):
        historized_current = self.device.currentTrend
        if lapse != None:
           try:
               historized_current = historized_current[-lapse:]
           except:
               print traceback.print_exc()
        return historized_current
    
    def get_current_trend(self, lapse=None):
        current_trend = np.array( zip(self.device.currentTrendTimes/1e3, self.device.currentTrend))
        if lapse != None:
           try:
               current_trend = current_trend[-lapse:, :]
           except:
               print traceback.print_exc()
        return current_trend
    
    def get_operator_message(self):
        return self.device.operatorMessage + self.device.operatorMessage2
    
    def get_message(self):
        return self.device.message
    
    def get_end_of_beam(self):
        return self.device.endOfCurrentFunctionMode
    
    def get_vertical_emmitance(self):
        return self.device.vEmittance
    
    def get_horizontal_emmitance(self):
        return self.device.hEmmitance
    
    def get_filling_mode(self):
        return self.device.fillingMode
    
    def get_average_pressure(self):
        return self.device.averagePressure
    
    def get_function_mode(self):
        return self.device.functionMode
    
    def is_beam_usable(self):
        return self.device.isBeamUsable
    
    def get_time_from_last_top_up(self):
        pass
        
    def get_time_to_last_top_up(self):
        pass
    
class machine_status_mockup:
    def __init__(self, default_current=450.):
        self.default_current = default_current
        
    def get_current(self):
        return self.default_current
    
    def get_current_trend(self):
        return 
    
    def get_operator_message(self):
        return 'mockup'
    
    def get_message(self):
        return 'mockup'
    
    def get_end_of_beam(self):
        return 
    
    def get_vertical_emmitance(self):
        return 38.3e-12
    
    def get_horizontal_emmitance(self):
        return 4480.e-12
    
    def get_filling_mode(self):
        return 'Hybrid'
    
    def get_average_pressure(self):
        return 1e-10
    
    def get_function_mode(self):
        return 'top-up'
    
    def is_beam_usable(self):
        return True
    
    def get_time_from_last_top_up(self):
        return 'inf'
        
    def get_time_to_last_top_up(self):
        return 'inf'

def main():
    from scipy.optimize import curve_fit, minimize
    import pylab
    def current(time, max_current, period, offset, constant):
        time -= offset
        time = time % period
        current = max_current - constant*time 
        return current
    
    mac = machine_status()
    
    def residual(x, measured_current_trend):
        max_current, period, offset, constant = x
        time = measured_current_trend[:,0]
        measured_current = measured_current_trend[:,1]
        diff = current(time, max_current, period, offset, constant) - measured_current
        return np.dot(diff, diff)
    
    max_current0 = 451.8
    period0 = 210.
    offset0 = 0.
    constant0 = 0.0048*max_current0/period0
    
    x0 = [max_current0, period0, offset0, constant0]
    mc = mac.get_current_trend()[85140:, :]
    fit = minimize(residual, x0, args=(mc,))
    print fit.x
    print 'fit'
    print fit
    #print mc
    pylab.plot(mc[:,0], mc[:,1], label='measured')
    max_current, period, offset, constant = fit.x
    pylab.plot(mc[:,0], current(mc[:,0], max_current, period, offset, constant), label='predicted')
    pylab.legend()
    pylab.grid(True)
    pylab.show()
    
if __name__ == '__main__':
    main()