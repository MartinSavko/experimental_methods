#!/usr/bin/env python
# -*- coding: utf-8 -*-
import gevent

import PyTango
import time
from monitor import monitor
from goniometer import goniometer

class fluorescence_detector(monitor):
    
    def __init__(self,
                 device_name='i11-ma-cx1/dt/dtc-mca_xmap.1',
                 channel='channel00',
                 sleeptime=0.001):
    
        self.device = PyTango.DeviceProxy(device_name)
        self.channel = channel
        self.goniometer = goniometer()
        self.sleeptime = sleeptime
        self._calibration = -16.1723871876, 9.93475667754, 0.0
        self.observe = None
    
    def set_integration_time(self, integration_time):
        self.device.presetValue = integration_time
        
    
    def get_integration_time(self):
        return self.device.presetValue
    
    
    def set_roi(self, start, end):
        self.device.SetRoisFromList(["0;%d;%d;%d;%d;%d;%d" % (start, end, 50, end, start, end+250)])
      
    
    def get_roi_start_and_roi_end(self):
        return map(int, self.device.getrois()[0].split(';')[1:])
        
    
    def insert(self):
        self.goniometer.insert_fluorescence_detector()
        try:
            self.device.accumulate = False
        except:
            pass
    
    def extract(self):
        self.goniometer.extract_fluorescence_detector()
        
    
    def cancel(self):
        self.device.Abort()
    
    
    def get_state(self):
        return self.device.State().name
    
    
    def get_counts_in_roi(self):
        return float(self.device.roi00_01)
    
    
    def get_counts_uptoend_roi(self):
        return float(self.device.roi00_02)
        
    
    def get_counts_compton_roi(self):
        return float(self.device.roi00_03)
    
    
    def get_real_time(self):
        return self.device.realTime00
    
    
    def get_dead_time(self):
        return self.device.deadTime00
    
    
    def get_trigger_live_time(self):
        return self.device.triggerLiveTime00
    
    
    def get_input_count_rate(self):
        return float(self.device.inputCountRate00)
    
    
    def get_output_count_rate(self):
        return float(self.device.outputCountRate00)
   
    
    def get_events_in_run(self):
        return float(self.device.eventsInRun00)
        
    
    def get_calculated_dead_time(self):
        icr = self.get_input_count_rate()
        ocr = self.get_output_count_rate()
        if icr == 0:
            return 0
        return 1e2 * (1 - (ocr / icr))
    
    
    def get_calibration(self):
        return self._calibration


    def snap(self):
        return self.device.Snap()


    def get_spectrum(self):
        spectrum = self.device.read_attribute(self.channel).value
        return spectrum
        
    
    def get_point(self, wait=True):
        self.snap()
        if wait:
            integration_time = self.get_integration_time()
            gevent.sleep(integration_time/10)
            while self.get_state() != 'STANDBY':
                gevent.sleep(self.sleeptime)
        return self.get_spectrum()
    
    
    def measure(self, wait=True):
        self.snap()
        if wait:
            integration_time = self.get_integration_time()
            gevent.sleep(integration_time)
            while self.get_state() != 'STANDBY':
                gevent.sleep(self.sleeptime)
                
    
    def get_single_observation(self, chronos=None):
        measure_start_time = time.time()
        self.measure()
        measure_end_time = time.time()
        
        readout_start_time = time.time()
        counts_in_roi = self.get_counts_in_roi()
        counts_uptoend_roi = self.get_counts_uptoend_roi()
        counts_compton = self.get_counts_compton_roi()
        normalized_counts = 1000. * counts_in_roi/counts_compton
        readout_end_time = time.time()
        spectrum = self.get_spectrum()
        dead_time = self.get_dead_time()
        input_count_rate = self.get_input_count_rate()
        output_count_rate = self.get_output_count_rate()
        real_time = self.get_real_time()
        events_in_run = self.get_events_in_run()
        readout_time = readout_end_time - readout_start_time
        measure_time = measure_end_time - measure_start_time
        return [chronos + measure_time/2., spectrum, counts_in_roi, normalized_counts, dead_time, input_count_rate, output_count_rate, real_time, events_in_run, measure_time, readout_time]
        
    
    def monitor(self, start_time):
        self.observations = []
        self.observation_fields = ['chronos', 'spectrum', 'counts_in_roi', 'normalized_counts', 'dead_time', 'input_count_rate', 'output_count_rate', 'real_time', 'events_in_run', 'measure_time', 'readout_time', 'duration']
        while self.observe == True:
            chronos = time.time() - start_time
            observation = self.get_single_observation(chronos)
            duration = time.time() - start_time - chronos
            self.observations.append(observation+[duration])
            
    
    def get_observations(self):
        return self.observations
        
    
    def get_observation_fields(self):
        return self.observation_fields
        
        
def main():
    
    fd = fluorescence_detector()
    fd.set_integration_time(0.5)
    fd.get_point()
    
if __name__ == '__main__':
    main()
