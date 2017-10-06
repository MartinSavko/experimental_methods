#!/usr/bin/env python
# -*- coding: utf-8 -*-

import PyTango
import logging
import traceback
import time

class energy_mockup:
    def set_energy(self, energy):
        self.photon_energy = energy
    def get_energy(self):
        return 12.650
    def check_energy(self):
        return
    def turn_off(self):
        return
    def turn_on(self):
        return
    def wait(self):
        return

class energy(object):
    def __init__(self, test=False):
        self.energy = PyTango.DeviceProxy('i11-ma-c00/ex/beamlineenergy')
        self.mono = PyTango.DeviceProxy('i11-ma-c03/op/mono1')
        self.mono_mt_rx = PyTango.DeviceProxy('i11-ma-c03/op/mono1-mt_rx')
        self.mono_mt_rx_fine = PyTango.DeviceProxy('i11-ma-c03/op/mono1-mt_rx_fine')
        self.undulator = PyTango.DeviceProxy('ans-c11/ei/m-u24_energy')
        self.test = test
        
    def set_energy(self, energy, wait=False, energy_tolerance=0.5):
        '''assuming energy specified in keV'''
        if self.test: return -1
        if energy < 100:
            '''probably specified in keV'''
            energy *= 1e3
        if abs(self.get_energy()-energy) <= energy_tolerance: 
            print 'energy_difference negligible', abs(self.get_energy()-energy)
            return 0
        else:
            
            self.turn_on()
            time.sleep(0.1)
            
        self.energy.write_attribute('energy', energy * 1e-3)
        if wait:
            self.wait()
        return energy
    
    def get_energy(self):
        return self.mono.read_attribute('energy').value * 1.e3

    def check_energy(self, gap_tolerance=0.01):
        if self.test: return
        if abs(self.undulator.gap - self.undulator.computedgap)/self.undulator.computedgap > gap_tolerance:
            self.undulator.gap = self.undulator.computedgap
    
    def turn_off(self, sleeptime=0.1):
        while not self.mono_mt_rx.state().name == 'OFF':
            try:
                self.mono_mt_rx.Off()
            except:
                logging.error(traceback.print_exc())
                time.sleep(sleeptime)
        while not self.mono_mt_rx_fine.state().name == 'OFF':
            try:
                self.mono_mt_rx_fine.Off()
            except:
                logging.error(traceback.print_exc())
                time.sleep(sleeptime)

    def turn_on(self):
        if self.test: return
        if self.mono_mt_rx.state().name == 'OFF':
            self.mono_mt_rx.On()
        #if self.mono_mt_rx_fine.state().name == 'OFF':
            #self.mono_mt_rx_fine.On()
    
    def get_state(self):
        #_start = time.time()
        state =  self.energy.state().name
        #print 'energy get_state took %s' % (time.time() - _start)
        return state
    
    def wait(self, sleeptime=0.1):
        while self.get_state() not in ['STANDBY', 'ALARM']:
            #if self.get_state() == 'ALARM':
                #self.turn_on()
                #time.sleep(sleeptime*2)
            time.sleep(sleeptime)
