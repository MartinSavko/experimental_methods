#!/usr/bin/env python
# -*- coding: utf-8 -*-

try:
    import tango
except:
    import PyTango as tango
import logging
import traceback
import gevent
import time
#from resolution import resolution_mockup
import numpy as np

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
        self.energy = tango.DeviceProxy('i11-ma-c00/ex/beamlineenergy')
        self.coupling = tango.DeviceProxy('i11-ma-c00/ex/ble-coupling')
        self.mono = tango.DeviceProxy('i11-ma-c03/op/mono1')
        self.mono_mt_rx = tango.DeviceProxy('i11-ma-c03/op/mono1-mt_rx')
        self.mono_mt_rx_fine = tango.DeviceProxy('i11-ma-c03/op/mono1-mt_rx_fine')
        self.undulator = tango.DeviceProxy('ans-c11/ei/m-u24_energy')
        self.experimental_table = tango.DeviceProxy('i11-ma-c05/ex/tab.2')
        self.test = test
        self.resolution = None #resolution_mockup()
        
    def abort(self):
        self.energy.Stop()
        self.mono_mt_rx.Stop()
        self.experimental_table.Stop()
        self.undulator.Stop()
        
    def energy_converged(self):
        theta_energy = self.resolution.get_energy_from_theta(self.mono_mt_rx.read_attribute('position'))
        beamline_energy = self.get_energy()
        return np.isclose(theta_energy, beamline_energy)
    
    def set_energy(self, energy, wait=False, energy_tolerance=0.5, tries=7, timeout=15, sleeptime=0.1):
        '''assuming energy specified in keV'''
        if self.test: return -1
        if energy < 100:
            '''probably specified in keV'''
            energy *= 1e3
        if abs(self.get_energy()-energy) <= energy_tolerance: 
            print('energy_difference negligible', abs(self.get_energy()-energy))
            if abs(self.undulator.energy - energy) >= energy_tolerance:
                try:
                    self.undulator.write_attribute('energy', energy * 1e-3)
                except:
                    print(traceback.print_exc())
                _startu = time.time()
                while self.undulator.state().name in ['MOVING'] and time.time()-_startu < timeout:
                    gevent.sleep(sleeptime)
        else:   
            self.turn_on()
            gevent.sleep(sleeptime)
            self.energy.write_attribute('energy', energy * 1e-3)
        #start = time.time()
        #attempt = 0
        #while not self.energy_converged() and attempt < tries and time.time()-start < timeout:
            #attempt += 1
            #try:
                #self.energy.write_attribute('energy', energy * 1e-3)
                #gevent.sleep(1)
            #except:
                #logging.getLogger().info('did not succeed to set energy on try no %d' % attempt)
                #logging.getLogger().exception(traceback.format_exc())
            #logging.getLogger().info('set energy on try no %d' % attempt)
            
        if wait:
            self.wait()
        return energy
    
    def get_energy(self):
        return self.mono.read_attribute('energy').value * 1.e3

    def get_wavelength(self):
        return self.mono.read_attribute('lambda').value
    
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
                gevent.sleep(sleeptime)
        while not self.mono_mt_rx_fine.state().name == 'OFF':
            try:
                self.mono_mt_rx_fine.Off()
            except:
                logging.error(traceback.print_exc())
                gevent.sleep(sleeptime)

    def turn_on(self):
        if self.test: return
        if self.mono_mt_rx.state().name == 'OFF':
            self.mono_mt_rx.On()
        if self.mono_mt_rx_fine.state().name == 'OFF':
            self.mono_mt_rx_fine.On()
    
    def get_state(self):
        try:
            state =  self.energy.state().name
        except:
            state = 'ALARM'
        return state
    
    def wait(self, sleeptime=0.1):
        while self.get_state() not in ['STANDBY', 'ALARM']:
            gevent.sleep(sleeptime)

    def get_current_coupling(self):
        return self.energy.currentCouplingName
    
    def set_coupling(self, coupling):
        self.energy.ChangeCoupling(coupling)
