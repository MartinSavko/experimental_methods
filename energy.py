#!/usr/bin/env python

import PyTango
import logging
import traceback
import time

class energy(object):
    def __init__(self, test=False):
        self.energy = PyTango.DeviceProxy('i11-ma-c00/ex/beamlineenergy')
        self.mono = PyTango.DeviceProxy('i11-ma-c03/op/mono1')
        self.mono_mt_rx = PyTango.DeviceProxy('i11-ma-c03/op/mono1-mt_rx')
        self.mono_mt_rx_fine = PyTango.DeviceProxy('i11-ma-c03/op/mono1-mt_rx_fine')
        self.undulator = PyTango.DeviceProxy('ans-c11/ei/m-u24_energy')
        self.test = test
        
    def set_energy(self, energy, wait=False):
        if self.test: return
        self.turn_on()
        self.energy.write_attribute('energy', energy)
        if wait:
            self.wait()

    def get_energy(self):
        return self.mono.read_attribute('energy').value

    def check_energy(self):
        if self.test: return
        if abs(self.undulator.gap - self.undulator.computedgap)/self.undulator.computedgap > 0.01:
            self.undulator.gap = self.undulator.computedgap

    def turn_off(self):
        while not self.mono_mt_rx.state().name == 'OFF':
            try:
                self.mono_mt_rx.Off()
            except:
                logging.error(traceback.print_exc())
                time.sleep(0.1)
        while not self.mono_mt_rx_fine.state().name == 'OFF':
            try:
                self.mono_mt_rx_fine.Off()
            except:
                logging.error(traceback.print_exc())
                time.sleep(0.1)

    def turn_on(self):
        if self.test: return
        if self.mono_mt_rx.state().name == 'OFF':
            self.mono_mt_rx.On()
        if self.mono_mt_rx_fine.state().name == 'OFF':
            self.mono_mt_rx_fine.On()

    def wait(self):
        while self.energy.state().name != 'STANDBY':
            time.sleep(0.1)
