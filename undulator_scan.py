#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Undulator scan. Execute monochromator scan over the useful photon energy range of the beamline, i.e. roughly between 5 and 30 degrees of theta angle (i.e. between 5000 eV and 20000 eV) at a given undulator gap setting.

The range available gap settings 7.801 - 30. mm

'''

import gevent

import traceback
import logging
import time
import itertools
import os
import pickle
import numpy as np
import pylab

from xray_experiment import xray_experiment
from scipy.constants import eV, h, c, angstrom, kilo, degree
from monitor import Si_PIN_diode

class undulator_scan(xray_experiment):
            
    specific_parameter_fields = [{'name': 'start_energy', 'type': 'float', 'description': 'Scan start photon energy in eV'},
                                 {'name': 'end_energy', 'type': 'float', 'description': 'Scan end photon energy in eV'},
                                 {'name': 'scan_speed', 'type': 'float', 'description': 'Scan speed in degrees per second'},
                                 {'name': 'gap', 'type': 'float', 'description': 'Undulator gap intention in mm'},
                                 {'name': 'start_wavelength', 'type': 'float', 'description': 'Scan start wavelength in A'},
                                 {'name': 'end_wavelength', 'type': 'float', 'description': 'Scan end wavelength in A'},
                                 {'name': 'start_theta', 'type': 'float', 'description': 'Scan start wavelength in rad'},
                                 {'name': 'end_theta', 'type': 'float', 'description': 'Scan end wavelength in rad'}]
    def __init__(self,
                 name_pattern,
                 directory,
                 gap=8,
                 start_energy=5.e3,
                 end_energy=20.e3,
                 scan_speed=0.025,
                 default_speed=0.5,
                 darkcurrent_time=10.,
                 optimize=True,
                 transmission=None,
                 diagnostic=True,
                 analysis=None,
                 conclusion=None,
                 simulation=None,
                 display=False,
                 extract=False):
        
        if hasattr(self, 'parameter_fields'):
            self.parameter_fields += undulator_scan.specific_parameter_fields
        else:
            self.parameter_fields = undulator_scan.specific_parameter_fields[:]
        
        xray_experiment.__init__(self, 
                                 name_pattern, 
                                 directory,
                                 transmission=transmission,
                                 diagnostic=diagnostic,
                                 analysis=analysis,
                                 conclusion=conclusion,
                                 simulation=simulation)
    
        self.description = 'Energy scan between %6.1f and %6.1f eV at gap=%5.2f mm, Proxima 2A, SOLEIL, %s' % (start_energy, end_energy, gap, time.ctime(self.timestamp))
        
        self.gap = gap
        self.start_energy = start_energy
        self.end_energy = end_energy
        self.scan_speed = scan_speed
        self.default_speed = default_speed
        self.darkcurrent_time = darkcurrent_time
        
        self.optimize = optimize
        self.transmission = transmission
        self.diagnostic = diagnostic
        self.display = display
        self.extract = extract
        
        self.calibrated_diode = Si_PIN_diode()
        
        self.monitors_dictionary['calibrated_diode'] = self.calibrated_diode
        
        self.monitor_names += ['calibrated_diode']
        self.monitors += [self.calibrated_diode]
            
        self.actuator = self.monochromator_rx_motor
        
    def get_theta_from_wavelength(self, wavelength, units_theta=degree, units_energy=kilo*eV, units_wavelength=angstrom, d=3.1347507142511746):
        theta = np.arcsin((angstrom/units_wavelength)*wavelength/(2*d)) / units_theta
        return theta
        
    def get_wavelength_from_theta(self, theta, units_theta=degree, units_energy=kilo*eV, units_wavelength=angstrom, d=3.1347507142511746):
        wavelength = 2*d*np.sin(units_theta*theta)
        return wavelength
        
    def get_energy_from_theta(self, theta, units_theta=degree, units_energy=kilo*eV, units_wavelength=angstrom):
        wavelength = self.get_wavelength_from_theta(theta, units_wavelength=units_wavelength, units_theta=units_theta)
        energy = self.get_energy_from_wavelength(wavelength, units_energy=units_energy, units_wavelength=units_wavelength)
        return energy
        
    def get_theta_from_energy(self, energy, units_theta=degree, units_energy=kilo*eV, units_wavelength=angstrom):
        wavelength = self.get_wavelength_from_energy(energy, units_energy=units_energy, units_wavelength=units_wavelength)
        theta = self.get_theta_from_wavelength(wavelength, units_theta=units_theta, units_energy=units_energy, units_wavelength=units_wavelength)
        return theta
        
    def get_wavelength_from_energy(self, energy, units_energy=kilo*eV, units_wavelength=angstrom):
        wavelength = h*c/(units_wavelength*units_energy)/energy
        return wavelength
        
    def get_energy_from_wavelength(self, wavelength, units_energy=kilo*eV, units_wavelength=angstrom):
        energy = h*c/(units_wavelength*units_energy)/wavelength
        return energy
            
    def set_efficient_start_end(self):
        current_energy = self.get_energy_from_theta(self.actuator.get_position(), units_energy=eV, units_theta=degree, units_wavelength=angstrom)
        if abs(self.start_energy - current_energy) > abs(self.end_energy - current_energy):
            self.start_energy, self.end_energy = self.end_energy, self.start_energy
   
    def prepare(self):
        
        self.check_directory(self.directory)
        self.write_destination_namepattern(self.directory, self.name_pattern)
        
        if self.optimize:
            self.set_efficient_start_end()
            
        self.start_position = self.get_theta_from_energy(self.start_energy, units_energy=eV, units_theta=degree, units_wavelength=angstrom)
        self.end_position = self.get_theta_from_energy(self.end_energy, units_energy=eV, units_theta=degree, units_wavelength=angstrom)

        self.actuator.set_speed(self.default_speed)
        
        initial_settings = []
        
        for k in [1, 2, 3, 5, 6]:
            initial_settings.append(gevent.spawn(getattr(getattr(self, 'slits%d' % k), 'set_horizontal_gap'), 4))
            initial_settings.append(gevent.spawn(getattr(getattr(self, 'slits%d' % k), 'set_vertical_gap'), 4))
            initial_settings.append(gevent.spawn(getattr(getattr(self, 'slits%d' % k), 'set_horizontal_position'), 0))
            initial_settings.append(gevent.spawn(getattr(getattr(self, 'slits%d' % k), 'set_vertical_position'), 0))
            
        initial_settings.append(gevent.spawn(self.calibrated_diode.insert))
        initial_settings.append(gevent.spawn(self.undulator.set_position, self.gap, wait=True))
        initial_settings.append(gevent.spawn(self.actuator.set_position, self.start_position, wait=True))
        initial_settings.append(gevent.spawn(self.goniometer.set_transfer_phase, wait=True))
        if self.safety_shutter.closed():
           initial_settings.append(gevent.spawn(self.safety_shutter.open))
        
        gevent.joinall(initial_settings)
        
        self.actuator.set_speed(self.scan_speed)
        
    def run(self):
        # sleep for darkcurrent_time while observation is already running
        gevent.sleep(self.darkcurrent_time)
        self.fast_shutter.open()
        move = gevent.spawn(self.actuator.set_position, self.end_position, timeout=None, wait=True)
        move.join()
        self.fast_shutter.close()
        gevent.sleep(self.darkcurrent_time)
        
    def clean(self):
        self.save_parameters()
        self.save_results()
        #self.save_plot()
        self.actuator.set_speed(self.default_speed)
        if self.extract:
            self.calibrated_diode.extract()
            
    def norm(self, a):
        return (a - a.mean())/(a.max() - a.min())
            
    def save_plot(self):
        pylab.figure(figsize=(16, 9))
        for (monitor_name, monitor) in zip(self.monitor_names, self.monitors):
            r = np.array(self.results[monitor_name]['observations'])
            y = self.norm(r[:,1])
            pylab.plot(r[:,0], y, label=monitor_name)
        pylab.xlabel('chronos [s]')
        pylab.ylabel('intensity')
        pylab.title(self.description)
        pylab.grid(True)
        pylab.legend()
        pylab.savefig(os.path.join(self.directory, '%s_results.png' % (self.name_pattern,) ))
        
        if self.display == True:
            pylab.show()
      
    def get_start_wavelength(self):
        return self.get_wavelength_from_energy(self.start_energy, units_energy=eV, units_wavelength=angstrom)
    
    def get_end_wavelength(self):
        return self.get_wavelength_from_energy(self.end_energy, units_energy=eV, units_wavelength=angstrom)
    
    def get_start_theta(self):
        return self.get_theta_from_energy(self.start_energy, units_energy=eV, units_wavelength=angstrom, units_theta=degree)
    
    def get_end_theta(self):
        return self.get_theta_from_energy(self.end_energy, units_energy=eV, units_wavelength=angstrom, units_theta=degree)

        
def main():
    
    usage = '''Program will execute a series of energy scans at varying undulator gap scans
    
    ./undulator_scan.py -e <element> -s <edge> <options>
    
    '''
    
    import optparse
    
    parser = optparse.OptionParser(usage=usage)
        
    parser.add_option('-d', '--directory', type=str, default='/tmp/undulator_scan', help='Directory to store the results (default=%default)')
    parser.add_option('-n', '--name_pattern', type=str, default='undulator_scan', help='name_pattern')
    parser.add_option('-g', '--gap', type=float, default=8.0, help='Undulator gap')
    parser.add_option('-s', '--start_energy', type=float, default=5.e3, help='Lower bound of the energy scan range in eV')
    parser.add_option('-e', '--end_energy', type=float, default=20.e3, help='Upper bound of the energy scan range in eV')
    parser.add_option('-c', '--scan_speed', type=float, default=0.025, help='Scan speed')
    parser.add_option('-o', '--optimize', action='store_true', help='Check current monochromator position and shuffle start, end to speed up the startup')
    parser.add_option('-D', '--display', action='store_true', help='display plot')
    parser.add_option('-E', '--extract', action='store_true', help='Extract the calibrated diode after the scan')
            
    options, args = parser.parse_args()
    
    print('options', options)
    print('args', args)
    
    us = undulator_scan(options.name_pattern,
                        options.directory,
                        gap=options.gap,
                        start_energy=options.start_energy,
                        end_energy=options.end_energy,
                        scan_speed=options.scan_speed,
                        optimize=options.optimize,
                        display=options.display,
                        extract=options.extract)
  
    us.execute()
    
if __name__ == '__main__':
    main()
    
    
    