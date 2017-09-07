#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Slits scan. Execute scan on a pair of slits.

'''

import gevent
from gevent.monkey import patch_all
patch_all()

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

class slit_scan(xray_experiment):
    
    slit_types = {1: 1, 2: 1, 3: 2, 5: 2, 6: 2}
    
    def __init__(self,
                 name_pattern,
                 directory,
                 slits=1,
                 start=2.0,
                 end=-2.0,
                 scan_speed=0.1,
                 default_speed=0.5,
                 darkcurrent_time=5.,
                 photon_energy=None,
                 optimize=True,
                 transmission=None,
                 diagnostic=True,
                 analysis=None,
                 conclusion=None,
                 simulation=None,
                 display=False,
                 extract=False):
                 
        xray_experiment.__init__(self, 
                                 name_pattern, 
                                 directory,
                                 photon_energy=photon_energy,
                                 transmission=transmission,
                                 diagnostic=diagnostic,
                                 analysis=analysis,
                                 conclusion=conclusion,
                                 simulation=simulation)
    
        self.description = 'Slits %d scan between %6.1f and %6.1f mm, Proxima 2A, SOLEIL, %s' % (slits, start, end, time.ctime(self.timestamp))
        
        self.start = start
        self.end = end
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
        
        self.slit_type = self.slit_types[slits]
        self.slits = getattr(self, 'slits%d' % slits)
        
            
    def set_efficient_start_end(self, motor):
        current_position = self.motor.get_position()
        if abs(self.start - current_position) > abs(self.end - current_position):
            self.start, self.end = self.end, self.start
   
    def prepare(self):
        
        self.check_directory(self.directory)
        self.write_destination_namepattern(self.directory, self.name_pattern)
        
        if self.optimize:
            self.set_efficient_start_end()
            
        self.actuator.set_speed(self.default_speed)
        
        initial_settings = []
        
        for k in [1, 2, 3, 5, 6]:
            initial_settings.append(gevent.spawn(getattr(getattr(self, 'slits%d' % k), 'set_horizontal_gap'), 4))
            initial_settings.append(gevent.spawn(getattr(getattr(self, 'slits%d' % k), 'set_vertical_gap'), 4))
            initial_settings.append(gevent.spawn(getattr(getattr(self, 'slits%d' % k), 'set_horizontal_position'), 0))
            initial_settings.append(gevent.spawn(getattr(getattr(self, 'slits%d' % k), 'set_vertical_position'), 0))
            
        initial_settings.append(gevent.spawn(self.calibrated_diode.insert))
        initial_settings.append(gevent.spawn(self.undulator.set_position, self.gap, wait=True))
        initial_settings.append(gevent.spawn(self.actuator.set_position, self.start, wait=True))
        initial_settings.append(gevent.spawn(self.goniometer.set_transfer_phase, wait=True))
        if self.safety_shutter.closed():
           initial_settings.append(gevent.spawn(self.safety_shutter.open))
        
        gevent.joinall(initial_settings)
        
        self.actuator.set_speed(self.scan_speed)
        
    def execute(self):
        self.start_time = time.time()
        try:
            self.prepare()
            self.run()
        except:
            print 'Problem in preparation or execution %s' % self.__module__
            print traceback.print_exc()
        finally:
            self.end_time = time.time()
            self.clean()
        if self.analysis == True:
            self.analyze()
        if self.conclusion == True:
            self.conclude()
            
        print 'experiment execute took %s' % (time.time() - self.start_time  
    
    def run(self):                    

        for actuator in self.slits.alignment_actuators:
            self.start_monitor()
            
            # sleep for darkcurrent_time while observation is already running
            gevent.sleep(self.darkcurrent_time)
            
            self.fast_shutter.open()
            move = gevent.spawn(self.actuator.set_position, self.end, timeout=None, wait=True)
            move.join()
            self.fast_shutter.close()
            
            gevent.sleep(self.darkcurrent_time)
            
            self.actuator.observe = False
            
            self.stop_monitor()
        
    def clean(self):
        self.save_parameters()
        self.save_results()
        self.save_plot()
        self.actuator.set_speed(self.default_speed)
        
        final_settings = []
        
        for k in [1, 2, 3, 5, 6]:
            final_settings.append(gevent.spawn(getattr(getattr(self, 'slits%d' % k), 'set_horizontal_gap'), 4))
            final_settings.append(gevent.spawn(getattr(getattr(self, 'slits%d' % k), 'set_vertical_gap'), 4))
            final_settings.append(gevent.spawn(getattr(getattr(self, 'slits%d' % k), 'set_horizontal_position'), 0))
            final_settings.append(gevent.spawn(getattr(getattr(self, 'slits%d' % k), 'set_vertical_position'), 0))
            
        final_settings.append(gevent.spawn(self.calibrated_diode.insert))
        final_settings.append(gevent.spawn(self.undulator.set_position, self.gap, wait=True))
        final_settings.append(gevent.spawn(self.actuator.set_position, self.start, wait=True))
        final_settings.append(gevent.spawn(self.goniometer.set_transfer_phase, wait=True))
        if self.extract:
            final_settings.append(gevent.spawn(self.calibrated_diode.extract))
        
        gevent.joinall(final_settings)
        
        
    def get_results(self):
        results = {}
        
        results['actuator'] = {'observation_fields': self.actuator.get_observation_fields(),
                               'observations': self.actuator.get_observations()}
        
        for (monitor_name, monitor) in zip(self.monitor_names, self.monitors):
            results[monitor_name] = {'observation_fields': monitor.get_observation_fields(),
                                     'observations': monitor.get_observations()}
        
        return results
        
    def save_results(self):
        self.results = self.get_results()
        
        f = open(os.path.join(self.directory, '%s_results.pickle' % self.name_pattern), 'w')
        pickle.dump(self.results, f)
        f.close()
    
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
            
    def save_parameters(self):
        self.parameters = {}
        
        self.parameters['timestamp'] = self.timestamp
        self.parameters['name_pattern'] = self.name_pattern
        self.parameters['directory'] = self.directory
        self.parameters['gap'] = self.gap
        self.parameters['start_energy'] = self.start_energy
        self.parameters['end_energy'] = self.end_energy
        self.parameters['scan_speed'] = self.scan_speed
        
        self.parameters['start_wavelength'] = self.get_wavelength_from_energy(self.start_energy, units_energy=eV, units_wavelength=angstrom)
        self.parameters['end_wavelength'] = self.get_wavelength_from_energy(self.end_energy, units_energy=eV, units_wavelength=angstrom)
        self.parameters['start_theta'] = self.get_theta_from_energy(self.start_energy, units_energy=eV, units_wavelength=angstrom, units_theta=degree)
        self.parameters['end_theta'] = self.get_theta_from_energy(self.end_energy, units_energy=eV, units_wavelength=angstrom, units_theta=degree)
        self.parameters['start_time'] = self.start_time
        self.parameters['end_time'] = self.end_time
        self.parameters['duration'] = self.end_time - self.start_time
        
        for k in [1, 2, 3, 5, 6]:
            self.parameters['slits%d_horizontal_gap' % k] = getattr(getattr(self, 'slits%d' % k), 'get_horizontal_gap')()
            self.parameters['slits%d_vertical_gap' % k] = getattr(getattr(self, 'slits%d' % k), 'get_vertical_gap')()
            self.parameters['slits%d_horizontal_position' % k] = getattr(getattr(self, 'slits%d' % k), 'get_horizontal_position')()
            self.parameters['slits%d_vertical_position' % k] = getattr(getattr(self, 'slits%d' % k), 'get_vertical_position')()
        
        self.parameters['undulator_gap'] = self.undulator.get_position()
        self.parameters['undulator_gap_encoder_position'] = self.undulator.get_encoder_position()
        
        f = open(os.path.join(self.directory, '%s_parameters.pickle' % self.name_pattern), 'w')
        pickle.dump(self.parameters, f)
        f.close()
        
def main():
    
    usage = '''Program will execute a series of energy scans at varying undulator gap scans
    
    ./undulator_scan.py -e <element> -s <edge> <options>
    
    '''
    
    import optparse
    
    parser = optparse.OptionParser(usage=usage)
        
    parser.add_option('-d', '--directory', type=str, default='/tmp/undulato_scan', help='Directory to store the results (default=%default)')
    parser.add_option('-n', '--name_pattern', type=str, default='undulator_scan', help='name_pattern')
    parser.add_option('-g', '--gap', type=float, default=8.0, help='Undulator gap')
    parser.add_option('-s', '--start_energy', type=float, default=5.e3, help='Lower bound of the energy scan range in eV')
    parser.add_option('-e', '--end_energy', type=float, default=20.e3, help='Upper bound of the energy scan range in eV')
    parser.add_option('-c', '--scan_speed', type=float, default=0.025, help='Scan speed')
    parser.add_option('-o', '--optimize', action='store_true', help='Check current monochromator position and shuffle start, end to speed up the startup')
    parser.add_option('-D', '--display', action='store_true', help='display plot')
    parser.add_option('-E', '--extract', action='store_true', help='Extract the calibrated diode after the scan')
            
    options, args = parser.parse_args()
    
    print 'options', options
    print 'args', args
    
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
    
    
    