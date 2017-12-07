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
import optparse
from motor import tango_motor

from analysis import slit_scan_analysis

class slit_scan(xray_experiment):
    
    slit_types = {1: 1, 2: 1, 3: 2, 5: 2, 6: 2}
    
    def __init__(self,
                 name_pattern,
                 directory,
                 slits=1,
                 start_position=2.0,
                 end_position=-2.0,
                 scan_speed=None,
                 default_speed=None,
                 darkcurrent_time=5.,
                 photon_energy=None,
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
                                 diagnostic=diagnostic,
                                 analysis=analysis,
                                 conclusion=conclusion,
                                 simulation=simulation)
    
        self.description = 'Slits %d scan between %6.1f and %6.1f mm, Proxima 2A, SOLEIL, %s' % (slits, start_position, end_position, time.ctime(self.timestamp))
        
        self.start_position = start_position
        self.end_position = end_position
        self.scan_speed = scan_speed
        self.default_speed = default_speed
        self.darkcurrent_time = darkcurrent_time
        
        self.diagnostic = diagnostic
        self.display = display
        self.extract = extract
        
        self.calibrated_diode = Si_PIN_diode()
        
        self.monitors_dictionary['calibrated_diode'] = self.calibrated_diode
        
        self.monitor_names += ['calibrated_diode']
        self.monitors += [self.calibrated_diode]
        
        self.slits = slits
        self.slit_type = self.slit_types[slits]
        self.alignment_slits = getattr(self, 'slits%d' % slits)
        

    def prepare(self):
        
        self.check_directory(self.directory)
        self.write_destination_namepattern(self.directory, self.name_pattern)
            
        initial_settings = []

        if self.slit_type == 1:
            self.alignment_slits.set_independent_mode()
                
        if self.simulation != True: 
            initial_settings.append(gevent.spawn(self.goniometer.set_transfer_phase, wait=True))
            initial_settings.append(gevent.spawn(self.set_photon_energy, self.photon_energy, wait=True))
            
        for k in [1, 2, 3, 5, 6]:
            initial_settings.append(gevent.spawn(getattr(getattr(self, 'slits%d' % k), 'set_horizontal_gap'), 4))
            initial_settings.append(gevent.spawn(getattr(getattr(self, 'slits%d' % k), 'set_vertical_gap'), 4))
            initial_settings.append(gevent.spawn(getattr(getattr(self, 'slits%d' % k), 'set_horizontal_position'), 0))
            initial_settings.append(gevent.spawn(getattr(getattr(self, 'slits%d' % k), 'set_vertical_position'), 0))
            
        initial_settings.append(gevent.spawn(self.calibrated_diode.insert))
        
        if self.safety_shutter.closed():
           initial_settings.append(gevent.spawn(self.safety_shutter.open))
        
        gevent.joinall(initial_settings)
        
        if self.scan_speed == None:
            self.scan_speed = self.alignment_slits.scan_speed
        if self.default_speed == None:
            self.default_speed = self.alignment_slits.default_speed
        
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
            
        print 'experiment execute took %s' % (time.time() - self.start_time  )
            
    def run(self):                    
        
        self.res = {}
        
        for k, actuator in enumerate(self.alignment_slits.get_alignment_actuators()):
            print 'k, actuator', k ,actuator
            
            self.actuator = actuator
            actuator.wait()
            actuator.set_position(self.start_position, timeout=None, wait=True)
            
            actuator.set_speed(self.scan_speed)
            
            if self.slit_type == 2:
                self.alignment_slits.set_pencil_scan_gap(k, scan_gap=0.1, wait=True)
                
            self.start_monitor()
            
            # sleep for darkcurrent_time while observation is already running
            gevent.sleep(self.darkcurrent_time)
            
            self.fast_shutter.open()
            
            move = gevent.spawn(actuator.set_position, self.end_position, timeout=None, wait=True)
            move.join()
            
            actuator.set_speed(self.default_speed)
            
            self.fast_shutter.close()
            
            gevent.sleep(self.darkcurrent_time)
                        
            self.stop_monitor()
            
            actuator.wait()
            
            
            if self.slit_type == 2:
                self.alignment_slits.set_pencil_scan_gap(k, scan_gap=4, wait=True)
                actuator.set_position(0.)
            elif self.slit_type == 1:
                actuator.set_position(self.start_position, wait=True)
            
            res = self.get_results()
            self.res[actuator.get_name()] = res
            
    def clean(self):
        self.save_parameters()
        self.save_results()
        self.save_log()
        #self.save_plot()
        
        final_settings = []
        
        for k in [1, 2, 3, 5, 6]:
            final_settings.append(gevent.spawn(getattr(getattr(self, 'slits%d' % k), 'set_horizontal_gap'), 4))
            final_settings.append(gevent.spawn(getattr(getattr(self, 'slits%d' % k), 'set_vertical_gap'), 4))
            final_settings.append(gevent.spawn(getattr(getattr(self, 'slits%d' % k), 'set_horizontal_position'), 0))
            final_settings.append(gevent.spawn(getattr(getattr(self, 'slits%d' % k), 'set_vertical_position'), 0))
            
        if self.extract:
            final_settings.append(gevent.spawn(self.calibrated_diode.extract))
        
        gevent.joinall(final_settings)
        
    def save_results(self):        
        print 'self.res'
        print self.res.keys()
        f = open(os.path.join(self.directory, '%s_results.pickle' % self.name_pattern), 'w')
        pickle.dump(self.res, f)
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
        self.parameters['slits'] = self.slits
        self.parameters['photon_energy'] = self.photon_energy
        self.parameters['darkcurrent_time'] = self.darkcurrent_time
        self.parameters['start_position'] = self.start_position
        self.parameters['end_position'] = self.end_position
        self.parameters['scan_speed'] = self.scan_speed
        self.parameters['default_speed'] = self.default_speed
        for actuator in self.alignment_slits.get_alignment_actuators():
            self.parameters['%s_offset' % actuator.get_name()] = actuator.device.offset
            
        self.parameters['start_time'] = self.start_time
        self.parameters['end_time'] = self.end_time
        self.parameters['duration'] = self.end_time - self.start_time
        self.parameters['description'] = self.description
        
        f = open(os.path.join(self.directory, '%s_parameters.pickle' % self.name_pattern), 'w')
        pickle.dump(self.parameters, f)
        f.close()

    def analyze(self):
        ssa = slit_scan_analysis(os.path.join(self.directory, '%s_parameters.pickle' % self.name_pattern))
        ssa.analyze()
        
    def conclude(self):
        ssa = slit_scan_analysis(os.path.join(self.directory, '%s_parameters.pickle' % self.name_pattern))
        ssa.conclude()
        
def main():
    
    usage = '''Program will execute a slit scan
    
    ./slit_scan.py <options>
    
    '''
    parser = optparse.OptionParser(usage=usage)
        
    parser.add_option('-d', '--directory', type=str, default='/tmp/slit_scan', help='Directory to store the results (default=%default)')
    parser.add_option('-n', '--name_pattern', type=str, default='slit_scan', help='name_pattern')
    parser.add_option('-s', '--slits', type=int, default=1, help='Slits')
    parser.add_option('-b', '--start_position', type=float, default=2., help='Start position')
    parser.add_option('-e', '--end_position', type=float, default=-2., help='End position')
    parser.add_option('-p', '--photon_energy', type=float, default=12650, help='Photon energy')
    parser.add_option('-D', '--display', action='store_true', help='display plot')
    parser.add_option('-E', '--extract', action='store_true', help='Extract the calibrated diode after the scan')
    parser.add_option('-A', '--analysis', action='store_true', help='Analyze the scan')
    parser.add_option('-C', '--conclude', action='store_true', help='Apply the offsets')
            
    options, args = parser.parse_args()
    
    print 'options', options
    print 'args', args
    
    filename = os.path.join(options.directory, options.name_pattern) + '_parameters.pickle'
    
    slscan = slit_scan(options.name_pattern,
                       options.directory,
                       slits=options.slits,
                       start_position=options.start_position,
                       end_position=options.end_position,
                       photon_energy=options.photon_energy,
                       display=options.display,
                       extract=options.extract)
                           
    if not os.path.isfile(filename):
        slscan.execute()
    if options.analysis == True:
        slscan.analyze()
    if options.conclude == True:
        slscan.conclude()
    
def analysis():
    parser = optparse.OptionParser()
        
    parser.add_option('-f', '--filename', type=str, default='/tmp/slit_scan_parameters.pickle', help='File storing parameters of the slit scan')
    
    options, args = parser.parse_args()
    
    print 'options', options
    print 'args', args
    
    ssa = slit_scan_analysis(options.filename)

    ssa.analyze()
    ssa.conclude()
    
if __name__ == '__main__':
    main()
    #analysis()
    
    
    
    