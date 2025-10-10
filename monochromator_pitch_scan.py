!#/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Monochromator pitch scan. Execute scan on monochromator pitch motor.
'''

import gevent
from gevent.monkey import patch_all
patch_all()

import traceback
import logging
import time
import os
import pickle
import numpy as np
import pylab
import sys

from xray_experiment import xray_experiment
from scipy.constants import eV, h, c, angstrom, kilo, degree
from monitor import Si_PIN_diode
from motor import monochromator_pitch_motor

#from analysis import fast_shutter_scan_analysis

class monochromator_pitch_scan(xray_experiment):
    
    specific_parameter_fields = [{'name': 'darkcurrent_time', 'type': 'float', 'description': 'Period of measuring the dark current'},
                                 {'name': 'default_position', 'type', 'float', 'description': 'default position'},
                                 {'name': 'start_position', 'type', 'float', 'description': 'Scan start position'},
                                 {'name': 'end_position', 'type', 'float', 'description': 'Scan end position'}]
          
    def __init__(self,
                 name_pattern,
                 directory,
                 start_position=-0.8,
                 end_position=0.,
                 default_position=-0.4400,
                 darkcurrent_time=5.,
                 photon_energy=None,
                 diagnostic=True,
                 analysis=None,
                 conclusion=None,
                 simulation=None,
                 display=False,
                 extract=False):
                 
        if hasattr(self, 'parameter_fields'):
            self.parameter_fields += monochromator_pitch_scan.specific_parameter_fields
        else:
            self.parameter_fields = monochromator_pitch_scan.specific_parameter_fields[:]
            
        self.default_experiment_name = f"Monochromator rocking curve. Scan between {start_position:.1f} and {end_position:.1f} mm"
        xray_experiment.__init__(self, 
                                 name_pattern, 
                                 directory,
                                 photon_energy=photon_energy,
                                 diagnostic=diagnostic,
                                 analysis=analysis,
                                 conclusion=conclusion,
                                 simulation=simulation)
      
        self.start_position = start_position
        self.end_position = end_position
        self.default_position = default_position
        self.darkcurrent_time = darkcurrent_time
        
        self.diagnostic = diagnostic
        self.display = display
        self.extract = extract
        
        self.calibrated_diode = Si_PIN_diode()
        
        self.monitors_dictionary['calibrated_diode'] = self.calibrated_diode
        
        self.monitor_names += ['calibrated_diode']
        self.monitors += [self.calibrated_diode]
        
        self.actuator = monochromator_pitch_motor()
    
    def prepare(self):
        self.check_directory(self.directory)
        self.write_destination_namepattern(self.directory, self.name_pattern)
        
        initial_settings = []
        
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
        
        self.actuator.wait()
        
        self.actuator.set_position(self.start_position, timeout=None, wait=True)
        
        
    def run(self):

        gevent.sleep(self.darkcurrent_time)
        self.fast_shutter.open()
        
        move = gevent.spawn(self.actuator.set_position, self.end_position, timeout=None, wait=True)
        move.join()
        
        self.fast_shutter.close()
        gevent.sleep(self.darkcurrent_time)
                    
    def clean(self):
        self.actuator.set_position(self.default_position)
        self.save_parameters()
        self.save_results()
        self.save_log()
        #self.save_plot()
        
        final_settings = []
        
        if self.extract:
            final_settings.append(gevent.spawn(self.calibrated_diode.extract))
        
        gevent.joinall(final_settings)
    
    def analyze(self):
        pass
    
    def conclude(self):
        pass
    
    
def main():
    usage = '''Program will execute a fast shutter scan
    
    ./monochromator_pitch_motor.py <options>
    
    '''
    import optparse
    parser = optparse.OptionParser(usage=usage)
        
    parser.add_option('-d', '--directory', type=str, default='/tmp/fast_shutter_scan', help='Directory to store the results (default=%default)')
    parser.add_option('-n', '--name_pattern', type=str, default='fast_shutter_scan', help='name_pattern')
    parser.add_option('-V', '--start_position', type=float, default=-0.8, help='start position')
    parser.add_option('-R', '--end_position', type=float, default=0., help='end position')
    parser.add_option('-p', '--photon_energy', type=float, default=12650, help='Photon energy')
    parser.add_option('-D', '--display', action='store_true', help='display plot')
    parser.add_option('-E', '--extract', action='store_true', help='Extract the calibrated diode after the scan')
    parser.add_option('-A', '--analysis', action='store_true', help='Analyze the scan')
    parser.add_option('-C', '--conclude', action='store_true', help='Apply the offsets')
            
    options, args = parser.parse_args()
    
    print('options', options)
    print('args', args)
    
    filename = os.path.join(options.directory, options.name_pattern) + '_parameters.pickle'
    
    #if os.path.isfile(filename):
        #ssa = slit_scan_analysis(filename)
        #if options.analysis == True:
            #ssa.analyze()
        #if options.conclude == True:
            #ssa.conclude()
    #else:
    mpscan = monochromator_pitch_scan(options.name_pattern,
                                      options.directory,
                                      start_position=options.start_position,
                                      end_position=options.end_position,
                                      photon_energy=options.photon_energy,
                                      display=options.display,
                                      extract=options.extract)
    mpscan.execute()
        
if __name__ == '__main__':
    main()
    
