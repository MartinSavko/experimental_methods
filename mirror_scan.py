#!/usr/bin/env python2
# -*- coding: utf-8 -*-
'''
Slits scan. Execute scan on a pair of slits.

'''
import gevent

import PyTango

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

from motor import tango_motor
from monitor import xray_camera, analyzer
from slit_scan import slit_scan

from analysis import slit_scan_analysis

class mirror_scan_analysis(slit_scan_analysis):
    
    def analyze(self):
        parameters = self.get_parameters()
        results = self.get_results()
        for lame_name in results.keys():
            actuator_chronos, actuator_position = self.get_observations(results[lame_name], 'actuator')
            fast_shutter_chronos, fast_shutter_state = self.get_observations(results[lame_name], 'fast_shutter')
            monitor_chronos, monitor_points = self.get_observations(results[lame_name], self.monitor)
            if 'tz' in lame_name:
                monitor_points = np.array([item[1] for item in monitor_points])
            else:
                monitor_points = np.array([item[0] for item in monitor_points])
            
            start_chronos, end_chronos = self.get_fast_shutter_open_close_times(fast_shutter_chronos, fast_shutter_state)
            
            dark_current_indices = np.logical_or(monitor_chronos < start_chronos - self.fast_shutter_chronos_uncertainty, monitor_chronos > end_chronos + self.fast_shutter_chronos_uncertainty)
            #dark_current = monitor_points[dark_current_indices].mean()
            #monitor_points -= dark_current
            
            actuator_scan_indices = self.get_scan_indices(actuator_chronos, start_chronos, end_chronos, self.fast_shutter_chronos_uncertainty)
            actuator_scan_chronos = actuator_chronos[actuator_scan_indices]
            actuator_scan_position = actuator_position[actuator_scan_indices]
            
            position_chronos_predictor = self.get_position_chronos_predictor(actuator_scan_chronos, actuator_scan_position)
            
            monitor_scan_indices = self.get_scan_indices(monitor_chronos, start_chronos, end_chronos, self.fast_shutter_chronos_uncertainty)
            monitor_scan_chronos = monitor_chronos[monitor_scan_indices]
            self.monitor_scan_points = monitor_points[monitor_scan_indices]
            
            self.monitor_scan_positions = position_chronos_predictor(monitor_scan_chronos)
            
            results[lame_name]['analysis'] = {}
            
            results[lame_name]['analysis']['actuator_position'] = self.monitor_scan_positions
            
            results[lame_name]['analysis']['beam_position'] = self.monitor_scan_points
            
        self.save_results(results)
            
class adaptive_mirror(object):
    
    mirror_address = {'vfm': 'i11-ma-c05/op/mir2-vfm', 'hfm': 'i11-ma-c05/op/mir3-hfm'}

    def __init__(self, mirror='vfm', check_time=1.):
        
        self.mirror = mirror
        self.check_time = check_time
        self.mirror_device = PyTango.DeviceProxy(self.mirror_address[self.mirror])
        channel_base = self.mirror_address[mirror].replace(mirror, 'ch.%02d')
        self.channels = [PyTango.DeviceProxy(channel_base % k) for k in range(12)]
        self.pitch = tango_motor(self.mirror_address[mirror].replace('mir2-vfm', 'mir.2-mt_rx').replace('mir3-hfm', 'mir.3-mt_rz'))
        self.translation = tango_motor(self.mirror_address[mirror].replace('mir2-vfm', 'mir.2-mt_tz').replace('mir3-hfm', 'mir.3-mt_tx'))
        
    def get_channel_values(self):
        return [getattr(c, 'voltage') for c in self.channels]
    
    def get_channel_target_values(self):
        return [getattr(c, 'targetVoltage') for c in self.channels]
    
    def set_channel_target_values(self, channel_values, number_of_channels=12):
        if len(channel_values) == 0:
            print('not modifying target values as none specified')
        elif len(channel_values) != number_of_channels:
            print('not modifying target values as the value vector length is not consistent with number of channels (%d)' % number_of_channels)
        else:
            for k, c in enumerate(channel_values):
                if c not in [None, np.nan]:
                    setattr(self.channels[k], 'targetVoltage', c)
                    gevent.sleep(1)
                else:
                    print('not modifying channel %d, current value %.1f' % (k, getattr(c, 'voltage')))
        print('current target voltages: %s' % self.get_channel_target_values())
        
    def set_voltages(self, channel_values):
        self.set_channel_target_values(channel_values)
        self.mirror_device.SetChannelsTargetVoltage()
        self.wait()
        
    def wait(self):
        print 'Setting %s mirror voltages' % self.mirror
        print 'Please wait for %s mirror tensions to settle ...' % self.mirror
        gevent.sleep(10*self.check_time)
        while self.mirror_device.State().name != 'STANDBY':
            gevent.sleep(self.check_time)
            print 'wait ',
        print
        print 'done!'
        print '%s mirror tensions converged' % self.mirror
        
    def get_pitch_position(self):
        return self.pitch.get_position()
    
    def get_translation_position(self):
        return self.translation.get_position()
    
    def set_pitch_position(self, position):
        self.pitch.set_position(position)
        
    def set_translation_position(self, position):
        self.translation.set_position(position)
        
                 
class mirror_scan(slit_scan):
    
    mirrors = {'vfm': 'i11-ma-c05/op/mir2-vfm', 'hfm': 'i11-ma-c05/op/mir3-hfm'}
    
    specific_parameter_fields = [{'name': 'mirror_name', 'type': 'str', 'description': 'Target mirror'},
                                 {'name': 'channel_values', 'type': 'list', 'description': 'Mirror tensions'},
                                 {'name': 'mirror_pitch', 'type': 'float', 'description': 'Mirror pitch'},
                                 {'name': 'mirror_translation', 'type': 'float', 'description': 'Mirror translation'}]
                                
    def __init__(self,
                 name_pattern,
                 directory,
                 mirror_name='vfm',
                 channel_values=[],
                 start_position=1.0,
                 end_position=-1.0,
                 scan_gap=0.035,
                 scan_speed=None,
                 darkcurrent_time=5.,
                 photon_energy=None,
                 diagnostic=True,
                 analysis=None,
                 conclusion=None,
                 simulation=None,
                 display=False,
                 extract=False):
        
        if hasattr(self, 'parameter_fields'):
            self.parameter_fields += mirror_scan.specific_parameter_fields
        else:
            self.parameter_fields = mirror_scan.specific_parameter_fields[:]
            
        slit_scan.__init__(self,
                           name_pattern,
                           directory,
                           slits=3,
                           start_position=start_position,
                           end_position=end_position,
                           scan_speed=scan_speed,
                           scan_gap=scan_gap,
                           darkcurrent_time=darkcurrent_time,
                           photon_energy=photon_energy,
                           diagnostic=diagnostic,
                           analysis=analysis,
                           conclusion=conclusion,
                           simulation=simulation,
                           display=display,
                           extract=extract)
        
        self.description = 'Scan of %s mirror scan scan between %6.1f and %6.1f mm, Proxima 2A, SOLEIL, %s' % (mirror_name, start_position, end_position, time.ctime(self.timestamp))
        
        self.mirror_name = mirror_name
        self.mirror = adaptive_mirror(self.mirror_name)


    def set_up_monitor(self):
        self.monitor_device = xray_camera()
        
        #self.monitors_dictionary['xray_camera'] = self.monitor_device
        #self.monitor_names += ['xray_camera']
        #self.monitors += [self.monitor_device]        

        self.auxiliary_monitor_device = analyzer()
        self.monitors_dictionary['analyzer'] = self.auxiliary_monitor_device
        self.monitor_names += ['analyzer']
        self.monitors += [self.auxiliary_monitor_device]        
 
    
    def get_clean_slits(self):
        return [1, 2, 5, 6]
    
    def get_alignment_actuators(self):
        alignment_actuators = self.alignment_slits.get_alignment_actuators()
        print('alignment_actuators', alignment_actuators)
        if self.mirror_name == 'vfm':
            actuators = alignment_actuators[1]
        else:
            actuators = alignment_actuators[0]
        print('actuators', actuators)
        return actuators
            
    def get_channel_values(self):
        return self.mirror.get_channel_values()
    
    def get_mirror_pitch(self):
        return self.mirror.get_pitch_position()
    
    def get_mirror_translation(self):
        return self.mirror.get_translation_position()
    
    def run(self):                    
        
        self.res = {}
        
        actuator = self.get_alignment_actuators()
        k = 1 if self.mirror_name == 'vfm' else 0
        
        print 'actuator', actuator
        
        self.actuator = actuator
        #self.actuator_names = [self.actuator.get_name()]
        actuator.wait()
        actuator.set_position(self.start_position, timeout=None, wait=True)
        
        actuator.set_speed(self.scan_speed)
        
        if self.slit_type == 2:
            self.alignment_slits.set_pencil_scan_gap(k, scan_gap=self.get_scan_gap(), wait=True)
            
        self.start_monitor()
        
        print 'sleep for darkcurrent_time while observation is already running'
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
            self.alignment_slits.set_pencil_scan_gap(k, scan_gap=self.default_gap, wait=True)
            actuator.set_position(0.)
        elif self.slit_type == 1:
            actuator.set_position(self.start_position, wait=True)
        
        res = self.get_results()
        self.res[actuator.get_name()] = res
            
    def analyze(self):
        a = mirror_scan_analysis(os.path.join(self.directory, '%s_parameters.pickle' % self.name_pattern), monitor='analyzer')
        a.analyze()
    
    def conclude(self):
        pass
    
def main():
    
    import optparse
    
    usage = '''Program will execute a slit scan
    
    ./mirror_scan.py <options>
    
    '''
    parser = optparse.OptionParser(usage=usage)
        
    parser.add_option('-d', '--directory', type=str, default='/tmp/slit_scan', help='Directory to store the results (default=%default)')
    parser.add_option('-n', '--name_pattern', type=str, default='slit_scan', help='name_pattern')
    parser.add_option('-m', '--mirror_name', type=str, default='vfm', help='mirror_name')
    parser.add_option('-b', '--start_position', type=float, default=1., help='Start position')
    parser.add_option('-e', '--end_position', type=float, default=-1., help='End position')
    parser.add_option('-p', '--photon_energy', type=float, default=12650, help='Photon energy')
    parser.add_option('-D', '--display', action='store_true', help='display plot')
    parser.add_option('-E', '--extract', action='store_true', help='Extract the calibrated diode after the scan')
    parser.add_option('-A', '--analysis', action='store_true', help='Analyze the scan')
    parser.add_option('-C', '--conclusion', action='store_true', help='Apply the offsets')
            
    options, args = parser.parse_args()
    
    print 'options', options
    print 'args', args
    
    filename = os.path.join(options.directory, options.name_pattern) + '_parameters.pickle'
    
    mscan = mirror_scan(**vars(options))
                           
    if not os.path.isfile(filename):
        mscan.execute()
    if options.analysis == True:
        mscan.analyze()
    if options.conclusion == True:
        mscan.conclude()

def scan_mirror():
    vfm = adaptive_mirror('vfm')
    
    base_voltages = [255.0, 215.0, 12.0, 170.0, 185.0, 443.0, 322.0, 188.0, 40.0, -47.0, -3.0, 88.0] # vfm.get_channel_values()
    
    directory = '/nfs/data3/Martin/Commissioning/mirrors/2020-06-15/'
    
    start_stop = [1, -1]
    for increment in [+20, -20, -40, +40, +60, -60, -80, +80, +100, -100]:
        for k in range(12):
            print('channel %02d, increment %d' % (k, increment))
            #new_voltages = base_voltages[:]
            #new_voltages[k] += increment
            #vfm.set_voltages(new_voltages)
            
            mscan1 = mirror_scan('channel_%02d_increment_%d_a' % (k, increment), directory, mirror_name='vfm', start_position=start_stop[0], end_position=start_stop[1])
            mscan1.analyze()
            start_stop = start_stop[::-1]
            mscan2 = mirror_scan('channel_%02d_increment_%d_b' % (k, increment), directory, mirror_name='vfm', start_position=start_stop[0], end_position=start_stop[1])
            mscan2.analyze()
            
    vfm.set_voltages(base_voltages)
    
if __name__ == '__main__':
    #main()
    scan_mirror()