#!/usr/bin/env python
# -*- coding: utf-8 -*-
import gevent
from gevent.monkey import patch_all
patch_all()

import time
import os
import numpy as np
import pickle
import traceback

from experiment import experiment
from detector import detector as detector
from goniometer import goniometer
from energy import energy as energy_motor
from motor import undulator, monochromator_rx_motor
from resolution import resolution as resolution_motor
from transmission import transmission as transmission_motor
from machine_status import machine_status

# from flux import flux
# from filters import filters
#from beam import beam

from beam_center import beam_center
from safety_shutter import safety_shutter
from fast_shutter import fast_shutter
from camera import camera
from monitor import xbpm
from slits import slits1, slits2, slits3, slits5, slits6

class xray_experiment(experiment):
    
    def __init__(self,
                 name_pattern, 
                 directory,
                 position=None,
                 photon_energy=None,
                 resolution=None,
                 detector_distance=None,
                 detector_vertical=None,
                 detector_horizontal=None,
                 transmission=None,
                 flux=None,
                 ntrigger=1,
                 snapshot=False,
                 zoom=None,
                 diagnostic=None,
                 analysis=None,
                 conclusion=None,
                 simulation=None):
        
        experiment.__init__(self, 
                            name_pattern=name_pattern, 
                            directory=directory,
                            diagnostic=diagnostic,
                            analysis=analysis,
                            conclusion=conclusion,
                            simulation=simulation)
        
        self.position = position
        self.photon_energy = photon_energy
        self.resolution = resolution
        self.detector_distance = detector_distance
        self.detector_vertical = detector_vertical
        self.detector_horizontal = detector_horizontal
        self.transmission = transmission
        self.flux = flux
        self.ntrigger = ntrigger
        self.snapshot = snapshot
        self.zoom = zoom
        
        # Necessary equipment
        self.goniometer = goniometer()
        try:
            self.beam_center = beam_center()
        except:
            from beam_center import beam_center_mockup
            self.beam_center = beam_center_mockup()
        try:
            self.detector = detector()
        except:
            from detector_mockup import detector_mockup
            self.detector = detector_mockup()
        try:
            self.energy_motor = energy_motor()
        except:
            from energy import energy_mockup
            self.energy_motor = energy_mockup()
        try:
            self.resolution_motor = resolution_motor()
        except:
            from resolution import resolution_mockup
            self.resolution_motor = resolution_mockup()
        try:
            self.transmission_motor = transmission_motor()
        except:
            from transmission import transmission_mockup
            self.transmission_motor = transmission_mockup()
        try:
            self.machine_status = machine_status()
        except:
            from machine_status import machine_status_mockup
            self.machine_status = machine_status_mockup()
        
        try:
            self.undulator = undulator()
        except:
            from undulator import undulator_mockup
            self.undulator = undulator_mockup()
        
        try:
            self.monochromator_rx_motor = monochromator_rx_motor()
        except:
            from motor import monochromator_rx_motor_mockup
            self.monochromator_rx_motor_mockup = monochromator_rx_motor_mockup()
            
        self.safety_shutter = safety_shutter()
        self.fast_shutter = fast_shutter()
        self.camera = camera()
        
        if self.photon_energy == None and self.simulation != True:
            self.photon_energy = self.energy_motor.get_energy()
        else:
            self.photon_energy = 12650.
        self.wavelength = self.resolution_motor.get_wavelength_from_energy(self.photon_energy)

        if self.resolution != None:
            self.detector_distance = self.resolution_motor.get_distance_from_resolution(self.resolution, wavelength=self.wavelength)
            if self.detector_distance == None:
                self.detector_distance = self.detector.position.get_ts()
            self.resolution = self.resolution_motor.get_resolution_from_distance(self.detector_distance, wavelength=self.wavelength)
        
        self.slits1 = slits1()
        self.slits2 = slits2()
        self.slits3 = slits3()
        self.slits5 = slits5()
        self.slits6 = slits6()
        
        self.xbpm1 = xbpm('i11-ma-c04/dt/xbpm_diode.1-base')
        self.cvd1 = xbpm('i11-ma-c05/dt/xbpm-cvd.1-base')
        self.xbpm5 = xbpm('i11-ma-c06/dt/xbpm_diode.5-base')
        self.psd5 = xbpm('i11-ma-c06/dt/xbpm_diode.psd.5-base')
        self.psd6 = xbpm('i11-ma-c06/dt/xbpm_diode.6-base')
        
        self.monitor_names = ['xbpm1', 
                              'cvd1', 
                              #'xbpm5', 
                              'psd5', 
                              'psd6', 
                              'machine_status', 
                              'fast_shutter']
                              
        self.monitors = [self.xbpm1, 
                         self.cvd1, 
                         #self.xbpm5, 
                         self.psd5, 
                         self.psd6, 
                         self.machine_status, 
                         self.fast_shutter]
                         
        self.monitors_dictionary = {'xbpm1': self.xbpm1,
                                    'cvd1': self.cvd1,
                                    #'xbpm5': self.xbpm5,
                                    'psd5': self.psd5,
                                    'psd6': self.psd6,
                                    'machine_status': self.machine_status,
                                    'fast_shutter': self.fast_shutter}
                                    
        self.monitor_sleep_time = 0.05
    
    def get_duration(self):
        return time.time() - self._start    
    
    def get_progression(self):
        def changes(a):
            '''Helper function to remove consecutive indices. 
            -- trying to get out of np.gradient what skimage.match_template would return'''
            if len(a) == 0:
                return
            elif len(a) == 1:
                return a
            indices = [True]
            for k in range(1, len(a)):
                if a[k]-1 == a[k-1]:
                    indices.append(False)
                else:
                    indices.append(True)
            return a[np.array(indices)]

        def get_on_segments(on, off):
            if len(on) == 0:
                return
            elif off == None:
                segments = [(on[-1], -1)]
            elif len(on) == len(off):
                segments = zip(on, off)
            elif len(on) > len(off):
                segments = zip(on[:-1], off)
                segments.append((on[-1], -1))
            return segments
        
        def get_complete_incomplete_wedges(segments):
            nsegments = len(segments)
            if segments[-1][-1] == -1:
                complete = nsegments - 1
                incomplete = 1
            else:
                complete = nsegments
                incomplete = 0
            return complete, incomplete
                
        observations = np.array(self.fast_shutter.get_observations())

        if observations == []:
            return 0
        try:
            g = np.gradient(observations[:, 1])
            ons = changes(np.where(g == 0.5)[0])
            offs = changes(np.where(g == -0.5)[0])
            if ons == None:
                return 0
        except:
            print 'observations'
            print observations
            print traceback.print_exc()
            return 0
            
        segments = get_on_segments(ons, offs)
        print 'segments'
        print segments
        if segments == None:
            return 0
        
        chronos = observations[:, 0]
        total_exposure_time = 0
        
        for segment in segments:
            total_exposure_time += chronos[segment[1]] - chronos[segment[0]]
        
        progression = total_exposure_time/self.total_expected_exposure_time
        
        complete, incomplete = get_complete_incomplete_wedges(segments)
        
        if progression > 1:
            progression = 1
        if complete == self.total_expected_wedges:
            progression = 2            
        return progression
    
    def get_point(self, start_time):
        chronos = time.time() - start_time
        position = self.goniometer.get_position()
        point = [chronos] +  [position[motor_name] for motor_name in self.actuator_names] + [self.get_progression()]
        return point
            
    def start_monitor(self):
        print 'start_monitor'
        self.observe = True
        self.observers = [gevent.spawn(self.actuator.monitor, self.start_time)]
        for monitor in self.monitors:
            monitor.observe = True
            self.observers.append(gevent.spawn(monitor.monitor, self.start_time))
        
    def actuator_monitor(self, start_time):
        print 'this is actuator_monitor'
        self.observations = []
        self.observations_fields = ['chronos'] + self.actuator_names
        
        while self.observe == True:
            point = self.get_point(start_time)
            print 'actuator monitor: point', point
            self.observations.append(point)
            gevent.sleep(self.monitor_sleep_time)
   
    def get_observations(self):
        return self.observations
    
    def get_observations_fields(self):
        return self.observations_fields
       
    def get_points(self):
        return np.array(self.observations)[:,1]
        
    def get_chronos(self):
        return np.array(self.observations)[:,0]        
    
    def stop_monitor(self):
        print 'stop_monitor'
        self.observe = False
        for monitor in self.monitors:
            monitor.observe = False
        gevent.joinall(self.observers)
        
    def set_photon_energy(self, photon_energy=None, wait=False):
        _start = time.time()
        if photon_energy > 1000: #if true then it was specified in eV not in keV
            photon_energy *= 1e-3
        if photon_energy is not None:
            self.energy_moved = self.energy_motor.set_energy(photon_energy, wait=wait)
        else:
            self.energy_moved = 0
        print 'set_photon_energy took %s' % (time.time() - _start)
        
    def get_photon_energy(self):
        return self.energy_motor.get_energy()

    def set_transmission(self, transmission=None):
        if transmission is not None:
            self.transmission = transmission
            self.transmission_motor.set_transmission(transmission)
            
    def get_transmission(self):
        return self.transmission_motor.get_transmission()

    def program_detector(self):
        _start = time.time()
        pass
        print 'program_detector took %s' % (time.time()-_start)
    
    def program_goniometer(self):
        self.goniometer.set_scan_number_of_frames(1)
        self.goniometer.set_detector_gate_pulse_enabled(True)
        
    def prepare(self):
        pass
        
    def collect(self):
        return self.run()
    def measure(self):
        return self.run()
    def acquire(self):
        return self.run()
    
    def get_observations(self):
        all_observations = {}
        all_observations['actuator_monitor'] = {}
        actuator_observations_fields = self.get_observations_fields()
        all_observations['actuator_monitor']['observations_fields'] = actuator_observations_fields
        actuator_observations = self.get_observations()
        all_observations['actuator_monitor']['observations'] = actuator_observations
        
        for monitor_name, mon in zip(self.monitor_names, self.monitors):
            all_observations[monitor_name] = {}
            all_observations[monitor_name]['observations_fields'] = mon.get_observations_fields()
            all_observations[monitor_name]['observations'] = mon.get_observations()
        
        return all_observations
        
    def get_all_observations(self):
        return self.get_observations()
    
    def save_diagnostic(self):
        f = open(os.path.join(self.directory, '%s_observations.pickle' % self.name_pattern), 'w')
        pickle.dump(self.get_all_observations(), f)
        f.close()
        
    def finalize(self):
        self.clean()
        
    def clean(self):
        self.detector.disarm()
        self.save_parameters()
        self.save_results()
        self.save_log()
        if self.diagnostic == True:
            self.save_diagnostic()
    
    def stop(self):
        self.goniometer.abort()
        self.detector.abort()