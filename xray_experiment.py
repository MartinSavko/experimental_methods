#!/usr/bin/env python
# -*- coding: utf-8 -*-
import gevent

import time
import os
import numpy as np
import pickle
import traceback
import logging

from experiment import experiment
from detector import detector as detector
from goniometer import goniometer
from energy import energy as energy_motor
from motor import undulator, monochromator_rx_motor
from resolution import resolution as resolution_motor
from transmission import transmission as transmission_motor
from machine_status import machine_status

from flux import flux as flux_monitor
# from filters import filters
#from beam import beam

from beam_center import beam_center
from safety_shutter import safety_shutter
from fast_shutter import fast_shutter
from camera import camera
from monitor import xbpm, xbpm_mockup, eiger_en_out, fast_shutter_close, fast_shutter_open, trigger_eiger_on, trigger_eiger_off, Si_PIN_diode
from slits import slits1, slits2, slits3, slits5, slits6, slits_mockup

class xray_experiment(experiment):
    
    specific_parameter_fields = [{'name': 'photon_energy', 'type': 'float', 'description': 'photon energy of the experiment in eV'},
                                 {'name': 'wavelength', 'type': 'float', 'description': 'experiment photon wavelength in A'},
                                 {'name': 'transmission_intention', 'type': 'float', 'description': 'intended photon beam transmission in %'},
                                 {'name': 'transmission', 'type': 'float', 'description': 'measured photon beam transmission in %'},
                                 {'name': 'flux_intention', 'type': 'float', 'description': 'intended flux of the experiment in ph/s'},
                                 {'name': 'flux', 'type': 'float', 'description': 'intended flux of the experiment in ph/s'},
                                 {'name': 'slit_configuration', 'type': 'dict', 'description': 'slit configuration'},
                                 {'name': 'undulator_gap', 'type': 'float', 'description': 'experiment undulator gap in mm'},
                                 {'name': 'monitor_sleep_time', 'type': 'float', 'description': 'default pause between monitor measurements in s'}]
    
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
                 simulation=None,
                 monitor_sleep_time=0.05,
                 parent=None,
                 mxcube_parent_id=None,
                 mxcube_gparent_id=None):
        
        logging.debug('xray_experiment __init__ len(xray_experiment.specific_parameter_fields) %d' % len(xray_experiment.specific_parameter_fields))
        
        if hasattr(self, 'parameter_fields'):
            self.parameter_fields += xray_experiment.specific_parameter_fields
        else:
            self.parameter_fields = xray_experiment.specific_parameter_fields[:]
        
        logging.debug('xray_experiment __init__ len(self.parameters_fields) %d' % len(self.parameter_fields))
        
        experiment.__init__(self, 
                            name_pattern=name_pattern, 
                            directory=directory,
                            diagnostic=diagnostic,
                            analysis=analysis,
                            conclusion=conclusion,
                            simulation=simulation,
                            snapshot=snapshot,
                            mxcube_parent_id=mxcube_parent_id,
                            mxcube_gparent_id=mxcube_gparent_id)
        
        self.description = 'X-ray experiment, Proxima 2A, SOLEIL, %s' % time.ctime(self.timestamp)
        self.position = position
        self.photon_energy = photon_energy
        self.resolution = resolution
        self.detector_distance = detector_distance
        self.detector_vertical = detector_vertical
        self.detector_horizontal = detector_horizontal
        self.transmission = transmission # hack for single bunch transmission
        self.flux = flux
        self.ntrigger = ntrigger
        self.snapshot = snapshot
        self.zoom = zoom
        self.monitor_sleep_time = monitor_sleep_time
        self.parent = parent
        
        # Necessary equipment
        self.goniometer = goniometer()
        try:
            self.flux_monitor = flux_monitor()
        except:
            self.flux_monitor = None
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
            from motor import undulator_mockup
            self.undulator = undulator_mockup()
        
        try:
            self.monochromator_rx_motor = monochromator_rx_motor()
        except:
            from motor import monochromator_rx_motor_mockup
            self.monochromator_rx_motor_mockup = monochromator_rx_motor_mockup()
            
        self.safety_shutter = safety_shutter()
        
        try:
            self.fast_shutter = fast_shutter()
        except:
            self.fast_shutter = fast_shutter_mockup()
            
        try:
            self.camera = camera()
        except:
            self.camera = None
        
        if self.photon_energy == None and self.simulation != True:
            self.photon_energy = self.get_current_photon_energy()

        self.wavelength = self.resolution_motor.get_wavelength_from_energy(self.photon_energy)

        try:
            self.slits1 = slits1()
        except:
            self.slits1 = slits_mockup(1)
            
        try:
            self.slits2 = slits2()
        except:
            self.slits2 = slits_mockup(2)
            
        try:
            self.slits3 = slits3()
        except:
            self.slits3 = slits_mockup(3)
            
        try:
            self.slits5 = slits5()
        except:
            self.slits5 = slits_mockup(5)
            
        try:
            self.slits6 = slits6()
        except:
            self.slits6 = slits_mockup(6)

        try:
            self.xbpm1 = xbpm('i11-ma-c04/dt/xbpm_diode.1-base')
        except:
            self.xbpm1 = xbpm_mockup('i11-ma-c04/dt/xbpm_diode.1-base')
        try:
            self.cvd1 = xbpm('i11-ma-c05/dt/xbpm-cvd.1-base')
        except:
            self.cvd1 = xbpm_mockup('i11-ma-c05/dt/xbpm-cvd.1-base')
        try:
            self.xbpm5 = xbpm('i11-ma-c06/dt/xbpm_diode.5-base')
        except:
            self.xbpm5 = xbpm_mockup('i11-ma-c06/dt/xbpm_diode.5-base')
        try:
            self.psd5 = xbpm('i11-ma-c06/dt/xbpm_diode.psd.5-base')
        except:
            self.psd5 = xbpm_mockup('i11-ma-c06/dt/xbpm_diode.psd.5-base')
        try:
            self.psd6 = xbpm('i11-ma-c06/dt/xbpm_diode.6-base')
        except:
            self.psd6 = xbpm_mockup('i11-ma-c06/dt/xbpm_diode.6-base')
        
        self.eiger_en_out = eiger_en_out()
        self.trigger_eiger_on = trigger_eiger_on()
        self.trigger_eiger_off = trigger_eiger_off()
        self.fast_shutter_open = fast_shutter_open()
        self.fast_shutter_close = fast_shutter_close()
        self.Si_PIN_diode = Si_PIN_diode()
        
        self.monitor_names = ['xbpm1', 
                              'cvd1', 
                              #'xbpm5', 
                              'psd5', 
                              'psd6', 
                              'machine_status', 
                              'fast_shutter',
                              #'eiger_en_out',
                              #'trigger_eiger_on',
                              #'trigger_eiger_off',
                              #'fast_shutter_open',
                              #'fast_shutter_close',
                              'self']
                              
        self.monitors = [self.xbpm1, 
                         self.cvd1, 
                         #self.xbpm5, 
                         self.psd5, 
                         self.psd6, 
                         self.machine_status, 
                         self.fast_shutter,
                         #self.eiger_en_out,
                         #self.trigger_eiger_on,
                         #self.trigger_eiger_off,
                         #self.fast_shutter_open,
                         #self.fast_shutter_close,
                         self]
                         
        self.monitors_dictionary = {'xbpm1': self.xbpm1,
                                    'cvd1': self.cvd1,
                                    #'xbpm5': self.xbpm5,
                                    'psd5': self.psd5,
                                    'psd6': self.psd6,
                                    'machine_status': self.machine_status,
                                    'fast_shutter': self.fast_shutter,
                                    #'eiger_en_out': self.eiger_en_out,
                                    #'trigger_eiger_on': self.trigger_eiger_on,
                                    #'trigger_eiger_off': self.trigger_eiger_off,
                                    #'fast_shutter_open': self.fast_shutter_open,
                                    #'fast_shutter_close': self.fast_shutter_close,
                                    'self': self}
        
        self.image = None
        self.rgbimage = None
        self._stop_flag = False


    def get_undulator_gap(self):
        return self.undulator.get_encoder_position()
        

    def get_slit_configuration(self):
        slit_configuration = {}
        for k in [1, 2, 3, 5, 6]:
            for direction in ['vertical', 'horizontal']:
                for attribute in ['gap', 'position']:
                    slit_configuration['slits%d_%s_%s' % (k, direction, attribute)] = getattr(getattr(self, 'slits%d' % k), 'get_%s_%s' % (direction, attribute))()
        return slit_configuration
    

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
                #if a[k]-1 == a[k-1]:
                if a[k] == a[k-1]:
                    indices.append(False)
                else:
                    indices.append(True)
            return a[np.array(indices)]

        def get_on_segments(on, off):
            if len(on) == 0:
                return
            elif off is None:
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
                
        fs_observations = self.fast_shutter.get_observations()
        if len(fs_observations) >= 2 and int(fs_observations[0][1]) == 1 and int(fs_observations[1][1]) == 1  :
            fs_observations[0][1] = 0
            fs_observations[1][1] = 0
        observations = np.array(fs_observations)

        if fs_observations == [] or len(observations) < 3:
            return 0
        try:
            g = np.gradient(observations[:, 1])
            ons = changes(np.where(g == 0.5)[0])
            offs = changes(np.where(g == -0.5)[0])
            if ons is None:
                return 0
            bons = [on for k, on in enumerate(ons[:-1]) if on == ons[k+1]-1]
            if offs is not None:
                boffs = [off for k, off in enumerate(offs[:-1]) if off == offs[k+1]-1]
            else:
                boffs = offs
        except:
            print 'observations'
            print observations
            print 'ons'
            print ons
            print traceback.print_exc()
            return 0
            
        segments = get_on_segments(bons, boffs)
        if segments is None:
            return 0
        
        chronos = observations[:, 0]
        total_exposure_time = 0
        
        for segment in segments:
            total_exposure_time += chronos[segment[1]] - chronos[segment[0]]
        
        progression = 100 * total_exposure_time/self.total_expected_exposure_time
        complete, incomplete = get_complete_incomplete_wedges(segments)
        if progression > 100:
            progression = 100
        if complete == self.total_expected_wedges:
            progression = 100
        return progression
    


    def get_point(self, start_time):
        chronos = time.time() - start_time
        progress = self.get_progression()
        return [chronos, progress]
    

    def monitor(self, start_time):
        #print 'xray_experiment monitor start'
        if not hasattr(self, 'observations'):
            self.observations = []
        self.observation_fields = ['chronos', 'progress']
        last_point = [None, None]
        while self.observe == True:
            point = self.get_point(start_time)
            progress = point[1]
            self.observations.append(point)
            if self.parent != None:
                if point[0] != None and last_point[0] != point[0] and last_point[1] != progress and progress > 0:
                    self.parent.emit('progressStep', (progress))
                    last_point = point
            gevent.sleep(self.monitor_sleep_time)
            
            
    def start_monitor(self):
        #print 'start_monitor'
        self.observe = True
        if hasattr(self, 'actuator'):
            self.actuator.observe = True
            if hasattr(self, 'actuator_names'):
                self.observers = [gevent.spawn(self.actuator.monitor, self.start_time, self.actuator_names)]
            else:
                self.observers = [gevent.spawn(self.actuator.monitor, self.start_time)]
        else:
            self.observers = []
        for monitor in self.monitors:
            monitor.observe = True
            self.observers.append(gevent.spawn(monitor.monitor, self.start_time))
        
    
    def stop_monitor(self):
        #print 'stop_monitor'
        self.observe = False
        if hasattr(self, 'actuator'):
            self.actuator.observe = False
        for monitor in self.monitors:
            monitor.observe = False
        gevent.joinall(self.observers)
    
    
    def get_observations(self):
        return self.observations
    
    
    def get_observation_fields(self):
        return self.observation_fields
    
    
    def get_points(self):
        return np.array(self.observations)[:,1]
    
    
    def get_chronos(self):
        return np.array(self.observations)[:,0]        
    
    def stop_monitor(self):
        print 'stop_monitor'
        self.observe = False
        self.actuator.observe = False
        for monitor in self.monitors:
            monitor.observe = False
        gevent.joinall(self.observers)
        
    
    def get_results(self):
        results = {}
        
        results['actuator'] = {'observation_fields': self.actuator.get_observation_fields(),
                               'observations': self.actuator.get_observations()}
        
        for (monitor_name, monitor) in zip(self.monitor_names, self.monitors):
            results[monitor_name] = {'observation_fields': monitor.get_observation_fields(),
                                     'observations': monitor.get_observations()}
        
        return results

        
    def set_photon_energy(self, photon_energy=None, wait=False):
        _start = time.time()
        if photon_energy > 1000: #if true then it was specified in eV not in keV
            photon_energy *= 1e-3
        if photon_energy != None:
            self.energy_moved = self.energy_motor.set_energy(photon_energy, wait=wait)
        else:
            self.energy_moved = 0
        
        #print 'set_photon_energy took %s' % (time.time() - _start)
        


    def get_current_photon_energy(self):
        return self.energy_motor.get_energy()


    def set_transmission(self, transmission=None):
        if transmission is not None:
            self.transmission = transmission
            self.transmission_motor.set_transmission(transmission)
            

    def get_transmission(self):
        return self.transmission_motor.get_transmission()


    def get_transmission_intention(self):
        return self.transmission
    

    def get_flux(self):
        if self.flux_monitor != None:
            flux = self.flux_monitor.get_flux()
        else:
            flux = None
        return flux
    
    def get_flux_intention(self):
        return self.flux
    
    def program_detector(self):
        _start = time.time()
        pass
        #print 'program_detector took %s' % (time.time()-_start)
    

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
    
    
    def get_all_observations(self):
        all_observations = {}
        if hasattr(self, 'actuator'):
            all_observations['actuator_monitor'] = {}
            actuator_observation_fields = self.actuator.get_observation_fields()
            all_observations['actuator_monitor']['observation_fields'] = actuator_observation_fields
            actuator_observations = self.actuator.get_observations()
            all_observations['actuator_monitor']['observations'] = actuator_observations
        
        for monitor_name, mon in zip(self.monitor_names, self.monitors):
            all_observations[monitor_name] = {}
            all_observations[monitor_name]['observation_fields'] = mon.get_observation_fields()
            all_observations[monitor_name]['observations'] = mon.get_observations()
        
        return all_observations

        
    def get_observations(self):
        return self.observations

    
    def save_diagnostic(self):
        _start = time.time()
        f = open(os.path.join(self.directory, '%s_diagnostics.pickle' % self.name_pattern), 'w')
        pickle.dump(self.get_all_observations(), f)
        f.close()
        logging.info('save_diagnostic took %s' % (time.time() - _start))

    def clean(self):
        _start = time.time()
        self.detector.disarm()
        logging.info('detector disarm finished')
        self.collect_parameters()
        logging.info('collect_parameters finished')
        clean_jobs = []
        clean_jobs.append(gevent.spawn(self.save_parameters))
        clean_jobs.append(gevent.spawn(self.save_log))
        if self.diagnostic == True:
            clean_jobs.append(gevent.spawn(self.save_diagnostic))
        gevent.joinall(clean_jobs)        
        logging.info('clean took %s' % (time.time() - _start))

    def stop(self):
        self._stop_flag = True
        self.goniometer.abort()
        self.detector.abort()
        

    def get_image(self):
        if self.image is not None:
            return self.image
        return self.camera.get_image()
    

    def get_rgbimage(self):
        if self.rgbimage is not None:
            return self.rgbimage
        return self.camera.get_rgbimage()
    

    def get_zoom(self):
        return self.camera.get_zoom()


