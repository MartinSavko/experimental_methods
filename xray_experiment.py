#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time

from experiment import experiment
from detector import detector as detector
from goniometer import goniometer
from energy import energy as energy_motor
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
                 analysis=None):
        
        experiment.__init__(self, 
                            name_pattern=name_pattern, 
                            directory=directory,
                            analysis=analysis)
        
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
        
        self.safety_shutter = safety_shutter()
        self.fast_shutter = fast_shutter()
        self.camera = camera()
        
        if self.photon_energy == None:
            self.photon_energy = self.energy_motor.get_energy()
        self.wavelength = self.resolution_motor.get_wavelength_from_energy(self.photon_energy)

        if self.resolution != None:
            self.detector_distance = self.resolution_motor.get_distance_from_resolution(self.resolution, wavelength=self.wavelength)
            if self.detector_distance == None:
                self.detector_distance = self.detector.position.get_ts()
            self.resolution = self.resolution_motor.get_resolution_from_distance(self.detector_distance, wavelength=self.wavelength)
        
        self.monitor_names = ['xbpm1', 'cvd1', 'psd6']
        self.monitors = [xbpm('i11-ma-c04/dt/xbpm_diode.1', 'i11-ma-c00/ca/sai.1'),
                         xbpm('i11-ma-c05/dt/xbpm-cvd.1', 'i11-ma-c00/ca/sai.3'),
                         xbpm('i11-ma-c06/dt/xbpm_diode.6', 'i11-ma-c00/ca/sai.5')]
        
    def get_ntrigger(self):
        return self.ntrigger
    
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

    def set_resolution(self, resolution=None):
        if resolution is not None:
            self.resolution = resolution
            self.resolution_motor.set_resolution(resolution)
            
    def get_resolution(self):
        return self.resolution_motor.get_resolution()

    def get_detector_distance(self):
        return self.detector.position.ts.get_position()
    
    def set_detector_distance(self, position, wait=True):
        self.detector_ts_moved = self.detector.position.ts.set_position(position, wait=wait)
        
    def get_detector_vertical_position(self):
        return self.detector.position.tz.get_position()
    
    def set_detector_vertical_position(self, position, wait=True):
        self.detector_tz_moved = self.detector.position.tz.set_position(position, wait=wait)
        
    def get_detector_horizontal_position(self):
        return self.detector.position.tx.get_position()
    
    def set_detector_horizontal_position(self, position, wait=True):
        self.detector_tx_moved = self.detector.position.tx.set_position(position, wait=wait)
        
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
    def run():
        pass
    
    def finalize(self):
        self.clean()
        
    def clean(self):
        self.detector.disarm()
        self.save_parameters()
        self.save_results()
        self.save_log()
    
    def stop(self):
        self.goniometer.abort()
        self.detector.abort()