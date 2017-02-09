#!/usr/bin/env python
'''
single position oscillation scan
'''
import traceback
import logging
import time

from experiment import experiment
from detector import detector
from goniometer import goniometer
#from beam import beam
from beam_center import beam_center
from transmission import transmission as transmission_motor
from energy import energy as energy_motor
from resolution import resolution as resolution_motor
# from flux import flux
# from filters import filters
# from camera import camera
from protective_cover import protective_cover


class scan(experiment):

    def __init__(self, scan_range, scan_exposure_time, scan_start_angle, angle_per_frame, name_pattern, directory, image_nr_start, position=None, photon_energy=None, flux=None, transmission=None):
        self.goniometer = goniometer()
        self.detector = detector()
        self.beam_center = beam_center()
        self.energy_motor = energy_motor()
        self.resolution_motor = resolution_motor()
        self.protective_cover = protective_cover()
        self.transmission_motor = transmission_motor()
        self.scan_range = scan_range
        self.scan_exposure_time = scan_exposure_time
        self.scan_start_angle = scan_start_angle
        self.angle_per_frame = angle_per_frame
        self.image_nr_start = image_nr_start
        self.position = position
        self.photon_energy = photon_energy
        self.flux = flux
        self.transmission = transmission
        self.name_pattern = name_pattern
        self.directory = directory
        self._ntrigger = 1
        super(self, experiment).__init__()

    def get_nimages(self):
        nimages, rest = divmod(self.scan_range, self.angle_per_frame)
        if rest > 0:
            nimages += 1
            self.scan_range += rest*self.angle_per_frame
            self.scan_exposure_time += rest*self.angle_per_frame/self.scan_range
        return nimages

    def get_frame_time(self):
        return self.scan_exposure_time/self.get_nimages()

    def get_position(self):
        if self.position is None:
            return self.goniometer.get_position()
        else:
            return self.position

    def set_position(self, position=None):
        if position is None:
            self.position = self.goniometer.get_position()
        else:
            self.position = position
            self.goniometer.set_position(self.position)
            self.goniometer.wait()
        self.goniometer.save_position()

    def set_photon_energy(self, photon_energy=None):
        if photon_energy is not None:
            self.photon_energy = photon_energy
            self.energy_motor.set_energy(photon_energy)

    def get_photon_energy(self):
        return self.photon_energy

    def set_resolution(self, resolution=None):
        if resolution is not None:
            self.resolution = resolution
            self.resolution_motor.set_resolution(resolution)

    def get_resolution(self):
        return self.resolution

    def set_transmission(self, transmission=None):
        if transmission is not None:
            self.transmission = transmission
            self.transmission_motor.set_transmission(transmission)

    def get_transmission(self):
        return self.transmission

    def program_detector(self):
        self.detector.set_ntrigger(self._ntrigger)
        self.detector.set_standard_parameters()
        self.detector.clear_monitor() 	
        self.detector.set_name_pattern(self.get_full_name_pattern())
        self.detector.set_frame_time(self.get_frame_time())
        count_time = self.get_frame_time() - self.detector.get_detector_readout_time()
        self.detector.set_count_time(count_time)
        self.detector.set_nimages(self.nimages)
        self.detector.set_omega(self.start_angle)
        self.detector.set_omega_increment(self.angle_per_frame)
        if self.detector.get_photon_energy() != self.photon_energy:
            self.detector.set_photon_energy(self.photon_energy)
        if self.detector.get_image_nr_start() != self.image_nr_start:
            self.detector.set_image_nr_start(self.image_nr_start)
        beam_center_x, beam_center_y = self.beam_center.get_beam_center()
        self.detector.set_beam_center_x(beam_center_x)
        self.detector.set_beam_center_y(beam_center_y)
        self.detector.set_detector_distance(self.beam_center.get_detector_distance()/1000.)
        self.detector.arm()

    def program_goniometer(self):
        self.nimages = self.get_nimages()
        self.goniometer.set_scan_start_angle(self.scan_start_angle)
        self.goniometer.set_scan_range(self.scan_range)
        self.goniometer.set_scan_exposure_time(self.scan_exposure_time)
        self.goniometer.set_scan_number_of_frames(1)
        self.goniometer.set_detector_gate_pulse_enabled(True)
        self.goniometer.set_data_collection_phase()

    def prepare(self):
        self.check_directory()
        
        self.program_goniometer()
        
        self.set_photon_energy(self.photon_energy)
        self.set_resolution(self.resolution)
        self.set_transmission(self.transmission)
        self.protective_cover.extract()
        
        self.resolution_motor.wait()
        self.energy_motor.wait()
        self.goniometer.wait()

        self.program_detector()
        if self.goniometer.backlight_is_on():
            self.goniometer.remove_backlight()

        self.energy_motor.turn_off()

    def execute(self):
        self.prepare()
        self.run()
        self.clean()

    def run(self):
        self.goniometer.set_position(self.get_position())
        self.goniometer.point_scan(self.scan_start_angle, self.scan_range, self.scan_exposure_time, wait=True) #self.goniometer.start_scan(wait=True)

    def clean(self):
        self.detector.disarm()

    def stop(self):
        self.goniometer.abort()
        self.detector.abort()
