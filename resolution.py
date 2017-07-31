# -*- coding: utf-8 -*-

import PyTango
from math import tan, asin
import numpy as np
from energy import energy
from beam_center import beam_center
import time
from scipy.constants import c, eV, h, angstrom

class resolution_mockup:
    def __init__(self, x_pixels_in_detector=3110, y_pixels_in_detector=3269, x_pixel_size=75e-6, y_pixel_size=75e-6, distance=None, wavelength=None, photon_energy=None):
        self.x_pixel_size = x_pixel_size
        self.y_pixel_size = y_pixel_size
        self.x_pixels_in_detector = x_pixels_in_detector
        self.y_pixels_in_detector = y_pixels_in_detector
        self.distance = distance
        self.wavelength = wavelength
        self.photon_energy = photon_energy
        
    def get_detector_radii(self):
        return
    def get_detector_min_radius(self):
        return self.x_pixels_in_detector/2 * self.x_pixel_size
    def get_detector_max_radius(self):
        return self.y_pixels_in_detector * self.y_pixel_size
    def set_distance(self, distance, wait=False):
        return
    def get_distance(self):
        return 1
    def get_wavelength(self):
        return
    def set_energy(self, energy):
        return
    def get_energy(self):
        return
    
    def get_energy_from_wavelength(self, wavelength):
        '''energy in eV, wavelength in angstrom'''
        return (h*c)/(eV*angstrom)/wavelength
    
    def get_wavelength_from_energy(self, energy):
        '''energy in eV, wavelength in angstrom'''
        return (h*c)/(eV*angstrom)/energy
    
    def get_resolution(self, distance=None, wavelength=None, radius=None):
        if distance is None:
            distance = self.get_distance()
        if radius is None:
            detector_radius = self.get_detector_min_radius()
        if wavelength is None:
            wavelength = self.get_wavelength()
        
        two_theta = np.math.atan(detector_radius/distance)
        resolution = 0.5 * wavelength / np.sin(0.5*two_theta)
        return resolution
        
    def get_resolution_from_distance(self, distance, wavelength=None):
        return self.get_resolution(distance=distance, wavelength=wavelength)
        
    def get_distance_from_resolution(self, resolution, wavelength=None, radius=None):
        if wavelength is None:
            wavelength = self.get_wavelength()
        two_theta = 2*asin(0.5*wavelength/resolution)
        if radius is None:
            radius = self.get_detector_min_radius()
        distance = radius/tan(two_theta)
        return distance

    def set_resolution(self, resolution, wavelength=None, wait=False):
        if wavelength is None:
            wavelength = self.get_wavelength()
        else:
            energy = self.get_energy_from_wavelength(wavelength)
            self.set_energy(energy)
        distance = self.get_distance_from_resolution(resolution, wavelength)
        self.set_distance(distance)
        if wait:
            self.wait()

    def wait_distance(self):
        return
    def wait_energy(self):
        return 
    def wait(self):
        self.wait_distance()
        self.wait_energy()

class resolution(object):
    def __init__(self, x_pixels_in_detector=3110, y_pixels_in_detector=3269, x_pixel_size=75e-6, y_pixel_size=75e-6, distance=None, wavelength=None, photon_energy=None, test=False):
        self.distance_motor = PyTango.DeviceProxy('i11-ma-cx1/dt/dtc_ccd.1-mt_ts')
        self.horizontal_motor = PyTango.DeviceProxy('i11-ma-cx1/dt/dtc_ccd.1-mt_tx')
        self.vertical_motor = PyTango.DeviceProxy('i11-ma-cx1/dt/dtc_ccd.1-mt_tz')
        self.wavelength_motor = PyTango.DeviceProxy('i11-ma-c03/op/mono1')
        self.energy_motor = energy()
        self.bc = beam_center()

        self.x_pixel_size = x_pixel_size
        self.y_pixel_size = y_pixel_size
        self.x_pixels_in_detector = x_pixels_in_detector
        self.y_pixels_in_detector = y_pixels_in_detector
        self.distance = distance
        self.wavelength = wavelength
        self.photon_energy = photon_energy

        self.test = test
        
    def get_detector_radii(self):
        beam_center_x, beam_center_y = self.bc.get_beam_center()
        detector_size_x = self.x_pixel_size * self.x_pixels_in_detector
        detector_size_y = self.y_pixel_size * self.y_pixels_in_detector
        
        beam_center_distance_x = self.x_pixel_size * beam_center_x
        beam_center_distance_y = self.y_pixel_size * beam_center_y
        
        distances_x = np.array([detector_size_x - beam_center_distance_x, beam_center_distance_x])
        distances_y = np.array([detector_size_y - beam_center_distance_y, beam_center_distance_y])
        
        edge_distances = np.hstack([distances_x, distances_y])
        corner_distances = np.array([(x**2 + y**2)**0.5 for x in distances_x for y in distances_y])
        
        distances = np.hstack([edge_distances, corner_distances]) * 1000.
        return distances
        
    def get_detector_min_radius(self):
        distances = self.get_detector_radii()
        return distances.min()
        
    def get_detector_max_radius(self):
        distances = self.get_detector_radii()
        return distances.max()
        
    def get_distance(self):
        return self.distance_motor.position
    
    def get_horizontal_position(self):
        return self.horizontal_motor.position
    
    def get_vertical_position(self):
        return self.vertical_motor.position
        
    def set_distance(self, distance, wait=False):
        if self.distance_motor.position != distance:
            self.distance_motor.position = distance
        if wait:
            self.wait_distance()

    def get_wavelength(self):
        return self.wavelength_motor.read_attribute('lambda').value
        
    def set_energy(self, energy):
        self.energy_motor.set_energy(energy)

    def get_energy(self):
        return self.wavelength_motor.energy


    def get_energy_from_wavelength(self, wavelength):
        '''energy in eV, wavelength in angstrom'''
        return (h*c)/(eV*angstrom)/wavelength
    
    def get_wavelength_from_energy(self, energy):
        '''energy in eV, wavelength in angstrom'''
        return (h*c)/(eV*angstrom)/energy

    def get_wavelength_from_theta(self, theta, d2=6.2696):
        '''wavelength in angstrom'''
        return d2*np.sin(np.radians(theta))
    
    def get_resolution(self, distance=None, wavelength=None, radius=None):
        if distance is None:
            distance = self.get_distance()
        if radius is None:
            detector_radius = self.get_detector_min_radius()
        if wavelength is None:
            wavelength = self.get_wavelength()
        
        two_theta = np.math.atan(detector_radius/distance)
        resolution = 0.5 * wavelength / np.sin(0.5*two_theta)
        return resolution
        
    def get_resolution_from_distance(self, distance, wavelength=None):
        return self.get_resolution(distance=distance, wavelength=wavelength)
        
    def get_distance_from_resolution(self, resolution, wavelength=None, radius=None):
        if wavelength is None:
            wavelength = self.get_wavelength()
        two_theta = 2*asin(0.5*wavelength/resolution)
        if radius is None:
            radius = self.get_detector_min_radius()
        distance = radius/tan(two_theta)
        return distance

    def set_resolution(self, resolution, wavelength=None, photon_energy=None, wait=False):
        if photon_energy != None:
            wavelength = self.get_wavelength_from_energy(photon_energy)
        elif wavelength == None:
            wavelength = self.get_wavelength()
        distance = self.get_distance_from_resolution(resolution, wavelength)
        self.set_distance(distance)
        if wait:
            self.wait()

    def wait_distance(self):
        while self.distance_motor.state().name != 'STANDBY':
            time.sleep(0.1)

    def wait_energy(self):
        self.energy_motor.wait()

    def wait(self):
        self.wait_distance()
        self.wait_energy()


