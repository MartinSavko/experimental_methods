# /usr/bin/env python
# -*- coding: utf-8 -*-
import logging
from math import tan, asin, atan, sin
import numpy as np
import gevent
import traceback

try:
    import tango
except ImportError:
    import PyTango as tango

from useful_routines import (
    get_energy_from_wavelength,
    get_wavelength_from_energy,
    get_theta_from_wavelength,
    get_wavelength_from_theta,
    get_theta_from_energy,
    get_energy_from_theta,
    get_resolution_from_radial_distance,
    get_detector_distance_from_resolution,
    get_resolution_from_detector_distance,
)

# from energy import energy
# from beam_center import beam_center
DEFAULT_ENERGY = 12650.0  # 13179.2 #15355.6 #15306.0 #15348.5 #15370. # 13215.0


class resolution_mockup:
    def __init__(
        self,
        x_pixels_in_detector=3110,
        y_pixels_in_detector=3269,
        x_pixel_size=75e-6,
        y_pixel_size=75e-6,
        distance=0.180,
        wavelength=0.981,
        photon_energy=DEFAULT_ENERGY,
        tunable=True,
    ):
        self.x_pixel_size = x_pixel_size
        self.y_pixel_size = y_pixel_size
        self.x_pixels_in_detector = x_pixels_in_detector
        self.y_pixels_in_detector = y_pixels_in_detector
        self.distance = distance
        self.wavelength = wavelength
        self.photon_energy = photon_energy
        self.tunable = tunable

    def get_detector_distance(self):
        return self.distance

    def get_detector_radii(self):
        return

    def get_detector_min_radius(self):
        return self.x_pixels_in_detector / 2 * self.x_pixel_size * 1.0e3

    def get_detector_max_radius(self):
        return self.y_pixels_in_detector / 2 * self.y_pixel_size * 1.0e3

    def set_distance(self, distance, wait=False):
        return

    def get_distance(self):
        return self.distance

    def get_wavelength(self):
        return self.wavelength

    def set_energy(self, energy):
        return

    def get_energy(self):
        return self.photon_energy

    # def get_resolution(self, distance=None, wavelength=None, radius=None):
    # if distance is None:
    # distance = self.get_distance()
    # if radius is None:
    # radial_distance = self.get_detector_min_radius()
    # if wavelength is None:
    # wavelength = self.get_wavelength()

    # two_theta = atan(radial_distance / distance)
    # resolution = 0.5 * wavelength / np.sin(0.5 * two_theta)
    # return resolution

    def get_resolution(self, detector_distance=None, wavelength=None, radius=None):
        if detector_distance is None:
            detector_distance = self.get_distance()
        if radius is None:
            radial_distance = self.get_detector_min_radius()
        if wavelength is None:
            wavelength = self.get_wavelength()

        resolution = get_resolution_from_radial_distance(
            radial_distance, wavelength, detector_distance
        )
        return resolution

    def get_distance_from_resolution(self, resolution, wavelength=None, radius=None):
        logging.getLogger("user_level_log").info(
            "get_distance_from_resolution 1: resolution %s, wavelength %s, radius %s"
            % (resolution, wavelength, radius)
        )
        if wavelength is None:
            wavelength = self.get_wavelength()
        logging.getLogger("user_level_log").info(
            "get_distance_from_resolution 2: resolution %s, wavelength %s, radius %s"
            % (resolution, wavelength, radius)
        )
        two_theta = 2 * asin(0.5 * wavelength / resolution)
        if radius is None:
            radius = self.get_detector_min_radius()
        distance = radius / tan(two_theta)
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

    def get_distance_limits(self):
        return (113, 650)

    def get_resolution_limits(self, wavelength=None, photon_energy=None):
        if wavelength == None:
            wavelength = self.get_wavelength()
        if photon_energy != None:
            if photon_energy < 1e3:
                photon_energy *= 1e3
            wavelength = self.get_wavelength_from_energy(photon_energy)

        min_distance, max_distance = self.get_distance_limits()
        high = self.get_resolution(
            detector_distance=min_distance, wavelength=wavelength
        )
        low = self.get_resolution(detector_distance=max_distance, wavelength=wavelength)
        return high, low


class resolution(object):
    def __init__(
        self,
        x_pixels_in_detector=3110,
        y_pixels_in_detector=3269,
        x_pixel_size=75e-6,
        y_pixel_size=75e-6,
        distance=None,
        wavelength=None,
        photon_energy=None,
        test=False,
        tunable=True,
    ):
        self.tunable = tunable
        self.distance_motor = tango.DeviceProxy("i11-ma-cx1/dt/dtc_ccd.1-mt_ts")
        self.horizontal_motor = tango.DeviceProxy("i11-ma-cx1/dt/dtc_ccd.1-mt_tx")
        self.vertical_motor = tango.DeviceProxy("i11-ma-cx1/dt/dtc_ccd.1-mt_tz")

        if self.tunable:
            self.wavelength_motor = tango.DeviceProxy("i11-ma-c03/op/mono1")
        else:
            self.wavelength_motor = None
        # self.energy_motor = energy()
        # self.bc = beam_center()

        self.x_pixel_size = x_pixel_size
        self.y_pixel_size = y_pixel_size
        self.x_pixels_in_detector = x_pixels_in_detector
        self.y_pixels_in_detector = y_pixels_in_detector
        self.distance = distance
        self.wavelength = wavelength
        self.photon_energy = photon_energy

        self.test = test

    def get_detector_distance(self):
        return self.distance_motor.position

    def get_detector_radii(self):
        beam_center_x, beam_center_y = (
            self.x_pixels_in_detector / 2,
            self.y_pixels_in_detector / 2,
        )  # self.bc.get_beam_center()
        detector_size_x = self.x_pixel_size * self.x_pixels_in_detector
        detector_size_y = self.y_pixel_size * self.y_pixels_in_detector

        beam_center_distance_x = self.x_pixel_size * beam_center_x
        beam_center_distance_y = self.y_pixel_size * beam_center_y

        distances_x = np.array(
            [detector_size_x - beam_center_distance_x, beam_center_distance_x]
        )
        distances_y = np.array(
            [detector_size_y - beam_center_distance_y, beam_center_distance_y]
        )

        edge_distances = np.hstack([distances_x, distances_y])
        corner_distances = np.array(
            [(x**2 + y**2) ** 0.5 for x in distances_x for y in distances_y]
        )

        distances = np.hstack([edge_distances, corner_distances]) * 1.0e3
        return distances

    def get_distance_limits(self):
        distance_motor_config = self.distance_motor.get_attribute_config("position")
        return float(distance_motor_config.min_value), float(
            distance_motor_config.max_value
        )

    def get_resolution_limits(self, wavelength=None, photon_energy=None):
        if wavelength == None:
            wavelength = self.get_wavelength()
        if photon_energy != None:
            if photon_energy < 1e3:
                photon_energy *= 1e3
            wavelength = self.get_wavelength_from_energy(photon_energy)

        min_distance, max_distance = self.get_distance_limits()
        high = self.get_resolution(
            detector_distance=min_distance, wavelength=wavelength
        )
        low = self.get_resolution(detector_distance=max_distance, wavelength=wavelength)
        return high, low

    def get_detector_min_radius(self):
        # radii = self.get_detector_radii()
        # return radii.min()
        return self.x_pixels_in_detector / 2 * self.x_pixel_size * 1.0e3

    def get_detector_max_radius(self):
        # radii = self.get_detector_radii()
        # return radii.max()
        return self.y_pixels_in_detector / 2 * self.y_pixels_size * 1.0e3

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
        if self.tunable:
            wavelength = self.wavelength_motor.read_attribute("lambda").value
        else:
            wavelength = self.get_wavelength_from_energy(self.photon_energy)
        return wavelength

    # def set_energy(self, energy):
    # self.energy_motor.set_energy(energy)

    def get_energy(self):
        if self.tunable:
            energy = self.photon_energy
        else:
            energy = self.wavelength_motor.energy
        return energy

    def get_energy_from_wavelength(self, wavelength):
        """energy in eV, wavelength in angstrom"""
        energy = get_energy_from_wavelength(wavelength)
        return energy

    def get_energy_from_wavelength(self, wavelength):
        return get_energy_from_wavelength(wavelength)

    def get_wavelength_from_energy(self, energy):
        """energy in eV, wavelength in angstrom"""
        wavelength = get_wavelength_from_energy(energy)
        return wavelength

    def get_wavelength_from_theta(self, theta, d2=6.2696):
        """wavelength in angstrom"""
        wavelength = get_wavelength_from_theta(theta, d2=d2)
        return wavelength

    def get_theta_from_wavelength(self, wavelength, d2=6.2696):
        theta = get_theta_from_wavelength(wavelength, d2=d2)
        return theta

    def get_energy_from_theta(self, theta):
        energy = get_energy_from_theta(theta)
        return energy

    def get_theta_from_energy(self, energy):
        theta = get_theta_from_energy(energy)
        return theta

    def get_resolution(self, detector_distance=None, wavelength=None, radius=None):
        if detector_distance is None:
            detector_distance = self.get_distance()
        if radius is None:
            radial_distance = self.get_detector_min_radius()
        if wavelength is None:
            wavelength = self.get_wavelength()
        if wavelength > 3:
            # print(f"get_resolution, is someone asking resolution with energy instead of wavelength {wavelength} ?")
            if wavelength < 1.0e2:
                wavelength *= 1.0e3
            wavelength = get_wavelength_from_energy(wavelength)
        resolution = get_resolution_from_radial_distance(
            radial_distance, wavelength, detector_distance
        )
        # logging.info(f"get_resolution:  {resolution:.3f}\n\tdetector distance: {detector_distance:.3f}\twavelength: {wavelength:.3f}\tradial_distance: {radial_distance:.3f}")
        return resolution

    def get_resolution_from_distance(self, distance, wavelength=None):
        return self.get_resolution_from_detector_distance(
            distance, wavelength=wavelength
        )

    def get_resolution_from_detector_distance(self, detector_distance, wavelength=None):
        return self.get_resolution(
            detector_distance=detector_distance, wavelength=wavelength
        )

    def get_detector_distance_from_resolution(self, resolution, wavelength=None):
        if wavelength is None:
            wavelength = self.get_wavelength()
        elif wavelength > 3:
            print(
                f"get_detector_distance_from_resolution, is someone asking resolution with energy instead of wavelength {wavelength} ?"
            )
            if wavelength < 1.0e2:
                wavelength *= 1.0e3
            wavelength = get_wavelength_from_energy(wavelength)
        detector_distance = get_detector_distance_from_resolution(
            resolution, wavelength
        )
        return detector_distance

    def get_distance_from_resolution(self, resolution, wavelength=None, radius=None):
        # logging.getLogger('user_level_log').info('get_distance_from_resolution 1: resolution %s, wavelength %s, radius %s' % (resolution, wavelength, radius))
        if wavelength is None:
            wavelength = self.get_wavelength()
        elif wavelength > 3:
            wavelength *= 1e-3
        if wavelength > 5:
            wavelength = get_wavelength_from_energy(wavelength * 1e3)
        # logging.getLogger('user_level_log').info('get_distance_from_resolution 2: resolution %s, wavelength %s, radius %s' % (resolution, wavelength, radius))
        try:
            print("wavelength", wavelength)
            print("resolution", resolution)

            two_theta = 2 * asin(0.5 * wavelength / resolution)

            if radius is None:
                radius = self.get_detector_min_radius()
            detector_distance = radius / tan(two_theta)
        except:
            traceback.print_exc()
            detector_distance = get_detector_distance_from_resolution(
                resolution, wavelength
            )
        return detector_distance

    def set_resolution(
        self, resolution, wavelength=None, photon_energy=None, wait=False
    ):
        if photon_energy != None:
            wavelength = self.get_wavelength_from_energy(photon_energy)
        elif wavelength == None:
            wavelength = self.get_wavelength()
        distance = self.get_distance_from_resolution(resolution, wavelength)
        self.set_distance(distance)
        if wait:
            self.wait()

    def wait_distance(self):
        while self.distance_motor.state().name != "STANDBY":
            gevent.sleep(0.1)

    # def wait_energy(self):22926.8777
    # self.energy_motor.wait()

    def wait(self):
        self.wait_distance()
        self.wait_energy()

    def stop(self):
        if self.tunable:
            self.wavelength_motor.Stop()
        self.distance_motor.Stop()
        # self.horizontal_motor.Stop()
        # self.vertical_motor.Stop()

    def abort(self):
        self.stop()
