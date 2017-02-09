#!/usr/bin/env python
'''
helical scan
'''
import traceback
import logging
import time

from scan import scan

class helical_scan(scan):
	def __init__(self, scan_range, scan_exposure_time, scan_start_angle, angle_per_frame, name_pattern, directory, image_nr_start, position1=None, position2=None, photon_energy=None, flux=None, transmission=None): 
		super(self, scan).__init__(scan_range, scan_exposure_time, scan_start_angle, angle_per_frame, name_pattern, directory, image_nr_start, photon_energy=photon_energy, flux=flux, transmission=transmission)
		self.position1 = position1
		self.position2 = position2

	def run(self):
		self.goniometer.helical_scan(self.position1, self.position2, self.scan_start_angle, self.scan_range, self.scan_exposure_time)