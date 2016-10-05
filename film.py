#!/usr/bin/env python
'''
The purpose of this object is to record a film (a series of images) of a rotating sample on the goniometer as a function of goniometer axis (axes) position(s).
'''

from camera import camera
from goniometer import goniometer

class film(experiment):
	def __init__(self):
		super(film, self).__init__()
		camera = camera()
		goniometer = goniometer()
		
    def set_scan_range(self, scan_range):
    	self.scan_range = scan_range
    def get_scan_range(self):
    	return self.scan_range


    def set_scan_exposure_time(self, scan_exposure_time):
    	self.scan_exposure_time = scan_exposure_time
    def get_scan_exposure_time(self):
    	return self.scan_exposure_time


