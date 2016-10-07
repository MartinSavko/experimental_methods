#!/usr/bin/env python

'''general experiment template. It should support all of the experimental methods we will ever come up with e.g.:

1. single scan oscillation crystallography experiment
2. helical scans
3. raster scans
4. x-ray centerings
5. nested helical scans
6. x-ray fluorescence spectra
7. xanes
8. inverse beam scans
9. interleaved energy data collections
10. interleaved energy helical data collections
11. interleaved energy inverse beam helical data collections
12. multi positional experiments
13. translational type of experiments (e.g. neha, tranlational scans, regression sweep acquisitions)
14. Burn strategy
15. Reference image acquisition
'''

import traceback
import logging

class experiment(object):
	'''properties to set and get:
	experiment_id, directory, name, project, user, group, sample, method, position, positions,
	photon_energy, resolution, flux, transmission, filters, beam_size'''
	def __init__(self, experiment_id):
		self.experiment_id = experiment_id

	def get_protect(get_method, *args):
		try:
			return get_method(*args)
	    except:
	    	logging.error(traceback.print_exc())
	    	return None

    def set_directory(self, directory):
    	self.directory = directory
	dsef get_directory(self):
		return self.directory

    def set_name_pattern(self, name_pattern):
    	self.name_pattern = name_pattern
    def get_name_pattern(self):
    	return self.name_pattern

    def set_project(self, project):
    	self.project = project
    def get_project(self):
    	return self.project 

    def set_user(self, user):
    	self.user = user
    def get_user(self):
    	return self.user
    
    def set_group(self, group):
    	self.group = group
    def get_group(self):
    	return self.group

    def set_sample(self, sample):
    	self.sample = sample
    def get_sample(self):
    	return self.sample 

    def set_method(self, method):
    	self.method = method
    def get_method(self):
    	return self.method

 	def set_position(self, position):
 		self.position = position
    def get_position(self):
    	return self.position

    def set_positions(self, positions):
    	self.positions = positions
    def get_positions(self):
 		return self.positions

	def set_photon_energy(self, photon_energy):
		self.photon_energy = photon_energy
	def get_photon_energy(self):
		return self.photon_energy

    def set_resolution(self, resolution):
    	self.resolution = resolution
    def get_resolution(self):
    	return self.resolution

    def set_flux(self, flux):
    	self.flux = flux
    def get_flux(self):
    	return self.flux

    def set_transmission(self, transmission):
    	self.transmission = transmission
    def get_transmission(self):
    	return self.transmission

    def set_filters(self, filters):
    	self.filters = filters
    def get_filters(self, filters):
    	return self.filters

    def set_beam_size(self, beam_size):
    	self.beam_size = beam_size
    def get_beam_size(self):
    	return self.beam_size

    def prepare(self):
    	pass
    def cancel(self):
     	pass
    def abort(self):
    	pass
    def start(self):
    	pass
    def stop(self):
    	return self.abort()
    def run(self):
    	return self.start()
    def clean(self):
    	pass
    def analyze(self):
    	pass
    def save_log(self):
    	pass
    def store_ispyb(self):
    	pass

    def check_dir(self, download):
        if os.path.isdir(download):
            pass
        else:
            os.makedirs(download)
