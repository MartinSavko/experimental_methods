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
import time
import os

class experiment(object):
	'''properties to set and get:
	experiment_id, directory, name, project, user, group, sample, method, position, positions,
	photon_energy, resolution, flux, transmission, filters, beam_size'''
	def __init__(self):
        self.time_stamp = time.time()

    def set_experiment_id(self, experiment_id=None):
        if experiment_id is None:
            self.experiment_id = time.time()
        else:
            self.experiment_id = experiment_id
    def get_experiment_id(self):
        return self.experiment_id
    
    def set_user_id(self, user_id=None):
        if user_id is None:
            self.user_id = os.getuid()
    
    def get_user_id(self):
        return self.user_id

    def set_group_id(self, group_id):
        if group_id is None:
            self.group_id = os.getguid()
        else:
            self.group_id = group_id
    
    def get_group_id(self):
        return self.group_id

    def set_user_name(self, user_name):
        if user_name is None:
            self.user_name = os.getlogin()
        else:
            self.user_name = user_name

    def get_user_name(self):
        return self.user_name

	def get_protect(get_method, *args):
		try:
			return get_method(*args)
	    except:
	    	logging.error(traceback.print_exc())
	    	return None

    def set_directory(self, directory):
    	self.directory = directory
	def get_directory(self):
		return self.directory

    def set_name_pattern(self, name_pattern):
    	self.name_pattern = name_pattern
    def get_name_pattern(self):
    	return self.name_pattern
    def get_full_name_pattern(self):
        full_name_pattern = '/'.join(('', 
                                     str(self.get_user_id(), 
                                     str(self.get_user_id()),
                                     self.get_directory()[1:]),
                                     self.get_name_pattern()))
        return full_name_pattern
        
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
    	pass
    def run(self):
        pass
    def clean(self):
    	pass
    def analyze(self):
    	pass
    def save_log(self):
    	pass
    def store_ispyb(self):
    	pass

    
    def get_instrument_configuration(self):
        '''the purpose of this method is to gather and return all relevant information about the beamline and the machine
        
        Information to collect:
        0. machine status, current, mode
        1. slit positions
        2. tables positions
        3. intensity monitor values
        4. undulator settings 
        5. mirrors motors and tensions
        6. pressure monitors
        7. monochoromator motors
        8. thermometers readings
        9. diffractometer parameters
        10. aperture settings
        '''
        instrument_configuration = {}


        pass
        
    def check_directory(self):
        if os.path.isdir(self.directory):
            pass
        else:
            os.makedirs(self.directory)

    def write_destination_namepattern(self, image_path, name_pattern, goimgfile='/927bis/ccd/log/.goimg/goimg.db'):
        try:
            f = open(goimgfile, 'w')
            f.write('%s %s' % (os.path.join(image_path, 'process'), name_pattern))
            f.close()
        except IOError:
            logging.info('Problem writing goimg.db %s' % (traceback.format_exc()))
