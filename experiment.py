#!/usr/bin/env python
'''generic experiment template. It should support all of the experimental methods we will ever come up with e.g.:

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

'''
class experiment(object):
	def __init__(self, experiment_id):
		self.experiment_id = experiment_id

    def set_directory(self, directory):
    	self.directory = directory
	def get_directory(self):
		return self.directory

    def set_name(self, name):
    	self.name = name
    def get_name(self):
    	return self.name

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

