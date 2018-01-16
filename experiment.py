#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''general experiment template. It will support all of the experimental methods we will ever come up with e.g.:

1. single scan oscillation crystallography experiment
2. helical scans
3. raster scans
4. x-ray centerings
5. nested helical scans (neha)
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
import pickle
import scipy.misc

class experiment:
    
    parameter_fields = set(['timestamp', 
                            'name_pattern', 
                            'directory', 
                            'diagnostic', 
                            'analysis', 
                            'conclusion', 
                            'simulation', 
                            'display',
                            'start_time',
                            'end_time',
                            'duration'])
    
    def __init__(self, 
                 name_pattern=None, 
                 directory=None,
                 diagnostic=None,
                 analysis=None,
                 conclusion=None,
                 simulation=None,
                 display=None,
                 snapshot=None):
        
        self.timestamp = time.time()
        self.name_pattern = name_pattern
        self.directory = directory
        self.diagnostic = diagnostic
        self.analysis = analysis
        self.conclusion = conclusion
        self.simulation = simulation
        self.display = display
        self.snapshot = snapshot
        
        if type(self.directory) == str:
            self.process_directory = os.path.join(self.directory, 'process')
        
        self.parameters = {}
        
        self.parameter_fields = experiment.parameter_fields
                
    def get_protect(get_method, *args):
        try:
            return get_method(*args)
        except:
            logging.error(traceback.print_exc())
            return None

    def get_full_name_pattern(self):
        full_name_pattern = '/'.join(('', str(self.get_user_id()), self.get_directory()[1:],  self.get_name_pattern()))
        return full_name_pattern
    
    def set_directory(self, directory):
        self.directory = directory
    def get_directory(self):
        return self.directory
    
    def set_name_pattern(self, name_pattern):
        self.name_pattern = name_pattern
    def get_name_pattern(self):
        return self.name_pattern
    
    def set_timestamp(self, timestamp):
        self.timestamp = timestamp
    def get_timestamp(self):
        return self.timestamp
    
    def set_diagnostic(self, diagnostic):
        self.diagnostic = diagnostic
    def get_diagnostic(self):
        return self.diagnostic

    def set_analysis(self, analysis):
        self.analysis = analysis
    def get_analysis(self):
        return self.analysis
    
    def set_conclusion(self, conclusion):
        self.conclusion = conclusion
    def get_conclusion(self):
        return self.conclusion
    
    def set_simulation(self, simulation):
        self.simulation = simulation
    def get_simulation(self):
        return self.simulation
    
    def set_display(self, display):
        self.display = display
    def get_display(self):
        return self.display
    
    def get_user_id(self):
        return os.getuid()
        
    def set_start_time(self, start_time):
        self.start_time = start_time
    def get_start_time(self):
        return self.start_time
    
    def set_end_time(self, end_time):
        self.end_time = end_time
    def get_end_time(self):
        return self.end_time
    
    def get_duration(self):
        return self.get_end_time() - self.get_start_time()
    
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
    def conclude(self):
        pass
    def execute(self):
        self.start_time = time.time()
        try:
            self.prepare()
            print 'self.diagnostic', self.diagnostic
            if self.diagnostic == True:
                print 'Starting monitoring'
                self.start_monitor()
            self.run()
            if self.diagnostic == True:
                print 'Stopping monitors'
                self.stop_monitor()
        except:
            print 'Problem in preparation or execution %s' % self.__module__
            print traceback.print_exc()
        finally:
            self.end_time = time.time()
            self.clean()
        if self.analysis == True:
            self.analyze()
            self.save_results()
        if self.conclusion == True:
            self.conclude()
            
        print 'experiment execute took %s' % (time.time() - self.start_time)
        
    #def save_log(self, template, experiment_information):
    def save_results(self):
        pass
    
    def collect_parameters(self):
        for parameter in self.parameter_fields:
            if parameter != 'slit_configuration':
                try:
                    self.parameters[parameter] = getattr(self, 'get_%s' % parameter)()
                except AttributeError:
                    self.parameters[parameter] = None
                    
            else:
                slit_configuration = self.get_slit_configuration()
                for key in slit_configuration:
                    self.parameters[key] = slit_configuration[key]
        
        if self.snapshot == True:
            self.parameters['camera_zoom'] = self.camera.get_zoom()
            self.parameters['camera_calibration_horizontal'] = self.camera.get_horizontal_calibration()
            self.parameters['camera_calibration_vertical'] = self.camera.get_vertical_calibration()
            self.parameters['beam_position_vertical'] = self.camera.md2.beampositionvertical
            self.parameters['beam_position_horizontal'] = self.camera.md2.beampositionhorizontal
            self.parameters['image'] = self.image
            self.parameters['rgb_image'] = self.rgbimage.reshape((self.image.shape[0], self.image.shape[1], 3))
            scipy.misc.imsave(os.path.join(self.directory, '%s_optical_bw.png' % self.name_pattern), self.image)
            scipy.misc.imsave(os.path.join(self.directory, '%s_optical_rgb.png' % self.name_pattern), self.rgbimage.reshape((self.image.shape[0], self.image.shape[1], 3)))
            
    def save_parameters(self):
        parameters = self.get_parameters()
        f = open(os.path.join(self.directory, '%s_parameters.pickle' % self.name_pattern), 'w')
        pickle.dump(parameters, f)
        f.close()
    
    def get_parameters(self):
        return self.parameters
    
    def add_parameters(self, parameters=[]):
        for parameter in parameters:
            try:
                self.parameters[parameter] = getattr(self, 'get_%s' % parameter)()
            except AttributeError:
                self.parameters[parameter] = None
    
    def save_log(self, exclude_parameters=['image', 'rgb_image']):
        '''method to save the experiment details in the log file'''
        parameters = self.get_parameters()
        f = open(os.path.join(self.directory, '%s.log' % self.name_pattern), 'w')
        keyvalues = parameters.items()
        keyvalues.sort()
        for key, value in keyvalues:
            if key not in exclude_parameters:
                f.write('%s: %s\n' % (key, value)) 
        f.close()

    def store_ispyb(self):
        pass

    def store_lims(self, database, table, information_dictionary):
        d = sqlite3.connect(database)
        values, keys = [], []
        for k,v in iteritems(information_dictionary):
            values.append(v)
            keys.append(k)
        values = ','.join(values)
        keys = ','.join(keys)
        insert_dictionary =  {}
        insert_dictionary['table'] = table
        insert_dictionary['values'] = values
        insert_dictionary['keys'] = keys
        insert_statement='insert ({values}) into {table}({keys});'.format(**insert_dictionary)
        d.execute(insert_statement)
        d.commit()
        d.close()

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
        
        try:
            from instrument import instrument
            self.instrument = instrument()
        except ImportError:
            print 'Not possible to import instrument'
            return None

        return self.instrument.get_state()

    def check_directory(self, directory=None):
        if directory == None:
            directory = self.directory
        if os.path.isdir(directory):
            pass
        else:
            os.makedirs(directory)

    def write_destination_namepattern(self, directory, name_pattern, goimgfile='/927bis/ccd/log/.goimg/goimg.db'):
        try:
            f = open(goimgfile, 'w')
            f.write('%s %s' % (os.path.join(directory, 'process'), name_pattern))
            f.close()
        except IOError:
            logging.info('Problem writing goimg.db %s' % (traceback.format_exc()))
