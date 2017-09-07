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

class experiment:

    def __init__(self, 
                 name_pattern=None, 
                 directory=None,
                 diagnostic=None,
                 analysis=None,
                 conclusion=None,
                 simulation=None):
        
        self.timestamp = time.time()
        self.name_pattern = name_pattern
        self.directory = directory
        self.diagnostic = diagnostic
        self.analysis = analysis
        self.conclusion = conclusion
        self.simulation = simulation
        
        self.process_directory = os.path.join(self.directory, 'process')
        
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
    def get_timestamp(self):
        return self.timestamp
    
    def get_user_id(self):
        return os.getuid()
    
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
        if self.conclusion == True:
            self.conclude()
            
        print 'experiment execute took %s' % (time.time() - self.start_time)
        
    #def save_log(self, template, experiment_information):
    def save_results(self):
        pass
    
    def save_parameters(self):
        pass
    
    def save_log(self):
        '''method to save the experiment details in the log file'''
        f = open(os.path.join(self.directory, '%s.log' % self.name_pattern), 'w')
        keyvalues = self.parameters.items()
        keyvalues.sort()
        for key, value in keyvalues:
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
