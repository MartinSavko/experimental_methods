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

class experiment(object):
    
    specific_parameter_fields = [{'name': 'name_pattern', 'type': 'str', 'description': 'root name of the files of all results of the experiment'}, 
                                 {'name': 'directory', 'type': 'str', 'description': 'directory to store the results in'},
                                 {'name': 'description', 'type': 'str', 'description': 'high level human readable designation of the  experiment'},
                                 {'name': 'diagnostic', 'type': 'bool', 'description': 'whether or not to record diagnostic from available monitors'}, 
                                 {'name': 'analysis', 'type': 'bool', 'description': 'whether or not to run analysis'}, 
                                 {'name': 'conclusion', 'type': 'bool', 'description': 'whether or not to act upon analysis results (implies analysis=True)'}, 
                                 {'name': 'simulation', 'type': 'bool', 'description': 'whether or not to run in simulation mode'}, 
                                 {'name': 'display', 'type': 'bool', 'description': 'whether or not to display interactively any graphics generated during the experiment or its analysis'},
                                 {'name': 'snapshot', 'type': 'bool', 'description': 'whether or not to take a snapshot of the sample'},
                                 {'name': 'duration', 'type': 'float', 'description': 'duration of the experiment in seconds'},
                                 {'name': 'start_time', 'type': 'float', 'description': 'start of the execution the experiment in seconds (since the beginning of the epoch)'},
                                 {'name': 'end_time', 'type': 'float', 'description': 'time of end of the experiment in seconds (since the beginning of the epoch)'},
                                 {'name': 'timestamp', 'type': 'float', 'description': 'time of initialization of base class in seconds (since the beginning of the epoch)'},
                                 {'name': 'user_id', 'type': 'int', 'description': 'User id'}]
                
        
    def __init__(self, 
                 name_pattern=None, 
                 directory=None,
                 description=None,
                 diagnostic=None,
                 analysis=None,
                 conclusion=None,
                 simulation=None,
                 display=None,
                 snapshot=None):
        
        self.parameter_fields += experiment.specific_parameter_fields
        
        self.timestamp = time.time()
        if description == None:
            self.description = 'Experiment, Proxima 2A, SOLEIL, %s' % time.ctime(self.timestamp)
        else:
            self.description = description
        self.name_pattern = os.path.basename(name_pattern)
        self.directory = os.path.abspath(directory).replace('nfslocal', 'nfs')
        self.diagnostic = diagnostic
        self.analysis = analysis
        self.conclusion = conclusion
        self.simulation = simulation
        self.display = display
        self.snapshot = snapshot
        
        #if type(self.directory) == str:
        self.process_directory = os.path.join(self.directory, 'process')
        
        self.parameters = {}
       
        #print 'len self.parameter_fileds'
        #print len(self.parameter_fields)
        
        k = 0 
        for parameter in self.parameter_fields:
            k+=1
            #print 'k %d' % k
            
            getter = 'get_%s' % parameter['name']
            if not hasattr(self, getter):
                
                #print 'creating %s' % getter
                getter_code = '''def get_{name:s}(self): 
                                    """
                                    Gets {name:s}: {description}
                            
                                    :param None
                                    :returns {name:s}
                                    :rtype {type:s}
                                    """
                                    return self.{name:s}'''.format(**parameter)
                
                result = {}
                exec getter_code.strip() in result
                setattr(self.__class__, getter, result[getter])
            #else:
                #print '%s already created' % getter
            
            setter = 'set_%s' % parameter['name']
            if not hasattr(self, setter):
                #print 'creating %s' % setter
                setter_code = '''def set_{name:s}(self, {name:s}): 
                                    """
                                    Sets {name:s}: {description}
                            
                                    :param {name:s}
                                    :type {type:s}
                                    """
                                    self.{name:s} = {name:s}'''.format(**parameter)
                
                result = {}
                exec setter_code.strip() in result
                setattr(self.__class__, setter, result[setter])
            #else:
                #print '%s already created' % setter
            
    
    def get_protect(get_method, *args):
        try:
            return get_method(*args)
        except:
            logging.error(traceback.print_exc())
            return None


    def get_template(self):
        return os.path.join(self.directory, self.name_pattern)
        

    def get_full_name_pattern(self):
        full_name_pattern = '/'.join(('', str(self.get_user_id()), self.get_directory()[1:],  self.get_name_pattern()))
        return full_name_pattern
    

    def get_user_id(self):
        return os.getuid()
        
    
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


    def save_results(self):
        pass


    def store_ispyb(self):
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



    def collect_parameters(self):
        for parameter in self.parameter_fields:
            if parameter['name'] != 'slit_configuration':
                try:
                    self.parameters[parameter['name']] = getattr(self, 'get_%s' % parameter['name'])()
                except:
                    logging.info('experiment collect_parameters %s' % traceback.format_exc())
                    logging.info('parameter %s' % parameter['name'])
                    self.parameters[parameter['name']] = None
                    
            else:
                slit_configuration = self.get_slit_configuration()
                for key in slit_configuration:
                    self.parameters[key] = slit_configuration[key]
        
        if self.snapshot == True:
            self.parameters['camera_zoom'] = self.get_zoom()
            self.parameters['camera_calibration_horizontal'] = self.camera.get_horizontal_calibration()
            self.parameters['camera_calibration_vertical'] = self.camera.get_vertical_calibration()
            self.parameters['beam_position_vertical'] = self.camera.get_beam_position_vertical()
            self.parameters['beam_position_horizontal'] = self.camera.get_beam_position_horizontal()
            self.parameters['image'] = self.get_image()
            self.parameters['rgbimage'] = self.get_rgbimage()
        

    def get_parameters(self):
        return self.parameters

    def get_results_filename(self):
        return '%s_results.pickle' % self.get_template()
    
    
    def get_parameters_filename(self):
        return '%s_parameters.pickle' % self.get_template()
    
    
    def save_parameters(self):
        parameters = self.get_parameters()
        f = open(self.get_parameters_filename(), 'w')
        pickle.dump(parameters, f)
        f.close()
    

    def load_parameters_from_file(self):
        try:
            return pickle.load(open(os.path.join(self.directory, '%s_parameters.pickle' % self.name_pattern)))
        except IOError:
            return None
        

    def add_parameters(self, parameters=[]):
        for parameter in parameters:
            try:
                self.parameters[parameter] = getattr(self, 'get_%s' % parameter)()
            except AttributeError:
                self.parameters[parameter] = None
    

    def save_log(self, exclude_parameters=['image', 'rgbimage']):
        '''method to save the experiment details in the log file'''
        parameters = self.get_parameters()
        f = open(os.path.join(self.directory, '%s.log' % self.name_pattern), 'w')
        keyvalues = parameters.items()
        keyvalues.sort()
        for key, value in keyvalues:
            if key not in exclude_parameters:
                f.write('%s: %s\n' % (key, value)) 
        f.close()


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

    
    def write_destination_namepattern(self, directory, name_pattern, goimgfile='/927bis/ccd/log/.goimg/goimg.db'): #/927bis/ccd/log/.goimg/goimg.db'
        try:
            f = open(goimgfile, 'w')
            f.write('%s %s' % (os.path.join(directory, 'process'), name_pattern))
            f.close()
        except IOError:
            logging.info('Problem writing goimg.db %s' % (traceback.format_exc()))

