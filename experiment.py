#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""general experiment template. It will support all of the experimental methods we will ever come up with e.g.:

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

"""

import os
import time
import logging
import traceback
import gevent
import pickle
import scipy.misc
import subprocess
import pprint

#from camera import camera
from beamline import beamline

class experiment(object):
    """
    specific_parameter_fields = [
        {
            "name": ,
            "type": ,
            "description": ,
        },
    ]
    """
    specific_parameter_fields = [
        {
            "name": "name_pattern",
            "type": "str",
            "description": "root name of the files of all results of the experiment",
        },
        {
            "name": "directory",
            "type": "str",
            "description": "directory to store the results in",
        },
        {
            "name": "description",
            "type": "str",
            "description": "high level human readable designation of the  experiment",
        },
        {
            "name": "diagnostic",
            "type": "bool",
            "description": "whether or not to record diagnostic from available monitors",
        },
        {
            "name": "analysis",
            "type": "bool",
            "description": "whether or not to run analysis",
        },
        {
            "name": "conclusion",
            "type": "bool",
            "description": "whether or not to act upon analysis results (implies analysis=True)",
        },
        {
            "name": "simulation",
            "type": "bool",
            "description": "whether or not to run in simulation mode",
        },
        {
            "name": "display",
            "type": "bool",
            "description": "whether or not to display interactively any graphics generated during the experiment or its analysis",
        },
        {
            "name": "snapshot",
            "type": "bool",
            "description": "whether or not to take a snapshot of the sample",
        },
        {
            "name": "duration",
            "type": "float",
            "description": "duration of the experiment in seconds",
        },
        {
            "name": "prepare_duration",
            "type": "float",
            "description": "duration of the experiment in seconds",
        },
        {
            "name": "run_duration",
            "type": "float",
            "description": "duration of the run in seconds",
        },
        {
            "name": "analysis_duration",
            "type": "float",
            "description": "duration of the analysis in seconds",
        },
        {
            "name": "conclusion_duration",
            "type": "float",
            "description": "duration of conclusion experiment in seconds",
        },
        {
            "name": "clean_duration",
            "type": "float",
            "description": "duration of conclusion experiment in seconds",
        },
        {
            "name": "start_time",
            "type": "float",
            "description": "start of the execution the experiment in seconds (since the beginning of the epoch)",
        },
        {
            "name": "start_run_time",
            "type": "float",
            "description": "start of the run of the experiment in seconds (since the beginning of the epoch)",
        },
        {
            "name": "end_run_time",
            "type": "float",
            "description": "end of the run of the experiment in seconds (since the beginning of the epoch)",
        },
        {
            "name": "start_prepare_time",
            "type": "float",
            "description": "start of the prepare of the experiment in seconds (since the beginning of the epoch)",
        },
        {
            "name": "end_prepare_time",
            "type": "float",
            "description": "end of the prepare of the experiment in seconds (since the beginning of the epoch)",
        },
        {
            "name": "start_clean_time",
            "type": "float",
            "description": "start of the clean of the experiment in seconds (since the beginning of the epoch)",
        },
        {
            "name": "end_clean_time",
            "type": "float",
            "description": "end of the clean of the experiment in seconds (since the beginning of the epoch)",
        },
        {
            "name": "end_time",
            "type": "float",
            "description": "time of end of the experiment in seconds (since the beginning of the epoch)",
        },
        {
            "name": "timestamp",
            "type": "float",
            "description": "time of initialization of base class in seconds (since the beginning of the epoch)",
        },
        {"name": "user_id", "type": "int", "description": "User id"},
        {"name": "mxcube_parent_id", "type": "int", "description": ""},
        {"name": "mxcube_gparent_id", "type": "int", "description": ""},
        {"name": "mounted_sample", "type": "tuple", "description": "mounted_sample"},
        {"name": "cameras", "type": "list", "description": "cameras"},
    ]

    def __init__(
        self,
        name_pattern=None,
        directory=None,
        description=None,
        diagnostic=None,
        analysis=None,
        conclusion=None,
        simulation=None,
        display=None,
        snapshot=None,
        mxcube_parent_id=None,
        mxcube_gparent_id=None,
        name="experiment",
        cameras=[
            "sample_view",
            "goniometer",
            "cam14_quad",
        ],
        
    ):
        self.name = name
        if hasattr(self, "parameter_fields"):
            self.parameter_fields += experiment.specific_parameter_fields
        else:
            self.parameter_fields = experiment.specific_parameter_fields[:]

        if not hasattr(self, "timestamp"):
            self.timestamp = time.time()
            
        if description is None and not hasattr(self, "description"):
            self.description = "Experiment, Proxima 2A, SOLEIL, %s" % time.ctime(
                self.timestamp
            )
        else:
            self.description = description
        self.name_pattern = os.path.basename(name_pattern)
        self.directory = os.path.abspath(directory).replace("nfslocal", "nfs")
        self.diagnostic = diagnostic
        self.analysis = analysis
        self.conclusion = conclusion
        self.simulation = simulation
        self.display = display
        self.snapshot = snapshot
        self.mxcube_parent_id = mxcube_parent_id
        self.mxcube_gparent_id = mxcube_gparent_id
        self.results = {}

        self.process_directory = os.path.join(self.directory, "process")

        self.parameters = {}

        self.logger = logging.getLogger("experiment")
        # logging.basicConfig(
        # format="%(asctime)s |%(module)s |%(levelname)s | %(message)s",
        ##datefmt="%Y-%m-%d %H:%M:%S",
        # level=logging.INFO,
        # )
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt=("%(asctime)s |%(module)s |%(levelname)s | %(message)s")
        )
        stream_handler.setFormatter(formatter)
        if not self.logger.handlers:
            self.logger.handlers = [stream_handler]
        # self.logger.addHandler(stream_handler)

        self.logger.debug(
            "experiment __init__ len(experiment.specific_parameter_fields) %d"
            % len(experiment.specific_parameter_fields)
        )
        self.logger.debug(
            "experiment __init__ len(self.parameters_fields) %d"
            % len(self.parameter_fields)
        )

        k = 0
        for parameter in self.parameter_fields:
            k += 1
            getter = "get_%s" % parameter["name"]
            if not hasattr(self, getter):
                getter_code = '''def get_{name:s}(self): 
                                    """
                                    Gets {name:s}: {description}
                            
                                    :param None
                                    :returns {name:s}
                                    :rtype {type:s}
                                    """
                                    return self.{name:s}'''.format(
                    **parameter
                )

                exec(getter_code.strip())
                setattr(self.__class__, getter, locals()[getter])

            setter = "set_%s" % parameter["name"]
            if not hasattr(self, setter):
                setter_code = '''def set_{name:s}(self, {name:s}): 
                                    """
                                    Sets {name:s}: {description}
                            
                                    :param {name:s}
                                    :type {type:s}
                                    """
                                    self.{name:s} = {name:s}'''.format(
                    **parameter
                )

                exec(setter_code.strip())
                setattr(self.__class__, setter, locals()[setter])
        
        self.cameras = cameras
        if not hasattr(self, "instrument"):
            self.instrument = beamline()
        
    def get_protect(get_method, *args):
        try:
            return get_method(*args)
        except:
            self.logger.error(traceback.format_exc())
            return None

    def get_template(self):
        return os.path.join(self.directory, self.name_pattern)

    def get_full_name_pattern(self):
        full_name_pattern = "/".join(
            (
                "",
                str(self.get_user_id()),
                self.get_directory()[1:],
                self.get_name_pattern(),
            )
        )
        return full_name_pattern

    def get_directory(self):
        return self.directory

    def get_process_directory(self):
        return "%s/process" % self.get_directory()

    def get_user_id(self):
        return os.getuid()

    def get_login(self):
        try:
            login = os.getlogin()
        except:
            try:
                login = subprocess.getoutput("whoami")
            except:
                login = "com-proixma2a"
        return login

    # def get_duration(self):
    ##duration = self.get_end_time() - self.get_start_time()
    # return self.end_time - self.start_time

    def prepare(self):
        self.check_camera()
        self.check_directory()

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
        _s = time.time()
        self.save_optical_history()
        _e = time.time()
        print(f"save_optical_history took {_e - _s:.3f} seconds")
        self.collect_parameters()
        clean_jobs = []
        clean_jobs.append(gevent.spawn(self.save_parameters))
        clean_jobs.append(gevent.spawn(self.save_log))
        gevent.joinall(clean_jobs)

    def analyze(self):
        pass

    def conclude(self):
        pass

    def get_results(self):
        return self.results

    def save_results(self, mode="wb"):
        _start = time.time()
        if len(self.results) == 0:
            try:
                self.results = self.get_results()
            except:
                self.results = []
                self.logger.debug(traceback.format_exc())
        if self.results:
            f = open(self.get_results_filename(), mode)
            pickle.dump(self.results, f)
            f.close()
        self.logger.debug("save_results took %.4f" % (time.time() - _start))

    def store_ispyb(self):
        pass

    def execute(self):
        self.start_time = time.time()
        self.prepared = False
        self.executed = False
        self.completed = False
        self.cleaned = False
        self.analyzed = False
        self.concluded = False

        try:
            # prepare
            self.start_prepare_time = time.time()
            self.logger.info("preparing the experiment ...")

            self.prepare()

            self.prepared = True
            self.end_prepare_time = time.time()
            self.prepare_duration = self.end_prepare_time - self.start_prepare_time
            self.logger.info(
                "experiment prepared in %.4f seconds" % self.prepare_duration
            )

            if self.diagnostic == True:
                self.logger.info("starting monitors")
                self.start_monitor()
                self.logger.info("experiment monitors started")
                
            # run
            self.logger.info("about to perform innermost part of the experiment ...")
            
            self.start_run_time = time.time()

            self.run()

            self.executed = True
            self.end_run_time = time.time()
            self.run_duration = self.end_run_time - self.start_run_time
            self.logger.info(
                "innermost part of the experiment completed in %.4f seconds"
                % self.run_duration
            )

            self.completed = True

        except:
            self.logger.debug(
                "Problem in preparation or execution %s" % self.__module__
            )
            self.logger.info(traceback.print_exc())

        finally:
            # clean
            self.start_clean_time = time.time()
            self.logger.info("cleaning after the experiment ...")
            if self.diagnostic == True:
                self.stop_monitor()
                self.logger.debug("experiment monitors stopped")

            self.clean()

            self.cleaned = True
            self.end_clean_time = time.time()
            self.clean_duration = self.end_clean_time - self.start_clean_time
            self.logger.debug("cleaned after the in %.4f seconds" % self.clean_duration)

        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.logger.info("experiment took %.4f seconds" % (self.duration))

        if self.analysis == True:
            self.start_analysis_time = time.time()
            self.analyze()
            try:
                self.save_results()
            except:
                self.logger.debug(traceback.format_exc())
            self.logger.debug("experiment analysis finished")
            self.analyzed = True
            self.end_analysis_time = time.time()
            self.analysis_duration = self.end_analysis_time - self.start_analysis_time
        if self.conclusion == True:
            self.start_conclusion_time = time.time()
            self.conclude()
            self.logger.debug("experiment conclusion finished")
            self.concluded = True
            self.end_conclusion_time = time.time()
            self.conclusion_duration = (
                self.end_conclusion_time - self.start_conclusion_time
            )
        self.total_end_time = time.time()
        self.logger.info(
            "experiment execute took %.4f seconds"
            % (self.total_end_time - self.start_time)
        )

    def get_end_run_time(self, beware_of_conclusion=True):
        end_time = self.end_run_time
        if beware_of_conclusion:
            if hasattr(self, "conclusion_end_time"):
                end_time = self.end_conclusion_time
        return end_time

    #def save_optical_history(self):
        #save_history_command = (
            #"history_saver.py -s %.2f -e %.2f -d %s -n %s -S _sample_view &"
            #% (
                #self.get_start_run_time(),
                #self.get_end_run_time(),
                #self.get_directory(),
                #self.get_name_pattern(),
            #)
        #)
        #self.logger.debug("save_optical_history_command: %s" % save_history_command)
        #os.system(save_history_command)
    
    def save_optical_history(self):
        save_history_command = (
            "history_saver.py -s %.2f -e %.2f -d %s -n %s -m 'modern' -c '%s' &"
            % (
                self.get_start_run_time(),
                self.get_end_run_time(),
                self.get_directory(),
                self.get_name_pattern(),
                str(self.cameras).replace("'", "\""),
            )
        )
        self.logger.debug("save_optical_history_command: %s" % save_history_command)
        os.system(save_history_command)

    def collect_parameters(self):
        _start = time.time()
        self.logger.debug(
            "collect_parameters len(self.parameters_fields) %d"
            % len(self.parameter_fields)
        )
        for parameter in self.parameter_fields:
            # if parameter['name'] not in ['slit_configuration', 'xbpm_readings']:
            try:
                self.parameters[parameter["name"]] = getattr(
                    self, "get_%s" % parameter["name"]
                )()
            except:
                self.logger.debug(
                    "experiment collect_parameters %s" % traceback.format_exc()
                )
                self.logger.debug("parameter %s" % parameter["name"])
                self.parameters[parameter["name"]] = None

            # elif parameter['name'] == 'slit_configuration':
            # slit_configuration = self.get_slit_configuration()
            # for key in slit_configuration:
            # self.parameters[key] = slit_configuration[key]
            # elif parameter['name'] == 'xbpm_readings':
            # xbpm_readings = self.get_xbpm_readings()
            # for key in xbpm_readings:
            # self.parameters[key] = xbpm_readings[key]

        if self.snapshot == True:
            self.parameters["camera_zoom"] = self.get_zoom()
            self.parameters[
                "camera_calibration_horizontal"
            ] = self.instrument.camera.get_horizontal_calibration()
            self.parameters[
                "camera_calibration_vertical"
            ] = self.instrument.camera.get_vertical_calibration()
            self.parameters[
                "beam_position_vertical"
            ] = self.instrument.camera.get_beam_position_vertical()
            self.parameters[
                "beam_position_horizontal"
            ] = self.instrument.camera.get_beam_position_horizontal()
            self.parameters["image"] = self.get_image()
            self.parameters["rgbimage"] = self.get_rgbimage()
        self.logger.debug(
            "collect_parameters took %.4f seconds" % (time.time() - _start)
        )

    def get_parameters(self):
        filename = self.get_parameters_filename()
        if os.path.isfile(filename):
            return self.get_pickled_file(filename)
        return self.parameters

    def get_diagnostics(self):
        filename = self.get_diagnostics_filename()
        if os.path.isfile(filename):
            return self.get_pickled_file(filename)

    def get_results_filename(self):
        return "%s_results.pickle" % self.get_template()

    def get_parameters_filename(self):
        return "%s_parameters.pickle" % self.get_template()

    def get_diagnostics_filename(self):
        return "%s_diagnostics.pickle" % self.get_template()

    def get_log_filename(self):
        return "%s.log" % self.get_template()

    def get_pickled_file(self, filename, mode="rb"):
        try:
            try:
                pickled_file = pickle.load(open(filename, mode))
            except:
                pickled_file = pickle.load(open(filename, mode), encoding="latin1")
        except IOError:
            pickled_file = None
        return pickled_file

    def save_parameters(self, mode="wb"):
        _start = time.time()
        parameters = self.get_parameters()
        f = open(self.get_parameters_filename(), mode)
        pickle.dump(parameters, f)
        f.close()
        self.logger.debug("save_parameters took %.4f seconds" % (time.time() - _start))

    def save_diagnostics(self, mode="wb"):
        _start = time.time()
        f = open(self.get_diagnostics_filename(), mode)
        self.logger.debug(
            "opening diagnostics file took %.4f seconds" % (time.time() - _start)
        )
        _d_start = time.time()
        pickle.dump(self.get_all_observations(), f)
        self.logger.debug(
            "writing to diagnostics file took %.4f seconds" % (time.time() - _d_start)
        )
        f.close()
        _c_start = time.time()
        self.logger.debug(
            "closing diagnostics file took %.4f seconds" % (time.time() - _c_start)
        )
        self.logger.debug("save_diagnostics took %.4f seconds" % (time.time() - _start))

    def load_parameters_from_file(self):
        try:
            return self.get_pickled_file(self.get_parameters_filename())
        except IOError:
            return None

    def add_parameters(self, parameters=[]):
        for parameter in parameters:
            try:
                self.parameters[parameter] = getattr(self, "get_%s" % parameter)()
            except AttributeError:
                self.parameters[parameter] = None

    def print_dictionary(self, parameters, prepend="", prepend_increment=8 * " "):
        dict_string = ""
        keyvalues = list(parameters.items())
        keyvalues.sort()
        for key, value in keyvalues:
            if type(value) is not dict:
                dict_string += "%s%s: %s\n" % (prepend, key, value)
            else:
                dict_string += "%s%s:\n" % (prepend, key)
                dict_string += self.print_dictionary(
                    value, prepend=prepend + prepend_increment
                )

        return dict_string

    def get_log_string(self, parameters, pp=True):
        log_string = ""
        if pp:
            log_string += pprint.pformat(parameters)
        else:
            log_string += self.print_dictionary(parameters)
        return log_string

    def save_log(self, exclude_parameters=["image", "rgbimage"], mode="w"):
        """method to save the experiment details in the log file"""
        _start = time.time()
        parameters = self.get_parameters()
        reduced_parameters = {}
        for parameter in parameters:
            if parameter not in exclude_parameters:
                reduced_parameters[parameter] = parameters[parameter]
        f = open(self.get_log_filename(), mode)
        f.write(self.get_log_string(reduced_parameters))
        f.close()
        self.logger.debug("save_log took %.4f seconds" % (time.time() - _start))

    def store_lims(self, database, table, information_dictionary):
        d = sqlite3.connect(database)
        values, keys = [], []
        for k, v in iteritems(information_dictionary):
            values.append(v)
            keys.append(k)
        values = ",".join(values)
        keys = ",".join(keys)
        insert_dictionary = {}
        insert_dictionary["table"] = table
        insert_dictionary["values"] = values
        insert_dictionary["keys"] = keys
        insert_statement = "insert ({values}) into {table}({keys});".format(
            **insert_dictionary
        )
        d.execute(insert_statement)
        d.commit()
        d.close()

    def get_instrument_configuration(self):
        """the purpose of this method is to gather and return all relevant information about the beamline and the machine

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
        """

        try:
            from instrument import instrument

            self.instrument = instrument()
        except ImportError:
            self.logger.debug("Not possible to import instrument")
            return None

        return self.instrument.get_state()

    def check_directory(self, directory=None):
        if directory == None:
            directory = self.directory
        if os.path.isdir(directory):
            pass
        else:
            os.makedirs(directory)

    def write_destination_namepattern(
        self, directory, name_pattern, goimgfile="/nfs/data2/log/goimg.db"
    ):  # /927bis/ccd/log/.goimg/goimg.db'
        try:
            f = open(goimgfile, "w")
            f.write("%s %s" % (directory, name_pattern))
            f.close()
        except IOError:
            self.logger.debug("Problem writing goimg.db %s" % (traceback.format_exc()))

    def check_server(self, server_name):
        server_status = self.get_server_status(server_name)
        if "running" and "you are the owner" in server_status:
            logging.getLogger("user_level_log").info("%s OK" % server_name)
        elif "%s is running. The owner is" % server_name in server_status:
            logging.getLogger("user_level_log").warning(
                "%s is running but you are not the owner\nYou might consider restarting the %s server under your account"
                % (server_name, server_name)
            )
        else:
            logging.getLogger("user_level_log").error("%s is NOT running" % server_name)
            logging.getLogger("user_level_log").info(
                "Restarting the %s ..." % server_name
            )
            server_start = subprocess.getoutput("%s start &" % server_name)
            logging.getLogger("user_level_log").info(server_start)

    def get_server_status(self, server_name):
        server_status = subprocess.getoutput("%s status" % server_name)
        return server_status

    def check_camera(self):
        return
        self.check_server("camera")

    def check_hbpc(self):
        self.check_server("hbpc")

    def check_vbpc(self):
        self.check_server("vbpc")
        
    def get_image(self):
        if self.image is not None:
            return self.image
        return self.instrument.camera.get_image()

    def get_rgbimage(self):
        if self.rgbimage is not None:
            return self.rgbimage
        return self.instrument.camera.get_rgbimage()

    def get_zoom(self):
        return self.instrument.camera.get_zoom()

    def get_mounted_sample(self):
        mounted_sample = None
        try:
            mounted_sample = self.instrument.sample_changer.get_mounted_puck_and_sample()
        except:
            print("could not determine mounted sample, please check")
            traceback.print_exc()
        return mounted_sample
    
    
    
