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
import re
import time
import logging
import traceback
import gevent
import pickle
import scipy.misc
import subprocess
import pprint
from useful_routines import (
    get_string_from_timestamp,
    get_element,
    adjust_filename,
    get_pickled_file,
)

try:
    from cats import cats
except:
    cats = None

from speech import speech
from oav_camera import oav_camera as camera


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
        {"name": "run_number", "type": "int", "description": "run number"},
        {"name": "prefix", "type": "str", "description": "prefix"},
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
        run_number=None,
        cats_api=None,
        init_camera=True,
        default_experiment_name=None,
        port=5556,
    ):
        self.name = name
        if hasattr(self, "parameter_fields"):
            self.parameter_fields += experiment.specific_parameter_fields
        else:
            self.parameter_fields = experiment.specific_parameter_fields[:]

        self.default_experiment_name = default_experiment_name
        
        if not hasattr(self, "timestamp"):
            self.timestamp = time.time()

        if description is None and not hasattr(self, "description"):
            self.description = self.get_description()
        else:
            self.description = description

        prefix = os.path.basename(name_pattern)
        if run_number is None:
            print(f"setting name_pattern {prefix}")
            self.name_pattern = prefix
        else:
            print(f"setting name_pattern {prefix}_{run_number:d}")
            self.name_pattern = f"{prefix}_{run_number:d}"

        self.prefix = prefix
        self.run_number = run_number

        self.directory = os.path.abspath(directory).replace("nfslocal", "nfs")
        self.diagnostic = diagnostic
        self.analysis = analysis
        self.conclusion = conclusion
        self.simulation = simulation
        self.display = display
        self.snapshot = snapshot
        self.mxcube_parent_id = mxcube_parent_id
        self.mxcube_gparent_id = mxcube_gparent_id
        self.port = port
        
        self.results = {}

        self.process_directory = os.path.join(self.directory, "process")

        self.parameters = {}

        self.ispyb = speech(service="ispyb", server=False, port=self.port)

        self.logger = logging.getLogger("experiment")
        # logging.basicConfig(
        # format="%(asctime)s |%(module)s |%(levelname)s | %(message)s",
        ##datefmt="%Y-%m-%d %H:%M:%S",
        # level=logging.INFO,
        # )
        # stream_handler = logging.StreamHandler()
        # formatter = logging.Formatter(
        # fmt=("%(asctime)s |%(module)s |%(levelname)s | %(message)s")
        # )
        # stream_handler.setFormatter(formatter)
        # if not self.logger.handlers:
        # self.logger.handlers = [stream_handler]
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
        if init_camera:
            self.camera = camera()
        else:
            self.camera = camera(mode="vimba")
        if cats_api is None and cats is not None:
            self.sample_changer = cats()
        elif cats_api is not None:
            self.sample_changer = cats_api
        else:
            self.sample_changer = None
        
        
    def get_protect(get_method, *args):
        try:
            return get_method(*args)
        except:
            self.logger.error(traceback.format_exc())
            return None


    def get_default_experiment_name(self):
        if hasattr(self, "default_experiment_name") and self.default_experiment_name is None:
            self.default_experiment_name = re.findall("\<class \'.*\.(.*)\'>", str(self.__class__))[0].replace("_", " ").capitalize()
        return self.default_experiment_name
    
    def get_beamline_name(self):
        return "Proxima 2A, SOLEIL"
    
    def get_description(self):
        experiment = self.get_default_experiment_name()
        beamline = self.get_beamline_name()
        timestring = self.get_timestring(modify=False)
        description = f"{experiment}, {beamline}, {timestring}"
        return description
        
    def get_template(self):
        return os.path.join(self.directory, self.name_pattern)


    def get_element(self, puck=None, sample=None):
        if puck is None and self.puck is not None:
            puck = self.puck
        if sample is None and self.sample is not None:
            sample = self.sample
        element = get_element(puck, sample)
        return element 


    def get_timestring(self, timestamp=None, modify=True):
        if timestamp is None and self.timestamp is not None:
            timestamp = self.timestamp
        else:
            timestamp = time.time()
        timestring = get_string_from_timestamp(timestamp, modify=modify)
        return timestring
    
    
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
        print("in clean")
        self.save_optical_history()
        print(f"save_optical_history took {time.time() - _s:.3f} seconds")
        self.collect_parameters()
        clean_jobs = []
        clean_jobs.append(gevent.spawn(self.save_parameters))
        clean_jobs.append(gevent.spawn(self.save_log))
        if self.diagnostic == True:
            clean_jobs.append(gevent.spawn(self.save_diagnostics))
        gevent.joinall(clean_jobs)
        self.logger.info(f"clean took {time.time() - _s:.3f} seconds")

    def analyze(self):
        pass

    def conclude(self):
        pass

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
            self.logger.info("preparing ...")

            self.prepare()

            self.prepared = True
            self.end_prepare_time = time.time()
            self.prepare_duration = self.end_prepare_time - self.start_prepare_time
            self.logger.info("readied in %.4f seconds" % self.prepare_duration)

            if self.diagnostic == True:
                self.logger.info("starting monitors")
                self.start_monitor()
                self.logger.info("experiment monitors started")

            # run
            self.logger.info("executing the innermost part ...")

            self.start_run_time = time.time()

            self.run()

            self.executed = True
            self.end_run_time = time.time()
            self.run_duration = self.end_run_time - self.start_run_time
            self.logger.info(
                "the innermost part completed in %.4f seconds" % self.run_duration
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
            self.logger.info("cleaning after ...")
            if self.diagnostic == True:
                self.stop_monitor()
                self.logger.debug("experiment monitors stopped")

            self.logger.info("about to clean ...")
            self.clean()
            self.logger.info("cleaned!")

            self.cleaned = True
            self.end_clean_time = time.time()
            self.clean_duration = self.end_clean_time - self.start_clean_time
            self.logger.debug("cleaned after the in %.4f seconds" % self.clean_duration)

        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.logger.info("execution took %.4f seconds" % (self.duration))

        if self.analysis == True:
            self.start_analysis_time = time.time()
            self.analyze()
            try:
                self.save_results()
            except:
                self.logger.debug(traceback.format_exc())
            self.logger.debug("analysis finished")
            self.analyzed = True
            self.end_analysis_time = time.time()
            self.analysis_duration = self.end_analysis_time - self.start_analysis_time
        if self.conclusion == True:
            self.start_conclusion_time = time.time()
            self.conclude()
            self.logger.debug("concluded")
            self.concluded = True
            self.end_conclusion_time = time.time()
            self.conclusion_duration = (
                self.end_conclusion_time - self.start_conclusion_time
            )
        self.total_end_time = time.time()
        self.logger.info(
            "execution + analysis took %.4f seconds"
            % (self.total_end_time - self.start_time)
        )

    def get_end_run_time(self, beware_of_conclusion=True):
        end_time = self.end_run_time
        if beware_of_conclusion:
            if (
                hasattr(self, "end_conclusion_time")
                and self.end_conclusion_time is not None
            ):
                end_time = self.end_conclusion_time
        return end_time

    def save_optical_history(self):
        save_history_command = (
            "nice -n 99 history_saver.py -s %.2f -e %.2f -d %s -n %s -m 'modern' -c '%s' &"
            % (
                self.get_start_run_time(),
                self.get_end_run_time(),
                self.get_directory(),
                self.get_name_pattern(),
                str(self.cameras).replace("'", '"'),
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
            ] = self.camera.get_horizontal_calibration()
            self.parameters[
                "camera_calibration_vertical"
            ] = self.camera.get_vertical_calibration()
            self.parameters[
                "beam_position_vertical"
            ] = self.camera.get_beam_position_vertical()
            self.parameters[
                "beam_position_horizontal"
            ] = self.camera.get_beam_position_horizontal()
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
        return self.diagnostics

    def get_results(self):
        filename = self.get_results_filename()
        if os.path.isfile(filename):
            return self.get_pickled_file(filename)
        return self.results

    def get_results_filename(self):
        return "%s_results.pickle" % self.get_template()

    def get_parameters_filename(self):
        return "%s_parameters.pickle" % self.get_template()

    def get_diagnostics_filename(self):
        return "%s_diagnostics.pickle" % self.get_template()

    def get_log_filename(self):
        return "%s.log" % self.get_template()

    def get_cartography_filename(self, archive=True, ispyb=False):
        filename = f"{self.get_template()}.png"
        filename = adjust_filename(filename, archive=archive, ispyb=ispyb)
        return filename
    
    def get_csv_filename(self, archive=True, ispyb=False):
        filename = f"{self.get_template()}.csv"
        filename = adjust_filename(filename, archive=archive, ispyb=ispyb)
        return filename
    
    def get_pickled_file(self, filename, mode="rb"):
        return get_pickled_file(filename, mode=mode)

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

    def load_parameters_from_file(self):
        try:
            return self.get_pickled_file(self.get_parameters_filename())
        except IOError:
            traceback.print_exc()
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
        f.write("\n")
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
        try:
            server_status = self.get_server_status(server_name)
            if "running" and "you are the owner" in server_status:
                logging.getLogger("user_level_log").info("%s OK" % server_name)
            elif "%s is running. The owner is" % server_name in server_status:
                logging.getLogger("user_level_log").warning(
                    "%s is running but you are not the owner\nYou might consider restarting the %s server under your account"
                    % (server_name, server_name)
                )
            else:
                logging.getLogger("user_level_log").error(
                    "%s is NOT running" % server_name
                )
                logging.getLogger("user_level_log").info(
                    "Restarting the %s ..." % server_name
                )
                server_start = subprocess.getoutput("%s start &" % server_name)
                logging.getLogger("user_level_log").info(server_start)
        except:
            pass

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
        return self.camera.get_image()

    def get_rgbimage(self):
        if self.rgbimage is not None:
            return self.rgbimage
        return self.camera.get_rgbimage()

    def get_zoom(self):
        return self.camera.get_zoom()

    def get_mounted_sample(self):
        mounted_sample = None
        try:
            mounted_sample = self.sample_changer.get_mounted_puck_and_sample()
        except:
            print("could not determine mounted sample, please check")
            traceback.print_exc()
        return mounted_sample

    def get_proposal(self, number=20250023):
        proposal = self.ispyb.talk(
            {
                "get_proposal": {
                    "args": (
                        "mx",
                        number,
                    )
                }
            }
        )
        return proposal

    def store_data_collection_group(self, experiment_type="OSC"):
        group_data = {
            "sessionId": self.session_id,
            "experimentType": experiment_type,
        }
        group_id = self.ispyb.talk(
            {"_store_data_collection_group": {"args": (group_data,)}}
        )
        return group_id

    def get_samples(self, proposal_id=3113, session_id=46530):
        samples = self.ispyb.talk(
            {
                "get_samples": {
                    "args": (
                        proposal_id,
                        session_id,
                    )
                }
            }
        )
        return samples
