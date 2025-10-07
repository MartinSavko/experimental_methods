#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import pickle
import logging
import traceback
import h5py
import gevent
import datetime

import re
import glob
import json
import gzip

import numpy as np

sys.path.insert(
    -1, "/home/experiences/proxima2a/com-proxima2a/.local/lib/python3.11/site-packages"
)
import cv2
import subprocess
from scipy.spatial import distance_matrix
import scipy.interpolate as si
import scipy.ndimage as ndi


from experimental_methods.utils.perfect_realignment import (
    get_both_extremes_from_pcd,
    get_likely_part,
    get_position_from_vector,
)
from experimental_methods.utils.useful_routines import (
    get_ddv, 
    get_d_min_for_ddv,
    get_resolution_from_distance,
    get_vector_from_position
)
from experimental_methods.experiment.experiment import experiment

class diffraction_experiment_analysis(experiment):
    def __init__(
        self,
        name_pattern,
        directory,
        spot_threshold=7,
    ):
        experiment.__init__(
            self,
            name_pattern=name_pattern,
            directory=directory,
            init_camera=False,
        )

        self.spot_threshold = spot_threshold

        self.format_dictionary = {
            "directory": self.directory,
            "name_pattern": self.name_pattern,
        }

        self.timestamp = time.time()

        self.description = (
            "Diffraction experiment analysis, Proxima 2A, SOLEIL, %s"
            % time.ctime(self.timestamp)
        )

        self.dozor_launched = 0

        self.match_number_in_spot_file = re.compile(".*([\d]{6}).adx.gz")
        self.match_number_in_cbf = re.compile(".*([\d]{6}).cbf.gz")
        
        self.lines = {}
        self.spots_per_line = {}
        self.spots_per_frame = {}

    def get_master(self):
        master = h5py.File(self.get_master_filename(), "r")
        return master

    def get_master_filename(self):
        return "%s_master.h5" % self.get_template()

    def get_parameters(self):
        if os.path.isfile(self.get_parameters_filename()):
            if self.parameters == {}:
                self.parameters = self.load_parameters_from_file()
        if self.parameters == {} and os.path.isfile(self.get_master_filename()):
            parameters = {}
            master = self.get_master()
            for key in ["detector_distance", "beam_center_x", "beam_center_y"]:
                parameters[key] = master["/entry/instrument/detector/%s" % key][()]
            for key in ["nimages", "ntrigger"]:
                parameters[key] = int(
                    master["/entry/instrument/detector/detectorSpecific/%s" % key][()]
                )
            parameters["wavelength"] = master["/entry/sample/beam/incident_wavelength"][
                ()
            ]
            parameters["detector_distance"] *= 1e3
            parameters[
                "resolution"
            ] = self.resolution_motor.get_resolution_from_distance(
                parameters["detector_distance"], parameters["wavelength"]
            )
            parameters["timestamp"] = datetime.datetime.timestamp(
                datetime.datetime.fromisoformat(
                    master[
                        "/entry/instrument/detector/detectorSpecific/data_collection_date"
                    ][()].decode()
                )
            )
            self.parameters = parameters
        return self.parameters

    def get_total_number_of_images(self):
        return self.get_nimages() * self.get_ntrigger()

    def get_expected_files(self):
        expected_files = ["%s_master.h5" % self.name_pattern]
        nimages_per_file = self.get_nimages_per_file()
        total_number_of_images = self.get_total_number_of_images()
        accounted_for = 0
        k = 0
        while accounted_for < total_number_of_images:
            k += 1
            data_file = "%s_data_%06d.h5" % (self.name_pattern, k)
            expected_files.append(data_file)
            accounted_for += nimages_per_file
        expected_files.sort()
        return expected_files

    def get_downloaded_files(self, expected_files=None, directory=None):
        if expected_files is None:
            expected_files = self.get_expected_files()
        if directory is None:
            directory = self.directory
        listdir = os.listdir(directory)
        already = []
        for f in expected_files:
            if f in listdir:
                already.append(f)
        return already

    def expected_files_present(self, expected_files=None, directory=None):
        if expected_files is None:
            expected_files = self.get_expected_files()
        if directory is None:
            directory = self.directory
        listdir = os.listdir(directory)
        for f in expected_files:
            if f not in listdir:
                return False
        return True

    def get_download_progress(self, expected_files=None, directory=None):
        if expected_files is None:
            expected_files = self.get_expected_files()
        if directory is None:
            directory = self.directory
        already = self.get_downloaded_files(
            expected_files=expected_files, directory=directory
        )
        download_progress = 100 * len(already) / len(expected_files)
        return download_progress

    def wait_for_expected_files(
        self,
        timeout=60,
        naptime=0.5,
        expected_files=None,
        directory=None,
        symbols=["-", "\\", "|", "/"],
    ):
        if expected_files is None:
            expected_files = self.get_expected_files()
        if directory is None:
            direcotry = self.directory
        self.logger.info("waiting for expected files %s" % expected_files)
        _start = time.time()
        k = 0
        while (
            not self.expected_files_present(
                expected_files=expected_files, directory=directory
            )
            and time.time() - _start < timeout
        ):
            print(
                "{progress:.1f}% {symbol:s}".format(
                    progress=self.get_download_progress(expected_files, directory),
                    symbol="%s\r" % symbols[k % len(symbols)],
                ),
                end=" ",
            )
            sys.stdout.flush()
            gevent.sleep(naptime)
            k += 1

        self.logger.info(
            "wait for expected files took %.4f seconds" % (time.time() - _start)
        )
        
    def _get_parameter(self, parameter_name):
        parameter = None
        if parameter_name in self.get_parameters():
            parameter = self.get_parameters()[parameter_name]
        elif hasattr(self, parameter_name):
            parameter = int(getattr(self, parameter_name))
        return parameter

    def get_nimages_per_file(self):
        return self._get_parameter("nimages_per_file")
    
    def get_reference_position(self):
        return self._get_parameter("reference_position")

    def get_nimages(self):
        return self._get_parameter("nimages")

    def get_ntrigger(self):
        return self._get_parameter("ntrigger")

    def get_overlap(
        self, overlap=0.0, scan_start_angles=[], scan_range=0.0, min_scan_range=0.025
    ):
        parameters = self.get_parameters()

        if "scan_start_angles" in parameters:
            scan_start_angles = parameters["scan_start_angles"]
            scan_range = parameters["scan_range"]
        elif hasattr(self, "scan_start_angles"):
            scan_start_angles = self.get_scan_start_angles()
            scan_range = self.get_scan_range()
        if scan_range < min_scan_range:
            scan_range = 0.0
        if len(scan_start_angles) > 1:
            delta = scan_start_angles[0] - scan_start_angles[1]
            overlap = delta + scan_range

        self.logger.info("get_overlap returns %.2f" % overlap)
        return overlap

    def get_dozor_directory(self):
        return os.path.join(f"{self.directory}/process/dozor_{self.name_pattern}")

    def get_dozor_control_card_filename(self):
        return "%s.dat" % self.name_pattern

    def get_dozor_log_filename(self):
        dozor_control_card_filename = self.get_dozor_control_card_filename()
        dozor_log_filename = dozor_control_card_filename.replace(".dat", "_dozor.log")
        return dozor_log_filename

    def create_dozor_control_card(self, dozor_major_version=2):
        """
        dozor_major_version == 1
        dozor parameters
        !-----------------
        detector eiger9m
        !
        exposure 0.025
        spot_size 3
        detector_distance 200
        X-ray_wavelength 0.9801
        fraction_polarization 0.99
        pixel_min 0
        pixel_max 100000
        !
        ix_min 1
        ix_max 1067
        iy_min 1029
        iy_max 1108
        !
        orgx 1454.26
        orgy 1731.02
        oscillation_range 0.001
        image_step 1
        starting_angle 75
        !
        first_image_number 1
        number_images 1
        name_template_image mesh1_?????.cbf

        dozor_major_version == 2
        dozor parameters
        !-----------------
        detector eiger9m
        nx 3110
        ny 3269
        pixel 0.075
        exposure 0.0043
        detector_distance 115
        X-ray_wavelength 0.980133704735
        fraction_polarization 0.99
        orgx 1453
        orgy 1731
        oscillation_range 0.1
        starting_angle 0.0
        number_images 3600
        first_image_number 1
        name_template_image /dev/shm/s1_1_h5/s1_1_??????.h5
        library /nfs/data/plugin.so
        pixel_min 0
        pixel_max 10712
        spot_size 3

        beamstop_distance 20
        beamstop_size 0.5
        beamstop_vertical 0
        image_step 1
        end

        """

        try:
            os.makedirs(self.get_dozor_directory())
        except OSError:
            pass

        parameters = self.get_parameters()
        if "frame_time" not in parameters:
            parameters["frame_time"] = 1.0 / parameters["frames_per_second"]
        if "angle_per_frame" not in parameters:
            parameters["angle_per_frame"] = 0.001

        if parameters["angle_per_frame"] <= 0.001:
            parameters["angle_per_frame_dozor"] = 0.0
        else:
            parameters["angle_per_frame_dozor"] = parameters["angle_per_frame"]

        if "scan_start_angles" in parameters:
            starting_angle = parameters["scan_start_angles"][0]
        elif "scan_start_angle" in parameters:
            starting_angle = parameters["scan_start_angle"]
        else:
            starting_angle = 0.0

        if "nimages" in parameters:
            self.nimages = parameters["nimages"]
        else:
            self.nimages = (
                parameters["total_expected_exposure_time"] / parameters["frame_time"]
            )
        if "ntrigger" in parameters:
            self.ntrigger = parameters["ntrigger"]
        elif "scan_start_angles" in parameters:
            self.ntrigger = len(parameters["scan_start_angles"])
        else:
            self.ntrigger = 1

        name_template_image = (
            f"{self.get_cbf_directory()}/{self.name_pattern}_??????.cbf.gz"
        )

        self.logger.info(
            "deciding whether to create ordered cbf links get_overlap: %.4f, angle_per_frame_dozor: %.4f"
            % (self.get_overlap(), parameters["angle_per_frame_dozor"])
        )

        if self.get_overlap() != 0.0:
            self.create_ordered_cbf_links()
            name_template_image = os.path.join(
                self.get_dozor_directory(),
                "{name_pattern}_ordered_??????.cbf.gz".format(**self.format_dictionary),
            )

        dozor_parameters = {
            "detector": "eiger9m",
            "exposure": parameters["frame_time"],
            "spot_size": 3,
            "nx": 3110,
            "ny": 3269,
            "pixel": 0.075,
            "detector_distance": parameters["detector_distance"],
            "X-ray_wavelength": parameters["wavelength"],
            "fraction_polarization": 0.99,
            "pixel_min": 0,
            "pixel_max": self.detector.get_countrate_correction_count_cutoff(),
            "ix_min": 1,
            "ix_max": int(parameters["beam_center_x"]) + 50,
            "iy_min": int(parameters["beam_center_y"]) - 50,
            "iy_max": int(parameters["beam_center_y"]) + 50,
            "orgx": int(parameters["beam_center_x"]),
            "orgy": int(parameters["beam_center_y"]),
            "oscillation_range": parameters["angle_per_frame_dozor"],
            "image_step": 1,
            "beamstop_distance": 20,
            "beamstop_size": 1.0,
            "beamstop_vertical": 0,
            "starting_angle": starting_angle,
            "first_image_number": 1,
            "library": "/nfs/data/xds-zcbf.so",
            "number_images": self.nimages * self.ntrigger,
            "name_template_image": name_template_image,
        }

        input_file = "dozor parameters\n"
        input_file += "!-----------------\n"
        for parameter in [
            "detector",
            "exposure",
            "spot_size",
            "detector_distance",
            "X-ray_wavelength",
            "fraction_polarization",
            "pixel_min",
            "pixel_max",
            "orgx",
            "orgy",
            "oscillation_range",
            "image_step",
            "starting_angle",
            "first_image_number",
            "number_images",
            "name_template_image",
        ]:
            input_file += "%s %s\n" % (parameter, dozor_parameters[parameter])

        if dozor_major_version < 2:
            for parameter in ["ix_min", "ix_max", "iy_min", "iy_max"]:
                input_file += "%s %s\n" % (parameter, dozor_parameters[parameter])
        else:
            for parameter in [
                "nx",
                "ny",
                "pixel",
                "beamstop_distance",
                "beamstop_size",
                "beamstop_vertical",
                "library",
            ]:
                input_file += "%s %s\n" % (parameter, dozor_parameters[parameter])

        input_file += "end\n"

        f = open(
            os.path.join(
                self.get_dozor_directory(), self.get_dozor_control_card_filename()
            ),
            "w",
        )
        f.write(input_file)
        f.close()

    def get_cbf_directory(self):
        # cbf_directory = '{directory}/{name_pattern}_cbf'.format(**self.format_dictionary)
        cbf_directory = f"{self.directory}"
        return cbf_directory

    def get_cbf_template(self):
        return os.path.join(
            self.get_cbf_directory(), f"{self.name_pattern:s}_%06d.cbf.gz"
        )

    def get_spot_list_ordered_directory(self):
        return os.path.join(self.get_cbf_directory(), "spot_list_ordered")
    
    def get_spot_list_directory(self):
        return os.path.join(self.get_cbf_directory(), "spot_list")

    def get_spot_file_template(self):
        if os.path.isdir(self.get_spot_list_ordered_directory()):
            sld = self.get_spot_list_ordered_directory()
        else:
            sld = self.get_spot_list_directory()
        return os.path.join(
            sld, f"{self.name_pattern:s}_%06d.adx.gz"
        )

    def get_list_of_spot_files(self):
        return glob.glob(self.get_spot_file_template().replace("%06d", 6 * "?"))

    def get_list_of_cbf_files(self):
        return glob.glob(self.get_cbf_template().replace("%06d", 6 * "?"))

    def create_ordered_cbf_links(self):
        self.logger.info("create_ordered_cbf_links")
        os.chdir(self.get_dozor_directory())
        os.system("touch %s" % self.get_cbf_directory())

        cbfs = glob.glob(
            "%s/%s_*.cbf.gz" % (self.get_cbf_directory(), self.name_pattern)
        )
        if cbfs == []:
            self.logger.info("no cbfs generated please check")
            return
        cbfs.sort()
        for k, cbf in enumerate(cbfs):
            try:
                os.symlink(cbf, "%s_ordered_%06d.cbf.gz" % (self.name_pattern, k + 1))
            except OSError:
                pass

    def run_dials(self):
        self.logger.info("run_dials")

        if os.path.isfile(
            f"{self.directory}/process/dials_{self.name_pattern}/dials.find_spots.log"
        ):
            return
        spot_find_line = f"source /usr/local/dials/dials_env.sh; mkdir -p {self.directory}/process/dials_{self.name_pattern}; cd {self.directory}/process/dials_{self.name_pattern}; touch {self.directory}; echo $(pwd); dials.find_spots shoebox=False per_image_statistics=True spotfinder.filter.ice_rings.filter=True nproc=80 ../../{self.name_pattern}_master.h5 &"

        if os.uname()[1] != "process1":
            spot_find_line = 'ssh process1 "%s"&' % spot_find_line
        self.logger.info("spot_find_line %s" % spot_find_line)
        os.system(spot_find_line)

    def get_dials_results(self):
        if not os.path.isdir(self.get_process_dir()):
            os.mkdir(self.get_process_dir())
        if not os.path.isfile("%s/dials.find_spots.log" % self.get_process_dir()):
            while not os.path.exists("%s_master.h5" % self.get_template()):
                os.system("touch %s" % self.directory)
                gevent.sleep(0.25)

            self.run_dials()

        dials_process_output = "%s/dials.find_spots.log" % self.get_process_dir()
        print("dials process output", dials_process_output)
        os.system("touch %s" % os.path.dirname(dials_process_output))
        if os.path.isfile(dials_process_output):
            a = subprocess.getoutput("grep '|' %s" % dials_process_output).split("\n")
        # dials_results = self.get_nspots_nimage(a)
        dials_results = np.array(
            list(
                map(
                    int,
                    subprocess.getoutput(
                        f"grep '|' {self.directory}/process/dials_{self.name_pattern}/dials.find_spots.log | grep -v image | cut -d '|' -f 3"
                    ).split("\n"),
                )
            )
        )
        return dials_results

    def get_dials_raw_results(self):
        results_file = self.get_dials_raw_results_filename()

        if not os.path.isfile(results_file):
            self.results = self.get_dials_results()
        elif self.results == None:
            self.results = pickle.load(open(results_file))
        return self.results

    def get_nspots_nimage(self, a):
        results = {}
        for line in a:
            try:
                nimage, nspots, nspots_no_ice, total_intensity = list(
                    map(
                        int,
                        re.findall(
                            "\| (\d*)\s*\| (\d*)\s*\| (\d*)\s*\| (\d*)\s*\|", line
                        )[0],
                    )
                )
                results[nimage] = {}
                results[nimage]["dials_spots"] = nspots_no_ice
                results[nimage]["dials_all_spots"] = nspots
                results[nimage]["dials_total_intensity"] = total_intensity
            except:
                print(traceback.print_exc())
        return results

    def parse_find_spots(self):
        def get_nspots_nimage(a):
            results = {}
            for line in a:
                try:
                    nimage, nspots, nspots_no_ice, total_intensity = list(
                        map(
                            int,
                            re.findall(
                                "\| (\d*)\s*\| (\d*)\s*\| (\d*)\s*\| (\d*)\s*\|", line
                            )[0],
                        )
                    )
                    # nspots, nimage = map(int, re.findall('Found (\d*) strong pixels on image (\d*)', line)[0])
                    results[nimage] = {}
                    results[nimage]["dials_spots"] = nspots_no_ice
                    results[nimage]["dials_all_spots"] = nspots
                    results[nimage]["dials_total_intensity"] = total_intensity
                except:
                    self.logger.info(traceback.format_exc())
            return results

        search_line = f"grep '|' {self.directory}/process/dials_{self.name_pattern}/dials.find_spots.log"
        a = commands.get_output(search_line).split("\n")
        results = get_nspots_nimage(a)

        return results

    def run_dozor(self, force=False, binning=1, blocking=False):
        self.logger.info("run_dozor force=%s blocking=%s" % (force, blocking))
        _start = time.time()
        process_directory = self.get_dozor_directory()
        if not force and os.path.isfile(
            os.path.join(
                self.get_dozor_directory(), self.get_dozor_control_card_filename()
            )
        ):
            self.logger.info(
                "dozor was already executed, please use force flag if you want to run it again"
            )
            return
        if not force and os.path.isfile(
            os.path.join(process_directory, "dozor_average.dat")
        ):
            self.logger.info(
                "dozor was already executed, and completed, please use force flag if you want to run it again"
            )
            return
        if not os.path.isfile(self.get_dozor_control_card_filename()):
            self.create_dozor_control_card()

        dozor_line = "cd {directory}; dozor -bin {binning:d} -b -p -wg -rd -s -pall {control_card} | tee {log_file}".format(
            **{
                "directory": process_directory,
                "control_card": self.get_dozor_control_card_filename(),
                "log_file": os.path.join(
                    self.get_dozor_directory(), self.get_dozor_log_filename()
                ),
                "binning": binning,
            }
        )
        if not blocking:
            dozor_line += "&"
        if os.uname()[1] != "process1":
            dozor_line = 'ssh process1 "%s"' % dozor_line
        if not blocking:
            dozor_line += "&"
        self.logger.info("dozor_line %s" % dozor_line)
        os.system(dozor_line)
        if blocking:
            self.logger.info(
                "dozor analysis took %.4f seconds" % (time.time() - _start)
            )

    def get_dozor_raw_results(self):
        return open(
            os.path.join(self.get_dozor_directory(), self.get_dozor_log_filename())
        ).read()

    def get_dozor_results(self, blocking=False, timeout=30):
        self.run_dozor(blocking=blocking)
        _start = time.time()
        no_spots_mainscore_resolution = re.compile(
            "\\s+([\\d]+).*\|\\s+([\\d]+).*\|.*\|\\s+([\\d\.]+)\\s+[\\d\.]+\\s+([\\d\.]+).*"
        )
        # spots = re.compile('[\\s]+[\\d]+[\\s]+([\\d]+)[\\s]+[\\d\\.]+[\\s]+[\\d\\.]+[\\s]+[\\d\\.]+[\\s]+[\\d\\.]+[\\s]+[\\d\\.]+')
        results_filename = os.path.join(self.get_dozor_directory(), "dozor_average.dat")
        while not os.path.isfile(results_filename) and time.time() - _start < timeout:
            time.sleep(0.5)
            self.logger.info("Waiting for dozor results to appear")
            os.system("touch %s" % self.get_dozor_directory())
        results_file = self.get_dozor_raw_results()
        raw_results = no_spots_mainscore_resolution.findall(results_file)
        results = np.array([list(map(float, item)) for item in raw_results])
        return results

    def write_xds_inp_init(self):
        template = """ JOB= {jobs:s}    
 DATA_RANGE= {img_start:d} {img_end:d}
 SPOT_RANGE= {img_start:d} {img_end:d}
 BACKGROUND_RANGE= {background_img_start:d} {background_img_end:d}
 SPOT_MAXIMUM-CENTROID= 2.0   
 MINIMUM_NUMBER_OF_PIXELS_IN_A_SPOT= 3  
 SIGNAL_PIXEL= 3.0   
 OSCILLATION_RANGE= {angle_per_frame:.4f}
 STARTING_ANGLE= 0.0
 STARTING_FRAME= 1   
 X-RAY_WAVELENGTH= {wavelength:.4f}
 NAME_TEMPLATE_OF_DATA_FRAMES= img/{name_pattern:s}_??????.cbf.gz 
 DETECTOR_DISTANCE= {detector_distance:.2f}
 DETECTOR= EIGER    MINIMUM_VALID_PIXEL_VALUE= 0    OVERLOAD= 12457   
 DIRECTION_OF_DETECTOR_X-AXIS= 1.0 0.0 0.0   
 DIRECTION_OF_DETECTOR_Y-AXIS= 0.0 1.0 0.0   
 NX= 3110    NY= 3269    QX= 0.075    QY= 0.075
 ORGX= {beam_center_x:.4f}    ORGY= {beam_center_y:.4f}
 ROTATION_AXIS=  1.000000  0.000000  0.000000   
 INCIDENT_BEAM_DIRECTION= 0.0 0.0 1.0   
 FRACTION_OF_POLARIZATION= 0.99   
 POLARIZATION_PLANE_NORMAL= 0.0 1.0 0.0   
 SPACE_GROUP_NUMBER= 0   
 UNIT_CELL_CONSTANTS= 0 0 0 0 0 0   
 VALUE_RANGE_FOR_TRUSTED_DETECTOR_PIXELS= 5500 30000   
 INCLUDE_RESOLUTION_RANGE= 50 {resolution:.2f}
 RESOLUTION_SHELLS= 15.0 7.0 {resolution:.2f}   
 TRUSTED_REGION= 0.0 1.42   
 STRICT_ABSORPTION_CORRECTION= FALSE   
 TEST_RESOLUTION_RANGE= 20 4.97   
 GAIN= 1.0   
 LIB= /nfs/data/xds-zcbf.so  

!=== Added Keywords ===!

 MAXIMUM_NUMBER_OF_JOBS= {maximum_number_of_jobs:d}
 MAXIMUM_NUMBER_OF_PROCESSORS= {maximum_number_of_processors:d}
 FRIEDEL'S_LAW= TRUE   
 EXCLUDE_RESOLUTION_RANGE= 3.930 3.870 !ice-ring at 3.897 Angstrom
 EXCLUDE_RESOLUTION_RANGE= 3.700 3.640 !ice-ring at 3.669 Angstrom
 EXCLUDE_RESOLUTION_RANGE= 3.470 3.410 !ice-ring at 3.441 Angstrom
 EXCLUDE_RESOLUTION_RANGE= 2.700 2.640 !ice-ring at 2.671 Angstrom
 EXCLUDE_RESOLUTION_RANGE= 2.280 2.220 !ice-ring at 2.249 Angstrom
 EXCLUDE_RESOLUTION_RANGE= 2.102 2.042 !ice-ring at 2.072 Angstrom - strong
 EXCLUDE_RESOLUTION_RANGE= 1.978 1.918 !ice-ring at 1.948 Angstrom - weak
 EXCLUDE_RESOLUTION_RANGE= 1.948 1.888 !ice-ring at 1.918 Angstrom - strong
 EXCLUDE_RESOLUTION_RANGE= 1.913 1.853 !ice-ring at 1.883 Angstrom - weak
 EXCLUDE_RESOLUTION_RANGE= 1.751 1.691 !ice-ring at 1.721 Angstrom - weak
 SENSOR_THICKNESS= 0.45   
 DATA_RANGE_FIXED_SCALE_FACTOR= 1 100 1.0   
"""
        self.format_dictionary["name_pattern"] = self.get_name_pattern()
        self.format_dictionary["directory"] = self.directory
        print("%s" % self.format_dictionary)
        xds_inp_text = template.format(**self.format_dictionary)
        process_directory = "{directory}/process/xds_{name_pattern}".format(
            **self.format_dictionary
        )
        xds_inp_file = open(os.path.join(process_directory, "XDS.INP"), "w")
        xds_inp_file.write(xds_inp_text)
        print("xds_inp_text")
        xds_inp_file.close()

    def execute_xds(self):
        self.logger.info("execute_xds")
        self.write_xds_inp_init()
        execute_line = "cd {process_directory}; touch {directory}; echo $(pwd); ln -s ../../ img; xds_par &".format(
            **self.format_dictionary
        )
        if os.uname()[1] != "process1":
            execute_line = 'ssh process1 "%s"' % execute_line
        self.logger.info("spot_find_line %s" % execute_line)
        os.system(execute_line)

    def run_xds(self, force=False, background_images=18, binning=None, blocking=True):
        self.logger.info("run_xds")

        process_directory = "{directory}/process/xds_{name_pattern}".format(
            **self.format_dictionary
        )
        print("process_directory", process_directory)
        colspot = os.path.join(process_directory, "COLSPOT.LP")
        print("colspot", colspot)
        if not force and os.path.isfile(colspot):
            return
        if not os.path.isdir(process_directory):
            os.makedirs(process_directory)
        parameters = self.get_parameters()
        for key in [
            "resolution",
            "wavelength",
            "detector_distance",
            "beam_center_x",
            "beam_center_y",
        ]:
            self.format_dictionary[key] = parameters[key]

        self.format_dictionary["process_directory"] = process_directory
        self.format_dictionary["img_start"] = 1
        self.format_dictionary["img_end"] = int(
            parameters["ntrigger"] * parameters["nimages"]
        )
        self.format_dictionary["nimages"] = int(parameters["nimages"])
        self.format_dictionary[
            "angle_per_frame"
        ] = 0.001  # parameters['angle_per_frame']

        if not os.path.isfile(os.path.join(process_directory, "INIT.LP")) or force:
            self.format_dictionary["jobs"] = "XYCORR INIT"
            self.format_dictionary["maximum_number_of_jobs"] = 1
            self.format_dictionary["maximum_number_of_processors"] = min(
                background_images, 99
            )
            self.format_dictionary["background_img_start"] = max(
                1, int(parameters["nimages"] / 2 - background_images / 2)
            )
            self.format_dictionary["background_img_end"] = min(
                parameters["nimages"] * parameters["ntrigger"],
                int(parameters["nimages"] / 2 + background_images / 2),
            )
            self.execute_xds()

        self.format_dictionary["jobs"] = "COLSPOT"

        self.format_dictionary["maximum_number_of_jobs"] = min(self.get_ntrigger(), 99)
        self.format_dictionary["maximum_number_of_processors"] = min(
            self.get_nimages(), 99
        )
        self.execute_xds()

    def get_xds_directory(self):
        return os.path.join(
            "{directory}/process/xds_{name_pattern}".format(**self.format_dictionary)
        )

    def get_xds_raw_results_filename(self):
        return os.path.join(self.get_xds_directory(), "COLSPOT.LP")

    def get_xds_raw_results(self):
        return open(self.get_xds_raw_results_filename()).read()

    def get_xds_results(self, blocking=False, timeout=30):
        _start = time.time()
        self.run_xds()
        while (
            not os.path.isfile(self.get_xds_raw_results_filename())
            and time.time() - _start < timeout
        ):
            time.sleep(0.5)
            self.logger.info("Waiting for xds results to appear")
            os.system("touch %s" % self.get_xds_directory())
        spots = re.compile("[\s]+[\d]+[\s]+[\d]+[\s]+([\d]+)[\s]+[\d]+")
        colspot = self.get_xds_raw_results()
        results = np.array(list(map(int, spots.findall(colspot))))
        return results

    def get_spots_array(self, spots_file):
        spots = self.get_spots(spots_file)
        spots_array = np.array(spots)
        return spots_array

    # In [30]: %timeit sl = get_spot_lines('tomo_a_pos_10_a_006259.adx.gz', mode='rt')
    # 47.3 µs ± 300 ns per loop (mean ± std. dev. of 7 runs, 10,000 loops each)

    # In [31]: %timeit sl = get_spot_lines('tomo_a_pos_10_a_006259.adx.gz', mode='rb')
    # 42.2 µs ± 133 ns per loop (mean ± std. dev. of 7 runs, 10,000 loops each)

    # In [32]: def get_spot_lines(spot_file, mode="rb", encoding="ascii", newline="\n"):
    # ...:     if mode == "rb":
    # ...:         sf = gzip.open(spot_file, mode=mode)
    # ...:         sfr = sf.read()
    # ...:         sfd = sfr.decode(encoding=encoding)
    # ...:         spot_lines = sfd.split(newline)[:-1]
    # ...:     elif mode == "rt":
    # ...:         sf = gzip.open(spot_file, mode=mode, encoding=encoding, newline=newline)
    # ...:         sfr = sf.read()
    # ...:         spot_lines = sfr.split(newline)[:-1]
    # ...:     return spot_lines

    def get_spots_lines(self, spots_file, mode="rb", encoding="ascii"):
        try:
            spots_lines = (
                gzip.open(spots_file, mode=mode)
                .read()
                .decode(encoding=encoding)
                .split("\n")[:-1]
            )
        except:
            spots_lines = []
        return spots_lines

    def get_number_of_spots(self, spots_file):
        spots_lines = self.get_spots_lines(spots_file)
        number_of_spots = len(spots_lines)
        return number_of_spots

    def get_spots(self, spots_file, mode="rb", encoding="ascii"):
        spots_lines = self.get_spots_lines(spots_file, mode=mode, encoding=encoding)
        spots = [list(map(float, line.split())) for line in spots_lines]

        return spots

    def get_ddv_all(self):
        total_number_of_images = self.get_total_number_of_images()
        tioga_results = np.zeros((total_number_of_images,))
        spot_file_template = self.get_spot_file_template()
        image_number_range = range(1, total_number_of_images + 1)
        spot_files = [spot_file_template % d for d in image_number_range]
        ddv_all = np.array([])
        for sf in spot_files:
            if os.path.isfile(sf):
                try:
                    ddv = self.get_ddv(sf)
                    ddv_all = np.append(ddv_all, ddv)
                except:
                    print("sf", sf)
        return ddv_all

    def get_d_min_for_ddv(self, r_min=25., wavelength=None, detector_distance=None):
        if wavelength is None:
            wavelength = self.get_wavelength
        if detector_distance is None:
            detector_distance = self.get_detector_distance()
        
        d_min = get_d_min_for_ddv(r_min, wavelength, detector_distance)
        
        return d_min
    
    def get_ddv(self, spots_file, r_min=25):
        spots_mm = self.get_spots_mm(spots_file)

        detector_distance = self.get_detector_distance()
        wavelength = self.get_wavelength()
        
        valu, reso = get_ddv(spots_mm, r_min, wavelength, detector_distance)
        
        return valu

    def get_resolution_from_distance(self, distance):
        
        detector_distance = self.get_detector_distance()
        wavelength = self.get_wavelength()
        resolution = get_resolution_from_distance(distance, detector_distance, wavelength)

        return resolution

    def get_annotated_spots(self, spots_file, pixel_size=0.075, debug=False):
        # bragg's n * lambda = 2 * d * sin(theta)
        spots = np.array(self.get_spots(spots_file))
        beam_center = np.array(self.get_beam_center())
        detector_distance = self.get_detector_distance()
        wavelength = self.get_wavelength()

        centered_spots_px = spots[:, :2] - beam_center
        centered_spots_mm = centered_spots_px * pixel_size
        # r = np.sqrt(np.power(centered_spots_mm, 2).sum(axis=1))
        r = np.linalg.norm(centered_spots_mm, ord=2, axis=1)
        k = np.sqrt(r**2 + detector_distance**2)
        tans = r / detector_distance
        twotheta = np.arctan(tans)
        resolution = wavelength / (2 * np.sin(twotheta / 2.0))
        if debug:
            print(f"centered_spots_mm, {centered_spots_mm}")
            print(f"r, {r.shape}")
            print(f"k, {k.shape}")
            print(f"tans, {tans.shape}")
            print(f"twotheta, {twotheta.shape}")
            print(f"resolution, {resolution.shape}")

        r = np.expand_dims(r, 1)
        k = np.expand_dims(k, 1)
        tans = np.expand_dims(tans, 1)
        twotheta = np.expand_dims(twotheta, 1)
        resolution = np.expand_dims(resolution, 1)

        if debug:
            print(f"r, {r.shape}")
            print(f"k, {k.shape}")
            print(f"tans, {tans.shape}")
            print(f"twotheta, {twotheta.shape}")
            print(f"resolution, {resolution.shape}")

        annotated_spots = np.hstack(
            [
                centered_spots_px,
                centered_spots_mm,
                r,
                k,
                tans,
                twotheta,
                resolution,
            ]
        )
        return annotated_spots

    def get_spots_mm(self, spots_file, pixel_size=0.075):
        spots = np.array(self.get_spots(spots_file))
        beam_center = np.array(self.get_beam_center())

        centered_spots_px = spots[:, :2] - beam_center
        centered_spots_mm = centered_spots_px * pixel_size

        return centered_spots_mm

    def get_scattered_rays(self, spots_file):
        spots = self.get_spots_mm(spots_file)
        detector_distance = self.get_detector_distance()
        scattered_rays = np.hstack(
            (
                spots,
                np.ones((spots.shape[0], 1)) * detector_distance,
            )
        )
        scattered_rays /= np.linalg.norm(scattered_rays, axis=1, keepdims=True)
        return scattered_rays

    def get_wavelength(self):
        wavelength = self.get_parameters()["wavelength"]
        return wavelength

    def get_detector_distance(self):
        detector_distance = self.get_parameters()["detector_distance"]
        return detector_distance

    def get_beam_center(self):
        x = self.get_parameters()["beam_center_x"]
        y = self.get_parameters()["beam_center_y"]
        beam_center = (x, y)
        return beam_center

    def get_tioga_results(self):
        total_number_of_images = self.get_total_number_of_images()
        tioga_results = np.zeros((total_number_of_images,))
        spot_file_template = self.get_spot_file_template()
        image_number_range = range(1, total_number_of_images + 1)
        spot_files = [spot_file_template % d for d in image_number_range]
        for sf in spot_files:
            if os.path.isfile(sf):
                s = self.get_number_of_spots(sf)
                ordinal = self.get_ordinal_from_spot_file_name(sf)
                if ordinal != -1:
                    tioga_results[ordinal - 1] = s
        return tioga_results

    def get_rays_filename(self):
        rays_filename = os.path.join(
            self.get_directory(),
            "spot_list",
            f"{self.get_template()}_rays.pickle",
        )
        return rays_filename

    def get_rays_from_all_images(self):
        total_number_of_images = self.get_total_number_of_images()
        spot_file_template = self.get_spot_file_template()
        image_number_range = range(1, total_number_of_images + 1)
        spot_files = [spot_file_template % d for d in image_number_range]

        rays_filename = self.get_rays_filename()
        if os.path.isfile(rays_filename):
            rays_from_all_images = pickle.load(open(rays_filename, "rb"))
        else:
            rays_from_all_images = {}
            for sf in spot_files:
                if os.path.isfile(sf):
                    rays = self.get_scattered_rays(sf)
                    ordinal = self.get_ordinal_from_spot_file_name(sf)
                    rays_from_all_images[ordinal] = rays
            rays_file = open(rays_filename, "wb")
            pickle.dump(rays_from_all_images, rays_file)

        return rays_from_all_images

    def get_max_len(self, rays_from_all_images=None):
        if rays_from_all_images is None:
            rays_from_all_images = self.get_rays_from_all_images()
        max_len = 0
        for rays in rays_from_all_images.values():
            if rays.shape[0] > max_len:
                max_len = rays.shape[0]
        return max_len

    def get_rays_array(self):
        rays_from_all_images = self.get_rays_from_all_images()
        max_len = self.get_max_len(rays_from_all_images)
        rays_array = np.empty((len(rays_from_all_images), max_len, 3))

        for k, key in enumerate(sorted(list(rays_from_all_images.keys()))):
            rays = rays_from_all_images[key]
            rays_array[k][: rays.shape[0], :] = rays

        return rays_array

    def get_complete_rays_array(self):
        total_number_of_images = self.get_total_number_of_images()
        rays_from_all_images = self.get_rays_from_all_images()
        max_len = self.get_max_len(rays_from_all_images)
        complete_rays_array = np.empty((total_number_of_images, max_len, 3))
        for k in rays_from_all_images:
            rays = rays_from_all_images[k]
            complete_rays_array[k - 1][: rays.shape[0], :] = rays
        return complete_rays_array

    def get_ordinal_from_spot_file_name(self, spot_file_name):
        ordinal = -1
        try:
            ordinal = int(self.match_number_in_spot_file.findall(spot_file_name)[0])
        except:
            pass
        return ordinal

    def get_ordinal_from_cbf_file_name(self, cbf_file_name):
        ordinal = -1
        try:
            ordinal = int(self.match_number_in_cbf.findall(cbf_file_name)[0])
        except:
            pass
        return ordinal


    def get_seed_positions(self):
        try:
            seed_positions = self.get_parameters()["seed_positions"]
        except:
            traceback.print_exc()
            seed_positions = []
        return seed_positions

    def _line_analysis(self, k, sleeptime=0.001, timeout=0):
        img_start = k * self.get_nimages() + 1
        img_end = img_start + self.get_nimages()
        cbf_files = [self.get_cbf_template() % d for d in range(img_start, img_end)]
        spot_files = [
            self.get_spot_file_template() % d for d in range(img_start, img_end)
        ]
        print(f"start: end {img_start}: {img_end}")
        _start = time.time()
        while (
            not all([os.path.isfile(cf) for cf in cbf_files])
            and time.time() - _start < timeout
        ):
            time.sleep(sleeptime)
        print(f"{k} all files present or timeout reached")
        spots = 0
        line = []
        for sf in spot_files:
            s = len(self.get_spots(sf))
            ordinal = self.get_ordinal_from_spot_file_name(sf)
            if ordinal != -1:
                self.spots_per_frame[ordinal - 1] = s
            spots += s
            line.append(s)
        print(f"number of spots for line {k} is {spots}")
        self.spots_per_line[k] = spots
        self.lines[k] = line
        print(f"line analysis {k} took {time.time()-_start:.4f}")
        return spots

    def analyze(self):
        print("analyze")
        self.analyze_initial_raster()

    def analyze_initial_raster(self):
        for k in range(self.get_ntrigger()):
            self._line_analysis(k)

    def get_results(self):
        self.analyze()

        results = []

        seed_positions = self.get_seed_positions()

        for k, position in enumerate(seed_positions):
            try:
                print(f"point {k} has {self.spots_per_line[k]} spots")
                if self.spots_per_line[k] >= self.spot_threshold:
                    p = get_position_from_vector(position)
                    p["AlignmentZ"] = self.get_reference_position()["AlignmentZ"]
                    p["Omega"] = self.get_reference_position()["Omega"]
                    results.append(
                        {
                            "spots": self.spots_per_line[k],
                            "line": self.lines[k],
                            "result_position": p,
                        }
                    )
            except:
                traceback.print_exc()

        if results:
            results.sort(key=lambda x: x["spots"])
            results.reverse()

        return results

    def get_distance(self, spots1, spots2, d_threshold=0.1):
        if len(spots1) > len(spots2):
            spots1, spots2 = spots2, spots1

        N = len(spots1)
        D = 0.0
        for spot in spots1:
            d_min = np.dot(spot, spots2).min()
            d_min = min(d_min, d_threshold)
            D += d_min**2

        distance = np.sqrt(D) / N

        return distance

    def get_k(self, spots_file):
        return self.get_scattered_rays(spots_file)

    def get_distance_fast(self, rays1, rays2, d_threshold=0.1):
        sasha_distance(rays1, rays2, d_threshold=d_threshold)

        return distance

    def get_raster_omegas(self):
        raster_omegas = []
        initial_raster = self.get_initial_raster()
        for line in initial_raster:
            if line[2] not in raster_omegas:
                raster_omegas.append(line[2])
        return raster_omegas

    def get_horizontal_step_size(self):
        horizontal_step_size = self.get_parameters()["horizontal_step_size"]
        return horizontal_step_size

    def get_vertical_step_size(self):
        vertical_step_size = self.get_parameters()["vertical_step_size"]
        return vertical_step_size

    def get_initial_raster(self):
        initial_raster = self.get_parameters()["initial_raster"]
        return initial_raster

    def get_raster_lines(
        self,
        complete_rays_array=None,
        nimages=None,
        ntrigger=None,
        categorical=False,
    ):
        if complete_rays_array is None:
            complete_rays_array = self.get_complete_rays_array()
        if nimages is None:
            nimages = self.get_nimages()
        if ntrigger is None:
            ntrigger = self.get_ntrigger()

        spots = complete_rays_array.any(axis=2)
        if categorical:
            spots = spots.any(axis=1)
        else:
            spots = spots.sum(axis=1)

        lines = np.reshape(spots, (ntrigger, nimages)).T

        return lines

    def get_rasters_from_complete_rays_array(
        self,
        complete_rays_array=None,
        nimages=None,
        ntrigger=None,
        categorical=False,
        omegas=None,
        horizontal_step_size=None,
        vertical_step_size=None,
    ):
        if complete_rays_array is None:
            complete_rays_array = self.get_complete_rays_array()
        if nimages is None:
            nimages = self.get_nimages()
        if ntrigger is None:
            ntrigger = self.get_ntrigger()
        if omegas is None:
            omegas = self.get_raster_omegas()
        if horizontal_step_size is None:
            horizontal_step_size = self.get_horizontal_step_size()
        if vertical_step_size is None:
            vertical_step_size = self.get_vertical_step_size()

        lines = self.get_raster_lines(complete_rays_array, nimages, ntrigger)

        integral = lines.sum(axis=0)

        rasters = []
        scaled_but_uncorrected_projections = []
        normalized_integral = np.zeros(ntrigger)

        nrasters = len(omegas)
        for k in range(nrasters):
            raster = lines[:, k::nrasters]
            rasters.append(raster)

            i = integral[k::nrasters]
            i = i / np.linalg.norm(i)
            normalized_integral[k::nrasters] = i

        return lines, rasters, normalized_integral

    def raster_magic(
        self,
        ntrigger=None,
        nimages=None,
        horizontal_step_size=None,
        vertical_step_size=None,
        initial_raster=None,
    ):
        if nimages is None:
            nimages = self.get_nimages()
        if ntrigger is None:
            ntrigger = self.get_ntrigger()
        if horizontal_step_size is None:
            horizontal_step_size = self.get_horizontal_step_size()
        if vertical_step_size is None:
            vertical_step_size = self.get_vertical_step_size()
        if initial_raster is None:
            initial_raster = self.get_initial_raster()

        (
            lines,
            rasters,
            normalized_integral,
        ) = self.get_rasters_from_complete_rays_array()

        seed_positions = self.get_seed_positions()
        nlines = len(seed_positions)

        s_start = seed_positions[0]
        s_end = seed_positions[-1]
        s_vector = s_end - s_start
        s_length = np.linalg.norm(s_vector)
        s_unit_vector = s_vector / s_length

        line_models = []
        for k, (ir, line) in enumerate(zip(initial_raster, lines.T)):
            s = get_vector_from_position(
                ir[0], keys=["CentringX", "CentringY", "AlignmentY"]
            )
            e = get_vector_from_position(
                ir[1], keys=["CentringX", "CentringY", "AlignmentY"]
            )
            start, end = -1, 1
            vector = e - s
            distance = np.linalg.norm(vector)
            unit_vector = vector / distance
            center = np.mean([s, e], axis=0)
            omega = ir[2]
            length = np.sum(line > 0)
            length = length / nimages
            if length:
                center_of_mass = ndi.center_of_mass(line)
                center_of_mass = start + (end - start) * (
                    np.array(center_of_mass) / nimages
                )
                center_of_mass = center_of_mass[0]
            else:
                center_of_mass = None

            line_model = si.interp1d(
                np.linspace(-1, 1, nimages), line, bounds_error=False, fill_value=0
            )

            seed_position = seed_positions[k]

            line_models.append(
                {
                    "line_model": line_model,
                    "omega": omega,
                    "center": center,
                    "seed_position": seed_position,
                    "center_from_vector": s_start
                    + (k / (nlines - 1)) * s_unit_vector * s_length,
                    "position_index": k,
                    "unit_vector": unit_vector,
                    "center_of_mass": center_of_mass,
                    "length": length,
                }
            )

        nrasters = len(rasters)
        coms = [(lm["position_index"], lm["center_of_mass"]) for lm in line_models]
        coms_models = []
        for k in range(nrasters):
            coms_p = [com for com in coms[k::nrasters] if com[1] is not None]
            coms_p = np.array(coms_p)
            coms_p_model = si.interp1d(
                coms_p[:, 0], coms_p[:, 1], bounds_error=False, fill_value="extrapolate"
            )
            coms_models.append((coms_p_model, coms_p))

        return line_models, coms_models

        # corrected_rasters = []

        # return corrected_rasters

    def get_scaled_but_uncorrected_projections(
        self,
        ntrigger=None,
        nimages=None,
        horizontal_step_size=None,
        vertical_step_size=None,
    ):
        if nimages is None:
            nimages = self.get_nimages()
        if ntrigger is None:
            ntrigger = self.get_ntrigger()
        if horizontal_step_size is None:
            horizontal_step_size = self.get_horizontal_step_size()
        if vertical_step_size is None:
            vertical_step_size = self.get_vertical_step_size()

        (
            lines,
            rasters,
            normalized_integral,
        ) = self.get_rasters_from_complete_rays_array()

        nrasters = len(rasters)
        scaled_but_uncorrected_projections = []

        for k in range(nrasters):
            raster = rasters[k]
            raster = raster / raster.max()
            raster = raster * 255
            raster = raster.astype(np.uint8)

            shape = (
                int(ntrigger * (horizontal_step_size / vertical_step_size)),
                nimages,
            )

            scaled_but_uncorrected_projection = cv2.resize(
                raster,
                shape,
                interpolation=cv2.INTER_LINEAR,
            )
            scaled_but_uncorrected_projections.append(scaled_but_uncorrected_projection)

        return scaled_but_uncorrected_projections


def sasha_distance(rays1, rays2, d_threshold=0.1):
    rays1 = rays1[rays1 != 0]
    rays2 = rays2[rays2 != 0]

    if len(rays1) > len(rays2):
        rays1, rays2 = rays2, rays1

    N = (rays1 != 0).sum() // 3

    if rays1.shape[-1] != 3:
        rays1 = np.reshape(rays1, (rays1.shape[0] // 3, 3))
    if rays2.shape[-1] != 3:
        rays2 = np.reshape(rays2, (rays2.shape[0] // 3, 3))

    distances = distance_matrix(rays1, rays2)

    mini_d = distances.min(axis=0)

    # Euclidean distance and dot product
    mini_a = np.degrees(np.arccos(1 - (mini_d**2) / 2))

    mini_a[mini_a > d_threshold] = d_threshold

    distance = np.sqrt(np.sum(mini_a**2)) / N

    return distance


def main():
    import pylab
    import glob
    import random
    import sys

    dea = diffraction_experiment_analysis(
        directory="/home/experiences/proxima2a/com-proxima2a/Documents/Martin/pos_10_a/tomo",
        name_pattern="tomo_a_pos_10_a",
    )

    line_models, com_models = dea.raster_magic()

    lines = len(line_models)
    valid_indices = [
        line["position_index"]
        for line in line_models
        if line["center_of_mass"] is not None
    ]
    mini = min(valid_indices)
    maxi = max(valid_indices)
    valid_indices.sort()

    shift1 = com_models[0][0](valid_indices) - (-com_models[2][0](valid_indices))
    shift2 = com_models[1][0](valid_indices) - (-com_models[3][0](valid_indices))
    print(f"shift1 {shift1.mean():.3f}, +- {shift1.std():.3f}")
    print(f"shift2 {shift2.mean():.3f}, +- {shift2.std():.3f}")
    shifts = [shift1, shift2]
    pylab.figure(1, figsize=(16, 9))
    for k, (com_p_model, com_p) in enumerate(com_models):
        if k in [0, 1]:
            pylab.plot(
                valid_indices,
                com_p_model(valid_indices) - shifts[k].mean(),
                "o-",
                label=f"model {k}",
            )
            pylab.plot(
                com_p[:, 0], np.array(com_p[:, 1]) - shifts[k].mean(), "o"
            )  # , label=f'{k}')
        else:
            pylab.plot(
                valid_indices,
                -com_p_model(valid_indices),
                "o-",
                label=f"model {k}^{-1}",
            )
            pylab.plot(com_p[:, 0], -np.array(com_p[:, 1]), "o")  # , label=f'{k}^{-1}')

    pylab.legend()
    pylab.ylabel("center of mass")
    pylab.xlabel("position")

    pylab.show()

    sys.exit()

    bins = 100

    adxs = glob.glob(
        os.path.join(dea.directory, "spot_list", "tomo_a_pos_10_a_??????.adx.gz")
    )

    for k in range(27):
        spots_file = random.choice(adxs)

        # for d in [2280, 2716, 4192, 4887, 6327]:
        # spots_file = os.path.join(dea.directory, 'spot_list', 'tomo_a_pos_10_a_%06d.adx.gz' % d)
        # spots_file = os.path.join(dea.directory, 'spot_list', 'tomo_a_pos_10_a_002280.adx.gz')
        # spots_file = os.path.join(dea.directory, 'spot_list', 'tomo_a_pos_10_a_002716.adx.gz')
        # spots_file = os.path.join(dea.directory, 'spot_list', 'tomo_a_pos_10_a_004192.adx.gz')
        # spots_file = os.path.join(dea.directory, 'spot_list', 'tomo_a_pos_10_a_004887.adx.gz')
        print(spots_file)
        # d = dea.get_annotated_spots(spots_file)
        ddv = dea.get_ddv(spots_file)
        # ddv = dea.get_ddv_all() #(spots_file, d_min=25)
        print(f"ddv.shape {ddv.shape}")

        # print(ddv[:100])
        # bins = min(500, int(len(ddv)/5))
        hist, edges = np.histogram(ddv, bins=bins, density=True)
        # hist = hist/2
        # hist[hist==1] = 0
        centers = np.array(
            [(edges[k] + edges[k + 1]) / 2.0 for k in range(len(edges) - 1)]
        )
        pylab.figure()
        # pylab.plot(d, 'o', label='d')

        # pylab.figure(2)
        # pylab.hist(ddv, bins=50)
        imageno = int(
            re.findall(
                dea.name_pattern + "_([\d]{6}).adx.gz", os.path.basename(spots_file)
            )[0]
        )
        pylab.title("image %d %d" % (k, imageno))
        pylab.xlabel("1/d [A^-1]")
        pylab.ylabel("probability")
        pylab.plot(1.0 / centers, hist)

    pylab.show()

    sys.exit()

    start = time.time()
    tr = dea.get_tioga_results()
    end = time.time()

    print("get_tioga_results() took %.4f seconds" % (end - start,))
    print("tr", tr)
    pylab.figure(3)
    pylab.plot(tr)

    start = time.time()
    results = dea.get_results()
    end = time.time()

    print("get_results() took %.4f seconds" % (end - start,))
    # print('results', results)

    pylab.show()


if __name__ == "__main__":
    main()
