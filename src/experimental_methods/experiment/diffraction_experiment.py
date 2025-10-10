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

import subprocess

from experimental_methods.experiment.xray_experiment import xray_experiment
from experimental_methods.utils.speech import speech


class diffraction_experiment(xray_experiment):
    specific_parameter_fields = [
        {"name": "kappa", "type": "float", "description": "kappa position in degrees"},
        {"name": "phi", "type": "float", "description": "phi position in degrees"},
        {"name": "resolution", "type": "float", "description": ""},
        {"name": "detector_distance", "type": "float", "description": ""},
        {"name": "detector_vertical_position", "type": "float", "description": ""},
        {"name": "detector_horizontal_position", "type": "float", "description": ""},
        {"name": "detector_ts_intention", "type": "float", "description": ""},
        {"name": "detector_tz_intention", "type": "float", "description": ""},
        {"name": "detector_tx_intention", "type": "float", "description": ""},
        {"name": "nimages_per_file", "type": "int", "description": ""},
        {"name": "image_nr_start", "type": "int", "description": ""},
        {"name": "total_expected_wedges", "type": "float", "description": ""},
        {"name": "total_expected_exposure_time", "type": "float", "description": ""},
        {"name": "beam_center_x", "type": "float", "description": ""},
        {"name": "beam_center_y", "type": "float", "description": ""},
        {"name": "sequence_id", "type": "int", "description": ""},
        {
            "name": "frames_per_second",
            "type": "float",
            "description": "number of frames per second",
        },
        {"name": "overlap", "type": "float", "description": "overlap"},
        {
            "name": "beware_of_download",
            "type": "bool",
            "description": "wait for expected files",
        },
        {"name": "generate_cbf", "type": "bool", "description": "generate CBFs"},
        {"name": "generate_h5", "type": "bool", "description": "generate h5"},
        {
            "name": "nimages",
            "type": "int",
            "description": "number of images per trigger",
        },
        {
            "name": "ntrigger",
            "type": "int",
            "description": "number of triggers",
        },
        {
            "name": "total_number_of_images",
            "type": "int",
            "description": "nimages*ntriggers",
        },
        {
            "name": "md_task_info",
            "type": "str",
            "description": "scan diagnostic information",
        },
        {"name": "keep_originals", "type": "bool", "description": "keep_originals"},
        {
            "name": "maximum_rotation_speed",
            "type": "float",
            "description": "maximum_rotation_speed",
        },
        {
            "name": "minimum_exposure_time",
            "type": "float",
            "description": "minimum_exposure_time",
        },
        {
            "name": "session_id",
            "type": "int",
            "description": "session id ",
        },
        {
            "name": "sample_id",
            "type": "int",
            "description": "sample id",
        },
        {
            "name": "collection_id",
            "type": "int",
            "description": "collection id",
        },
        {
            "name": "protein_acronym",
            "type": "str",
            "description": "protein acronym",
        },
        {
            "name": "use_server",
            "type": "bool",
            "description": "use server",
        },
    ]

    def __init__(
        self,
        name_pattern,
        directory,
        frames_per_second=None,
        nimages_per_file=400,
        nimages=10,
        position=None,
        kappa=None,
        phi=None,
        chi=None,
        photon_energy=None,
        resolution=None,
        detector_distance=None,
        detector_vertical=None,
        detector_horizontal=None,
        transmission=None,
        flux=None,
        ntrigger=1,
        snapshot=True,
        zoom=None,
        diagnostic=None,
        analysis=None,
        conclusion=None,
        simulation=None,
        parent=None,
        mxcube_parent_id=None,
        mxcube_gparent_id=None,
        beware_of_top_up=False,
        beware_of_download=False,
        generate_cbf=True,
        generate_h5=True,
        image_nr_start=1,
        keep_originals=False,
        maximum_rotation_speed=360.0,
        minimum_exposure_time=1.0 / 238,
        session_id=None,
        sample_id=None,
        collection_id=None,
        protein_acronym=None,
        use_server=False,
        run_number=None,
        cats_api=None,
    ):
        logging.debug(
            "diffraction_experiment __init__ len(diffraction_experiment.specific_parameter_fields) %d"
            % len(diffraction_experiment.specific_parameter_fields)
        )

        if hasattr(self, "parameter_fields"):
            self.parameter_fields += diffraction_experiment.specific_parameter_fields[:]
        else:
            self.parameter_fields = diffraction_experiment.specific_parameter_fields[:]

        logging.debug(
            "xray_experiment __init__ len(self.parameters_fields) %d"
            % len(self.parameter_fields)
        )

        self.resolution = resolution
        self.detector_distance = detector_distance
        self.detector_vertical = detector_vertical
        self.detector_horizontal = detector_horizontal

        xray_experiment.__init__(
            self,
            name_pattern,
            directory,
            position=position,
            photon_energy=photon_energy,
            transmission=transmission,
            flux=flux,
            ntrigger=ntrigger,
            snapshot=snapshot,
            zoom=zoom,
            diagnostic=diagnostic,
            analysis=analysis,
            conclusion=conclusion,
            simulation=simulation,
            parent=parent,
            mxcube_parent_id=mxcube_parent_id,
            mxcube_gparent_id=mxcube_gparent_id,
            beware_of_top_up=beware_of_top_up,
            run_number=run_number,
            cats_api=cats_api,
        )

        self.format_dictionary = {
            "directory": self.directory,
            "name_pattern": self.name_pattern,
        }

        self.actuator = self.goniometer

        self.frames_per_second = frames_per_second
        self.nimages_per_file = nimages_per_file
        self.nimages = nimages
        self.ntrigger = ntrigger
        self.beware_of_download = beware_of_download
        self.generate_cbf = generate_cbf
        self.generate_h5 = generate_h5
        self.set_image_nr_start(image_nr_start)
        self.keep_originals = keep_originals
        self.maximum_rotation_speed = maximum_rotation_speed
        self.minimum_exposure_time = minimum_exposure_time

        if kappa == None:
            try:
                self.kappa = self.goniometer.md.kappaposition
            except:
                self.kappa = None
        else:
            self.kappa = kappa
        if phi == None:
            try:
                self.phi = self.goniometer.md.phiposition
            except:
                self.phi = None
        else:
            self.phi = phi
        if chi == None:
            try:
                self.chi = self.goniometer.md.chiposition
            except:
                self.chi = None
        else:
            self.chi = chi

        # Set resolution: detector_distance takes precedence
        # if neither specified, takes currect detector_distance
        print(
            "diffraction_experiment current, specified detector_distance and specified, resolution start",
            self.get_detector_distance(),
            self.detector_distance,
            self.resolution,
        )

        if self.detector_distance is None and self.resolution is None:
            self.detector_distance = self.detector.position.ts.get_position()

        if self.detector_distance is not None:
            self.detector_distance_limits = self.detector.position.ts.get_limits()
            if None not in self.detector_distance_limits:
                self.detector_distance = max(
                    self.detector_distance, self.detector_distance_limits[0]
                )
                self.detector_distance = min(
                    self.detector_distance, self.detector_distance_limits[1]
                )

            self.resolution = self.resolution_motor.get_resolution_from_distance(
                self.detector_distance, wavelength=self.wavelength
            )
        elif self.resolution is not None:
            self.detector_distance = self.resolution_motor.get_distance_from_resolution(
                self.resolution, wavelength=self.wavelength
            )
        else:
            print(
                "There seem to be a problem with logic for detector distance determination. Please check"
            )

        print(
            "diffraction_experiment detector_distance and resolution end",
            self.detector_distance,
            self.resolution,
        )
        self.observations = []
        self.observation_fields = ["chronos", "progress"]
        self.dozor_launched = 0

        self.collect = speech(service="collect", server=False)

        self.session_id = session_id
        self.sample_id = sample_id
        self.collection_id = collection_id
        self.protein_acronym = protein_acronym
        self.use_server = use_server

    def get_master(self):
        master = h5py.File(self.get_master_filename(), "r")
        return master

    def get_master_filename(self):
        return "%s_master.h5" % self.get_template()

    def get_parameters(self):
        if os.path.isfile(self.get_parameters_filename()):
            self.parameters = self.load_parameters_from_file()
            return self.parameters
        if os.path.isfile(self.get_master_filename()) and self.parameters == {}:
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
        # else:
        # self.parameters = self.collect_parameters()
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
        answer = False
        for f in expected_files:
            if f not in listdir:
                return answer
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

    def get_degrees_per_frame(self):
        """get degrees per frame"""
        return self.angle_per_frame

    def get_fps(self):
        return self.get_frames_per_second()

    def get_frames_per_second(self):
        """get frames per second"""
        return self.get_nimages() / self.scan_exposure_time

    def get_dps(self):
        return self.get_degrees_per_second()

    def get_degrees_per_second(self):
        """get degrees per second"""
        return self.get_scan_speed()

    def get_scan_speed(self):
        """get scan speed"""
        return self.scan_range / self.scan_exposure_time

    def get_frame_time(self):
        """get frame time"""
        return self.scan_exposure_time / self.get_nimages()

    def get_reference_position(self):
        return self.get_position()

    def get_position(self):
        """get position"""
        if self.position is None:
            return self.goniometer.get_position()
        else:
            return self.position

    def set_position(self, position=None):
        """set position"""
        if position is None:
            self.position = self.goniometer.get_position()
        else:
            self.position = position
            self.goniometer.set_position(self.position)
            self.goniometer.wait()
        self.goniometer.save_position()

    def get_kappa(self):
        return self.kappa

    def set_kappa(self, kappa):
        self.kappa = kappa

    def get_phi(self):
        return self.phi

    def set_phi(self, phi):
        self.phi = phi

    def get_chi(self):
        return self.chi

    def set_chi(self):
        self.chi = chi

    def get_md_task_info(self):
        return self.md_task_info

    def set_nimages(self, nimages):
        self.nimages = nimages

    def get_nimages(self):
        try:
            nimages = self.parameters["nimages"]
        except:
            nimages = int(self.nimages)
        return nimages

    def get_scan_range(self):
        return self.scan_range

    def set_scan_range(self, scan_range):
        self.scan_range = scan_range

    def get_scan_exposure_time(self):
        return self.scan_exposure_time

    def set_scan_exposure_time(self, scan_exposure_time):
        self.scan_exposure_time = scan_exposure_time

    def get_scan_start_angle(self):
        return self.scan_start_angle

    def set_scan_start_angle(self, scan_start_angle):
        self.scan_start_angle = scan_start_angle

    def get_angle_per_frame(self):
        return self.angle_per_frame

    def set_angle_per_frame(self, angle_per_frame):
        self.angle_per_frame = angle_per_frame

    def set_ntrigger(self, ntrigger):
        self.ntrigger = ntrigger

    def get_ntrigger(self):
        try:
            ntrigger = self.parameters["ntrigger"]
        except:
            ntrigger = int(self.ntrigger)
        return int(self.ntrigger)

    def set_resolution(self, resolution=None):
        if resolution != None:
            self.resolution = resolution
            self.resolution_motor.set_resolution(resolution)

    def get_resolution(self):
        return self.resolution_motor.get_resolution()

    def get_detector_distance(self):
        return self.detector.position.ts.get_position()

    def set_detector_distance(self, position, wait=True):
        self.detector_ts_moved = self.detector.set_ts_position(position, wait=wait)

    def get_detector_vertical_position(self):
        return self.detector.position.tz.get_position()

    def set_detector_vertical_position(self, position, wait=True):
        self.detector_tz_moved = self.detector.position.tz.set_position(
            position, wait=wait
        )

    def get_detector_horizontal_position(self):
        return self.detector.position.tx.get_position()

    def set_detector_horizontal_position(self, position, wait=True):
        self.detector_tx_moved = self.detector.position.tx.set_position(
            position, wait=wait
        )

    def get_sequence_id(self):
        return self.sequence_id

    def get_detector_ts_intention(self):
        return self.detector_distance

    def get_detector_tz_intention(self):
        return self.detector_vertical

    def get_detector_tx_intention(self):
        return self.detector_horizontal

    def get_detector_ts(self):
        return self.get_detector_distance()

    def get_detector_tz(self):
        return self.get_detector_vertical_position()

    def get_detector_tx(self):
        return self.get_detector_horizontal_position()

    def get_detector_vertical(self):
        return self.detector_vertical

    def set_detector_vertical(self, detector_vertical):
        self.detector_vertical = detector_vertical

    def get_detector_horizontal(self):
        return self.detector_horizontal

    def set_detector_horizontal(self, detector_horizontal):
        self.detector_horizontal = detector_horizontal

    def set_nimages_per_file(self, nimages_per_file):
        self.nimages_per_file = nimages_per_file

    def get_nimages_per_file(self):
        return int(self.nimages_per_file)

    def set_image_nr_start(self, image_nr_start):
        self.image_nr_start = image_nr_start

    def get_image_nr_start(self):
        return int(self.image_nr_start)

    def set_total_expected_wedges(self, total_expected_wedges):
        self.total_expected_wedges = total_expected_wedges

    def get_total_expected_wedges(self):
        return self.total_expected_wedges

    def set_total_expected_exposure_time(self, total_expected_exposure_time):
        self.total_expected_exposure_time = total_expected_exposure_time

    def get_total_expected_exposure_time(self):
        return self.total_expected_exposure_time

    def set_beam_center_x(self, beam_center_x):
        self.beam_center_x = beam_center_x

    def get_beam_center_x(self):
        return self.beam_center_x

    def set_beam_center_y(self, beam_center_y):
        self.beam_center_y = beam_center_y

    def get_beam_center_y(self):
        return self.beam_center_y

    def check_downloader(self):
        if self.generate_h5:
            self.check_server(server_name="downloader")

    def get_overlap(
        self, overlap=0.0, scan_start_angles=[], scan_range=0.0, min_scan_range=0.025
    ):
        parameters = self.get_parameters()

        if parameters is not None and "scan_start_angles" in parameters:
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

    def get_spot_list_directory(self):
        return os.path.join(self.get_cbf_directory(), "spot_list")

    def get_spot_file_template(self):
        return os.path.join(
            self.get_spot_list_directory(), f"{self.name_pattern:s}_%06d.adx.gz"
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
        if not os.path.isdir(self.get_process_directory()):
            os.mkdir(self.get_process_directory())
        if not os.path.isfile("%s/dials.find_spots.log" % self.get_process_directory()):
            while not os.path.exists("%s_master.h5" % self.get_template()):
                os.system("touch %s" % self.directory)
                gevent.sleep(0.25)

            self.run_dials()

        dials_process_output = "%s/dials.find_spots.log" % self.get_process_directory()
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

    def run_dozor(self, force=False, binning=1, blocking=False, deport=False):
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
        if deport and os.uname()[1] != "process1":
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

    def execute_xds(self, deport=False):
        self.logger.info("execute_xds")
        self.write_xds_inp_init()
        execute_line = "cd {process_directory}; touch {directory}; echo $(pwd); ln -s ../../ img; xds_par &".format(
            **self.format_dictionary
        )
        if deport and os.uname()[1] != "process1":
            execute_line = 'ssh process1 "%s"' % execute_line
        self.logger.info("spot_find_line %s" % execute_line)
        os.system(execute_line)

    def run_xds(self, force=False, background_images=18, binning=None, blocking=True):
        self.logger.info("run_xds")

        process_directory = os.path.join(
            self.get_process_directory(), f"xds_{self.get_name_pattern()}"
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

    def get_stream_header_appendix(self):
        beam_center_x, beam_center_y = self.beam_center.get_beam_center(
            wavelength=self.wavelength,
            ts=self.detector_distance,
            tx=self.detector_horizontal,
            tz=self.detector_vertical,
        )
        header_appendix = {
            "htype": "dhappendix",
            "path": self.directory,  # os.path.join(self.directory, '%s_cbf' % self.name_pattern),
            # "path": self.directory.replace('/nfs/data2', '/dev/shm'),
            "template": "%s_??????" % self.name_pattern,
            "detector_distance": self.detector_distance / 1000.0,
            "wavelength": self.wavelength,
            "beam_center_x": beam_center_x,
            "beam_center_y": beam_center_y,
            "omega_start": self.scan_start_angle,
            "omega_increment": self.angle_per_frame,
            "two_theta_start": 0,
            "kappa_start": self.kappa,
            "phi_start": self.phi,
            "overlap": self.get_overlap(),
            "start_number": self.image_nr_start,
            "user": self.get_login(),
            "compression": 1,
            "processing": 1,
            "binning": 1,
            "beamstop_distance": 20,
            "beamstop_size": 1.0,
            "beamstop_vertical": 0,
            "frames_per_wedge": self.get_nimages(),
            "count_cutoff": self.detector.get_countrate_correction_count_cutoff(),
        }
        return json.dumps(header_appendix)

    def program_detector(
        self,
        photon_energy_delta=0.5,
        angle_delta=0.002,
        time_delta=0,
        filewriter=True,
        stream=True,
    ):
        _start = time.time()

        if stream:
            if self.detector.get_stream_disabled():
                self.detector.initialize_stream()
                self.detector.set_stream_enabled()
        else:
            if self.detector.get_stream_enabled():
                self.detector.set_stream_disabled()
        if filewriter:
            if self.detector.get_filewriter_disabled():
                self.detector.initialize_filewriter()
                self.detector.set_filewriter_enabled()
        else:
            if self.detector.get_filewriter_enabled():
                self.detector.set_filewriter_disabled()

        self.detector.set_standard_parameters(
            nimages_per_file=self.get_nimages_per_file()
        )
        self.detector.set_name_pattern(self.get_full_name_pattern())
        self.logger.info(
            "program_detector set_name reached after %.4f seconds"
            % (time.time() - _start)
        )

        # self.detector.clear_monitor()
        if self.detector.get_ntrigger() != self.get_ntrigger():
            self.detector.set_ntrigger(self.get_ntrigger())
        if self.detector.get_nimages() != self.get_nimages():
            self.detector.set_nimages(self.get_nimages())
        try:
            if self.detector.get_nimages_per_file() > self.get_nimages():
                self.detector.set_nimages_per_file(self.get_nimages())
        except:
            pass
        frame_time = self.get_frame_time()
        if abs(self.detector.get_frame_time() - frame_time) >= time_delta:
            self.detector.set_frame_time(frame_time)
        count_time = frame_time - self.detector.get_detector_readout_time()
        if abs(self.detector.get_count_time() - count_time) >= time_delta:
            self.detector.set_count_time(count_time)
        if abs(self.detector.get_omega() - self.scan_start_angle) >= angle_delta:
            self.detector.set_omega(self.scan_start_angle)
        if self.angle_per_frame <= 0.001:
            self.detector.set_omega_increment(0)
        else:
            self.detector.set_omega_increment(self.angle_per_frame)

        if abs(self.detector.get_kappa() - self.kappa) >= angle_delta:
            self.detector.set_kappa(self.kappa)
        if abs(self.detector.get_phi() - self.phi) >= angle_delta:
            self.detector.set_phi(self.phi)
        if abs(self.detector.get_chi() - self.chi) >= angle_delta:
            self.detector.set_chi(self.chi)
        if (
            abs(self.detector.get_photon_energy() - self.photon_energy)
            >= photon_energy_delta
        ):
            self.detector.set_photon_energy(self.photon_energy)

        if self.detector.get_image_nr_start() != self.image_nr_start:
            self.detector.set_image_nr_start(self.image_nr_start)

        self.logger.info(
            "program_detector set_image_nr_start reached after %.4f seconds"
            % (time.time() - _start)
        )

        if self.simulation != True:
            beam_center_x, beam_center_y = self.beam_center.get_beam_center(
                wavelength=self.wavelength,
                ts=self.detector_distance,
                tx=self.detector_horizontal,
                tz=self.detector_vertical,
            )
        else:
            beam_center_x, beam_center_y = 1430, 1550

        self.beam_center_x, self.beam_center_y = beam_center_x, beam_center_y

        if abs(self.detector.get_beam_center_x() - beam_center_x) >= angle_delta:
            self.detector.set_beam_center_x(beam_center_x)
        if abs(self.detector.get_beam_center_y() - beam_center_y) >= angle_delta:
            self.detector.set_beam_center_y(beam_center_y)

        self.logger.info(
            "program_detector beam_center handled after %.4f seconds"
            % (time.time() - _start)
        )
        if self.simulation == True:
            self.detector_distance = 250.0
        
        detector_distance_meter = self.detector_distance / 1000.0
        if abs(detector_distance_meter) >= 0:
            self.detector.set_detector_distance(detector_distance_meter)
        if stream:
            self.detector.set_stream_header_appendix(self.get_stream_header_appendix())

        self.logger.info(
            "program_detector stream handled after %.4f seconds"
            % (time.time() - _start)
        )
        self.logger.info("only arm() to complete ...")

        self.sequence_id = self.detector.arm()["sequence id"]

        self.logger.info("program_detector took %.4f seconds" % (time.time() - _start))

    def prepare(self):
        _start = time.time()

        if self.Si_PIN_diode.isinserted():
            self.Si_PIN_diode.extract()

        if self.position != None:
            self.goniometer.set_position(self.position, wait=True)
        
        #self.goniometer.set_beamstopposition("BEAM")
        self.goniometer.set_data_collection_phase(wait=True)
        
        if self.scan_start_angle is None:
            self.scan_start_angle = self.reference_position["Omega"]
        else:
            self.reference_position["Omega"] = self.scan_start_angle
        # self.goniometer.set_omega_position(self.scan_start_angle)

        if self.snapshot == True:
            print("taking image")
            self.goniometer.insert_backlight()
            self.goniometer.extract_frontlight()
            self.goniometer.set_position(self.reference_position, wait=True)
            self.image = self.get_image()
            self.rgbimage = self.get_rgbimage()

        if self.goniometer.backlight_is_on():
            self.goniometer.remove_backlight()

        initial_settings = []
        if self.simulation != True:
            #initial_settings.append(
                #gevent.spawn(self.goniometer.set_data_collection_phase, wait=True)
            #)
            initial_settings.append(
                gevent.spawn(self.set_photon_energy, self.photon_energy, wait=True)
            )
            # if self.detector_distance is not None:
            # print(f"sending detector distance to {self.detector_distance}")
            # initial_settings.append(
            # gevent.spawn(
            # self.set_detector_distance, self.detector_distance, wait=True
            # )
            # )
            if self.detector_horizontal is not None:
                initial_settings.append(
                    gevent.spawn(
                        self.set_detector_horizontal_position,
                        self.detector_horizontal,
                        wait=True,
                    )
                )
            if self.detector_vertical is not None:
                initial_settings.append(
                    gevent.spawn(
                        self.set_detector_vertical_position,
                        self.detector_vertical,
                        wait=True,
                    )
                )
            if self.transmission is not None:
                initial_settings.append(
                    gevent.spawn(self.set_transmission, self.transmission)
                )

        if self.diagnostic == True:
            try:
                if self.eiger_en_out.get_state() == "FAULT":
                    self.eiger_en_out.init()
                self.eiger_en_out.stop()
                self.eiger_en_out.set_total_buffer_duration(
                    2 * self.total_expected_exposure_time
                )
                self.eiger_en_out.start()
            except:
                print(
                    "Could not start the eiger_en_out monitor. Please check status of cpt.2 device"
                )
                print(traceback.print_exc())
                self.logger.info(traceback.format_exc())

        self.check_downloader()

        free_space = self.detector.get_free_space()
        if self.generate_h5 and free_space > 100.0:
            logging.getLogger("user_level_log").info(
                "Eiger DCU memory OK, free space: %.2f GB" % free_space
            )
        elif self.generate_h5 and free_space > 25.0:
            logging.getLogger("user_level_log").warning(
                "Eiger DCU memory NOT OK, free space only: %.2f GB" % free_space
            )
            logging.getLogger("user_level_log").error(
                "Please check that the downloader is running"
            )
        else:
            while self.generate_h5 and self.detector.get_free_space() < 25.0:
                message1 = (
                    "Eiger DCU memory critically low, free space only %.2f GB"
                    % self.detector.get_free_space()
                )
                logging.getLogger("user_level_log").error(message1)
                print(message1)
                message2 = "Please check the downloader server for any error. Is the network having an issue?"
                logging.getLogger("user_level_log").error(message2)
                print(message2)
                message3 = "Please call your local contact to investigate the anomaly"
                logging.getLogger("user_level_log").error(message3)
                print(message3)
                gevent.sleep(1)

        self.check_directory(self.process_directory)
        print("filesystem ready")
        # try:
        # self.goniometer.md.beamstopposition = 'BEAM'
        # except:
        # pass
        self.program_goniometer()
        print("goniometer ready")
        self.program_detector(filewriter=self.generate_h5, stream=self.generate_cbf)
        print("detector ready")

        print("wait for motors to reach destinations")
        gevent.joinall(initial_settings, timeout=1)
        print("all motors reached their destinations")

        if "$id" in self.name_pattern:
            self.name_pattern = self.name_pattern.replace("$id", str(self.sequence_id))

        if self.simulation != True:
            try:
                self.safety_shutter.open()
                if self.frontend_shutter.closed():
                    self.frontend_shutter.open()
            except:
                self.logger.info(traceback.print_exc())
                traceback.print_exc()
            if self.frontend_shutter.closed():
                sys.exit("Impossible to open frontend shutter, exiting gracefully ...")
            # self.detector.cover.extract(wait=True)

        self.write_destination_namepattern(self.directory, self.name_pattern)

        if self.simulation != True:
            self.energy_motor.turn_off()

        self.format_dictionary["total_number_of_images"] = (
            self.get_nimages() * self.get_ntrigger()
        )

        if self.detector_distance is not None:
            print(f"sending detector distance to {self.detector_distance}")
            a = self.set_detector_distance(self.detector_distance, wait=True)
            print(f"detector distance move completed {a}")

        if self.detector.cover.isclosed():
            self.detector.extract_protective_cover(wait=True)

        monitor_line = "image_monitor.py -n {name_pattern} -d {directory} -t {total_number_of_images} &".format(
            **self.format_dictionary
        )

        self.logger.info("launching image monitor %s " % monitor_line)
        os.system(monitor_line)
        self.goniometer.insert_frontlight()
        self.goniometer.set_frontlightlevel(50)
        self.md_task_info = []

    def clean(self):
        _start = time.time()
        self.detector.disarm()
        self.logger.info("detector disarm %.4f took" % (time.time() - _start))
        self.save_optical_history()
        self.goniometer.set_position(self.reference_position)
        self.collect_parameters()
        clean_jobs = []
        clean_jobs.append(gevent.spawn(self.save_parameters))
        clean_jobs.append(gevent.spawn(self.save_results))
        clean_jobs.append(gevent.spawn(self.save_log))
        if self.diagnostic == True:
            clean_jobs.append(gevent.spawn(self.save_diagnostics))
        if self.beware_of_download and self.generate_h5:
            clean_jobs.append(gevent.spawn(self.wait_for_expected_files))
        gevent.joinall(clean_jobs)

    def save_results(self):
        pass

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

    def get_oscillation_sequence(self):
        total_images = self.get_ntrigger() * self.get_nimages()
        oscillation_sequence = [
            {
                "exposure_time": self.get_frame_time(),
                "kappaStart": self.get_kappa(),
                "mesh_range": (),
                "num_images_per_trigger": self.get_nimages(),
                "num_triggers": self.get_ntrigger(),
                "number_of_images": total_images,
                "number_of_lines": 1,
                "number_of_passes": 1,
                "overlap": self.get_overlap(),
                "phiStart": self.get_scan_start_angle() + 360.0,
                "range": self.get_angle_per_frame(),
                "start": self.get_scan_start_angle(),
                "start_image_number": self.get_image_nr_start(),
            }
        ]
        return oscillation_sequence

    def get_sample_reference(self, sample_id=-2):
        sample_reference = {
            "blSampleId": sample_id,
            "cell": ",".join("0" * 6),
            "spacegroup": "",
            #"proteinAcronym": self.get_protein_acronym(),
            #"sampleName": self.get_prefix(),
        }
        return sample_reference

    def get_mxcube_collection_parameters(
        self, directory, name_pattern, session_id, sample_id, experiment_type="OSC"
    ):
        # proposal = self.get_proposal()
        # session_id = proposal["Session"]["SessionId"]
        print(f"get_mxcube_collection_parameters called with dire: {directory}, namp: {name_pattern}, seid: {session_id}, said: {sample_id}, expt: {experiment_type}")
        
        archive_directory = directory.replace("RAW_DATA", "ARCHIVE")
        snapshot = os.path.join(archive_directory, f"{name_pattern}.snapshot.jpeg")
        mcp = {
            "workflowTitle": "MSE",
            "workflowId": "MSE",
            "proteinAcronym": self.get_protein_acronym(),
            "protein_acronym": self.get_protein_acronym(),
            "sampleName": self.get_prefix(),
            "EDNA_files_dir": directory.replace("RAW_DATA", "PROCESSED_DATA"),
            "anomalous": False,
            "comments": "",
            "dark": False,
            "detector_binning_mode": 0,
            "detector_roi_mode": 0,
            "do_inducedraddam": False,
            "energy": self.get_photon_energy() / 1e3,
            "wavelength": self.get_wavelength(),
            "flux": self.get_flux(),
            # "flux_end": self.get_flux(),
            "experiment_type": experiment_type,
            "fileinfo": {
                "archive_directory": archive_directory,
                "compression": True,
                "directory": directory,
                "prefix": self.get_prefix(),
                "process_directory": directory.replace("RAW_DATA", "PROCESSED_DATA"),
                "run_number": self.get_run_number(),
                "run": self.get_run_number(),
                "template": f"{name_pattern}_%06d.h5",
            },
            "group_id": self.store_data_collection_group(experiment_type),
            "in_interleave": None,
            "in_queue": False,
            "motors": self.goniometer.translate_from_md_to_mxcube(
                self.reference_position
            ),
            "oscillation_sequence": self.get_oscillation_sequence(),
            "processing": "True",
            "processing_offline": True,
            "processing_online": False,
            "residues": 200,
            "resolution": {"upper": self.get_resolution()},
            "sample_reference": self.get_bl_sample(sample_id),
            "sessionId": session_id,
            "shutterless": True,
            "skip_images": True,
            "take_snapshots": True,
            "take_video": True,
            "transmission": self.get_transmission(),
            "xds_dir": directory.replace("RAW_DATA", "PROCESSED_DATA"),
            "synchrotronMode": "4/4",
            "xtalSnapshotFullPath1": snapshot,
        }
        return mcp

    def store_data_collection_in_lims(self, cp):
        # self.ispyb.talk({"store_data_collection": {"args": (cp,)}})
        self.collection_id = self.collect.talk(
            {"_store_data_collection_in_lims": {"args": (cp,)}}
        )
        return self.collection_id

    def store_sample_info_in_lims(self, cp):
        # self.ispyb.talk({"update_bl_sample": {"args": (cp,)}})
        self.collect.talk({"_store_sample_info_in_lims": {"args": (cp,)}})

    def update_data_collection_in_lims(self, cp):
        # self.ispyb.talk({"update_data_collection": {"args": (cp,)}})
        self.collect.talk({"_update_data_collection_in_lims": {"args": (cp,)}})

    def store_image_in_lims(self, cp, frame_number):
        # self.ispyb.talk({"update_data_collection": {"args": (cp,)}})
        self.collect.talk({"_store_image_in_lims": {"args": (cp, frame_number)}})

    def get_processing_filename(self, cp):
        processing_filename = self.collect.talk(
            {"get_processing_filename": {"args": (cp,)}}
        )
        return processing_filename

    def get_collection_id(self):
        collection_id = self.collect.talk({"get_collection_id": None})
        return collection_id

    def init_collect(self):
        self.collect.talk({"init": None})

    def get_bl_sample(self, sample_id):
        bl_sample = self.ispyb.talk({"get_bl_sample": {"args": (sample_id,)}})
        return bl_sample

    def run_analysis(self, processing_filename):
        # line = f'autoprocessing-px2 {processing_filename} &'
        # self.log.info(f"executing {line}")
        # os.system(line)
        self.collect.talk({"run_analysis": {"args": (processing_filename,)}})
