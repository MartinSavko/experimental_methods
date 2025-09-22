#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


class nested_helical(object):
    motors = ["AlignmentX", "AlignmentY", "AlignmentZ", "CentringY", "CentringX"]

    def __init__(
        self,
        name_pattern,
        directory,
        start,  # dictionary of motor positions
        stop,  # dictionary of motor positions
        vertical_range,  # vertical single sweep range in milimiters
        frame_time=0.1,  # in seconds
        number_of_points=None,  #
        oscillation_start_angle=0,  # angle in degrees
        total_oscillation_range=180,  # angle in degrees
        degrees_per_frame=0.1,  # angle per image in degrees
        degrees_of_overlap_between_neighboring_sweeps=1,  # angle in degrees
        beam_horizontal_size=0.01,
    ):  # beam width in mmm
        self.name_pattern = name_pattern
        self.directory = directory
        self.start = start
        self.stop = stop
        self.vertical_range = vertical_range
        self.beam_horizontal_size = beam_horizontal_size
        if number_of_points is None:
            start_vector = np.array([start[motor] for motor in self.motors])
            stop_vector = np.array([stop[motor] for motor in self.motors])
            scan_length = np.linalg.norm(stop_vector - start_vector)
            self.number_of_points = int(
                np.floor(scan_length / self.beam_horizontal_size)
            )
        print("total length of principal helical line is %s" % scan_length)
        print(
            "experiment will consist of %s vertical helical sweeps"
            % self.number_of_points
        )
        self.oscillation_start_angle = oscillation_start_angle
        self.total_oscillation_range = total_oscillation_range
        self.degrees_per_frame = degrees_per_frame
        self.degrees_of_overlap_between_neighboring_sweeps = (
            degrees_of_overlap_between_neighboring_sweeps
        )
        self.frame_time = frame_time

        self.detector = detector()
        self.goniometer = goniometer()

    def get_range_per_vertical_sweep(self):
        return (
            self.total_oscillation_range / self.number_of_points
            + self.degrees_of_overlap_between_neighboring_sweeps
        )

    def get_vertical_scan_exposure_time(self):
        return (
            self.frame_time
            * self.get_range_per_vertical_sweep()
            / self.degrees_per_frame
        )

    def get_number_of_images_per_vertical_sweep(self):
        return int(self.get_range_per_vertical_sweep() / self.degrees_per_frame)

    def get_collect_points(self):
        step_template = np.linspace(
            0.0, 1.0, self.number_of_points
        )  # actual individual motor positions will be calculated by multiplying this template by a corresponding range and adding the start value

        motor_helical_positions_dictionary = {}
        for motor in self.motors:
            motor_helical_positions_dictionary[motor] = (
                self.start[motor]
                + (self.stop[motor] - self.start[motor]) * step_template
            )

        motor_helical_positions_dictionary["Omega"] = (
            self.oscillation_start_angle + self.total_oscillation_range * step_template
        )

        motor_helical_positions_list = []
        for n in range(self.number_of_points):
            position = {}
            for motor in motor_helical_positions_dictionary:
                position[motor] = motor_helical_positions_dictionary[motor][n]
            motor_helical_positions_list.append(position)

        return motor_helical_positions_list

    def get_vertical_start_and_end_positions(self, position):
        start = copy.copy(position)
        end = copy.copy(position)

        start["AlignmentZ"] += self.vertical_range / 2.0
        end["AlignmentZ"] -= self.vertical_range / 2.0

        return start, end

    def program_detector(self):
        nimages = self.get_number_of_images_per_vertical_sweep()
        ntrigger = self.number_of_points
        self.detector_parameters = detector_parameters(
            self.name_pattern, self.directory, self.frame_time, nimages, ntrigger
        )
        self.detector_parameters.prepare()
        self.detector_parameters.set_parameters()

    def program_goniometer(self):
        self.goniometer.md.backlightison = False
        self.goniometer.set_position(self.start)
        self.goniometer.wait()

    def prepare(self):
        self.program_goniometer()
        self.program_detector()
        self.detector.arm()

    def collect(self):
        motor_helical_positions_list = self.get_collect_points()
        range_per_vertical_sweep = self.get_range_per_vertical_sweep()
        vertical_scan_exposure_time = self.get_vertical_scan_exposure_time()

        self.prepare()

        for position in motor_helical_positions_list:
            vertical_start, vertical_end = self.get_vertical_start_and_end_positions(
                position
            )
            start_angle = (
                position["Omega"] - self.degrees_of_overlap_between_neighboring_sweeps
            )
            self.goniometer.helical_scan(
                vertical_start,
                vertical_end,
                start_angle,
                range_per_vertical_sweep,
                vertical_scan_exposure_time,
            )
