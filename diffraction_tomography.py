#!/usr/bin/env python
# -*- coding: utf-8 -*-

import traceback
import logging
import time
import numpy as np
import copy
import os
import subprocess
import pickle
import re
import pylab
import sys
import gevent
import scipy.ndimage as nd
from scipy.optimize import minimize

from diffraction_experiment import diffraction_experiment
from diffraction_experiment_analysis import diffraction_experiment_analysis
from area import area
from useful_routines import fit_circle, get_model_parameters

class diffraction_tomography(diffraction_experiment):
    specific_parameter_fields = [
        {"name": "scan_start_angles", "type": "list", "description": ""},
        {"name": "vertical_range", "type": "bool", "description": ""},
        {"name": "vertical_step_size", "type": "bool", "description": ""},
        {"name": "reference_position", "type": "dict", "description": ""},
        {
            "name": "md_task_info",
            "type": "str",
            "description": "scan diagnostic information",
        },
        {
            "name": "motor_speed",
            "type": "float",
            "description": "translational scan speed",
        },
    ]

    def __init__(
        self,
        name_pattern="excenter_$id",
        directory="/nfs/data2/excenter",
        treatment_directory="/dev/shm",
        scan_start_angles="[0, 90, 180, 225, 315]",
        vertical_range=0.25,
        horizontal_range=0,
        scan_range=0.01,
        vertical_step_size=0.002,
        frame_time=0.005,
        transmission=None,
        position=None,
        photon_energy=None,
        resolution=None,
        diagnostic=False,
        analysis=True,
        conclusion=True,
        display=True,
        method="tioga",
        dont_move_motors=False,
        parent=None,
        beware_of_top_up=False,
        beware_of_download=False,
        generate_cbf=True,
        generate_h5=False,
        image_nr_start=1,
        cats_api=None,
    ):
        if hasattr(self, "parameter_fields"):
            self.parameter_fields += diffraction_tomography.specific_parameter_fields
        else:
            self.parameter_fields = diffraction_tomography.specific_parameter_fields[:]

        self.default_experiment_name = "X-ray diffraction tomgraphy"
        
        diffraction_experiment.__init__(
            self,
            name_pattern,
            directory,
            transmission=transmission,
            photon_energy=photon_energy,
            resolution=resolution,
            diagnostic=diagnostic,
            analysis=analysis,
            conclusion=conclusion,
            parent=parent,
            beware_of_top_up=beware_of_top_up,
            beware_of_download=beware_of_download,
            generate_cbf=generate_cbf,
            generate_h5=generate_h5,
            cats_api=cats_api,
        )

        self.display = display
        self.method = method

        self.scan_start_angles = eval(scan_start_angles)
        self.vertical_range = vertical_range
        self.horizontal_range = horizontal_range
        self.vertical_step_size = vertical_step_size
        self.frame_time = frame_time
        self.nimages = int(vertical_range / vertical_step_size)
        self.scan_exposure_time = self.frame_time * self.nimages
        self.scan_range = scan_range
        self.ntrigger = len(self.scan_start_angles)
        self.number_of_rows = int(vertical_range / vertical_step_size)
        self.number_of_columns = 1

        self.motor_speed = self.vertical_range / (self.number_of_rows * self.frame_time)
        print("number_of_rows", self.number_of_rows)
        print("number_of_columns", self.number_of_columns)
        print("motor_speed", self.motor_speed)
        print("scan_range", scan_range)

        if position == None:
            self.reference_position = self.goniometer.get_aligned_position()
        else:
            self.reference_position = position

        self.nimages_per_file = self.number_of_rows
        self.scan_start_angle = self.scan_start_angles[0]
        self.angle_per_frame = self.scan_range / self.nimages
        self.image_nr_start = image_nr_start
        self.treatment_directory = treatment_directory
        self.format_dictionary = {
            "directory": self.directory,
            "name_pattern": self.name_pattern,
            "treatment_directory": self.treatment_directory,
        }

        self.line_scan_time = self.frame_time * self.number_of_rows
        self.total_expected_exposure_time = self.line_scan_time * self.ntrigger
        self.total_expected_wedges = self.ntrigger
        self.overlap = 0.0
        self.dta = diffraction_experiment_analysis(
            directory=self.directory, name_pattern=self.name_pattern
        )

    
    def get_overlap(self):
        return self.overlap

    def get_helical_lines(self, model="MD3Up"):
        # start, stop, scan_start_angle, scan_range, scan_exposure_time
        helical_lines = []
        for scan_start_angle in self.scan_start_angles:
            position = copy.copy(self.reference_position)
            position["Omega"] = scan_start_angle
            position_start = (
                self.goniometer.get_aligned_position_from_reference_position_and_shift(
                    position,
                    self.vertical_range / 2,
                    0,
                    AlignmentZ_reference=position["AlignmentZ"],
                )
            )
            position_stop = (
                self.goniometer.get_aligned_position_from_reference_position_and_shift(
                    position,
                    -self.vertical_range / 2,
                    0,
                    AlignmentZ_reference=position["AlignmentZ"],
                )
            )

            helical_line = [
                position_start,
                position_stop,
                scan_start_angle,
                self.scan_range,
                self.scan_exposure_time,
            ]
            helical_lines.append(helical_line)
            print(f"helical line {helical_line}")

        return helical_lines

    def get_reference_position(self):
        if os.path.isfile(self.get_parameters_filename()):
            self.reference_position = self.load_parameters_from_file()[
                "reference_position"
            ]
        return self.reference_position

    def run(self):
        self.md_task_info = []
        for helical_line in self.get_helical_lines():
            start, stop, scan_start_angle, scan_range, scan_exposure_time = helical_line
            task_id = self.goniometer.helical_scan(
                start, stop, scan_start_angle, scan_range, scan_exposure_time
            )
            self.md_task_info.append(self.goniometer.get_task_info(task_id))

    def analyze(self, method=None):
        if method is None and self.method is not None:
            method = self.method
        else:
            method = "tioga"

        if method == "dozor":
            self.run_dozor(blocking=True)
        elif method == "xds":
            self.run_xds()
        elif method == "dials":
            self.run_dials()
        elif method == "tioga":
            pass

    def get_results(self, method=None):
        if method is None and self.method is not None:
            method = self.method
        else:
            method = "tioga"

        self.logger.info("get_results, method %s" % method)
        if not self.analysis and not self.conclusion:
            return []
        if method == "dozor":
            results = self.get_dozor_results()[:, 2]
            print("results", results.shape, results[:10])
        elif method == "dials":
            results = self.get_dials_results()
        elif method == "xds":
            results = self.dta.get_xds_results()
        elif method == "tioga":
            results = self.dta.get_tioga_results()
        return results

    def get_result_position(
        self,
        threshold=0.25,
        min_spots=7,
        alignmenty_direction=-1.0,
        alignmentz_direction=1.0,
        centringx_direction=-1.0,
        centringy_direction=-1.0,
        method=None,
        geometric_center=True,
    ):
        self.logger.info("get_result_position")

        parameters = self.get_parameters()
        results = self.get_results(method=method)

        nimages = parameters["nimages"]
        angles = parameters["scan_start_angles"]

        vertical_displacements = []
        # beam_position = 0.5 * (self.nimages-1.)
        # print('beam_position', beam_position)

        for k in range(parameters["ntrigger"]):
            line = results[k * nimages : (k + 1) * nimages]
            line[line < min_spots] = 0
            line[line <= line.max() * threshold] = 0
            if geometric_center:
                line[line > 0] = 1
            y = nd.center_of_mass(line)[0]
            print("center_of_mass", y)
            # y -= beam_position
            # print('position in steps', y)
            # y *= parameters['vertical_step_size']
            # print('shift in mm', y)
            vertical_displacements.append(y)

        angles_radians = np.radians(parameters["scan_start_angles"])
        print("vertical_displacements", vertical_displacements)
        vertical_displacements = np.array(vertical_displacements)
        # vertical_displacements *= 1e3
        initial_parameters = [
            np.mean(vertical_displacements),
            np.std(vertical_displacements),
            np.random.random(),
        ]
        print("initial_parameters", initial_parameters)

        fit_y = fit_circle(angles_radians, vertical_displacements)
        
        #fit_y = minimize(
            #circle_model_residual,
            #initial_parameters,
            #method="nelder-mead",
            #args=(angles_radians, vertical_displacements),
        #)
        print("fit_y", fit_y)
        c, r, alpha = get_model_parameters(fit_y.params, ["c", "r", "alpha"])
        
        omega_axis_position = c
        print("omega_axis_position", omega_axis_position)
        omega_axis_shift = omega_axis_position - 0.5 * nimages
        print("estimated omega_axis_shift in px", omega_axis_shift)
        print(
            "estimated omega_axis_shift in mm",
            omega_axis_shift * parameters["vertical_step_size"],
        )

        # c *= parameters['vertical_step_size']
        c = omega_axis_shift * parameters["vertical_step_size"]
        r *= parameters["vertical_step_size"]
        v = {"c": c, "r": r, "alpha": alpha}
        print("c, r, alpha", c, r, alpha)
        d_sampx = centringx_direction * r * np.sin(alpha)
        d_sampy = centringy_direction * r * np.cos(alpha)
        # d_y = alignmenty_direction * horizontal_center
        d_z = alignmentz_direction * c

        move_vector_dictionary = {
            "AlignmentZ": d_z,
            #'AlignmentY': d_y,
            "CentringX": d_sampx,
            "CentringY": d_sampy,
        }

        print("move_vector", move_vector_dictionary)

        result_position = {}
        reference_position = self.get_reference_position()
        for motor in reference_position:
            result_position[motor] = reference_position[motor]
            if motor in move_vector_dictionary:
                result_position[motor] += move_vector_dictionary[motor]
        self.logger.info(f"reference_position {reference_position}")
        self.logger.info(f"result_position {result_position}")
        return result_position

    def run_shape_reconstruction(self, display=True, method=None):
        if method is None and self.method is not None:
            method = self.method
        else:
            method = "tioga"

        line = (
            line
        ) = f"shape_from_diffraction_tomography.py -d {self.directory} -n {self.name_pattern} -M {method} -D 1>/dev/null &"
        if not display:
            line = line.replace("-D 1>/dev/null", " 1>/dev/null")
        print(line)
        os.system(line)

    def conclude(self, method=None, move_motors=True):
        self.logger.info("conclude")
        self.run_shape_reconstruction(display=self.display)
        result_position = self.get_result_position(method=method)
        if move_motors:
            self.logger.info("moving motors")
            self.goniometer.set_position(result_position)
            self.goniometer.save_position()


def main():
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,)

    parser.add_argument(
        "-n", "--name_pattern", default="excenter_$id", type=str, help="Prefix"
    )
    parser.add_argument(
        "-d",
        "--directory",
        default="/nfs/data2/excenter",
        type=str,
        help="Destination directory",
    )
    parser.add_argument(
        "-a",
        "--scan_start_angles",
        default="[0, 90, 180, 225, 315]",
        type=str,
        help="angles",
    )
    parser.add_argument(
        "-y", "--vertical_range", default=0.1, type=float, help="vertical range"
    )
    parser.add_argument(
        "-f", "--frame_time", default=0.005, type=float, help="frame time"
    )
    parser.add_argument(
        "-A",
        "--analysis",
        action="store_true",
        help="If set will perform automatic analysis.",
    )
    parser.add_argument(
        "-C",
        "--conclusion",
        action="store_true",
        help="If set will move the motors upon analysis.",
    )
    parser.add_argument(
        "-D",
        "--diagnostic",
        action="store_true",
        help="If set will record diagnostic information.",
    )
    parser.add_argument(
        "-m", "--method", type=str, default="tioga", help="analysis method"
    )
    parser.add_argument(
        "-S",
        "--dont_move_motors",
        action="store_true",
        help="Do not move after conclusion",
    )
    parser.add_argument(
        "-5", "--generate_h5", action="store_false", help="Do not generate h5 files"
    )
    options = parser.parse_args()

    print("options", options)
    print("vars(options)", vars(options))

    experiment = diffraction_tomography(**vars(options))
    print("get_parameters_filename", experiment.get_parameters_filename())
    if not os.path.isfile(experiment.get_parameters_filename()):
        experiment.execute()
    elif options.analysis == True:
        experiment.analyze(method=options.method)
        if options.conclusion == True:
            experiment.conclude(
                method=options.method, move_motors=not options.dont_move_motors
            )


if __name__ == "__main__":
    main()
