#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
The raster object allows to define and carry out a collection of series of diffraction still images on a grid specified over a rectangular area.
"""
import gevent

import time
import copy
import pickle
import scipy.misc
import numpy
import os
import numpy as np

from experimental_methods.experiment.diffraction_experiment import diffraction_experiment
from experimental_methods.utils.area import area
from experimental_methods.experiment.optical_alignment import optical_alignment
from experimental_methods.analysis.raster_scan_analysis import raster_scan_analysis


def height_model(angle, c, r, alpha, k):
    return c + r * np.cos(k * angle - alpha)


class raster_scan(diffraction_experiment):
    actuator_names = ["Omega", "AlignmentY", "AlignmentZ", "CentringX", "CentringY"]

    specific_parameter_fields = [
        {"name": "vertical_range", "type": "float", "description": ""},
        {"name": "horizontal_range", "type": "float", "description": ""},
        {"name": "number_of_rows", "type": "int", "description": ""},
        {"name": "number_of_columns", "type": "int", "description": ""},
        {"name": "scan_start_angle", "type": "float", "description": ""},
        {"name": "inverse_direction", "type": "bool", "description": ""},
        {"name": "use_centring_table", "type": "bool", "description": ""},
        {"name": "focus_center", "type": "float", "description": ""},
        {"name": "against_gravity", "type": "bool", "description": ""},
        {"name": "scan_axis", "type": "str", "description": ""},
        {"name": "scan_range", "type": "float", "description": ""},
        {"name": "frame_time", "type": "float", "description": ""},
        {"name": "jumps", "type": "array", "description": ""},
        {"name": "collect_sequence", "type": "array", "description": ""},
        {"name": "nframes", "type": "int", "description": ""},
        {"name": "beam_size", "type": "array", "description": ""},
        {"name": "reference_position", "type": "dict", "description": ""},
        {"name": "grid", "type": "array", "description": ""},
        {"name": "points", "type": "array", "description": ""},
        {"name": "shape", "type": "array", "description": ""},
        {"name": "angle_per_frame", "type": "float", "description": ""},
        {"name": "shutterless", "type": "bool", "description": ""},
        {
            "name": "nimages_per_scan",
            "type": "int",
            "description": "Number of points per grid point, only relevant in shuttered mode",
        },
        {
            "name": "npasses",
            "type": "int",
            "description": "Number of passes per grid point",
        },
        {
            "name": "dark_time_between_passes",
            "type": "float",
            "description": "Time in seconds between successive passes",
        },
        {"name": "motor_speed", "type": "float", "description": "Motor speed"},
        {
            "name": "maximum_motor_speed",
            "type": "float",
            "description": "Maximum motor speed",
        },
        {
            "name": "md_task_info",
            "type": "str",
            "description": "scan diagnostic information",
        },
    ]

    def __init__(
        self,
        name_pattern,
        directory,
        vertical_range,
        horizontal_range,
        beam_size=np.array([0.005, 0.010]),
        number_of_rows=None,
        number_of_columns=None,
        frame_time=0.005,
        scan_start_angle=None,
        scan_range=0.0,
        image_nr_start=1,
        position=None,
        kappa=None,
        phi=None,
        photon_energy=None,
        resolution=None,
        detector_distance=None,
        detector_vertical=None,
        detector_horizontal=None,
        transmission=None,
        flux=None,
        scan_axis="vertical",  # 'horizontal' or 'vertical'
        shutterless=True,
        nimages_per_scan=1,
        npasses=1,
        dark_time_between_passes=0.0,
        use_centring_table=True,
        inverse_direction=True,
        against_gravity=False,
        zoom=None,  # by default use the current zoom
        snapshot=True,
        diagnostic=None,
        analysis=None,
        simulation=None,
        conclusion=True,
        parent=None,
        beware_of_top_up=False,
        beware_of_download=False,
        generate_cbf=True,
        generate_h5=False,
        cats_api=None,
    ):
        if hasattr(self, "parameter_fields"):
            self.parameter_fields += raster_scan.specific_parameter_fields
        else:
            self.parameter_fields = raster_scan.specific_parameter_fields[:]

        self.vertical_range = vertical_range
        self.horizontal_range = horizontal_range
        if number_of_columns == None or number_of_rows == None:
            if type(beam_size) is str:
                beam_size = np.array(eval(beam_size))
            else:
                beam_size = beam_size
            shape = np.ceil(
                np.array((self.vertical_range, self.horizontal_range)) / beam_size
            ).astype(int)
            number_of_rows, number_of_columns = shape

        self.beam_size = beam_size
        self.shape = numpy.array((number_of_rows, number_of_columns))
        self.number_of_rows = number_of_rows
        self.number_of_columns = number_of_columns
        self.nframes = self.number_of_rows * self.number_of_columns
        self.frame_time = frame_time

        diffraction_experiment.__init__(
            self,
            name_pattern,
            directory,
            position=position,
            kappa=kappa,
            phi=phi,
            photon_energy=photon_energy,
            resolution=resolution,
            detector_distance=detector_distance,
            detector_vertical=detector_vertical,
            detector_horizontal=detector_horizontal,
            transmission=transmission,
            flux=flux,
            snapshot=snapshot,
            diagnostic=diagnostic,
            analysis=analysis,
            simulation=simulation,
            conclusion=conclusion,
            parent=parent,
            beware_of_top_up=beware_of_top_up,
            beware_of_download=beware_of_download,
            generate_cbf=generate_cbf,
            generate_h5=generate_h5,
            cats_api=cats_api,
        )

        print("number_of_rows", self.number_of_rows)
        print("number_of_columns", self.number_of_columns)
        print("scan_range", scan_range)

        if scan_start_angle == None:
            self.scan_start_angle = self.goniometer.get_omega_position()
        else:
            self.scan_start_angle = scan_start_angle
        self.scan_range = scan_range

        self.image_nr_start = image_nr_start
        if position == None:
            self.reference_position = self.goniometer.get_aligned_position()
        else:
            self.reference_position = position
        self.scan_axis = scan_axis
        self.shutterless = shutterless
        self.nimages_per_scan = nimages_per_scan
        self.npasses = npasses
        self.dark_time_between_passes = dark_time_between_passes
        self.inverse_direction = inverse_direction
        self.use_centring_table = use_centring_table
        self.against_gravity = against_gravity
        self.zoom = zoom

        if self.scan_axis in ["vertical", b"vertical"]:
            print("am vertical")
            self.line_scan_time = self.frame_time * self.number_of_rows
            self.motor_speed = self.vertical_range / self.line_scan_time
            self.angle_per_frame = scan_range / self.number_of_rows
            self.ntrigger = self.number_of_columns
            self.nimages = self.number_of_rows
            self.nimages_per_file = self.number_of_rows
        else:
            print("am horizontal")
            self.line_scan_time = self.frame_time * self.number_of_columns
            self.motor_speed = self.horizontal_range / self.line_scan_time
            self.angle_per_frame = scan_range / self.number_of_columns
            self.ntrigger = self.number_of_rows
            self.nimages = self.number_of_columns
            self.nimages_per_file = self.number_of_columns

        print("motor_speed", self.motor_speed)
        print("ntrigger", self.ntrigger)
        print("nimages", self.nimages)
        print("nimages_per_file", self.nimages_per_file)

        self.total_expected_exposure_time = self.line_scan_time * self.ntrigger
        self.total_expected_wedges = self.ntrigger

        self.description = (
            "X-ray Diffraction raster scan, Proxima 2A, SOLEIL, %s"
            % time.ctime(self.timestamp)
        )

    def get_distance(self):
        return self.get_extent()

    def get_motor_speed(self):
        return self.motor_speed

    def get_scan_axis(self):
        return self.scan_axis

    def get_frame_time(self):
        return self.frame_time

    def get_vertical_step_size(self):
        return self.get_step_sizes()[0]

    def get_horizontal_step_size(self):
        return self.get_step_sizes()[1]

    def get_beam_vertical_position(self):
        return self.camera.md.beampositionvertical

    def get_beam_horizontal_position(self):
        return self.camera.md.beampositionhorizontal

    def get_extent(self):
        return numpy.array((self.vertical_range, self.horizontal_range))

    def get_step_sizes(self):
        step_sizes = self.get_extent() / numpy.array((self.shape))
        return step_sizes

    def get_nimages_per_file(self):
        # if self.shutterless == True and self.scan_axis == "vertical":
        # nimages_per_file = self.number_of_rows
        # elif self.shutterless == True:
        # nimages_per_file = self.number_of_columns
        # elif self.npasses > 1:
        # nimages_per_file = self.npasses * self.nimages_per_scan
        # else:
        # nimages_per_file = self.nimages
        return int(self.nimages_per_file)

    def get_frames_per_second(self):
        return 1.0 / self.get_frame_time()

    def run(self):
        task_id, grid, shifts = self.goniometer.raster_scan(
            self.vertical_range,
            self.horizontal_range,
            number_of_rows=self.number_of_rows,
            number_of_columns=self.number_of_columns,
            position=self.reference_position,
            scan_start_angle=self.scan_start_angle,
            scan_range=self.scan_range,
            frame_time=self.get_frame_time(),
            inverse_direction=self.inverse_direction,
            use_centring_table=self.use_centring_table,
            number_of_passes=self.npasses,
            dark_time_between_passes=self.dark_time_between_passes,
            number_of_frames=self.nimages_per_scan,
        )
        self.task_id = task_id
        self.grid = grid
        self.points = shifts

    def analyze(self):
        pass
        # spot_find_line = 'ssh process1 "source /usr/local/dials-v1-4-5/dials_env.sh; cd %s ; echo $(pwd); dials.find_spots shoebox=False per_image_statistics=True spotfinder.filter.ice_rings.filter=True nproc=80 ../%s_master.h5"' % (self.process_directory, self.name_pattern)
        # os.system(spot_find_line)
        # area_sense_line = '/927bis/ccd/gitRepos/eiger/area_sense.py -d %s -n %s &' % (self.directory, self.name_pattern)
        # command = '/home/experiences/proxima2a/com-proxima2a/mxcube_local/HardwareRepository/HardwareObjects/SOLEIL/PX2/experimental_methods/raster_scan_analysis.py'

        # command = 'raster_scan_analysis.py'
        # area_sense_line = '%s -d %s -n %s &' % (command, self.directory, self.name_pattern)
        # print('raster_scan_analysis line: %s' % area_sense_line)
        # os.system(area_sense_line)

    def conclude(self):
        rsa = raster_scan_analysis(self.name_pattern, self.directory)
        optimum_position = rsa.get_optimum_position()
        rsa.save_overlay_image(imagename=self.get_overlay_image_name())
        print("optimum_position", optimum_position)
        self.goniometer.set_position(optimum_position)
        rsa.save_report()

    def get_overlay_image_name(self):
        return "%s_z.png" % self.get_template()


def main():
    import optparse

    parser = optparse.OptionParser()
    parser.add_option(
        "-n",
        "--name_pattern",
        default="raster_test_$id",
        type=str,
        help="Prefix default=%default",
    )
    parser.add_option(
        "-d",
        "--directory",
        default="/nfs/data/default",
        type=str,
        help="Destination directory default=%default",
    )
    parser.add_option(
        "-y", "--vertical_range", default=0.1, type=float, help="Vertical range in mm"
    )
    parser.add_option(
        "-x",
        "--horizontal_range",
        default=0.2,
        type=float,
        help="Horizontal range in mm",
    )
    parser.add_option(
        "-r", "--number_of_rows", default=None, type=int, help="Number of rows"
    )
    parser.add_option(
        "-c", "--number_of_columns", default=None, type=int, help="Number of columns"
    )
    parser.add_option(
        "-b", "--beam_size", default="(0.005, 0.010)", type=str, help="Beam size in mm"
    )
    parser.add_option(
        "-a",
        "--scan_start_angle",
        default=None,
        type=float,
        help="Scan start angle [deg]",
    )
    parser.add_option(
        "-i",
        "--position",
        default=None,
        type=str,
        help="Gonio alignment position [dict]",
    )
    parser.add_option(
        "-s",
        "--scan_range",
        default=0.1,
        type=float,
        help="Scan range [deg] per helical line (-> 0)",
    )
    parser.add_option(
        "-e",
        "--frame_time",
        default=0.05,
        type=float,
        help="Exposure time per image [s]",
    )
    parser.add_option(
        "-f", "--image_nr_start", default=1, type=int, help="Start image number [int]"
    )
    parser.add_option(
        "-p", "--photon_energy", default=None, type=float, help="Photon energy "
    )
    parser.add_option(
        "-t", "--detector_distance", default=None, type=float, help="Detector distance"
    )
    parser.add_option(
        "-o", "--resolution", default=None, type=float, help="Resolution [Angstroem]"
    )
    parser.add_option("-X", "--flux", default=None, type=float, help="Flux [ph/s]")
    parser.add_option(
        "-m",
        "--transmission",
        default=None,
        type=float,
        help="Transmission. Number in range between 0 and 1.",
    )
    parser.add_option(
        "-I", "--inverse_direction", action="store_true", help="Rastered acquisition"
    )
    parser.add_option(
        "-G",
        "--against_gravity",
        action="store_true",
        help="Vertical scan direction against gravity",
    )
    parser.add_option(
        "-T",
        "--do_not_use_centring_table",
        action="store_true",
        help="Do not use centring table for vertical sample movements.",
    )
    parser.add_option(
        "-z", "--zoom", default=None, type=int, help="Zoom to acquire optical image at."
    )
    parser.add_option(
        "-A",
        "--analysis",
        action="store_true",
        help="If set will perform automatic analysis.",
    )
    parser.add_option(
        "-C",
        "--conclusion",
        action="store_true",
        help="If set will move the motors upon analysis.",
    )
    parser.add_option(
        "-D",
        "--diagnostic",
        action="store_true",
        help="If set will record diagnostic information.",
    )
    parser.add_option(
        "-S", "--simulation", action="store_true", help="If set will simulate the run."
    )
    parser.add_option(
        "-O",
        "--optical_alignment_results",
        default=None,
        type=str,
        help="Use results from optical alignment analysis to specify the raster parameters",
    )
    parser.add_option(
        "--max",
        action="store_true",
        help="To be used with -O parameter, use parameters for max area raster.",
    )
    parser.add_option(
        "--min",
        action="store_true",
        help="To be used with -O parameter, use parameters for min area raster.",
    )
    parser.add_option(
        "-V",
        "--vertical_plus",
        default=0.0,
        type=float,
        help="To be used with -O parameter, specify in mm by how much the vertical scan range should be increased compared to values from optical scan analysis.",
    )
    parser.add_option(
        "-H",
        "--horizontal_plus",
        default=0.0,
        type=float,
        help="To be used with -O parameter, specify in mm by how much the horizontal scan range should be increased compared to values from optical scan analysis.",
    )
    parser.add_option(
        "-P",
        "--angle_offset",
        default=0.0,
        type=float,
        help="To be used with -O parameter, specify in degrees angle offset with respect to min (if --min specified) or max (if --max specified) orientations.",
    )
    parser.add_option(
        "-M", "--motor_speed", default=None, type=float, help="Motor speed [mm/s]"
    )
    parser.add_option(
        "-N",
        "--scan_duration",
        default=None,
        type=float,
        help="Scan duration (per line) [s]",
    )
    parser.add_option(
        "--scan_axis",
        default="vertical",
        type=str,
        help="Scan axis (vertical or horizontal) default=%default",
    )
    parser.add_option(
        "--shuttered",
        action="store_true",
        help="Collect in shuttered mode. The default is shutterless.",
    )
    parser.add_option(
        "--nimages_per_scan",
        default=1,
        type=int,
        help="Images per point. Only relevant in shuttered mode. [int]",
    )

    options, args = parser.parse_args()

    print("options", options)
    print("args", args)
    print()
    filename = (
        os.path.join(options.directory, options.name_pattern) + "_parameters.pickle"
    )

    if options.shuttered == True:
        options.shutterless = False

    if options.do_not_use_centring_table == True:
        options.use_centring_table = False
    else:
        options.use_centring_table = True

    del options.do_not_use_centring_table

    if options.optical_alignment_results != None:
        oar = pickle.load(open(options.optical_alignment_results))
        oap = pickle.load(
            open(
                options.optical_alignment_results.replace(
                    "_results.pickle", "_parameters.pickle"
                )
            )
        )
        if "result_position" in oar:
            position = oar["result_position"]
        else:
            oa = optical_alignment(oap["name_pattern"], oap["directory"])
            reference_optical_scan_position = oap["position"]
            optical_scan_move_vector_mm = oar["move_vector_mm"]
            position = oa.get_result_position(
                reference_position=reference_optical_scan_position,
                move_vector_mm=optical_scan_move_vector_mm,
            )

        if options.min == True:
            horizontal_range, vertical_range, scan_start_angle, zoom = oar[
                "min_raster_parameters"
            ]
        else:
            horizontal_range, vertical_range, scan_start_angle, zoom = oar[
                "max_raster_parameters"
            ]

        if options.angle_offset != 0:
            scan_start_angle += options.angle_offset
            height_fit = oar["fits"]["height"]
            c, r, alpha = height_fit[0].x
            k = height_fit[1]
            vertical_range = (
                height_model(np.radians(scan_start_angle), c, r, alpha, k)
                * oap["calibration"][0]
            )

        position["Omega"] = scan_start_angle

        if options.vertical_plus != 0.0:
            vertical_range += options.vertical_plus
        if options.horizontal_plus != 0.0:
            horizontal_range += options.horizontal_plus
            position["AlignmentY"] += options.horizontal_plus / 2.0

        options.position = position
        options.vertical_range = vertical_range
        options.horizontal_range = horizontal_range
        options.scan_start_angle = scan_start_angle
        options.zoom = zoom

    if options.inverse_direction != True:
        options.inverse_direction = False
    if options.against_gravity != True:
        options.against_gravity = False

    if options.scan_duration != None:
        motor_speed = options.vertical_range / options.scan_duration
        nimages = int(options.scan_duration / options.frame_time)
        vertical_step = options.vertical_range / nimages
        if type(options.beam_size) is str:
            beam_size = np.array(eval(options.beam_size))
        else:
            beam_size = options.beam_size
        beam_size[0] = vertical_step
        options.beam_size = beam_size

    elif options.motor_speed != None:
        scan_duration = options.vertical_range / options.motor_speed
        nimages = int(scan_duration / options.frame_time)
        vertical_step = options.vertical_range / nimages
        if type(options.beam_size) is str:
            beam_size = np.array(eval(options.beam_size))
        else:
            beam_size = options.beam_size
        beam_size[0] = vertical_step
        options.beam_size = beam_size

    print()
    print(
        "options after update from optical scan analysis results and priority options",
        options,
    )
    print()
    del options.min
    del options.max
    del options.optical_alignment_results
    del options.vertical_plus
    del options.horizontal_plus
    del options.angle_offset
    del options.motor_speed
    del options.scan_duration
    del options.shuttered

    r = raster_scan(**vars(options))

    if not os.path.isfile(filename):
        r.execute()
    elif options.analysis == True:
        r.analyze()
        if options.conclusion == True:
            r.conclude()


if __name__ == "__main__":
    main()
