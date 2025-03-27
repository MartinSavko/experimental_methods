#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
single position oscillation scan
"""

import os
import time
import pickle
import logging
import traceback
import gevent

from diffraction_experiment import diffraction_experiment


class omega_scan(diffraction_experiment):
    """Will execute single continuous omega scan"""

    actuator_names = ["Omega"]

    specific_parameter_fields = [
        {
            "name": "position",
            "type": "dict",
            "description": "dictionary with motor names as keys and their positions in mm as values",
        },
        {"name": "scan_range", "type": "float", "description": "scan range in degrees"},
        {
            "name": "scan_exposure_time",
            "type": "float",
            "description": "scan exposure time in s",
        },
        {
            "name": "scan_start_angle",
            "type": "float",
            "description": "scan start angle in degrees",
        },
        {
            "name": "angle_per_frame",
            "type": "float",
            "description": "angle per frame in degrees",
        },
        {"name": "frame_time", "type": "float", "description": "frame time in s"},
        {
            "name": "degrees_per_second",
            "type": "float",
            "description": "frame range in degrees",
        },
        {
            "name": "degrees_per_frame",
            "type": "float",
            "description": "angle per frame in degrees",
        },
        {
            "name": "scan_speed",
            "type": "float",
            "description": "scan speed in degrees per second",
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
        scan_range=180,
        scan_exposure_time=18,
        scan_start_angle=0,
        angle_per_frame=0.1,
        image_nr_start=1,
        frames_per_second=None,
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
        snapshot=False,
        ntrigger=1,
        nimages_per_file=400,
        zoom=None,
        diagnostic=None,
        analysis=None,
        simulation=None,
        shift=None,
        parent=None,
        mxcube_parent_id=None,
        mxcube_gparent_id=None,
        beware_of_top_up=True,
        beware_of_download=False,
        generate_cbf=True,
        generate_h5=True,
    ):
        if hasattr(self, "parameter_fields"):
            self.parameter_fields += omega_scan.specific_parameter_fields
        else:
            self.parameter_fields = omega_scan.specific_parameter_fields[:]

        logging.debug(
            "omega_scan __init__ len(omega_scan.specific_parameter_fields) %d"
            % len(omega_scan.specific_parameter_fields)
        )
        logging.debug(
            "omega_scan __init__ len(self.parameters_fields) %d"
            % len(self.parameter_fields)
        )
        diffraction_experiment.__init__(
            self,
            name_pattern,
            directory,
            frames_per_second=frames_per_second,
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
            ntrigger=ntrigger,
            zoom=zoom,
            diagnostic=diagnostic,
            analysis=analysis,
            simulation=simulation,
            parent=parent,
            mxcube_parent_id=mxcube_parent_id,
            mxcube_gparent_id=mxcube_gparent_id,
            beware_of_top_up=beware_of_top_up,
            beware_of_download=beware_of_download,
            generate_cbf=generate_cbf,
            generate_h5=generate_h5,
        )

        self.description = "Omega scan, Proxima 2A, SOLEIL, %s" % time.ctime(
            self.timestamp
        )
        # Scan parameters
        self.scan_range = float(scan_range)
        self.scan_exposure_time = float(scan_exposure_time)
        if type(scan_start_angle) is type(None):
            scan_start_angle = self.goniometer.get_omega_position()
        self.scan_start_angle = float(scan_start_angle) % 360
        self.angle_per_frame = float(angle_per_frame)
        self.image_nr_start = int(image_nr_start)
        self.position = self.goniometer.check_position(position)
        self.reference_position = self.position

        self.shift = shift

        if self.shift != None:
            self.position["AlignmentY"] += self.shift

        self.nimages_per_file = nimages_per_file
        self.total_expected_exposure_time = self.scan_exposure_time
        self.total_expected_wedges = 1

    def get_nimages(self, epsilon=1e-3):
        nimages = int(self.scan_range / self.angle_per_frame)
        if abs(nimages * self.angle_per_frame - self.scan_range) > epsilon:
            nimages += 1
        return nimages

    def run(self, wait=True):
        """execute omega scan."""

        if (
            self.beware_of_top_up
            and self.scan_exposure_time <= self.machine_status.get_top_up_period()
        ):
            self.check_top_up()

        task_id = self.goniometer.omega_scan(
            self.scan_start_angle, self.scan_range, self.scan_exposure_time, wait=wait
        )

        self.md_task_info = self.goniometer.get_task_info(task_id)

    def analyze(self):
        xdsme_process_line = (
            'ssh -X process1 "goxdsme --brute -p xdsme_auto_{name_pattern}"'.format(
                name_pattern=self.name_pattern
            )
        )
        print("xdsme process_line", xdsme_process_line)
        terminal = "gnome-terminal --title \"xdsme {name_pattern}\" --hide-menubar --geometry 80x40+0+0 --execute bash -c '{xdsme_process_line}; bash '".format(
            name_pattern=os.path.basename(self.name_pattern),
            xdsme_process_line=xdsme_process_line,
        )
        self.logger.info("xdsme_process_line %s" % xdsme_process_line)


def main():
    import optparse

    parser = optparse.OptionParser()
    parser.add_option(
        "-n",
        "--name_pattern",
        default="test_$id",
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
        "-r", "--scan_range", default=45, type=float, help="Scan range [deg]"
    )
    parser.add_option(
        "-e",
        "--scan_exposure_time",
        default=4.5,
        type=float,
        help="Scan exposure time [s]",
    )
    parser.add_option(
        "-s", "--scan_start_angle", default=0, type=float, help="Scan start angle [deg]"
    )
    parser.add_option(
        "-a", "--angle_per_frame", default=0.1, type=float, help="Angle per frame [deg]"
    )
    parser.add_option(
        "-f", "--image_nr_start", default=1, type=int, help="Start image number [int]"
    )
    parser.add_option(
        "-N",
        "--nimages_per_file",
        default=100,
        type=int,
        help="Number of images per data file [int]",
    )
    parser.add_option(
        "-i",
        "--position",
        default=None,
        type=str,
        help="Gonio alignment position [dict]",
    )
    parser.add_option(
        "-p", "--photon_energy", default=None, type=float, help="Photon energy "
    )
    parser.add_option(
        "-t", "--detector_distance", default=None, type=float, help="Detector distance"
    )
    parser.add_option(
        "-Z",
        "--detector_vertical",
        default=None,
        type=float,
        help="Detector vertical position",
    )
    parser.add_option(
        "-X",
        "--detector_horizontal",
        default=None,
        type=float,
        help="Detector horizontal position",
    )
    parser.add_option(
        "-o", "--resolution", default=None, type=float, help="Resolution [Angstroem]"
    )
    parser.add_option("-x", "--flux", default=None, type=float, help="Flux [ph/s]")
    parser.add_option(
        "-m",
        "--transmission",
        default=None,
        type=float,
        help="Transmission. Number in range between 0 and 1.",
    )
    parser.add_option(
        "-A",
        "--analysis",
        action="store_true",
        help="If set will perform automatic analysis.",
    )
    parser.add_option(
        "-D",
        "--diagnostic",
        action="store_true",
        help="If set will record diagnostic information.",
    )
    parser.add_option(
        "-S",
        "--simulation",
        action="store_true",
        help="If set will record diagnostic information.",
    )
    parser.add_option(
        "-k",
        "--shift",
        default=None,
        type=float,
        help="Horizontal shift compared to current position (in mm).",
    )

    options, args = parser.parse_args()

    print("options", options)
    print("args", args)

    scan = omega_scan(**vars(options))

    filename = "%s_parameters.pickle" % scan.get_template()
    if not os.path.isfile(filename):
        scan.execute()
    elif options.analysis == True:
        scan.analyze()


def test():
    scan_range = 180
    scan_exposure_time = 18.0
    scan_start_angle = 0
    angle_per_frame = 0.1

    s = omega_scan(
        scan_range=scan_range,
        scan_exposure_time=scan_exposure_time,
        scan_start_angle=scan_start_angle,
        angle_per_frame=angle_per_frame,
    )


if __name__ == "__main__":
    main()
