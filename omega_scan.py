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
import pprint

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
            "type": "list",
            "description": "scan diagnostic information",
        },
        {
            "name": "raw_analysis",
            "type": "bool",
            "description": "raw analysis",
        },
    ]

    XDSME_process_script = [
        "#!/bin/bash",
        "echo starting analysis using XDSME",
        "echo;",
        "cd {process_directory};",
        "aX.sh ../{name_pattern}_master.h5 > xdsme_auto_{name_pattern}.log;",
        "echo {name_pattern} results from XDSME ...;",
        "echo;",
        'grep -A 48 "Summary data for" xdsme_auto_{name_pattern}/{name_pattern}_aimless.log;',
        "echo;",
        "echo XDSME on {name_pattern} done!;",
    ]
    autoPROC_process_script = [
        "#!/bin/bash",
        "echo starting analysis using autoPROC",
        "echo;",
        "cd {process_directory};",
        'process -B -xml -nthreads 12 autoPROC_XdsKeyword_LIB="/data2/bioxsoft/progs/AUTOPROC/AUTOPROC/autoPROC/bin/linux64/plugins-x86_64/durin-plugin.so" autoPROC_XdsKeyword_MAXIMUM_NUMBER_OF_JOBS="12" -d autoPROC_{name_pattern} -h5 ../{name_pattern}_master.h5 | tee > autoPROC_{name_pattern}.log;',
        "echo {name_pattern} results from autoPROC ...;",
        "echo;",
        "cat autoPROC_{name_pattern}/staraniso_alldata-unique.table1;",
        "echo;",
        "echo autoPROC on {name_pattern} done!;",
    ]

    def __init__(
        self,
        name_pattern,
        directory,
        scan_range=180,
        scan_exposure_time=18,
        scan_start_angle=None,
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
        XDSME=True,
        autoPROC=True,
        sample_id=None,
        session_id=None,
        protein_acronym="not_specified",
        use_server=False,
        run_number=None,
        cats_api=None,
        raw_analysis=False,
    ):
        if hasattr(self, "parameter_fields"):
            self.parameter_fields += omega_scan.specific_parameter_fields[:]
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
            session_id=session_id,
            sample_id=sample_id,
            use_server=use_server,
            run_number=run_number,
            protein_acronym=protein_acronym,
            cats_api=cats_api,
        )

        self.XDSME = XDSME
        self.autoPROC = autoPROC

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
        self.sample_id = sample_id
        self.session_id = session_id
        self.use_server = use_server
        self.raw_analysis = raw_analysis
        
    def get_nimages(self, epsilon=1e-3):
        nimages = int(self.scan_range / self.angle_per_frame)
        if abs(nimages * self.angle_per_frame - self.scan_range) > epsilon:
            nimages += 1
        return nimages

    # def execute(self):
    # if self.use_server:
    # mcp = self.get_mxcube_collection_parameters(self.get_directory(), self.get_name_pattern(), self.get_session_id(), self.get_sample_id(), )
    # print("MXCuBE collection parameters:")
    # pprint.pprint(mcp)
    # self.collect.talk({"_pre_collect": {"args": ("workflow", mcp,)}})
    # self.collect.talk({"_collect": {"args": (mcp,)}})
    # self.collect.talk({"_post_collect": {"args": (mcp,)}})
    # else:
    # super().execute()

    def prepare(self):
        super().prepare()
        if self.use_server:
            self.cp = self.get_mxcube_collection_parameters(
                self.get_directory(),
                self.get_name_pattern(),
                self.get_session_id(),
                self.get_sample_id(),
            )
            self.collection_id = self.store_data_collection_in_lims(self.cp)
            self.cp["collection_id"] = self.collection_id
            self.store_sample_info_in_lims(self.cp)
            self.processing_filename = self.get_processing_filename(self.cp)
            print(f"processing_filename {self.processing_filename}")
            self.collection_id = self.get_collection_id()
            print(f"collection_id {self.collection_id}")

    def run(self, wait=True, steps=1, order=1):
        """execute omega scan."""

        if (
            self.beware_of_top_up
            and self.scan_exposure_time <= self.machine_status.get_top_up_period()
        ):
            self.check_top_up()

        scan_range = self.scan_range / steps
        scan_start_angle = self.scan_start_angle + scan_range * (order - 1)
        scan_exposure_time = self.scan_exposure_time / steps

        task_id = self.goniometer.omega_scan(
            scan_start_angle, scan_range, scan_exposure_time, wait=wait
        )

        self.md_task_info.append(self.goniometer.get_task_info(task_id))

    def clean(self):
        if self.use_server:
            self.update_data_collection_in_lims(self.cp)
            self.store_image_in_lims(self.cp, 1)

        super().clean()

    def analyze(
        self,
        remote=True,
        hostname="process1",
        names={"XDSME": "xdsme_auto", "autoPROC": "autoPROC"},
    ):
        process_directory = self.get_process_directory()

        if not os.path.isdir(process_directory):
            os.makedirs(process_directory)

        for tool in ["XDSME", "autoPROC"]:
            if getattr(self, tool):
                script_name = os.path.join(
                    process_directory, f"{names[tool]}_{self.get_name_pattern()}.sh"
                )

                script = getattr(self, f"{tool}_process_script")
                script = "\n".join(script)
                script = script.format(
                    name_pattern=self.get_name_pattern(),
                    process_directory=process_directory,
                )
                f = open(script_name, "w")
                f.write(script)
                f.close()

                process_line = f"/bin/bash {script_name}"

                if remote:
                    process_line = f'ssh {hostname} "{process_line}" &'

                if self.raw_analysis:
                    print(f"{tool} process_line {process_line}")
                    os.system(process_line)

        if self.use_server:
            # self.update_data_collection_in_lims(self.cp)
            print(
                f"running autoanalysis {self.processing_filename} {self.collection_id}"
            )
            self.run_analysis(self.processing_filename)

        # terminal = "gnome-terminal --title \"xdsme {name_pattern}\" --hide-menubar --geometry 80x40+0+0 --execute bash -c '{xdsme_process_line}; bash '".format(
        # name_pattern=os.path.basename(self.name_pattern),
        # xdsme_process_line=xdsme_process_line,
        # )
        # self.logger.info("xdsme_process_line %s" % xdsme_process_line)

    def conclude(self, position={"Omega": 0}):
        if type(position) is dict and "Omega" in position and len(position) == 1:
            position = position["Omega"]

        if type(position) is float:
            self.goniometer.omegaposition = position
        else:
            self.goniometer.set_position(position)


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
        "-r", "--scan_range", default=360, type=float, help="Scan range [deg]"
    )
    parser.add_option(
        "-e",
        "--scan_exposure_time",
        default=18,
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
        default=400,
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
    parser.add_option(
        "--session_id",
        default=-1,
        type=int,
        help="session id",
    )
    parser.add_option(
        "--sample_id",
        default=-1,
        type=int,
        help="sample id",
    )
    parser.add_option(
        "--use_server",
        action="store_true",
        help="use server",
    )

    parser.add_option(
        "--run_number",
        default=None,
        type=int,
        help="run number",
    )

    parser.add_option(
        "--protein_acronym",
        default="not_specified",
        type=str,
        help="run number",
    )
    options, args = parser.parse_args()

    print("options", options)
    print("args", args)

    scan = omega_scan(**vars(options))

    print(f"scan.get_template() {scan.get_template()}")
    filename = scan.get_parameters_filename()
    print("filename", filename)

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
