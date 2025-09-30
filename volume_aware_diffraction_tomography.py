#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import open3d as o3d
import pickle
import time
import numpy as np
import copy
import threading
import traceback

from diffraction_experiment import diffraction_experiment
from diffraction_tomography import diffraction_tomography
from perfect_realignment import (
    get_both_extremes_from_pcd,
    get_likely_part,
    get_position_from_vector,
    get_critical_points,
)
from area import area

# import pylab


class volume_aware_diffraction_tomography(diffraction_experiment):
    specific_parameter_fields = [
        {"name": "scan_start_angle", "type": "float", "description": ""},
        {"name": "scan_start_step", "type": "float", "description": ""},
        {"name": "scan_start_angles", "type": "list", "description": ""},
        {"name": "seed_positions", "type": "list", "description": ""},
        {"name": "orthogonal_step_size", "type": "float", "description": ""},
        {"name": "along_step_size", "type": "float", "description": ""},
        {"name": "reference_position", "type": "dict", "description": ""},
        {"name": "scan_range", "type": "float", "description": ""},
        {"name": "volume", "type": "str", "description": ""},
        {"name": "max_bounding_ray", "type": "float", "description": ""},
        {
            "name": "position",
            "type": "dict",
            "description": "dictionary with motor names as keys and their positions in mm as values",
        },
        {"name": "scan_range", "type": "float", "description": "scan range in degrees"},
        {
            "name": "scan_start_angle",
            "type": "float",
            "description": "scan start angle in degrees",
        },
        {"name": "frame_time", "type": "float", "description": "frame time in s"},
        {
            "name": "md_task_info",
            "type": "str",
            "description": "scan diagnostic information",
        },
        {
            "name": "initial_raster",
            "type": "list",
            "description": "initial analysis helical lines",
        },
    ]

    def __init__(
        self,
        name_pattern="pos2_tomography_1",
        directory="/nfs/data4/2024_Run2/com-proxima2a/Commissioning/automated_operation/px2-0042/pos2",
        volume="/nfs/data4/2024_Run2/com-proxima2a/Commissioning/automated_operation/px2-0042/pos2/zoom_X_c_after_kappa_phi_change_mm.pcd",
        orthogonal_step_size=0.002,
        along_step_size=0.015,
        frame_time=0.005,
        scan_range=0.0,
        scan_start_angle=None,
        scan_start_step=45.0,
        scan_start_angles="[0., 45., 90., 135.]",  # "[-60, +60, +135, -135, +180]",
        transmission=None,
        photon_energy=None,
        resolution=None,
        diagnostic=True,
        analysis=True,
        conclusion=True,
        display=True,
        method="xds",
        dont_move_motors=False,
        parent=None,
        beware_of_top_up=False,
        beware_of_download=False,
        generate_cbf=True,
        generate_h5=False,
        spot_threshold=20,
        cats_api=None,
    ):
        if hasattr(self, "parameter_fields"):
            self.parameter_fields += (
                volume_aware_diffraction_tomography.specific_parameter_fields
            )
        else:
            self.parameter_fields = (
                volume_aware_diffraction_tomography.specific_parameter_fields[:]
            )

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

        self.description = (
            "X-ray volume aware diffraction tomgraphy, Proxima 2A, SOLEIL, %s"
            % time.ctime(self.timestamp)
        )
        self.scan_range = scan_range
        self.frame_time = frame_time
        self.display = display
        self.orthogonal_step_size = orthogonal_step_size
        self.along_step_size = along_step_size

        self.volume = self.get_volume(volume)
        self.init_volume(scan_start_angle, scan_start_step, scan_start_angles)

        self.spot_threshold = spot_threshold

        self.match_number_in_spot_file = re.compile(".*([\d]{6}).adx.gz")

        # self.lines = {}
        # self.spots_per_line = {}
        # self.spots_per_frame = {}
        self.lines = None
        self.spots_per_line = None
        self.spots_per_frame = None

    def get_volume(self, volume=None):
        parameters = self.get_parameters()
        if "volume" in parameters:
            volume = parameters["volume"]
        return volume

    def init_volume(
        self,
        scan_start_angle=None,
        scan_start_step=45.0,
        scan_start_angles="[-60, +60, +135, -135, +180]",
    ):
        self.pcd = o3d.io.read_point_cloud(self.volume)
        self.volume_analysis = pickle.load(
            open(self.volume.replace("_mm.pcd", "_results.pickle"), "rb")
        )
        self.mlp = get_likely_part(self.pcd, self.volume_analysis)
        self.points = np.asarray(self.pcd.points)
        self.omega_max_angle = self.volume_analysis["omega_max"]

        if scan_start_angle is None:
            self.scan_start_angle = self.omega_max_angle
        else:
            self.scan_start_angle = scan_start_angle

        if scan_start_step is not None:
            scan_start_anlges = np.arange(0, 360, scan_start_step)

        if type(scan_start_angles) is str:
            scan_start_angles = eval(scan_start_angles)

        self.scan_start_angles = self.scan_start_angle + np.array(scan_start_angles)
        self.norientations = len(self.scan_start_angles)
        self.reference_position = self.volume_analysis["result_position"]
        self.reference_position["Omega"] = self.scan_start_angle

    def get_ordinal_from_spot_file_name(self, spot_file_name):
        ordinal = -1
        try:
            ordinal = int(self.match_number_in_spot_file.findall(spot_file_name)[0])
        except:
            pass
        return ordinal

    def get_seed_positions(self, margin=0.015):
        # pmax, pmin = get_both_extremes_from_pcd(self.mlp, axis=2, eigenbasis=False)
        cp = get_critical_points(self.volume_analysis)
        pmin = cp[0]
        if not np.any(np.isnan(cp[2])):
            pmax = cp[2]
        elif not np.any(np.isnan(cp[1])):
            d = cp[1] - pmin
            pmax = pmin + 2 * d
        else:
            pmax = copy.copy(pmin)
            pmax[2] += 0.5

        vector = pmax - pmin
        length = np.linalg.norm(vector)
        projected_length = pmax[2] - pmin[2]
        vector /= length
        margin = margin / vector[2]
        stepsize = self.along_step_size / vector[2]
        seed_positions = []
        seed = pmin - vector * margin
        while seed[2] <= pmax[2]:
            seed_positions.append(seed)
            seed = seed + vector * stepsize

        return seed_positions

    def get_cylinder_around_position(self, position, d=0.01):
        indices = np.argwhere(
            np.logical_and(
                self.points[:, 2] <= position[2] + d,
                self.points[:, 2] >= position[2] - d,
            )
        )
        cylinder = self.pcd.select_by_index(indices)
        return cylinder

    def get_bounding_cylinder_ray_for_position(self, position, margin=0.030):
        try:
            cylinder = self.get_cylinder_around_position(position, d=margin / 3.0)
            distances = np.linalg.norm(np.asarray(cylinder.points) - position, axis=1)
            max_ray = distances.max() + margin
        except:
            max_ray = 0.0
        return max_ray

    def get_bounding_rays(self, seed_positions=None):
        if seed_positions is None:
            seed_positions = self.get_seed_positions()
        bounding_rays = [
            self.get_bounding_cylinder_ray_for_position(sp) for sp in seed_positions
        ]
        return bounding_rays

    def get_initial_raster(
        self,
        seed_positions=None,
        max_bounding_ray=None,
        orientation="vertical",
        default_bounding_ray=0.4,
    ):
        helical_lines = []
        if seed_positions is None:
            seed_positions = self.get_seed_positions()
        if max_bounding_ray is None:
            bounding_rays = self.get_bounding_rays(seed_positions)
            try:
                max_bounding_ray = max(bounding_rays)
            except:
                max_bounding_ray = default_bounding_ray

        scan_exposure_time = self.nimages * self.frame_time
        for k, position in enumerate(seed_positions):
            p = get_position_from_vector(
                position, keys=["CentringX", "CentringY", "AlignmentY"]
            )
            p["AlignmentZ"] = self.reference_position["AlignmentZ"]
            scan_start_angle = self.scan_start_angles[k % self.norientations]
            scan_start_angle = scan_start_angle % 360
            p["Omega"] = scan_start_angle

            position_start = (
                self.goniometer.get_aligned_position_from_reference_position_and_shift(
                    p,
                    max_bounding_ray,
                    0,
                    AlignmentZ_reference=p["AlignmentZ"],
                )
            )
            position_stop = (
                self.goniometer.get_aligned_position_from_reference_position_and_shift(
                    p,
                    -max_bounding_ray,
                    0,
                    AlignmentZ_reference=p["AlignmentZ"],
                )
            )

            if abs(self.scan_range) > 0.0:
                scan_start_angle = scan_start_angle - self.scan_range / 2.0

            helical_line = [
                position_start,
                position_stop,
                scan_start_angle,
                self.scan_range,
                scan_exposure_time,
            ]
            helical_lines.append(helical_line)

        print(f"{len(helical_lines)} helical lines to measure")

        return helical_lines

    def get_nimages(self, seed_positions=None, max_bounding_ray=None):
        if seed_positions is None:
            seed_positions = self.get_seed_positions()
        if max_bounding_ray is None:
            bounding_rays = self.get_bounding_rays(seed_positions)
            max_bounding_ray = max(bounding_rays)

        nimages = int(np.ceil(max_bounding_ray * 2.0 / self.orthogonal_step_size))
        return nimages

    def get_ntrigger(self, seed_positions=None):
        if seed_positions is None:
            seed_positions = self.get_seed_positions()
        return len(seed_positions)

    def self_prepare(self, default_bounding_ray=0.40, margin=0.01):
        self.seed_positions = self.get_seed_positions()
        print(f"self.seed_positions {len(self.seed_positions)}")
        self.bounding_rays = self.get_bounding_rays(self.seed_positions)
        print(f"self.bounding_rays {self.bounding_rays}")
        try:
            self.max_bounding_ray = max(self.bounding_rays) + margin
        except:
            self.max_bounding_ray = default_bounding_ray
        print(f"self.max_bounding_ray {self.max_bounding_ray}")
        self.ntrigger = max(1, len(self.seed_positions))
        self.nimages = self.get_nimages(self.seed_positions, self.max_bounding_ray)
        self.orthogonal_steps = self.nimages
        self.angle_per_frame = self.scan_range / self.nimages
        self.scan_exposure_time = self.frame_time * self.nimages
        self.line_scan_time = self.frame_time * self.orthogonal_steps
        self.total_expected_exposure_time = self.line_scan_time * self.ntrigger
        self.total_expected_wedges = self.ntrigger

        self.initial_raster = self.get_initial_raster(
            self.seed_positions, self.max_bounding_ray
        )
        self.md_task_info = []
        self.cbf_template = self.get_cbf_template()
        self.spot_file_template = self.get_spot_file_template()

        self.lines = {}
        self.spots_per_line = {}
        self.spots_per_frame = np.zeros((self.nimages * self.ntrigger,))

    def prepare(self):
        self.self_prepare()

        super().prepare()

    def execute_initial_raster(self):
        for k, helical_line in enumerate(self.initial_raster):
            start, stop, scan_start_angle, scan_range, scan_exposure_time = helical_line
            task_id = self.goniometer.helical_scan(
                start, stop, scan_start_angle, scan_range, scan_exposure_time
            )
            self.line_analysis(k)
            self.md_task_info.append(self.goniometer.get_task_info(task_id))

    def line_analysis(self, k, sleeptime=0.05, timeout=10):
        lat = threading.Thread(
            target=self._line_analysis,
            args=(k, sleeptime, timeout),
        )
        lat.daemon = False
        lat.start()

    def _line_analysis(self, k, sleeptime, timeout):
        print(f"line analysis {k}")
        img_start = k * self.nimages + 1
        img_end = img_start + self.nimages
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
        return spots

    def analyze_initial_raster(self):
        initial_raster = self.get_initial_raster()
        print("initial_raster", initial_raster)
        for k in range(len(initial_raster)):
            self._line_analysis(k, 0.05, 1)

    def run(self):
        self.execute_initial_raster()

        # for k, position in enumerate(self.seed_positions):
        # print(f"point {k} has {self.spots_per_line[k]} spots")
        # if self.spots_per_line[k] >= self.spot_threshold:
        # p = get_position_from_vector(position)
        # p["AlignmentZ"] = self.reference_position["AlignmentZ"]
        # p["Omega"] = self.scan_start_angle
        # name_pattern = self.name_pattern + f"_dt_sp_{k}"
        # dt = diffraction_tomography(
        # name_pattern,
        # directory=self.directory,
        # scan_start_angles=str(list(self.scan_start_angles)),
        # vertical_range=self.bounding_rays[k] * 2,
        # position=p,
        # analysis=False,
        # conclusion=False,
        # display=False,
        # dont_move_motors=True,
        # )
        # dt.execute()
        # dt.run_shape_reconstruction(display=False)

    def get_tioga_results(self):
        total_number_of_images = self.get_total_number_of_images()
        tioga_results = np.zeros((total_number_of_images,))
        cbf_template = self.get_cbf_template()
        spot_file_template = self.get_spot_file_template()
        image_number_range = range(1, total_number_of_images + 1)
        # cbf_files = [cbf_template % d for d in image_number_range]
        spot_files = [spot_file_template % d for d in image_number_range]
        print("spot_files", len(spot_files))
        for sf in spot_files:
            s = len(self.get_spots(sf))
            ordinal = self.get_ordinal_from_spot_file_name(sf)
            if ordinal != -1:
                tioga_results[ordinal - 1] = s
        return tioga_results

    def analyze(self):
        print("analyze")
        self.analyze_initial_raster()

    def get_results(self):
        if self.spots_per_line == {}:
            self.self_prepare()
            self.analyze()

        results = []

        seed_positions = self.get_seed_positions()

        for k, position in enumerate(seed_positions):
            try:
                print(f"point {k} has {self.spots_per_line[k]} spots")
                if self.spots_per_line[k] >= self.spot_threshold:
                    p = get_position_from_vector(position)
                    p["AlignmentZ"] = self.reference_position["AlignmentZ"]
                    p["Omega"] = self.reference_position["Omega"]
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

        # import glob
        # results = glob.glob(os.path.join(self.directory, self.name_pattern + "*_dt_sp_*.results"))
        # results = [pickle.load(open(result, 'rb')) for result in results]
        # results.sort(key=lambda x: (x["solidity"], x["area"]))
        # results.reverse()

        # return results


def main():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-n", "--name_pattern", default="tomography_$id", type=str, help="Prefix"
    )
    parser.add_argument(
        "-d",
        "--directory",
        default="/nfs/data4/2024_Run2/com-proxima2a/Commissioning/automated_operation/px2-0042/pos2",
        type=str,
        help="Destination directory",
    )
    parser.add_argument(
        "-v",
        "--volume",
        default="/nfs/data4/2024_Run2/com-proxima2a/Commissioning/automated_operation/px2-0042/pos2/zoom_X_c_after_kappa_phi_change_mm.pcd",
        type=str,
        help="Destination directory",
    )
    parser.add_argument(
        "-r",
        "--scan_range",
        default=0.0,
        type=float,
        help="scan range",
    )
    parser.add_argument(
        "-s",
        "--scan_start_step",
        default=45.0,
        type=float,
        help="scan start step",
    )
    parser.add_argument(
        "-a",
        "--scan_start_angles",
        default="[-60, 60, 135, -135]",
        type=str,
        help="scan start angles",
    )
    parser.add_argument(
        "-f", "--frame_time", default=0.005, type=float, help="frame time"
    )
    parser.add_argument(
        "-H",
        "--along_step_size",
        default=0.02,
        type=float,
        help="along step size",
    )
    parser.add_argument(
        "-V",
        "--orthogonal_step_size",
        default=0.002,
        type=float,
        help="orthogonal step size",
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
        "-m", "--method", type=str, default="xds", help="analysis method"
    )
    parser.add_argument(
        "-S",
        "--dont_move_motors",
        action="store_true",
        help="Do not move after conclusion",
    )
    parser.add_argument(
        "-5", "--generate_h5", action="store_false", help="generate h5 files"
    )
    options = parser.parse_args()

    print("options", options)
    print("vars(options)", vars(options))

    experiment = volume_aware_diffraction_tomography(**vars(options))
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
