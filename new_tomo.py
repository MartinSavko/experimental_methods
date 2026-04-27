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
from useful_routines import (
    get_ordinal_from_spot_file_name,
    get_tioga_results,
    get_pickled_file,
    get_raster_from_opti,
    execute_raster,
    get_center_of_mass,
    get_aaoi,
    is_number,
    is_valid_number,
    get_index_of_max_or_min,
)

from area import area

# import pylab


class new_tomo(diffraction_experiment):
    specific_parameter_fields = [
        {"name": "scan_start_angle", "type": "float", "description": ""},
        {"name": "scan_start_step", "type": "float", "description": ""},
        {"name": "scan_start_angles", "type": "list", "description": ""},
        {"name": "seed_positions", "type": "list", "description": ""},
        {"name": "orthogonal_step_size", "type": "float", "description": ""},
        {"name": "along_step_size", "type": "float", "description": ""},
        {"name": "reference_position", "type": "dict", "description": ""},
        {"name": "scan_range", "type": "float", "description": ""},
        {"name": "volume", "type": "str", "description": "volume"},
        {"name": "opti", "type": "str", "description": "opti"},
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
        opti=None,
        volume=None,
        orthogonal_step_size=0.002,
        along_step_size=0.025,
        frame_time=0.005,
        scan_range=0.0,
        scan_start_angle=None,
        scan_start_step=45.0,
        scan_start_angles="[0., 45., 90., 135.]",  # "[-60, +60, +135, -135, +180]",
        heart_start=True,
        do_flat_raster=True,
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
                new_tomo.specific_parameter_fields
            )
        else:
            self.parameter_fields = (
                new_tomo.specific_parameter_fields[:]
            )

        self.default_experiment_name = "X-ray volume aware diffraction tomgraphy"

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

        self.scan_range = scan_range
        self.frame_time = frame_time
        self.display = display
        self.orthogonal_step_size = orthogonal_step_size
        self.along_step_size = along_step_size

        self.heart_start = heart_start
        self.do_flat_raster = do_flat_raster
        self.opti = get_pickled_file(opti)
        self.results = None

    def get_opti_max_width(self, margin=0.025):
        aaoi = get_aaoi(self.opti)
        max_width = aaoi[:, -1].max() 
        if margin > 0:
            max_width += margin
        return max_width
    
    def prepare(self):
        self.prepare_shutters()
        self.prepare_environment()
    
    def run(self):
        
        raster, reference_position = get_raster_from_opti(self.opti)
        
        raster, parameters = execute_raster(
            raster, 
            reference_position, 
            gonio=self.goniometer, 
            detector=self.detector,
            beam_center=self.beam_center,
            name_pattern=f"{self.name_pattern}_init",
            directory=self.directory,
            detector_distance=self.detector_distance,
            photon_energy=self.photon_energy,
            calibration=self.camera.get_calibration()
        )
        
        try:
            if raster == -1:
                print("no diffraction found")
                self.results = -1
                return -1
        except:
            print("raster seems to have found some diffraction !")

        along_shift, orthogonal_shift = np.array(get_index_of_max_or_min(raster, max_or_min="max") - np.array(raster.shape)/2.) * self.camera.get_calibration()
        self.logger.info(f"shift HxV: {np.round([orthogonal_shift, along_shift], 3)}")
        optimum = self.goniometer.get_aligned_position_from_reference_position_and_shift(
            parameters["reference_position"],
            orthogonal_shift,
            along_shift,
            omega=parameters["scan_start_angle"],
        )
        
        self.results = {"optimum": optimum}
        self.logger.info(f"optimum position {optimum}")
        self.goniometer.set_position(optimum)
            
    def get_results(self):
        return self.results
    
    def conclude(self):
        pass
    
    def clean(self):
        pass
        
def main():
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

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
        "-O",
        "--opti",
        default="/nfs/data4/2026_Run2/com-proxima2a/Commissioning/automated_operation/opti/zoom_4_stepped_i_descriptions.pickle",
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
        default="[0., 45., 90., 135.]",
        type=str,
        help="scan start angles",
    )
    parser.add_argument(
        "-f", "--frame_time", default=0.005, type=float, help="frame time"
    )
    parser.add_argument(
        "-H",
        "--along_step_size",
        default=0.025,
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
        "-M", "--method", type=str, default="xds", help="analysis method"
    )
    parser.add_argument(
        "-o", "--resolution", default=None, type=float, help="Resolution [Angstroem]"
    )
    parser.add_argument(
        "-m",
        "--transmission",
        default=None,
        type=float,
        help="Transmission. Number in range between 0 and 1.",
    )
    parser.add_argument(
        "-p", "--photon_energy", default=None, type=float, help="Photon energy "
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
    args = parser.parse_args()

    print("args", args)
    print("vars(args)", vars(args))

    experiment = new_tomo(**vars(args))
    print("get_parameters_filename", experiment.get_parameters_filename())
    if not os.path.isfile(experiment.get_parameters_filename()):
        experiment.execute()
    elif args.analysis:
        #experiment.analyze() #method=args.method)
        print("analysis")
        if args.conclusion:
            print("conclusion")
            experiment.conclude()


if __name__ == "__main__":
    main()
