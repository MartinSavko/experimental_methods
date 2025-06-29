#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
mechanized sample evaluation

"""

import os
import time

from beamline import beamline
from experiment import experiment

from udc import udc

class mechanized_sample_evaluation(experiment):
    specific_parameter_fields = [
        {"name": "puck", "type": "int", "description": "puck"},
        {"name": "sample", "type": "int", "description": "sample"},
        {"name": "photon_energy", "type": "float", "description": "photon energy"},
        {"name": "transmission", "type": "float", "description": "transmission"},
        {"name": "resolution", "type": "float", "description": "resolution"},
        {"name": "scan_range", "type": "float", "description": "scan range"},
        {
            "name": "frame_exposure_time",
            "type": "float",
            "description": "frame exposure time",
        },
        {
            "name": "characterization_scan_range",
            "type": "float",
            "description": "characterization scan range",
        },
        {
            "name": "characterization_scan_start_angles",
            "type": "str",
            "description": "characterization scan start angles",
        },
        {
            "name": "characterization_frame_exposure_time",
            "type": "float",
            "description": "characterization frame exposure time",
        },
        {
            "name": "characterization_angle_per_frame",
            "type": "float",
            "description": "characterization angle per frame",
        },
        {
            "name": "characterization_transmission",
            "type": "float",
            "description": "characterization transmission",
        },
        {"name": "wash", "type": "bool", "description": "wash"},
        {"name": "beam_align", "type": "bool", "description": "beam align"},
        {"name": "skip_tomography", "type": "bool", "description": "skip tomography"},
        {"name": "norient", "type": "int", "description": "norient"},
        {"name": "defrost", "type": "int", "description": "defrost"},
        {"name": "prealign", "type": "bool", "description": "prealign"},
        {
            "name": "enforce_scan_range",
            "type": "bool",
            "description": "enforce scan range",
        },
        {"name": "force_transfer", "type": "bool", "description": "force transfer"},
        {"name": "force_centring", "type": "bool", "description": "force centring"},
        {"name": "beware_of_top_up", "type": "bool", "description": "beware of top up"},
        {
            "name": "default_directory",
            "type": "str",
            "description": "default directory",
        },
    ]

    def __init__(
        self,
        name_pattern=None,
        directory=None,
        puck=None,
        sample=None,
        photon_energy=13000.0,
        transmission=15.0,
        resolution=1.5,
        scan_range=400.0,
        frame_exposure_time=0.005,
        characterization_scan_range=1.2,
        characterization_scan_start_angles="[0, 45, 90, 135, 180]",
        characterization_frame_exposure_time=0.1,
        characterization_angle_per_frame=0.1,
        characterization_transmission=15.0,
        wash=False,
        beam_align=False,
        skip_tomography=False,
        norient=1,
        defrost=0,
        prealign=False,
        enforce_scan_range=True,
        force_transfer=False,
        force_centring=False,
        beware_of_top_up=True,
        default_directory="/nfs/data4/mechanized_sample_evaluation",
    ):
        self.timestamp = time.time()
        self.instrument = beamline()

        if None in (puck, sample):
            puck, sample = self.instrument.sample_changer.get_mounted_puck_and_sample()

        if name_pattern is None:
            if not -1 in (puck, sample):
                designation = f"{puck}_{sample}"
            else:
                designation = "manually_mounted"
            name_pattern = f"{designation}_{time.ctime(self.timestamp).replace(' ', '_')}"
        self.puck = puck
        self.sample = sample

        if directory is None:
            directory = os.path.join(
                default_directory,
                os.environ["USER"],
                f"{time.ctime(self.timestamp).replace(' ', '_')}",
            )

        experiment.__init__(
            self,
            name_pattern=name_pattern,
            directory=directory,
        )

        self.description = f"Mechanized sample evaluation, Proxima 2A, SOLEIL, {time.ctime(self.timestamp)}"
        
        self.scan_range = scan_range
        self.photon_energy = photon_energy
        self.transmission = transmission
        self.resolution = resolution
        self.frame_exposure_time = frame_exposure_time
        self.characterization_frame_exposure_time = characterization_frame_exposure_time
        self.characterization_transmission = characterization_transmission
        self.characterization_scan_range = characterization_scan_range
        self.characterization_scan_start_angles = characterization_scan_start_angles
        self.characterization_angle_per_frame = characterization_angle_per_frame

        self.wash = wash
        self.beam_align = beam_align
        self.skip_tomography = skip_tomography
        self.norient = norient
        self.defrost = defrost
        self.prealign = prealign
        self.enforce_scan_range = enforce_scan_range

        self.force_transfer = force_transfer,
        self.force_centring = force_centring,
        self.beware_of_top_up = beware_of_top_up,

        self.default_directory = default_directory

    def run(self):
        udc(
            puck=self.puck,
            sample=self.sample,
            base_directory=self.directory,
            beam_align=self.beam_align,
            skip_tomography=self.skip_tomography,
            norient=self.norient,
            wash=self.wash,
            photon_energy=self.photon_energy,
            transmission=self.transmission,
            resolution=self.resolution,
            frame_exposure_time=self.frame_exposure_time,
            characterization_frame_exposure_time=self.characterization_frame_exposure_time,
            characterization_transmission=self.characterization_transmission,
            characterization_scan_range=self.characterization_scan_range,
            characterization_scan_start_angles=self.characterization_scan_start_angles,
            characterization_angle_per_frame=self.characterization_angle_per_frame,
            defrost=self.defrost,
            prealign=self.prealign,
            force_transfer=self.force_transfer,
            beware_of_top_up=self.beware_of_top_up,
            enforce_scan_range=self.enforce_scan_range,
        )

        
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--puck", default=7, type=int, help="puck")
    parser.add_argument("-s", "--sample", default=1, type=int, help="sample")
    parser.add_argument(
        "-d",
        "--directory",
        default="/nfs/data4/2025_Run2/com-proxima2a/Commissioning/automated_operation/px2-0049",
        help="directory",
    )
    parser.add_argument("-w", "--wash", action="store_true", help="wash")
    parser.add_argument("-b", "--beam_align", action="store_true", help="beam_align")
    parser.add_argument(
        "-t", "--skip_tomography", action="store_true", help="dont do tomography"
    )
    parser.add_argument("-n", "--norient", default=1, type=int, help="norient")
    parser.add_argument("-M", "--defrost", default=0, type=float, help="defrost")
    parser.add_argument(
        "-P", "--prealign", action="store_true", help="prealign"
    )
    parser.add_argument(
        "-B",
        "--dont_enforce_scan_range",
        action="store_true",
        help="dont_enforce_scan_range",
    )
    parser.add_argument(
        "-F",
        "--force_transfer",
        action="store_true",
        help="force_transfer before prealignment",
    )
    parser.add_argument(
        "-E",
        "--force_centring",
        action="store_true",
        help="force_centring befor prealignment",
    )
    parser.add_argument(
        "-T", "--ignore_top_up", action="store_true", help="ignore top up"
    )
    parser.add_argument(
        "-e", "--photon_energy", default=13000, type=float, help="photon energy"
    )
    parser.add_argument(
        "-r", "--transmission", default=15.0, type=float, help="transmission"
    )
    parser.add_argument(
        "-R", "--resolution", default=1.5, type=float, help="resolution"
    )
    parser.add_argument(
        "-f",
        "--frame_exposure_time",
        default=0.005,
        type=float,
        help="frame exposure time",
    )
    parser.add_argument(
        "-c",
        "--characterization_frame_exposure_time",
        default=0.01,
        type=float,
        help="characterization frame exposure time",
    )
    parser.add_argument(
        "-C",
        "--characterization_transmission",
        default=15.0,  # 5
        type=float,
        help="characterization transmission",
    )
    parser.add_argument(
        "-S",
        "--characterization_scan_range",
        default=1.2,  # 200
        type=float,
        help="characterization scan range",
    )
    parser.add_argument(
        "-A",
        "--characterization_scan_start_angles",
        default="[0, 45, 90, 135, 180]",  # "[0]"
        type=str,
        help="characterization scan start angles",
    )
    parser.add_argument(
        "-D",
        "--characterization_angle_per_frame",
        default=0.1,  # 0.5
        type=float,
        help="characterization angle_per_frame",
    )

    args = parser.parse_args()
    print("args", args)

    mse = mechanized_sample_evaluation(
        puck=args.puck,
        sample=args.sample,
        directory=args.directory,
        beam_align=bool(args.beam_align),
        skip_tomography=bool(args.skip_tomography),
        norient=args.norient,
        wash=bool(args.wash),
        photon_energy=args.photon_energy,
        transmission=args.transmission,
        resolution=args.resolution,
        frame_exposure_time=args.frame_exposure_time,
        characterization_frame_exposure_time=args.characterization_frame_exposure_time,
        characterization_transmission=args.characterization_transmission,
        characterization_scan_range=args.characterization_scan_range,
        characterization_scan_start_angles=eval(
            args.characterization_scan_start_angles
        ),
        characterization_angle_per_frame=args.characterization_angle_per_frame,
        defrost=float(args.defrost),
        prealign=bool(args.prealign),
        force_transfer=bool(args.force_transfer),
        beware_of_top_up=not bool(args.ignore_top_up),
        enforce_scan_range=not bool(args.dont_enforce_scan_range),
    )
        
    mse.execute()
    
