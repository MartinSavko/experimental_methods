#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cosmic scan
"""

from diffraction_experiment import diffraction_experiment


class cosmic_scan(diffraction_experiment):
    def __init__(
        self,
        name_pattern,
        directory,
        frame_time=60.0,
        nimages=1,
        ntrigger=1,
        trigger_mode="ints",
        beam=True,
        transmission=None,
        photon_energy=None,
        resolution=None,
        detector_distance=None,
        detector_vertical=None,
        detector_horizontal=None,
        nimages_per_file=1,
        generate_h5=True,
        generate_cbf=True,
        analysis=False,
        diagnostic=False,
        use_goniometer=False,
        extract_protective_cover=False,
        snapshot=False,
    ):
        if hasattr(self, "parameter_fields"):
            self.parameter_fields += cosmic_scan.specific_parameter_fields[:]
        else:
            self.parameter_fields = cosmic_scan.specific_parameter_fields[:]

        diffraction_experiment.__init__(
            self,
            name_pattern,
            directory,
            photon_energy=photon_energy,
            resolution=resolution,
            detector_distance=detector_distance,
            detector_vertical=detector_vertical,
            detector_horizontal=detector_horizontal,
            nimages_per_file=nimages_per_file,
            transmission=transmission,
            ntrigger=ntrigger,
            diagnostic=diagnostic,
            analysis=analysis,
            generate_cbf=generate_cbf,
            generate_h5=generate_h5,
            trigger_mode=trigger_mode,
            use_goniometer=use_goniometer,
            extract_protective_cover=extract_protective_cover,
            snapshot=snapshot,
        )

        self.beam = beam
        self.frame_time = frame_time
        self.nimages = nimages
        self.total_expected_exposure_time = (
            self.frame_time * self.nimages * self.ntrigger
        )

    def get_frame_time(self):
        return self.frame_time

    def prepare(self):
        super().prepare()
        self.detector.set_trigger_mode(self.trigger_mode)
        if self.beam:
            self.fast_shutter.open()
            print("fast shutter open")

    def run(self):
        for k in range(self.ntrigger):
            self.detector.trigger()

    def clean(self):
        if self.beam:
            self.fast_shutter.close()
            print("fast shutter closed")

        super().clean()


if __name__ == "__main__":
    import os
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-n",
        "--name_pattern",
        default="test_$id",
        type=str,
        help="Prefix ",
    )
    parser.add_argument(
        "-d",
        "--directory",
        default="/nfs/data/default",
        type=str,
        help="Destination directory",
    )

    parser.add_argument(
        "-f",
        "--frame_time",
        default=60.0,
        type=float,
        help="Frame time [s]",
    )

    parser.add_argument(
        "-N",
        "--nimages",
        default=10,
        type=int,
        help="nimages",
    )
    
    parser.add_argument(
        "-b",
        "--beam",
        action="store_true",
        help="beam",
    )
        

    args = parser.parse_args()

    print(args)

    cs = cosmic_scan(
        name_pattern=args.name_pattern,
        directory=args.directory,
        nimages=args.nimages,
        frame_time=args.frame_time,
        beam=args.beam,
    )

    if not os.path.isfile(cs.get_parameters_filename()):
        cs.execute()
