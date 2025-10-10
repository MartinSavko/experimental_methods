#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
mechanized sample evaluation

"""

import os
import time
import traceback

from experimental_methods.instrument.beamline import beamline
from experimental_methods.experiment.experiment import experiment
from experimental_methods.experiment.diffraction_experiment import diffraction_experiment
from experimental_methods.experiment.udc import udc, align_beam
from experimental_methods.utils.speech import speech
from experimental_methods.utils.useful_routines import get_string_from_timestamp

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
        {"name": "use_server", "type": "bool", "description": "use server"},
        {"name": "proposal_id", "type": "int", "description": "proposal id"},
        {"name": "session_id", "type": "int", "description": "session id"},
        {"name": "sample_id", "type": "int", "description": "sample id"},
        {"name": "sample_name", "type": "str", "description": "sample name"},
        {"name": "protein_acronym", "type": "str", "description": "protein acronym"},
        {
            "name": "raw_analysis",
            "type": "bool",
            "description": "raw analysis",
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
        use_server=False,
        proposal_id=3113,
        session_id=46635,
        sample_id=-1,
        sample_name=None,
        protein_acronym="not_specified",
        raw_analysis=False,
    ):
        if hasattr(self, "parameter_fields"):
            self.parameter_fields += self.specific_parameter_fields[:]
        else:
            self.parameter_fields = self.specific_parameter_fields[:]

        self.timestamp = time.time()
        self.instrument = beamline()

        if None in (puck, sample):
            puck, sample = self.instrument.sample_changer.get_mounted_puck_and_sample()

        if sample_name is not None:
            name_pattern = sample_name
        elif name_pattern is None:
            if not -1 in (puck, sample):
                designation = self.get_element(puck, sample)
            else:
                designation = "manually_mounted"
            timestring = self.get_timestring()
            name_pattern = f"{designation}_{timestring}"

        self.puck = puck
        self.sample = sample

        if directory is None:
            directory = os.path.join(
                default_directory,
                os.environ["USER"],
                f"{get_string_from_timestamp(self.timestamp)",
            )

        experiment.__init__(
            self,
            name_pattern=name_pattern,
            directory=directory,
        )

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

        self.force_transfer = force_transfer
        self.force_centring = force_centring
        self.beware_of_top_up = beware_of_top_up

        self.default_directory = default_directory
        self.use_server = use_server
        self.proposal_id = proposal_id
        self.session_id = session_id
        self.sample_id = sample_id
        self.sample_name = sample_name
        self.protein_acronym = protein_acronym
        self.raw_analysis = raw_analysis
        
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
            use_server=self.use_server,
            sample_id=self.sample_id,
            session_id=self.session_id,
            sample_name=self.sample_name,
            protein_acronym=self.protein_acronym,
            raw_analysis=self.raw_analysis,
        )

    def get_samples(self):
        samples = self.ispyb.talk(
            {"get_samples": {"args": (self.get_proposal_id(), self.get_session_id())}}
        )
        return samples

def main():
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
    parser.add_argument("-P", "--prealign", action="store_true", help="prealign")
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
        "-x",
        "--use_server",
        action="store_true",
        help="use server",
    )
    parser.add_argument(
        "-T", "--ignore_top_up", action="store_true", help="ignore top up"
    )
    parser.add_argument(
        "-e", "--photon_energy", default=13000, type=float, help="photon energy"
    )
    parser.add_argument(
        "-r", "--transmission", default=25.0, type=float, help="transmission"
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
        default=50.0,  # 5
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

    parser.add_argument(
        "--sample_id",
        default=1,
        type=int,
        help="sample id",
    )

    parser.add_argument(
        "--session_id",
        default=46529,
        type=int,
        help="session id",
    )

    parser.add_argument(
        "--sample_name",
        default="DatZ_pin16",
        type=str,
        help="sample name",
    )

    parser.add_argument(
        "--protein_acronym",
        default="not_specified",
        type=str,
        help="protein acronym",
    )
    parser.add_argument(
        "--raw_analysis",
        action="store_true",
        help="raw analysis",
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
        use_server=bool(args.use_server),
        session_id=args.session_id,
        sample_name=args.sample_name,
        protein_acronym=args.protein_acronym,
        raw_analysis=bool(args.raw_analysis),
    )

    mse.execute()

if __name__ == "__main__":
    main()

def get_puck_and_position(x):
    return int(x["containerSampleChangerLocation"]), int(x["sampleLocation"])


# def mse_20250023(session_id=46530, proposal_id=3113):
def mse_20250023(session_id=46686, proposal_id=3113, just_print=True):
    # base_directory = "/nfs/data4/2025_Run3/20250023/2025-07-04/RAW_DATA"
    base_directory = "/nfs/data4/2025_Run3/20250023/2025-07-28/RAW_DATA"
    de = diffraction_experiment(directory=base_directory, name_pattern="mse_20250023")
    samples = de.get_samples(session_id=session_id, proposal_id=proposal_id)

    # pucks = ["BX029A", "BX033A", "BX041A"]
    pucks = ["BX011A", "BX019A"]
    relevant = [sample for sample in samples if sample["containerCode"] in pucks]
    relevant.sort(key=get_puck_and_position)

    # align_beam(base_directory)
    _start_t = time.time()
    # relevant = relevant[15:]
    failed = 0
    for k, sample in enumerate(relevant):
        _start = time.time()
        puck = int(sample["containerSampleChangerLocation"])
        pin = int(sample["sampleLocation"])
        sample_id = int(sample["sampleId"])
        protein_acronym = sample["proteinAcronym"]
        sample_name = f"{protein_acronym}-{sample['sampleName']}"
        directory = f"{base_directory}/{protein_acronym}/{sample_name}"

        print(
            f"will investigate sample {sample_name} from basket {sample['containerCode']}"
        )
        print(f"sample {k+1} of {len(relevant)} in the current run")

        command_line = f"mse -d {directory} -p {puck} -s {pin} --sample_name {sample_name} --sample_id {sample_id} --session_id {session_id} --protein_acronym {protein_acronym} --use_server -r 50"
        if not os.path.isdir(os.path.join(directory, "opti")):
            if just_print:
                print(command_line)
            else:
                os.system(command_line)
        else:
            print(command_line)
            print(f"sample {sample_name} {puck} {pin} already measured")

        # try:
        # mse = mechanized_sample_evaluation(
        # puck=puck,
        # sample=pin,
        # directory=base_directory,
        # frame_exposure_time=0.005,
        # transmission=25,
        # characterization_transmission=25.0,
        # step_size_along_omega=0.025,
        # sample_name=sample_name,
        # wash=False,
        # sample_id=sample_id,
        # session_id=session_id,
        # use_server=True,
        # )
        # mse.execute()
        # except:
        # traceback.print_exc()
        # failed += 1

        duration = time.time() - _start
        print(
            f"sample {sample_name} from basket {sample['containerCode']} analyzed in {duration:.2f} seconds ({duration/60:.1f} minutes)"
        )
        print(15 * "==++==")
        print(7 * "\n")
    duration = time.time() - _start_t
    print(
        f"{len(relevant)} samples analyzed in {duration:.2f} seconds ({duration/len(relevant):.2f} per sample), failed {failed}"
    )
    print(15 * "==++==")
    print(7 * "\n")
