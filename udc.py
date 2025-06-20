#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
import sys
import time
import traceback
import pickle
import numpy as np
import subprocess
import glob

from beam_align import beam_align as bac
from mount import mount as mount_object
from optical_alignment import optical_alignment
from volume_aware_diffraction_tomography import volume_aware_diffraction_tomography
from diffraction_tomography import diffraction_tomography
from reference_images import reference_images
from omega_scan import omega_scan


from beamline import beamline

logging.basicConfig(
    format="%(asctime)s |%(module)s | %(message)s",
    level=logging.INFO,
)

instrument = beamline()


def align_beam(base_directory, photon_energy=None):
    directory = os.path.join(base_directory, "beam")
    previous = glob.glob(os.path.join(directory, "*_parameters.pickle"))
    ba = bac(
        name_pattern=f"beam_align_{len(previous) + 1}", 
        directory=directory,
        photon_energy=photon_energy,
    )
    ba.execute()

def mount(directory, puck, sample, wash=True, subdir="mount"):
    _start = time.time()
    mo = mount_object(puck, sample, wash=wash, directory=os.path.join(directory, subdir))
    mo.execute()
    _end = time.time()
    logging.info(f"sample mounted in {_end - _start:.3f} seconds")
    print(10 * "=", "mount done", 10 * "=")
    return mo.get_success()


#def mount(directory, puck, sample, wash=True):
    #_start = time.time()
    #if instrument.cats.isoff():
        #instrument.cats.on()

    #mpuck, msample = instrument.cats.get_mounted_puck_and_sample()
    #if mpuck != puck or msample != sample:
        #if wash:
            #prepare_centring = False
        #else:
            #prepare_centring = True
        #instrument.cats.mount(puck, sample, prepare_centring=prepare_centring)

    #if instrument.cats.get_mounted_sample_id()[0] == -1:
        #logging.info(f"sample {sample} puck {puck} not present?")
        #return -1
    #elif wash:
        #instrument.cats.mount(puck, sample, prepare_centring=True)
        #if instrument.cats.get_mounted_sample_id()[0] == -1:
            #logging.info(f"sample {sample} puck {puck} not present?")
            #return -1

    #_end = time.time()
    #logging.info(f"sample mounted in {_end - _start:.3f} seconds")
    #print(10 * "=", "mount done", 10 * "=")
    #save_start = time.time()
    #directory = os.path.join(directory, "mount")
    #name_pattern = f"{puck}_{sample}"
    #filename_template = os.path.join(directory, name_pattern)
    #instrument.camm.save_history(filename_template, _start, _end, local=True)
    #save_end = time.time()
    #logging.info(f"mount history saved in {save_end - save_start:.3f} seconds")
    #print(10 * "=", "mount history saved", 10 * "=")
    #print(5 * "\n")
    
def prealignment(directory):
    _start = time.time()

    oa = optical_alignment(
        name_pattern="zoom_1_eager_init",
        directory=os.path.join(directory, "opti"),
        scan_range=0,
        angles="(0, 0, 0, 90, 180, 235, 315)",
        backlight=True,
        analysis=True,
        conclusion=True,
        move_zoom=False,
        zoom=1,
        save_history=True,
    )
    if not os.path.isfile(oa.get_parameters_filename()):
        oa.execute()

    if not oa.sample_seen:
        logging.info(f"sample base detected but not visible, is pin damaged?")
        return -1

    logging.info(f"sample prealigned in {time.time() - _start:.3f} seconds")
    print(10 * "=", "prealignment done", 10 * "=")
    print(5 * "\n")

def opti_series(directory):
    _start = time.time()

    # instrument.goniometer.set_centring_phase()

    oa = optical_alignment(
        name_pattern="zoom_1_eager",
        directory=os.path.join(directory, "opti"),
        scan_range=0,
        angles="(0, 90, 180, 270)",
        backlight=True,
        analysis=True,
        conclusion=True,
        move_zoom=False,
        zoom=1,
        save_history=True,
    )
    if not os.path.isfile(oa.get_parameters_filename()):
        oa.execute()

        if not oa.sample_seen:
            return -1
    print(10 * "=", "optical_alignment_eager done", 10 * "=")
    print(5 * "\n")

    oa = optical_alignment(
        name_pattern="zoom_1_careful",
        directory=os.path.join(directory, "opti"),
        scan_range=360,
        backlight=True,
        analysis=True,
        conclusion=True,
        move_zoom=True,
        zoom=1,
        save_history=True,
    )
    if not os.path.isfile(oa.get_parameters_filename()):
        oa.execute()

    print(10 * "=", "optical_alignment_careful at zoom 1 done", 10 * "=")
    print(5 * "\n")
    if oa.results["optimum_zoom"] != 1:
        oa = optical_alignment(
            name_pattern="zoom_X_careful",
            directory=os.path.join(directory, "opti"),
            scan_range=360,
            backlight=True,
            analysis=True,
            conclusion=True,
            save_history=True,
            zoom=None,
            move_zoom=False,
        )
        if not os.path.isfile(oa.get_parameters_filename()):
            oa.execute()
        print(10 * "=", "optical_alignment_careful at high zoom done", 10 * "=")
        print(5 * "\n")
    logging.info(f"optical alignment took {time.time() - _start:.3f} seconds")
    return oa


def tomo_series(
    directory,
    oa,
    position=None,
    transmission=15.0,
    resolution=1.5,
    photon_energy=13000.0,
    step_size_along_omega=0.025,
    diagnostic=False,
    sample_name=None,
):
    _start = time.time()

    vadt = volume_aware_diffraction_tomography(
        name_pattern="vadt",
        directory=os.path.join(directory, "tomo"),
        volume=oa.get_pcd_mm_name(),
        step_size_along_omega=step_size_along_omega,
        resolution=resolution,
        photon_energy=photon_energy,
        transmission=transmission,
        diagnostic=diagnostic,
    )
    if not os.path.isfile(vadt.get_parameters_filename()):
        vadt.execute()

    results = vadt.get_results()

    logging.info(f"tomography took {time.time() - _start:.3f} seconds")
    print(10 * "=", "vadt done", 10 * "=")
    print(5 * "\n")
    try:
        position = results[0]["result_position"]
    except:
        # logging.info(traceback.format_exc())
        traceback.print_exc()
        print(f"results {results}")
        return -1

    dt = diffraction_tomography(
        name_pattern="opt",
        directory=os.path.join(directory, "tomo"),
        position=position,
        vertical_range=2.1 * max(vadt.get_bounding_rays()),
        analysis=True,
        conclusion=True,
        display=False,
        dont_move_motors=True,
        photon_energy=photon_energy,
        diagnostic=diagnostic,
    )
    if not os.path.isfile(dt.get_parameters_filename()):
        dt.execute()

    #position = dt.get_result_position()
    #instrument.goniometer.set_position(position)


def char_series(
    directory,
    transmission=15.0,
    resolution=1.5,
    photon_energy=13000.0,
    frame_exposure_time=0.01,
    scan_range=1.2,
    angle_per_frame=0.1,
    scan_start_angles=[0.0, 45.0, 90.0, 135, 180.0],
    sample_name=None,
    min_resolution=3.,
):
    _char_start = time.time()


    if sample_name is not None:
        name_pattern = f"ref-{sample_name}"
    else:
        name_pattern = f"reference"
        
    args = {
        "name_pattern": name_pattern,
        "directory": os.path.join(directory, "char"),
        "scan_range": scan_range,
        "angle_per_frame": angle_per_frame,
        "scan_exposure_time": (scan_range / angle_per_frame) * frame_exposure_time,
        "scan_start_angles": scan_start_angles,
        "resolution": resolution,
        "transmission": transmission,
        "photon_energy": photon_energy,
        "analysis": False,
        "diagnostic": False,
        "generate_sum": False,
    }

    ref = reference_images(**args)

    if not os.path.isfile(ref.get_parameters_filename()):
        ref.execute()

    ref.analyze(block=True)

    try:
        strategy = ref.parse_best()
    except:
        strategy = []
    dozor_results = ref.get_dozor_results()
    try:
        resolution = min(dozor_results[:, -1].min()-0.1, min_resolution)
    except:
        resolution = min_resolution
    logging.info(f"strategy from reference images {strategy}")
    logging.info(f"characterization took {time.time() - _char_start:.3f} seconds")
    print(10 * "=", "characterization done", 10 * "=")
    print(5 * "\n")
    return strategy, resolution


def main_series(
    directory,
    strategy=[],
    resolution=None,
    transmission=15.0,
    photon_energy=13000.0,
    angle_per_frame=0.1,
    scan_range=400,
    frame_exposure_time=0.0043,
    enforce_scan_range=True,
    diagnostic=False,
    beware_of_top_up=True,
    sample_name=None,
):
    _start = time.time()

    scan_exposure_time = (scan_range / angle_per_frame) * frame_exposure_time

    default_resolution = resolution

    if strategy != []:
        for k, wedge in enumerate(strategy):
            wedge["resolution"] = resolution
            best_scan_exposure_time = wedge["scan_exposure_time"]
            best_scan_range = wedge["nimages"] * wedge["angle_per_frame"]
            if enforce_scan_range and (k + 1) == len(strategy):
                best_scan_exposure_time = best_scan_exposure_time * (
                    scan_range / best_scan_range
                )
                best_scan_range = scan_range

            logging.info(f"best_scan_range {best_scan_range}")
            logging.info(f"best_scan_exposure_time {best_scan_exposure_time}")

            if sample_name is not None:
                name_pattern = f"{sample_name}_strategy_BEST_{wedge['order']}"
            else:
                name_pattern = f"strategy_BEST_{wedge['order']}"
                
            osc = omega_scan(
                name_pattern=name_pattern,
                directory=os.path.join(directory, "main"),
                scan_start_angle=wedge["scan_start_angle"],
                angle_per_frame=wedge["angle_per_frame"],
                scan_exposure_time=best_scan_exposure_time,
                scan_range=best_scan_range,
                transmission=wedge["transmission"],
                resolution=resolution,
                photon_energy=photon_energy,
                diagnostic=diagnostic,
                beware_of_top_up=beware_of_top_up,
                analysis=True,
            )
            if not os.path.isfile(osc.get_parameters_filename()):
                osc.execute()
                
        print(10 * "=", "BEST strategy collection done!", 10 * "=")
        print(5 * "\n")
    
    else:
        if sample_name is not None:
            name_pattern = f"{sample_name}_strategy_DEFAULT_1"
        else:
            name_pattern = f"strategy_DEFAULT_1"
        
        default_osc = omega_scan(
            name_pattern=name_pattern,
            directory=os.path.join(directory, "main"),
            scan_range=scan_range,
            scan_exposure_time=scan_exposure_time,
            angle_per_frame=angle_per_frame,
            transmission=transmission,
            resolution=default_resolution,
            photon_energy=photon_energy,
            diagnostic=diagnostic,
            beware_of_top_up=beware_of_top_up,
            analysis=True,
        )
        
        if not os.path.isfile(default_osc.get_parameters_filename()):
            default_osc.execute()
            print(10 * "=", "Default strategy collection done!", 10 * "=")
            print(5 * "\n")
    
    message = f"main took {time.time() - _start:.3f} seconds"
    logging.info(message)



def udc(
    puck=8,
    sample=9,
    modifier="a",
    base_directory="/nfs/data4/2025_Run3/com-proxima2a/Commissioning/automated_operation",
    beam_align=False,
    skip_tomography=False,
    defrost=0,
    resolution=1.5,
    photon_energy=13000,
    frame_exposure_time=0.0043,
    characterization_frame_exposure_time=0.01,
    characterization_transmission=15.0,
    characterization_scan_range=1.2,
    characterization_scan_start_angles=[0, 45, 90, 135, 180],
    characterization_angle_per_frame=0.1,
    step_size_along_omega=0.025,
    wash=False,
    transmission=15.0,
    norient=1,
    sleeptime=1,
    prealign=False,
    force_transfer=False,
    force_centring=False,
    beware_of_top_up=True,
    enforce_scan_range=True,
    sample_name=None,
):
    _start = time.time()

    directory = base_directory
    instrument.check_beam()

    # 
    # STEP 0: (OPTIONAL) ALIGN THE BEAM
    #
    
    if beam_align:
        align_beam(base_directory, photon_energy=photon_energy)


    # 
    # STEP 1: MOUNT THE SAMPLE
    #
    
    m = mount(base_directory, puck, sample, wash=wash)
    if int(m) <= 0:
        print("mount -1")
        return

    if force_centring:
        instrument.goniometer.set_centring_phase()
    elif force_transfer:
        instrument.goniometer.set_transfer_phase(phase=True)
    
    if prealign:
        print("prealignment !")
        p = prealignment(directory)

        if p == -1:
            print("prealignment -1")
            return

    if defrost > 0:
        time.sleep(defrost)

    
    # 
    # STEP 2: ALIGN THE SAMPLE OPTICALLY
    #
    
    oa = opti_series(directory)
    if oa == -1:
        print("opti -1")
        return

    # 
    # STEP 3: DIFFRACTION CONTRAST TOMOGRAPHY ON THE VOLUME LIKELY TO CONTAIN CRYSTAL (FROM OPTICAL STEP)
    #
    
    t = tomo_series(
        directory,
        oa,
        step_size_along_omega=step_size_along_omega,
        transmission=transmission,
        resolution=resolution,
        photon_energy=photon_energy,
        sample_name=sample_name,
    )

    if t == -1:
        print("tomo -1")
        return

    # 
    # STEP 4: CALCULATE THE STRATEGY
    #
    
    strategy, resolution = char_series(
        directory,
        resolution=resolution,
        photon_energy=photon_energy,
        frame_exposure_time=characterization_frame_exposure_time,
        transmission=characterization_transmission,
        scan_range=characterization_scan_range,
        scan_start_angles=characterization_scan_start_angles,
        angle_per_frame=characterization_angle_per_frame,
        sample_name=sample_name,
    )

    # 
    # STEP 5: FULL RECIPROCAL SPACE MAP 
    #
    
    main_series(
        directory,
        strategy=strategy,
        transmission=transmission,
        resolution=resolution,
        photon_energy=photon_energy,
        frame_exposure_time=frame_exposure_time,
        beware_of_top_up=beware_of_top_up,
        enforce_scan_range=enforce_scan_range,
        sample_name=sample_name,
    )

    message = f"sample fully analyzed in {time.time() - _start:.3f} seconds"
    logging.info(message)
    print(message)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--puck", default=7, type=int, help="puck")
    parser.add_argument("-s", "--sample", default=1, type=int, help="sample")
    parser.add_argument("-m", "--modifier", default="a", type=str, help="modifier")
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

    udc(
        puck=args.puck,
        sample=args.sample,
        base_directory=args.directory,
        beam_align=bool(args.beam_align),
        skip_tomography=bool(args.skip_tomography),
        modifier=args.modifier,
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

FULL = list(range(1, 17))

def get_puck_and_position(x):
    return int(x['containerSampleChangerLocation']), int(x['sampleLocation'])


def udc_20250023():
    proposal = {
        "code": "mx",
        "number": "20250023",
        "proposalId": "3113",
        "title": "ALPX 2024",
        "type": "MX",
        "personId": "13597",
        "beamlineName": "PROXIMA2A",
        "comments": "Session created by MXCuBE",
        "endDate": "2025-04-21 07:59:59",
        "lastUpdate": "2025-04-21 07:59:59+02:00",
        "nbShifts": "3",
        "scheduled": "0",
        "sessionId": "46289",
        "startDate": "2025-04-20 00:00:00",
        "timeStamp": "2025-04-20 23:27:45+02:00",
        "proposalId": "3113",
        "proposalName": "mx20250023",
    }
    
    from samples import samples
    #{
        #"containerCode": "BX029A",
        #"containerSampleChangerLocation": "1",
        #"crystalId": "15099",
        #"diffractionPlan": {"diffractionPlanId": "110836"},
        #"proteinAcronym": "hTF",
        #"sampleId": "110648",
        #"sampleLocation": "5",
        #"sampleName": "CD044620_B10-1_BX029A-05",
    #},
    
    
    pucks = ["BX029A", "BX033A", "BX041A"]
    relevant = [sample for sample in samples if sample["containerCode"] in pucks]
    relevant.sort(key=get_puck_and_position)
    
    base_directory = "/nfs/data4/2025_Run2/20250023/2025-04-20/RAW_DATA"
    align_beam(base_directory)
    _start_t = time.time()
    relevant = relevant[15:]
    failed = 0
    for k, sample in enumerate(relevant):
        _start = time.time()
        puck = int(sample['containerSampleChangerLocation'])
        pin = int(sample['sampleLocation'])
        
        protein_acronym = sample["proteinAcronym"]
        sample_name = f"{protein_acronym}-{sample['sampleName']}"
        directory = f"{base_directory}/{protein_acronym}/{sample_name}"
        
        print(f"will investigate sample {sample_name} from basket {sample['containerCode']}")
        print(f"sample {k+1} of {len(relevant)} in the current run")
        
        try:
            udc(
                puck=puck,
                sample=pin,
                modifier="",
                base_directory=directory,
                frame_exposure_time=0.005,
                transmission=25,
                characterization_transmission=25.0,
                step_size_along_omega=0.025,
                sample_name=sample_name,
                wash=False,
            )
        except:
            traceback.print_exc()
            failed += 1
        
        duration = time.time() - _start
        print(f"sample {sample_name} from basket {sample['containerCode']} analyzed in {duration:.2f} seconds ({duration/60:.1f} minutes)")
        print(10*"=====")
        print(7*"\n")
    duration = time.time() - _start_t
    print(f"{len(relevant)} samples analyzed in {duration:.2f} seconds ({duration/len(relevant):.2f} per sample), failed {failed}")
    print(10*"==++==")
    print(7*"\n")
    
def udc_20231175():
    samples = [
        (3, FULL),
        (4, FULL),
        (5, FULL),
        (2, list(set(FULL) - set([6, 7]))),
        (1, [2, 3, 13]),
    ]
    names = dict([(k + 1, "SVL-16%d" % k) for k in range(5)])
    k = 0
    modifier = "c"
    for puck, puck_samples in samples:
        print(puck, names[puck])
        for sample in sorted(puck_samples):
            k += 1
            print(
                f"puck {puck} ({names[puck]}), sample {sample}, ({k} of the current run)"
            )
            _start = time.time()
            base_directory = (
                f"/nfs/data4/2024_Run4/20231175/krey/2024-11-03/{names[puck]}"
            )
            popenargs = [
                "udc.py",
                "-p",
                f"{puck}",
                "-s",
                f"{sample}",
                "-d",
                base_directory,
                "--wash",
                "-m",
                modifier,
                "--prealign",
            ]

            if k % 16 == 0:
                popenargs.append("--beam_align")
            print("popenargs")
            print(popenargs)

            commandline = " ".join(popenargs)
            print(commandline)

            # os.system(commandline)

            try:
                p = subprocess.run(
                    popenargs,
                    timeout=20 * 60,
                    text=True,
                )
                f = open(
                    os.path.join(
                        os.path.dirname(base_directory),
                        f"completed_process_{puck}_{sample}_{modifier}.pickle",
                    ),
                    "wb",
                )
                pickle.dump(p, f)
                f.close()
            except:
                traceback.print_exc()

            print(f"sample {sample} evaluation took {time.time() - _start:.3f} seconds\n")

            # udc(
            # puck=puck,
            # sample=sample,
            # base_directory=f"/nfs/data4/2024_Run4/20231175/krey/2024-10-15/{names[puck]}",
            # prealign=True,
            # wash=True,
            # beware_of_top_up=True,
            # defrost=30,
            # )
