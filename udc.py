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

from optical_alignment import optical_alignment
from volume_aware_diffraction_tomography import volume_aware_diffraction_tomography
from reference_images import reference_images
from omega_scan import omega_scan
from beam_align import beam_align as bac
from diffraction_tomography import diffraction_tomography

from beamline import beamline

logging.basicConfig(
    format="%(asctime)s |%(module)s | %(message)s",
    level=logging.INFO,
)

b = beamline()


def opti_series(directory, samp):
    _start = time.time()

    b.goniometer.set_centring_phase()

    oa = optical_alignment(
        name_pattern="oa_%s_zoom_1_eager" % samp,
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

    oa = optical_alignment(
        name_pattern="oa_%s_zoom_1_careful" % samp,
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

    oa = optical_alignment(
        name_pattern="oa_%s_zoom_X_careful" % samp,
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

    logging.info(f"optical alignment took {time.time() - _start} seconds")
    return oa


def tomo_series(
    directory,
    samp,
    oa,
    position=None,
    transmission=33.3,
    resolution=1.3,
    photon_energy=13000.0,
    horizontal_step_size=0.015,
    diagnostic=False,
):
    _start = time.time()

    vadt = volume_aware_diffraction_tomography(
        name_pattern="tomo_a_%s" % samp,
        directory=os.path.join(directory, "tomo"),
        volume=oa.get_pcd_mm_name(),
        horizontal_step_size=horizontal_step_size,
        resolution=resolution,
        photon_energy=photon_energy,
        transmission=transmission,
        diagnostic=diagnostic,
    )
    if not os.path.isfile(vadt.get_parameters_filename()):
        vadt.execute()

    results = vadt.get_results()

    logging.info(f"tomography took {time.time() - _start} seconds")

    try:
        position = results[0]["result_position"]
    except:
        # logging.info(traceback.format_exc())
        traceback.print_exc()
        print(f"results {results}")
        return -1

    b.goniometer.set_position(position)

    dt = diffraction_tomography(
        name_pattern="tomo_opt_%s" % samp,
        directory=os.path.join(directory, "tomo"),
        position=position,
        vertical_range=2.5 * max(vadt.get_bounding_rays()),
        analysis=True,
        conclusion=True,
        display=False,
        dont_move_motors=True,
        photon_energy=photon_energy,
        diagnostic=diagnostic,
    )
    if not os.path.isfile(dt.get_parameters_filename()):
        dt.execute()

    position = dt.get_result_position()
    b.goniometer.set_position(position)


def char_series(
    directory,
    samp,
    transmission=33.3,
    resolution=1.3,
    photon_energy=13000.0,
    frame_exposure_time=0.01,
    scan_range=1.2,
    angle_per_frame=0.1,
    scan_start_angles=[0.0, 45.0, 90.0, 135, 180.0],
):
    _char_start = time.time()

    args = {
        "name_pattern": "reference_%s" % samp,
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
    resolution = dozor_results[:, -1].min()
    logging.info(f"strategy from reference images {strategy}")
    logging.info(f"characterization took {time.time() - _char_start} seconds")
    return strategy, resolution


def main_series(
    directory,
    samp,
    strategy=[],
    resolution=None,
    transmission=33.3,
    photon_energy=13000.0,
    angle_per_frame=0.1,
    scan_range=400,
    frame_exposure_time=0.0043,
    enforce_scan_range=True,
    diagnostic=False,
    beware_of_top_up=True,
):
    _start = time.time()

    scan_exposure_time = (scan_range / angle_per_frame) * frame_exposure_time

    default_resolution = resolution
    if default_resolution is not None:
        default_resolution -= 0.1
        
    default_osc = omega_scan(
        name_pattern="omega_scan_default_%s" % samp,
        directory=os.path.join(directory, "main"),
        scan_range=scan_range,
        scan_exposure_time=scan_exposure_time,
        angle_per_frame=angle_per_frame,
        transmission=transmission,
        resolution=default_resolution,
        photon_energy=photon_energy,
        diagnostic=diagnostic,
        beware_of_top_up=beware_of_top_up,
    )
    if strategy == []:
        if not os.path.isfile(default_osc.get_parameters_filename()):
            default_osc.execute()
    else:
        for k, wedge in enumerate(strategy):
            wedge["resolution"] = resolution
            best_scan_exposure_time = wedge["scan_exposure_time"]
            best_scan_range = wedge["nimages"] * wedge["angle_per_frame"]
            if enforce_scan_range and (k+1) == len(strategy):
                best_scan_exposure_time = best_scan_exposure_time * (
                    scan_range / best_scan_range
                )
                best_scan_range = scan_range

            logging.info("best_scan_range", best_scan_range)
            logging.info("best_scan_exposure_time", best_scan_exposure_time)

            osc = omega_scan(
                name_pattern=f"omega_scan_best_{wedge['order']}_{samp}",
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
            )
            if not os.path.isfile(osc.get_parameters_filename()):
                osc.execute()

        if not os.path.isfile(default_osc.get_parameters_filename()):
            default_osc.execute()
    message = f"main took {time.time() - _start} seconds"
    logging.info(message)


def mount(puck, sample, wash=True):
    _start = time.time()
    if b.cats.isoff():
        b.cats.on()

    mpuck, msample = b.cats.get_mounted_puck_and_sample()
    if mpuck != puck or msample != sample or wash:
        b.cats.mount(puck, sample, prepare_centring=False)

    if b.cats.get_mounted_sample_id()[0] == -1:
        logging.info(f"sample {sample} puck {puck} not present?")
        return -1
    elif wash:
        b.cats.mount(puck, sample)
        if b.cats.get_mounted_sample_id()[0] == -1:
            logging.info(f"sample {sample} puck {puck} not present?")
            return -1

    logging.info(f"sample mounted in {time.time() - _start} seconds")


def prealignment(directory, samp, force_transfer=False):
    _start = time.time()

    if force_transfer:
        b.goniometer.set_transfer_phase(phase=True)

    oa = optical_alignment(
        name_pattern="oa_%s_zoom_1_eager_init" % samp,
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
        logging.info(f"sample {samp} base detected but not visible, is pin damaged?")
        return -1

    logging.info(f"sample prealigned in {time.time() - _start} seconds")


def align_beam(base_directory):
    ba = bac(
        name_pattern="beam_align_1", directory=os.path.join(base_directory, "beam")
    )
    ba.execute()


def udc(
    puck=8,
    sample=9,
    modifier="a",
    base_directory="/nfs/data4/2024_Run3/com-proxima2a/Commissioning/automated_operation",
    beam_align=False,
    dont_do_tomography=False,
    defrost=30,
    resolution=1.35,
    photon_energy=13000,
    frame_exposure_time=0.0043,
    characterization_frame_exposure_time=0.01,
    characterization_transmission=33.3,
    characterization_scan_range=1.2,
    characterization_scan_start_angles=[0, 45, 90, 135, 180],
    characterization_angle_per_frame=0.1,
    horizontal_step_size=0.015,
    wash=True,
    transmission=33.3,
    norient=1,
    sleeptime=1,
    do_prealignment=False,
    force_transfer=False,
    beware_of_top_up=True,
    enforce_scan_range=True,
):
    _start = time.time()

    samp = f"pos_{sample:02d}_{modifier}"
    logging.info(f"samp {samp}")
    directory = os.path.join(base_directory, samp)

    b.check_beam()

    if beam_align:
        align_beam(base_directory)

    m = mount(puck, sample, wash=wash)
    if m == -1:
        print('mount -1')
        return

    if do_prealignment:
        p = prealignment(directory, samp, force_transfer=force_transfer)

        if p == -1:
            print('prealignment -1')
            return

    if defrost > 0:
        time.sleep(defrost)

    oa = opti_series(directory, samp)
    if oa == -1:
        print('opti -1')
        return

    t = tomo_series(
        directory,
        samp,
        oa,
        horizontal_step_size=horizontal_step_size,
        transmission=transmission,
        resolution=resolution,
        photon_energy=photon_energy,
    )

    if t == -1:
        print('tomo -1')
        return

    strategy, resolution = char_series(
        directory,
        samp,
        resolution=resolution,
        photon_energy=photon_energy,
        frame_exposure_time=characterization_frame_exposure_time,
        transmission=characterization_transmission,
        scan_range=characterization_scan_range,
        scan_start_angles=characterization_scan_start_angles,
        angle_per_frame=characterization_angle_per_frame,
    )

    main_series(
        directory,
        samp,
        strategy=strategy,
        transmission=transmission,
        resolution=resolution,
        photon_energy=photon_energy,
        frame_exposure_time=frame_exposure_time,
        beware_of_top_up=beware_of_top_up,
        enforce_scan_range=enforce_scan_range,
    )

    message = f"sample fully analysed in {time.time() - _start} seconds"
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
        default="/nfs/data4/2024_Run4/com-proxima2a/Commissioning/automated_operation/px2-0047",
        help="directory",
    )
    parser.add_argument("-w", "--wash", action="store_true", help="wash")
    parser.add_argument("-b", "--beam_align", action="store_true", help="beam_align")
    parser.add_argument(
        "-t", "--dont_do_tomography", action="store_true", help="dont do tomography"
    )
    parser.add_argument("-n", "--norient", default=1, type=int, help="norient")
    parser.add_argument("-M", "--defrost", default=0, type=float, help="defrost")
    parser.add_argument(
        "-P", "--prealignment", action="store_true", help="prealignment"
    )
    parser.add_argument(
        "-B", "--dont_enforce_scan_range", action="store_true", help="dont_enforce_scan_range"
    )
    parser.add_argument(
        "-F",
        "--force_transfer",
        action="store_true",
        help="force_transfer befor prealignment",
    )
    parser.add_argument(
        "-T", "--ignore_top_up", action="store_true", help="ignore top up"
    )
    parser.add_argument(
        "-e", "--photon_energy", default=13215, type=float, help="photon energy"
    )
    parser.add_argument(
        "-r", "--transmission", default=33.3, type=float, help="transmission"
    )
    parser.add_argument(
        "-R", "--resolution", default=1.35, type=float, help="resolution"
    )
    parser.add_argument(
        "-f",
        "--frame_exposure_time",
        default=0.0043,
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
        default=33.3,  # 5
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
        "-B",
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
        dont_do_tomography=bool(args.dont_do_tomography),
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
        do_prealignment=bool(args.prealignment),
        force_transfer=bool(args.force_transfer),
        beware_of_top_up=not bool(args.ignore_top_up),
        enforce_scan_range=not bool(args.dont_enforce_scan_range),
    )

FULL = list(range(1, 17))


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
            print(f"puck {puck} ({names[puck]}), sample {sample}, ({k} of the current run)")
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
                "--prealignment",
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
                    popenargs, timeout=20 * 60, text=True,
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
                
            print(f"sample {sample} evaluation took {time.time() - _start} seconds\n")

            # udc(
            # puck=puck,
            # sample=sample,
            # base_directory=f"/nfs/data4/2024_Run4/20231175/krey/2024-10-15/{names[puck]}",
            # do_prealignment=True,
            # wash=True,
            # beware_of_top_up=True,
            # defrost=30,
            # )
