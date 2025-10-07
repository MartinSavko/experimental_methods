#!/usr/bin/env python
# conding: utf-8

import os
import pickle
import glob
import random
import numpy as np
import pylab
import sys
import datetime
import time
import copy

from experimental_methods.instrument.goniometer import goniometer
from experimental_methods.utils.useful_routines import (
    get_vector_from_position,
    get_position_from_vector,
    get_distance,
)
from experimental_methods.instrument.transmission import transmission

t = transmission()
g = goniometer()


def get_positions_between(p1, p2, stepsize=0.015):
    v1 = get_vector_from_position(p1)
    v2 = get_vector_from_position(p2)
    distance = np.linalg.norm(v2 - v1)
    nsteps = int(distance / stepsize)
    positions = []
    for axis in zip(v1, v2):
        positions.append(np.linspace(axis[0], axis[1], nsteps))
    pa = np.array(positions)
    pa = pa.T
    positions = []
    for v in pa:
        positions.append(get_position_from_vector(v))
    return positions


def get_areas(tomography_results):
    points = glob.glob(os.path.join(tomography_results, "slice*xds.results"))

    points.sort(key=lambda x: int(os.path.basename(x).split("_")[1]))

    point_results = [pickle.load(open(item, "rb")) for item in points]

    areas = np.array([point["area"] for point in point_results])

    return point_results, areas


def check_beam(
    directory, name_pattern, camera_exposure_time, transmission, photon_energy
):
    os.system(
        "beam_align.py -d %s -n %s -e %.3f -t %.1f -p %.3f -m"
        % (directory, name_pattern, camera_exposure_time, transmission, photon_energy)
    )
    g.set_zoom(1, wait=True)


def measure_diffraction_tomography(
    p1,
    p2,
    directory,
    modifier="diffraction_tomography",
    resolution=1.28,
    photon_energy=12650.0,
    transmission=5.0,
    stepsize=0.015,
    scan_range=0.01,
    vertical_range=0.5,
    angles="[0, 90, 180, 225, 315]",
    camera_exposure_time=0.035,
):
    ba_directory = os.path.join(directory, "beam_align")
    name_pattern = "beam_align_before_a"
    check_beam(
        ba_directory, name_pattern, camera_exposure_time, transmission, photon_energy
    )

    from diffraction_tomography import diffraction_tomography

    positions = get_positions_between(p1, p2, stepsize=stepsize)
    for k, p in enumerate(positions):
        tomdir = os.path.join(directory, modifier)
        nampat = "slice_%d" % k
        if not os.path.isfile(os.path.join(tomdir, "%s_parameters.pickle" % nampat)):
            g.set_position(p, wait=True)
        diftom = diffraction_tomography(
            directory=tomdir,
            name_pattern=nampat,
            scan_start_angles=angles,
            photon_energy=photon_energy,
            resolution=resolution,
            transmission=transmission,
            vertical_range=vertical_range,
            analysis=False,
            conclusion=False,
            display=False,
        )
        if not os.path.isfile(diftom.get_parameters_filename()):
            diftom.execute()
        os.system(
            "/nfs/data2/Martin/Research/tomography/shape_from_diffraction_tomography.py -d %s -n %s"
            % (tomdir, nampat)
        )

    name_pattern = "beam_align_after_b"
    check_beam(
        ba_directory, name_pattern, camera_exposure_time, transmission, photon_energy
    )


def measure_dose_curves(
    directory,
    resolution=1.28,
    photon_energy=12650.0,
    transmission=5.0,
    shift_criterion=5.0,
    acceptable_area_criterion=250.0,
    repeats=27,
    camera_exposure_time=0.035,
    protocols=["single_point", "omega_offset", "helical"],
    beam_check_delta=15 * 60.0,
):
    from experimental_methods.experiment.omega_scan import omega_scan
    from helical_scan import helical_scan

    tomography_results = os.path.join(directory, "diffraction_tomography")

    dosing_directory = os.path.join(directory, "dosing")

    beam_align_directory = os.path.join(directory, "beam_align")

    points_above_criterion_filename = os.path.join(
        directory, "points_above_criterion.pickle"
    )

    if os.path.isfile(points_above_criterion_filename):
        points_above_criterion = pickle.load(
            open(points_above_criterion_filename, "rb")
        )
    else:
        point_results, areas = get_areas(tomography_results)
        points_above_criterion = []
        for point in point_results:
            if point["area"] >= acceptable_area_criterion:
                points_above_criterion.append(point)

        for P, point in enumerate(points_above_criterion):
            if P % 3 == 0:
                random.shuffle(protocols)
            point["protocol"] = protocols[P % 3]
            point["ordinal"] = P + 1
        f = open(points_above_criterion_filename, "wb")
        pickle.dump(points_above_criterion, f)
        f.close()

    print(points_above_criterion)

    no_points = len(points_above_criterion)
    print("no_points", no_points)
    last_check = time.time()
    checks = 0
    for k in range(repeats):
        print("pass %d. (of %d)" % (k + 1, repeats))
        if datetime.datetime(2023, 6, 23, 7, 30, 0, 0) < datetime.datetime.now():
            sys.exit()

        points_k = points_above_criterion[::]
        random.shuffle(points_k)

        for P, point in enumerate(points_k):
            result_position = copy.copy(point["result_position"])
            protocol = point["protocol"]
            ordinal = point["ordinal"]
            print(
                "point %d (of %d), point ordinal number %d, pass %d (of %d)"
                % (P + 1, no_points, ordinal, k + 1, repeats)
            )
            print("chosen protocol is %s" % protocol)

            if time.time() - last_check > beam_check_delta:
                print(
                    "going to check beam position, %.3f seconds since last check"
                    % (time.time() - last_check)
                )
                checks += 1
                ba_directory = beam_align_directory
                name_pattern = "beam_align_repeat_%d" % checks
                check_beam(
                    ba_directory,
                    name_pattern,
                    camera_exposure_time,
                    transmission,
                    photon_energy,
                )
                last_check = time.time()

            if protocol == "omega_offset":
                minor_axis = point["axis_minor_length"] * point["calibration"][-1]
                omega_offset = minor_axis / 6
                result_position["AlignmentZ"] += omega_offset
            else:
                omega_offset = 0.0

            g.set_position(result_position)

            name_pattern = "pass_%d" % (k + 1)
            if protocol == "single_point":
                directory = os.path.join(
                    dosing_directory,
                    "point_%d_%s_%.3f" % (ordinal, protocol, omega_offset),
                )
                osc = omega_scan(
                    directory=directory,
                    name_pattern=name_pattern,
                    scan_range=360.0,
                    scan_exposure_time=18,
                    angle_per_frame=0.1,
                    transmission=transmission,
                    photon_energy=photon_energy,
                    resolution=resolution,
                    zoom=1,
                    diagnostic=True,
                    beware_of_top_up=True,
                    beware_of_download=True,
                )
            elif protocol == "omega_offset":
                directory = os.path.join(
                    dosing_directory,
                    "point_%d_%s_%.3f" % (ordinal, protocol, omega_offset),
                )
                osc = omega_scan(
                    directory=directory,
                    name_pattern=name_pattern,
                    scan_range=360.0,
                    scan_exposure_time=18,
                    angle_per_frame=0.1,
                    transmission=transmission,
                    photon_energy=photon_energy,
                    resolution=resolution,
                    zoom=1,
                    diagnostic=True,
                    beware_of_top_up=True,
                    beware_of_download=True,
                )
            elif protocol == "helical":
                p1, p2 = point["pca_points"]
                line_segment = get_distance(
                    p1, p2, keys=["CentringX", "CentringY", "AlignmentY", "AlignmentZ"]
                )
                directory = os.path.join(
                    dosing_directory,
                    "point_%d_%s_%.3f" % (ordinal, protocol, line_segment),
                )
                osc = helical_scan(
                    directory=directory,
                    name_pattern=name_pattern,
                    scan_range=360.0,
                    scan_exposure_time=18,
                    angle_per_frame=0.1,
                    position_start=p1,
                    position_end=p2,
                    transmission=transmission,
                    photon_energy=photon_energy,
                    resolution=resolution,
                    zoom=1,
                    diagnostic=True,
                    beware_of_top_up=True,
                    beware_of_download=True,
                )
            if not os.path.isfile(osc.get_parameters_filename()):
                osc.execute()
                for method in ["xds", "dozor"]:
                    os.system(
                        "/nfs/data4/2023_Run3/com-proxima2a/2023-06-02/RAW_DATA/Nastya/px2-0007/run_spotfinder.py -d %s -n %s -m %s &"
                        % (directory, name_pattern, method)
                    )
            else:
                print("%s already collected, moving on ..." % osc.get_template())


if __name__ == "__main__":
    # 2023-06-03 pos6
    # p1 = {'Omega': 75.00800000000163,
    #'Kappa': 88.80976577626525,
    #'Phi': 142.79343603561483,
    #'CentringX': 0.16672149122807017,
    #'CentringY': 0.47151315789473686,
    #'AlignmentX': 0.01710074347418722,
    #'AlignmentY': -1.7211553006558837,
    #'AlignmentZ': 0.09103721082532701}

    # p2 = {'Omega': 75.00820000001113,
    #'Kappa': 88.80976577626525,
    #'Phi': 142.79343603561483,
    #'CentringX': 0.1978015350877193,
    #'CentringY': 0.47299890350877194,
    #'AlignmentX': 0.01710074347418722,
    #'AlignmentY': -1.2474730001913557,
    #'AlignmentZ': 0.09218213550464505}
    # 2023-06-04 pos7
    # p1 = {'Omega': 0.005299999997077975,
    #'Kappa': 49.204492276882014,
    #'Phi': 49.36781088963721,
    #'CentringX': 0.7648190789473684,
    #'CentringY': -0.6910745614035088,
    #'AlignmentX': 0.02694884841506573,
    #'AlignmentY': -2.7120693103687223,
    #'AlignmentZ': 0.08776533467060776}

    # p2 = {'Omega': 0.0054000000018277206,
    #'Kappa': 49.204492276882014,
    #'Phi': 49.36781088963721,
    #'CentringX': 0.825109649122807,
    #'CentringY': -0.687390350877193,
    #'AlignmentX': 0.02694884841506573,
    #'AlignmentY': -2.207341560157573,
    #'AlignmentZ': 0.09196304816168688}

    # 2023-06-06 pos8
    # p1 = {'Omega': 240.00229999999283,
    #'Kappa': 88.83191421379986,
    #'Phi': 297.36562321463384,
    #'CentringX': 0.7683936403508772,
    #'CentringY': 0.5661074561403509,
    #'AlignmentX': 0.026981840391417045,
    #'AlignmentY': -1.7974657419565574,
    #'AlignmentZ': 0.10188641604861681}

    # p2 = {'Omega': 49.99034687501262,
    #'Kappa': 88.83191421379986,
    #'Phi': 297.36562321463384,
    #'CentringX': 0.7669956140350878,
    #'CentringY': 0.5669078947368421,
    #'AlignmentX': 0.026981840391417045,
    #'AlignmentY': -1.5897482588484522,
    #'AlignmentZ': 0.10188641604861681}
    # 2023-06-08 pos9
    # p1 = {'Omega': 320.04869999999937,
    #'Kappa': 0.0015625125024403275,
    #'Phi': 0.0056233125087601366,
    #'CentringX': 0.4507072368421053,
    #'CentringY': 0.14850877192982456,
    #'AlignmentX': 0.026981840391417045,
    #'AlignmentY': -1.6656133084683837,
    #'AlignmentZ': 0.10174568464949285}

    # p2 = {'Omega': 140.0486000000019,
    #'Kappa': 0.0015625125024403275,
    #'Phi': 0.0056233125087601366,
    #'CentringX': 0.40701206140350876,
    #'CentringY': 0.18311951754385966,
    #'AlignmentX': 0.026981840391417045,
    #'AlignmentY': -1.4369954083416943,
    #'AlignmentZ': 0.10174568464949285}

    # 2023-06-10 pos10
    # p1 = {'Omega': 0.003599999996367842,
    #'Kappa': 0.0015625125024403275,
    #'Phi': 0.002812500004394531,
    #'CentringX': 0.024216008771929826,
    #'CentringY': 0.04211622807017544,
    #'AlignmentX': 0.023031566722973018,
    #'AlignmentY': -1.8471846503180451,
    #'AlignmentZ': 0.0902727248733104}

    # p2 = {'Omega': 0.003700000001117587,
    #'Kappa': 0.0015625125024403275,
    #'Phi': 0.002812500004394531,
    #'CentringX': 0.024216008771929826,
    #'CentringY': 0.04211622807017544,
    #'AlignmentX': 0.023031566722973018,
    #'AlignmentY': -1.4464641055545329,
    #'AlignmentZ': 0.0902727248733104}

    # 2023-06-14 pos11
    # p1 ={'Omega': 85.60149999998976,
    #'Kappa': 0.0015625125024403275,
    #'Phi': 0.005624437508799929,
    #'CentringX': 0.09102521929824561,
    #'CentringY': 0.10901864035087719,
    #'AlignmentX': 0.026981840391417045,
    #'AlignmentY': -1.8798467069058837,
    #'AlignmentZ': 0.08817000188054225}

    # p2 = {'Omega': 85.60289999999804,
    #'Kappa': 0.0015625125024403275,
    #'Phi': 0.005624437508799929,
    #'CentringX': 0.045712719298245615,
    #'CentringY': 0.11100328947368421,
    #'AlignmentX': 0.026981840391417045,
    #'AlignmentY': -1.5767282846811668,
    #'AlignmentZ': 0.09015725295608101}
    # 2023-06-17 pos12
    # p1 = {'Omega': 30.002250000001368,
    #'Kappa': 0.0015625125024403275,
    #'Phi': 0.005624437508799929,
    #'CentringX': 0.35161184210526314,
    #'CentringY': 0.3155208333333333,
    #'AlignmentX': 0.026981840391417045,
    #'AlignmentY': -1.953468302133924,
    #'AlignmentZ': 0.08142108071816923}

    # p2 = {'Omega': 30.005799999999,
    #'Kappa': 0.0015625125024403275,
    #'Phi': 0.005624437508799929,
    #'CentringX': 0.40981359649122806,
    #'CentringY': 0.22261513157894736,
    #'AlignmentX': 0.026981840391417045,
    #'AlignmentY': -1.6640131976153434,
    #'AlignmentZ': 0.08717147909628364}

    # 2023-06-18 pos13
    # p1 = {'Omega': 240.00560000000405,
    #'Kappa': 74.47761731637127,
    #'Phi': 134.97749964840236,
    #'CentringX': -0.5929769736842105,
    #'CentringY': -0.24947916666666667,
    #'AlignmentX': 0.026981840391417045,
    #'AlignmentY': -1.8077592385781784,
    #'AlignmentZ': 0.09675461722709011}

    # p2 = {'Omega': 240.0054999999993,
    #'Kappa': 74.47761731637127,
    #'Phi': 134.97749964840236,
    #'CentringX': -0.5787774122807018,
    #'CentringY': -0.24557565789473684,
    #'AlignmentX': 0.026981840391417045,
    #'AlignmentY': -1.316904614422775,
    #'AlignmentZ': 0.09067584558435371}

    # 2023-06-20 pos14
    # p1 = {'Omega': 150.00329999999667,
    #'Kappa': 117.31691425830766,
    #'Phi': 331.4714056429241,
    #'CentringX': 0.1438157894736842,
    #'CentringY': -0.11277412280701754,
    #'AlignmentX': 0.02699833637959248,
    #'AlignmentY': -2.042662110199803,
    #'AlignmentZ': 0.08541414085594345}

    # p2 = {'Omega': 150.00350000000617,
    #'Kappa': 117.31691425830766,
    #'Phi': 331.4714056429241,
    #'CentringX': 0.12211622807017544,
    #'CentringY': -0.08939144736842106,
    #'AlignmentX': 0.02699833637959248,
    #'AlignmentY': -1.4896670985866258,
    #'AlignmentZ': 0.08439138958905179}

    # 2023-06-22
    p1 = {
        "Omega": 90.0048999999999,
        "Kappa": 0.0015625125024403275,
        "Phi": 0.005624437508799929,
        "CentringX": 0.4972203947368421,
        "CentringY": 0.06971491228070176,
        "AlignmentX": 0.02701483236776836,
        "AlignmentY": -1.815033969363654,
        "AlignmentZ": 0.08082825614310574,
    }

    p2 = {
        "Omega": 90.0048999999999,
        "Kappa": 0.0015625125024403275,
        "Phi": 0.005624437508799929,
        "CentringX": 0.4739199561403509,
        "CentringY": 0.09412280701754386,
        "AlignmentX": 0.02701483236776836,
        "AlignmentY": -1.423551177978517,
        "AlignmentZ": 0.08339595980257553,
    }

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--directory",
        default="/nfs/data4/2023_Run3/com-proxima2a/2023-06-02/RAW_DATA/Nastya/px2-0007/pos15",
        type=str,
        help="directory",
    )
    parser.add_argument("-s", "--stepsize", default=0.015, type=float, help="stepsize")
    parser.add_argument("-r", "--repeats", default=27, type=float, help="repeats")
    parser.add_argument(
        "-t", "--transmission", default=10, type=float, help="transmission"
    )
    parser.add_argument(
        "-v", "--vertical_range", default=0.4, type=float, help="vertical_range"
    )
    parser.add_argument(
        "-o", "--resolution", default=1.28, type=float, help="Resolution [Angstroem]"
    )
    parser.add_argument(
        "-p", "--photon_energy", default=12650.0, type=float, help="Photon energy "
    )
    # parser.add_argument('-a', '--angles', default='[0, 90, 180, 225, 315]', type=str, help='tomography angles')
    parser.add_argument(
        "-a",
        "--angles",
        default="[0.0, 90.0, 180.0, 225.0, 315.0]",
        type=str,
        help="tomography angles",
    )
    args = parser.parse_args()
    print(args)

    # measure_diffraction_tomography(p1, p2, args.directory,modifier='diffraction_tomography', resolution=args.resolution, photon_energy=args.photon_energy, vertical_range=args.vertical_range, angles=args.angles, transmission=5)
    measure_dose_curves(
        args.directory,
        resolution=args.resolution,
        photon_energy=args.photon_energy,
        transmission=args.transmission,
        repeats=args.repeats,
    )
    measure_diffraction_tomography(
        p1,
        p2,
        args.directory,
        modifier="diffraction_tomography_after",
        resolution=args.resolution,
        photon_energy=args.photon_energy,
        vertical_range=args.vertical_range,
        angles=args.angles,
        transmission=5,
    )
    measure_diffraction_tomography(
        p1,
        p2,
        args.directory,
        modifier="diffraction_tomography_after_5deg_oscillation",
        resolution=args.resolution,
        photon_energy=args.photon_energy,
        scan_range=5.0,
        vertical_range=args.vertical_range,
        angles=args.angles,
        transmission=5,
    )
