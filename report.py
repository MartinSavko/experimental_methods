#!/usr/bin/env python3

import os
import sys
import subprocess
import re
import pickle
import traceback
import logging

"""
s1 = "./krey/161/4/a/collect1_2_master.h5"
s2 = "../../2024-10-15/SVL-164/pos_12_b/main/omega_scan_default_pos_12_b_master.h5"
s3 = "../../2024-10-15/SVL-163/pos_06/main/omega_scan_best_1_pos_06_master.h5"

int(re.findall(".*16./pos_([\d]*).*", s2)[0])
int(re.findall(".*16./([\d]*).*", s1)[0])
".*16./pos_([\d]*).*_master.h5|.*16./([\d]*).*"
"""
sys.path.insert(0, "/nfs/data4/Martin/Research/yam_scripts")
from load_xparm import XPARM
from xds_plot_integrate import IntegrateLp

uinteger = "[\d]+"
sinteger = "[-+\d]+"
space = "[\s]+"
# https://stackoverflow.com/questions/4703390/how-to-extract-a-floating-number-from-a-string
# ifloat = "[-+]?(\d+([.,]\d*)?|[.,]\d+)([eE][-+]?\d+)?"
# ifloat = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'
ifloat = "[-+\d]+\.[\d]*"
sample_re = re.compile(".*16./pos_([\d]*).*_master.h5|.*16./([\d]*).*")

pucks1 = list((map(str, range(160, 165))))
pucks2 = [f"SVL-{p}" for p in pucks1]
pucks = pucks1 + pucks2

report_puck_names = {}
for puck in pucks:
    if "SVL" in puck:
        puck_name = puck
    else:
        puck_name = f"SVL-{puck}"
    report_puck_names[puck] = puck_name


def get_puck(path):
    items = path.split("/")
    puck = None
    for item in items:
        if item in pucks:
            return report_puck_names[item]


def get_sample(path):
    sample = None
    candidates = sample_re.findall(path)
    candidates = candidates[0]
    for c in candidates:
        if c != "":
            sample = int(c)
    return sample


def get_parameters(master_filename):
    parameters_filename = master_filename.replace("_master.h5", "_parameters.pickle")
    parameters = pickle.load(open(parameters_filename, "rb"))
    return parameters


def get_template(master_filename):
    basename = os.path.basename(master_filename)
    template = basename.replace("_master.h5", "")
    return template


def get_dirname(master_filename):
    dirname = os.path.dirname(master_filename)
    return dirname


def get_table1(master_filename):
    directory = get_dirname(master_filename)
    template = get_template(master_filename)
    find_table1 = f'find {directory}/process -iname "staraniso_alldata-unique.table1" | grep {template}'
    table1s = subprocess.getoutput(find_table1)
    print(table1s)
    t1s = []
    for table1 in table1s.split("\n"):
        t1 = open(table1, "r").read()
        t1s.append(t1)
    return t1s


def get_correct(master_filename):
    directory = get_dirname(master_filename)
    template = get_template(master_filename)
    find_correct = f'find {directory}/process -iname "correct.lp" | grep {template} | grep -v status'
    corrects = subprocess.getoutput(find_correct).split("\n")
    return corrects[0]


def get_ISa_table(master_filename):
    correct = get_correct(master_filename)
    print(f"correct [{correct}]")
    if correct:
        find_ISa_table = f'grep "     a        b          ISa" -A1 {correct}'
        ISa_table = subprocess.getoutput(find_ISa_table)
    else:
        ISa_table = None
    return ISa_table


def get_ISa(ISa_table):
    if ISa_table:
        ls = ISa_table.split("\n")
        ns = ls[1]
        ISas = ns.split()[-1]
        ISa = float(ISas)
    else:
        ISa = None
    return ISa


def get_xparm(xds_directory="./"):
    xparm = XPARM(os.path.join(xds_directory, "XPARM.XDS"))
    return xparm


def get_info_from_integrate_lp(xds_directory="./"):
    integrate_lp = IntegrateLp(os.path.join(xds_directory, "INTEGRATE.LP"))
    return integrate_lp


def get_results(search_string, filepath):
    ress = re.compile(search_string)
    read = open(filepath, "r").read()
    raw_results = ress.findall(read)
    if raw_results and type(raw_results[0]) is tuple:
        results = list(list(map(float, item)) for item in raw_results)
    else:
        results = list(map(float, raw_results))
    results_array = np.array(results)
    return results_array


def get_colspot(xds_directory="./", full=False):
    """*** DEFINITION OF SYMBOLS ***
    NBKG      = NUMBER OF BACKGROUND PIXELS ON DATA IMAGE
    NSTRONG   = NUMBER OF STRONG PIXELS ON DATA IMAGE
    I/O-FLAG  = ERROR CODE AFTER ACCESSING DATA IMAGE
                 0: NO ERROR
                -1: CANNOT OPEN OR READ IMAGE FILE
                -3: WRONG DATA FORMAT
    FRAME #      NBKG     NSTRONG   I/O-FLAG
       3301     33999        1461       0
    """
    test_line1 = "    3530     34916        1474       0"
    test_line2 = "    3530     34916        1474      -1"
    # spots = re.compile("[\s]+([\d])+[\s]+([\d])+[\s]+([\d]+)[\s]+([-\d]+)")
    # full_line = "[\s]+([\d]+)[\s]+([\d]+)[\s]+([\d]+)[\s]+([\d]+)"
    nstrong = f"{space}({uinteger})"
    if full:
        frame, nbkg = 2 * (f"{space}({uinteger})",)
        error = f"{space}({sinteger})"
    else:
        frame, nbkg = 2 * (f"{space}{uinteger}",)
        error = f"{space}{sinteger}"

    search_string = f"{frame}{nbkg}{nstrong}{error}"

    filepath = os.path.join(xds_directory, "COLSPOT.LP")

    results_array = get_results(search_string, filepath)

    return results_array


def get_integrate(xds_directory="./", filename="INTEGRATE.LP"):
    """

    *** DEFINITION OF SYMBOLS ***
    IER     = ERROR CODE AFTER ACCESSING DATA IMAGE
                0: NO ERROR
                -1: CANNOT OPEN OR READ IMAGE FILE
                -3: WRONG DATA FORMAT
    SCALE   = SCALING FACTOR FOR THIS DATA IMAGE
    NBKG    = NUMBER OF BACKGROUND PIXELS ON DATA IMAGE
    NOVL    = NUMBER OF OVERLOADED REFLECTIONS ON DATA IMAGE
    NEWALD  = NUMBER OF REFLECTIONS CLOSE TO THE EWALD SPHERE
    NSTRONG = NUMBER OF STRONG REFLECTIONS ON DATA IMAGE
    NREJ    = NUMBER OF UNEXPECTED REFLECTIONS
    SIGMAB  = BEAM_DIVERGENCE_E.S.D.=SIGMAB
    SIGMAR  = REFLECTING_RANGE_E.S.D.=SIGMAR (MOSAICITY)

    IMAGE IER    SCALE     NBKG NOVL NEWALD NSTRONG  NREJ   SIGMAB   SIGMAR
    1282   0    0.982    27756    0   5622     236     1  0.02209  0.16834
    1283   0    0.981    28401    0   5614     257     1  0.02184  0.16843
    1284   0    0.978    28303    0   5615     268     1  0.02233  0.16768
    1285   0    0.978    28875    0   5604     263     1  0.02181  0.17911
    """
    test_line1 = (
        " 1282   0    0.982    27756    0   5622     236     1  0.02209  0.16834"
    )
    test_line2 = (
        " 1285  -3    0.978    28875    0   5604     263     1  0.02181  0.17911"
    )

    image = f"{space}({uinteger})"
    ier = f"{space}({sinteger})"
    scale = f"{space}({ifloat})"
    nbkg = f"{space}({uinteger})"
    novl = f"{space}({uinteger})"
    newald = f"{space}({uinteger})"
    nstrong = f"{space}({uinteger})"
    nrej = f"{space}({uinteger})"
    sigmab = f"{space}({ifloat})"
    sigmar = f"{space}({ifloat})"

    search_string = (
        image + ier + scale + nbkg + novl + newald + nstrong + nrej + sigmab + sigmar
    )

    filepath = os.path.join(xds_directory, filename)

    results_array = get_results(search_string, filepath)

    return results_array


def get_spot_xds(xds_directory="./", indices=False, filename="SPOT.XDS"):
    test_line1 = " 1113.45 2028.28 44.14 190249. "
    test_line2 = " 828.58 1767.77 2907.61 18080.  12 -7 -24"
    x, y, z, Intensity = 4 * (f"{space}({ifloat})",)
    search_string = f"{x}{y}{z}{Intensity}"
    if indices:
        h, k, l = 3 * (f"{space}({sinteger})",)
        search_string += f"{h}{k}{l}"

    filepath = os.path.join(xds_directory, filename)

    results_array = get_results(search_string, filepath)

    return results_array


def get_xds_results(xds_directory):
    logging.info("processing the xds results")
    errorMessage = "Something unusual happend during XDS processing, can't deal with it yet, sorry. Please check."

    if not os.path.isdir(xds_directory):
        return {}

    os.chdir(xds_directory)
    try:
        UNIT_CELL_CONSTANTS = subprocess.getoutput(
            "grep UNIT_CELL_CONSTANTS XDS_ASCII.HKL"
        ).split()[1:]
        SPACE_GROUP_NUMBER = subprocess.getoutput(
            "grep SPACE_GROUP_NUMBER XDS_ASCII.HKL"
        ).split()[1:]
        INCIDENT_BEAM_DIRECTION = subprocess.getoutput(
            "grep INCIDENT_BEAM_DIRECTION XDS_ASCII.HKL"
        ).split()[1:]
        INCLUDE_RESOLUTION_RANGE = subprocess.getoutput(
            "grep INCLUDE_RESOLUTION_RANGE XDS_ASCII.HKL"
        ).split()[1:]
        REFLECTION_RECORDS = subprocess.getoutput(
            'grep NUMBER OF REFLECTION RECORDS ON OUTPUT FILE "XDS_ASCII.HKL" CORRECT.LP'
        ).split()[-1]
        ACCEPTED_OBSERVATIONS = subprocess.getoutput(
            'grep "NUMBER OF ACCEPTED OBSERVATIONS (INCLUDING SYSTEMATIC ABSENCES)" CORRECT.LP'
        ).split()[-1]
        REJECTED = subprocess.getoutput(
            'grep "NUMBER OF REJECTED MISFITS & ALIENS (marked by -1\*SIGMA(IOBS))" CORRECT.LP'
        ).split()[-1]
        COMPLETNESS_AND_QUALITY = subprocess.getoutput(
            'grep -A 275 "STATISTICS OF SAVED DATA SET" CORRECT.LP | grep -A 75 "COMPLETENESS AND QUALITY OF DATA SET"'
        )
        DIFFRACTION_PARAMETERS = subprocess.getoutput(
            'grep -A 30 "REFINEMENT OF DIFFRACTION PARAMETERS USING ALL IMAGES" CORRECT.LP'
        )
    except:
        UNIT_CELL_CONSTANTS = errorMessage  # subprocess.getoutput('grep UNIT_CELL_CONSTANTS XDS_ASCII.HKL').split()[1:]
        SPACE_GROUP_NUMBER = errorMessage  #  subprocess.getoutput('grep SPACE_GROUP_NUMBER XDS_ASCII.HKL').split()[1:]
        INCIDENT_BEAM_DIRECTION = errorMessage  #  subprocess.getoutput('grep INCIDENT_BEAM_DIRECTION XDS_ASCII.HKL').split()[1:]
        INCLUDE_RESOLUTION_RANGE = errorMessage  #  subprocess.getoutput('grep INCLUDE_RESOLUTION_RANGE XDS_ASCII.HKL').split()[1:]
        REFLECTION_RECORDS = errorMessage  #  subprocess.getoutput('grep NUMBER OF REFLECTION RECORDS ON OUTPUT FILE "XDS_ASCII.HKL" CORRECT.LP').split()[-1]
        ACCEPTED_OBSERVATIONS = errorMessage  #  subprocess.getoutput('grep "NUMBER OF ACCEPTED OBSERVATIONS (INCLUDING SYSTEMATIC ABSENCES)" CORRECT.LP').split()[-1]
        REJECTED = errorMessage  #  subprocess.getoutput('grep "NUMBER OF REJECTED MISFITS & ALIENS (marked by -1\*SIGMA(IOBS))" CORRECT.LP').split()[-1]
        COMPLETNESS_AND_QUALITY = errorMessage  #  subprocess.getoutput('grep -A 132 "STATISTICS OF SAVED DATA SET" CORRECT.LP')
        DIFFRACTION_PARAMETERS = errorMessage  #  subprocess.getoutput('grep -A 30 "REFINEMENT OF DIFFRACTION PARAMETERS USING ALL IMAGES" CORRECT.LP')
        traceback.print_exc()

    xds_results = {
        "UNIT_CELL_CONSTANTS": UNIT_CELL_CONSTANTS,
        "SPACE_GROUP_NUMBER": SPACE_GROUP_NUMBER,
        "INCIDENT_BEAM_DIRECTION": INCIDENT_BEAM_DIRECTION,
        "INCLUDE_RESOLUTION_RANGE": INCLUDE_RESOLUTION_RANGE,
        "REFLECTION_RECORDS": REFLECTION_RECORDS,
        "ACCEPTED_OBSERVATIONS": ACCEPTED_OBSERVATIONS,
        "REJECTED": REJECTED,
        "COMPLETNESS_AND_QUALITY": COMPLETNESS_AND_QUALITY,
        "DIFFRACTION_PARAMETERS": DIFFRACTION_PARAMETERS,
    }
    rdp = DIFFRACTION_PARAMETERS
    caq = COMPLETNESS_AND_QUALITY

    new_keys = [
        "refined_distance",
        "refined_detector_origin",
        "refined_direct_beam_coordinates",
        "refined_mosaicity",
        "spots_used_for_refinement",
        "sigma_spot",
        "sigma_spindle",
        "refined_space_group",
        "total_spots",
        "unique_spots",
        "possible_spots",
        "completness",
        "r_observed",
        "r_expected",
        "compared",
        "i_sigma",
        "r_meas",
        "cc_half",
        "a_corr",
        "sig_ano",
        "nano",
    ]
    try:
        xds_results["refined_distance"] = re.findall(
            "CRYSTAL TO DETECTOR DISTANCE \(mm\)[\ ]*([\d\.]*)", rdp
        )[0]
        xds_results["refined_detector_origin"] = re.findall(
            "DETECTOR ORIGIN \(PIXELS\) AT[\ ]*([\d\.]*)[\ ]*([\d\.]*)", rdp
        )[0]
        xds_results["refined_direct_beam_coordinates"] = re.findall(
            "DETECTOR COORDINATES \(PIXELS\) OF DIRECT BEAM[\ ]*([\d\.]*)[\ ]*([\d\.]*)",
            rdp,
        )[0]
        xds_results["refined_mosaicity"] = re.findall(
            "CRYSTAL MOSAICITY \(DEGREES\)[\ ]*([\d\.]*)", rdp
        )[0]
        xds_results["spots_used_for_refinement"] = re.findall(
            "REFINED VALUES OF DIFFRACTION PARAMETERS DERIVED FROM[\ ]*([\d\.]*)[\ ]*INDEXED SPOTS",
            rdp,
        )[0]
        xds_results["sigma_spot"] = re.findall(
            "STANDARD DEVIATION OF SPOT    POSITION \(PIXELS\)[\ ]*([\d\.]*)", rdp
        )[0]
        xds_results["sigma_spindle"] = re.findall(
            "STANDARD DEVIATION OF SPINDLE POSITION \(DEGREES\)[\ ]*([\d\.]*)", rdp
        )[0]
        xds_results["refined_space_group"] = re.findall(
            "SPACE GROUP NUMBER[\ ]*([\d]*)", rdp
        )[0]
        stats = re.findall(
            "total[\ ]*(\d*)[\ ]*(\d*)[\ ]*(\d*)[\ ]*([\d\.\%]*)[\ ]*([\d\.\%]*)[\ ]*([\d\.\%]*)[\ ]*(\d*)[\ ]*([\d\.]*)[\ ]*([\d\.\%]*)[\ ]*([\d\.\*]*)[\ ]*([\d\-\+]*)[\ ]*([\d\.]*)[\ ]*(\d*)",
            caq,
        )[0]
        (
            total,
            unique,
            possible,
            completness,
            r_observed,
            r_expected,
            compared,
            i_sigma,
            r_meas,
            cc_half,
            a_corr,
            sig_ano,
            nano,
        ) = stats
        xds_results["total_spots"] = total
        xds_results["unique_spots"] = unique
        xds_results["possible_spots"] = possible
        xds_results["completness"] = completness
        xds_results["r_observed"] = r_observed
        xds_results["r_expected"] = r_expected
        xds_results["compared"] = compared
        xds_results["i_sigma"] = i_sigma
        xds_results["r_meas"] = r_meas
        xds_results["cc_half"] = cc_half
        xds_results["a_corr"] = a_corr
        xds_results["sig_ano"] = sig_ano
        xds_results["nano"] = nano
    except:
        traceback.print_exc()
        print("rdp")
        print(rdp)
        print("caq")
        print(caq)

        errorMessage = "Something unusual happend during XDS processing, can't deal with it yet, sorry. Please check."
        for key in new_keys:
            xds_results[key] = errorMessage

    return xds_results


def main(directory="./"):
    find_master_files = (
        f'find {directory} -iname "*_master.h5" | grep -v ref | grep -v process'
    )
    print(f"executing {find_master_files}")
    master_files = subprocess.getoutput(find_master_files)
    print(master_files)
    print()
    results = {}
    for m in master_files.split("\n"):
        if "master.h5" not in m:
            continue
        print(m)
        try:
            parameters = get_parameters(m)
        except:
            print(f"something is wrong with {m}")
            continue

        if "omega scan" not in parameters["description"].lower():
            print("not an omega scan, moving on ...")
            continue
        puck = get_puck(m)
        sample = get_sample(m)
        sid = f"{puck}_{sample}"

        print(get_puck(m), get_sample(m))
        try:
            t1s = get_table1(m)
            print(f"#tables1 {len(t1s)}")
        except:
            message = "staraniso_alldata-unique.table1 does not seem present"
            t1s = [message]
            print(message)

        try:
            ISa_table = get_ISa_table(m)
            ISa = get_ISa(ISa_table)
        except:
            traceback.print_exc()
            ISa_table = None
            ISa = None

        if sid not in results:
            results[sid] = {}
        if m not in results[sid]:
            results[sid][m] = {}

        results[sid][m]["table1"] = t1s[0]
        results[sid][m]["ISa_table"] = ISa_table
        results[sid][m]["ISa"] = ISa

    f = open("results.pickle", "wb")
    pickle.dump(results, f)
    f.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-d",
        "--directory",
        default="/nfs/data4/2024_Run4/20231175/krey",
        type=str,
        help="directory",
    )
    args = parser.parse_args()
    print("args", args)

    main(directory=args.directory)
