#!/usr/bin/env python

import os
import sys
import numpy as np
import pickle
import pylab
import glob

from useful_routines import get_vector_from_position
try:
    import lmfit
except:
    lmfit = None
    from scipy.optimize import minimize
# kappa_direction = [-0.866,  0.004,  0.079]
# kappa_position = [0.276, -0.15,  -0.105]
# phi_direction = [1.0, 0.0, 0.0]
# phi_position = [-0.04,  -0.277,  0.016]

# kappa_direction = [-0.9135,  0.006    ,  0.4067]
# kappa_position = [-0.594,  0.070,  0.436]
# kappa_position = [0.4471, -1.0142, -3.3715]
# phi_direction =  [1., -0.011, 0.]
# phi_position = [0.2650,  14.965,  11.063]

# 2025-07
# kappa_direction = [-0.91330000, 0.00230000, 0.40270000]
# kappa_position = [0.94720000, 0.06700000, -0.24560000]
# phi_direction = [1.00000000, -0.01580000, 0.01160000]
# phi_position = [-0.14660000, 0.00230000, 0.27080000]

# 20251201_150725/360_zoom_5_results.npy
kd1 = -0.91729918# (init = -0.9133)
kd2 = -9.4534e-04# (init = 0.0023)
kd3 =  0.39802965# (init = 0.4027)
kp1 =  5.69867168# (init = 0.9472)
kp2 = -0.10352071# (init = 0.067)
kp3 = -2.15049712# (init = -0.2456)
pd1 =  0.99602090# (init = 1)
pd2 =  0.02097216# (init = -0.0158)
pd3 =  0.10287612# (init = 0.0116)
pp1 =  12.0310533# (init = -0.1466)
pp2 =  0.11541553# (init = 0.0023)
pp3 =  1.79116004# (init =  0.2708)

kappa_direction = [kd1, kd2, kd3]
kappa_position = [kp1, kp2, kp3]
phi_direction = [pd1, pd2, pd3]
phi_position = [pp1, pp2, pp3]

ka_index = 0
ph_index = 1
az_index = 2
ay_index = 3
cx_index = 4
cy_index = 5


def get_rotation_matrix(axis, angle):
    rads = np.radians(angle)
    cosa = np.cos(rads)
    sina = np.sin(rads)
    I = np.diag([1] * 3)
    rotation_matrix = I * cosa + axis["mT"] * (1 - cosa) + axis["mC"] * sina
    return rotation_matrix


def get_axis(direction, position):
    axis = {}
    # dn = direction/np.linalg.norm(direction)
    d = np.array(direction)
    p = np.array(position)
    axis["direction"] = d
    axis["position"] = p
    axis["mT"] = get_mT(direction)
    axis["mC"] = get_mC(direction)
    return axis


def get_mC(direction):
    mC = np.array(
        [
            [0.0, -direction[2], direction[1]],
            [direction[2], 0.0, -direction[0]],
            [-direction[1], direction[0], 0.0],
        ]
    )

    return mC


def get_mT(direction):
    mT = np.outer(direction, direction)

    return mT


def get_align_vector(t1, t2, kappa, phi, kappa_axis, phi_axis, align_direction):
    t1 = np.array(t1)
    t2 = np.array(t2)
    x = t1 - t2
    Rk = get_rotation_matrix(kappa_axis, -kappa)
    Rp = get_rotation_matrix(phi_axis, -phi)
    x = np.dot(Rp, np.dot(Rk, x)) / np.linalg.norm(x)
    c = np.dot(phi_axis["direction"], x)
    if c < 0.0:
        c = -c
        x = -x
    cos2a = pow(np.dot(kappa_axis[direction], align_direction), 2)

    d = (c - cos2a) / (1 - cos2a)

    if abs(d) > 1.0:
        new_kappa = 180.0
    else:
        new_kappa = np.degrees(np.arccos(d))

    Rk = get_rotation_matrix(kappa_axis, new_kappa)
    pp = np.dot(Rk, phi_axis["direction"])
    xp = np.dot(Rk, x)
    d1 = align_direction - c * pp
    d2 = xp - c * pp

    new_phi = np.degrees(
        np.arccos(np.dot(d1, d2) / np.linalg.norm(d1) / np.linalg.norm(d2))
    )

    newaxis = {}
    newaxis["mT"] = get_mT(pp)
    newaxis["mC"] = get_mC(pp)

    Rp = get_rotation_matrix(newaxis, new_phi)
    d = np.abs(np.dot(align_direction, np.dot(Rp, xp)))
    check = np.abs(np.dot(align_direction, np.dot(xp, Rp)))

    if check > d:
        new_phi = -new_phi

    position = get_position(
        kappa, phi, 0.5 * (t1 + t2), new_kappa, new_phi, kappa_axis, phi_axis
    )

    align_vector = new_kappa, new_phi, position

    return align_vector


def getmkc_at_zero(mkc, kappa_axis):
    mkc_at_zero = np.apply_along_axis(getmkc_line_at_kappa_zero, 1, mkc, kappa_axis)
    return mkc_at_zero


def getmkc_line_at_kappa_zero(mkc_line, kappa_axis):
    [ka, ph, az, ay, cx, cy] = mkc_line
    tk = kappa_axis["position"]
    R = get_rotation_matrix(kappa_axis, -ka)
    p = np.array([ay, cx, cy])
    ay_at_zero, cx_at_zero, cy_at_zero = tk - np.dot(R, (p - tk))
    return np.array([ka, ph, az, ay_at_zero, cx_at_zero, cy_at_zero])


def bring_to_kappa_zero(kappa_axis, kappa_source, position_source):
    tk = kappa_axis["position"]

    R = get_rotation_matrix(kappa_axis, -kappa_source)
    position_destination = tk + np.dot(R, (position_source - tk))

    return position_destination


def position_error(parameters, position_start, observations, along_axis):

    kappa_direction, kappa_position, phi_direction, phi_position = get_kdkppdpp(
        parameters, along_axis
    )

    kappa_axis = get_axis(kappa_direction, kappa_position)
    phi_axis = get_axis(phi_direction, phi_position)

    kappas_obs = observations[:, 0]
    phis_obs = observations[:, 1]
    xyz_obs = observations[:, [2, 3, 4]]

    xyz_model = np.array(
        [
            get_position(position_start, kappa, phi, kappa_axis, phi_axis)
            for kappa, phi in zip(kappas_obs, phis_obs)
        ]
    )

    # error = np.sum(np.linalg.norm(xyz_model - xyz_obs, axis=1), axis=0) / len(xyz_obs)
    error = np.linalg.norm(xyz_model - xyz_obs, axis=1)
    # error = np.mean(np.linalg.norm((model - observation)))
    # error = np.sum(np.sum(( model - observation ) ** 2, axis=1), axis=0)
    # error = np.sum(np.lin(xyz_model - xyz_obs) ** 2)
    return error


def get_kdkppdpp(
    parameters,
    along_axis=None,
    kd=kappa_direction,
    kp=kappa_position,
    pd=phi_direction,
    pp=phi_position,
):

    if type(parameters) is lmfit.parameter.Parameters:
        v = parameters.valuesdict()
        kappa_direction = [v["kd1"], v["kd2"], v["kd3"]]
        kappa_position = [v["kp1"], v["kp2"], v["kp3"]]
        phi_direction = [v["pd1"], v["pd2"], v["pd3"]]
        phi_position = [v["pp1"], v["pp2"], v["pp3"]]

    elif along_axis not in ["phi", "kappa"]:
        kappa_direction, kappa_position, phi_direction, phi_position = [
            parameters[k : k + 3] for k in range(len(parameters) // 3)
        ]
        # kappa_position, phi_position = [
        # parameters[k : k + 3] for k in range(len(parameters) // 3)
        # ]
        # kappa_direction, phi_direction = kd, pd
    elif along_axis == "kappa":
        phi_direction, phi_position = [
            parameters[k : k + 3] for k in range(len(parameters) // 3)
        ]
        kappa_direction, kappa_position = kd, kp
        # phi_direction = pd.copy()
    elif along_axis == "phi":
        kappa_direction, kappa_position = [
            parameters[k : k + 3] for k in range(len(parameters) // 3)
        ]
        phi_direction, phi_position = pd, pp
    return kappa_direction, kappa_position, phi_direction, phi_position


def clean_entries(entries, sort=True):
    entries = list(entries[np.apply_along_axis(np.any, 1, np.isnan(entries)) == False])
    entries.sort(key=lambda x: (x[0], x[1]))
    entries = np.array(entries)
    return entries


def show_all(mkc, figsize=(16, 9)):
    pylab.figure(figsize=figsize)
    pylab.plot(mkc[:, az_index], "b-", label="az")
    pylab.plot(mkc[:, ay_index], "b-", label="ay")
    pylab.plot(mkc[:, cx_index], "b-", label="cx")
    pylab.plot(mkc[:, cy_index], "b-", label="cy")
    pylab.show()


def plot_observations_and_model(
    observations,
    model,
    name_pattern="mkc",
    along_axis="",
    angle="",
    ay_index=0,
    cx_index=1,
    cy_index=2,
    figsize=(16, 9),
):

    pylab.figure(figsize=figsize)
    pylab.title(f"{along_axis.capitalize()} is {angle}")

    pylab.plot(observations[:, ay_index], "bo", label="ay experiment")
    pylab.plot(observations[:, cx_index], "ro", label="cx experiment")
    pylab.plot(observations[:, cy_index], "go", label="cy experiment")

    pylab.plot(model[:, ay_index], "b-", label="ay model")
    pylab.plot(model[:, cx_index], "r-", label="cx model")
    pylab.plot(model[:, cy_index], "g-", label="cy model")

    pylab.legend()

    if along_axis not in ["", None]:
        name_pattern += f"_{along_axis}"
    if angle not in ["", None]:
        name_pattern += f"_{angle}"

    pylab.savefig(f"{name_pattern}.png")


def explore(
    mkc,
    name_pattern,
    figsize=(16, 9),
    along_axis="kappa",
    kd=kappa_direction,
    kp=kappa_position,
    pd=phi_direction,
    pp=phi_position,
):

    er = []
    fr = []

    if along_axis == "kappa":
        unique = list(set(mkc[:, ka_index]))
        initial_parameters = pd + pp
    elif along_axis == "phi":
        unique = list(set(mkc[:, ph_index]))
        initial_parameters = kd + kp
    else:
        unique = ["all"]
        initial_parameters = kd + kp + pd + pp
        # initial_parameters = kp + pp
    unique.sort()

    print("unique", unique)
    print("along_axis", along_axis)
    print("initial_parameters", initial_parameters)
    print(
        "get_kdkppdpp(initial_parameters)",
        get_kdkppdpp(initial_parameters, along_axis=along_axis),
    )
    for angle in unique:
        if along_axis == "phi":
            mkc_work = mkc[mkc[:, ph_index] == angle]
        elif along_axis == "kappa":
            mkc_work = mkc[mkc[:, ka_index] == angle]
        else:
            mkc_work = mkc.copy()

        parameters = fit_mkc(
            mkc_work, initial_parameters, fr, er, along_axis=along_axis, angle=angle
        )

    report_fit_and_error(fr, er, unique)

    pylab.show()


def fit_mkc(
    mkc,
    initial_parameters,
    fr,
    er,
    name_pattern="mkc",
    along_axis=None,
    angle=None,
    ka_index=ka_index,
    ph_index=ph_index,
    ay_index=ay_index,
    cx_index=cx_index,
    cy_index=cy_index,
    kd=kappa_direction,
    kp=kappa_position,
    pd=phi_direction,
    pp=phi_position,
    plot=True,
    library="lmfit",
):

    observations = mkc[:, [ka_index, ph_index, ay_index, cx_index, cy_index]]
    position_start = observations[0]
    xyz_obs = mkc[:, [ay_index, cx_index, cy_index]]
    ka_obs = mkc[:, ka_index]
    ph_obs = mkc[:, ph_index]

    if library == "lmfit" and lmfit is not None:
        initial_parameters = lmfit.Parameters()
        if along_axis not in ["phi", "kappa"]:
            initial_parameters.add_many(
                ("kd1", kd[0], True, -1.0, 1.0, None, None),
                ("kd2", kd[1], True, -1.0, 1.0, None, None),
                ("kd3", kd[2], True, -1.0, 1.0, None, None),
                ("kp1", kp[0], True, None, None, None, None),
                ("kp2", kp[1], True, None, None, None, None),
                ("kp3", kp[2], True, None, None, None, None),
                ("pd1", pd[0], True, -1.0, 1.0, None, None),
                ("pd2", pd[1], True, -1.0, 1.0, None, None),
                ("pd3", pd[2], True, -1.0, 1.0, None, None),
                ("pp1", pp[0], True, None, None, None, None),
                ("pp2", pp[1], True, None, None, None, None),
                ("pp3", pp[2], True, None, None, None, None),
            )
        elif along_axis == "phi":
            initial_parameters.add_many(
                ("kd1", kd[0], True, -1.0, 1.0, None, None),
                ("kd2", kd[1], True, -1.0, 1.0, None, None),
                ("kd3", kd[2], True, -1.0, 1.0, None, None),
                ("kp1", kp[0], True, None, None, None, None),
                ("kp2", kp[1], True, None, None, None, None),
                ("kp3", kp[2], True, None, None, None, None),
                ("pd1", pd[0], False, -1.0, 1.0, None, None),
                ("pd2", pd[1], False, -1.0, 1.0, None, None),
                ("pd3", pd[2], False, -1.0, 1.0, None, None),
                ("pp1", pp[0], False, None, None, None, None),
                ("pp2", pp[1], False, None, None, None, None),
                ("pp3", pp[2], False, None, None, None, None),
            )
        elif along_axis == "kappa":
            initial_parameters.add_many(
                ("kd1", kd[0], False, -1.0, 1.0, None, None),
                ("kd2", kd[1], False, -1.0, 1.0, None, None),
                ("kd3", kd[2], False, -1.0, 1.0, None, None),
                ("kp1", kp[0], False, None, None, None, None),
                ("kp2", kp[1], False, None, None, None, None),
                ("kp3", kp[2], False, None, None, None, None),
                ("pd1", pd[0], True, -1.0, 1.0, None, None),
                ("pd2", pd[1], True, -1.0, 1.0, None, None),
                ("pd3", pd[2], True, -1.0, 1.0, None, None),
                ("pp1", pp[0], True, None, None, None, None),
                ("pp2", pp[1], True, None, None, None, None),
                ("pp3", pp[2], True, None, None, None, None),
            )
        fit = lmfit.minimize(
            position_error,
            initial_parameters,
            args=(position_start, observations, along_axis),
            method="nelder",
            #method="leastsq",
            # method="ampgo",
        )

        print(lmfit.fit_report(fit))
        parameters = fit.params

        print("-------------------------------")
        print("Parameter    Value       Stderr")
        for name, param in fit.params.items():
            try:
                print(f"{name:7s} {param.value:11.5f} {param.stderr:11.5f}")
            except:
                print(f"{name} {param.value} {param.stderr}")

    else:
        fit = minimize(
            position_error,
            initial_parameters,
            args=(position_start, observations, along_axis),
            method="Nelder-Mead",
        )
        parameters = fit.x

    kappa_direction, kappa_position, phi_direction, phi_position = get_kdkppdpp(
        parameters, along_axis=along_axis
    )

    kappa_axis = get_axis(kappa_direction, kappa_position)
    phi_axis = get_axis(phi_direction, phi_position)

    xyz_model = np.array(
        [
            get_position(position_start, kappa, phi, kappa_axis, phi_axis)
            for kappa, phi in zip(ka_obs, ph_obs)
        ]
    )

    fr.append(parameters)
    error = xyz_model - xyz_obs
    _er = np.mean(np.abs(error), axis=0)
    er.append(_er)

    plot_observations_and_model(
        xyz_obs,
        xyz_model,
        name_pattern=name_pattern,
        along_axis=along_axis,
        angle=angle,
    )

    return parameters


def report_fit_and_error(fr, er, unique):
    fr = np.array(fr)
    fr = np.round(fr, 4)
    print("fit results:")
    for a, r, e in zip(unique, fr, er):
        print(f"{a}: {np.round(r,4)} {np.round(e,4)}")
    er = np.array(er)
    er = np.round(er, 4)
    print("errors stats:")
    print("error =", np.sum(np.linalg.norm(er, axis=1), axis=0))
    print("median =", np.round(np.median(er, axis=0), 4))
    print("mean =", np.round(np.mean(er, axis=0), 4))
    print("std =", np.round(np.std(er, axis=0), 4))

    print("parameters stats:")
    print("median =", np.round(np.median(fr, axis=0), 3))
    print("mean =", np.round(np.mean(fr, axis=0), 3))
    print("std =", np.round(np.std(fr, axis=0), 3))


def get_xyz(position, xyz_keys=["AlignmentY", "CentringX", "CentringY"]):
    xyz = [position[key] for key in xyz_keys]
    return xyz

def get_position(
    position_start,
    kappa_end,
    phi_end,
    kappa_axis,
    phi_axis,
    debug=False,
    mode=1,
    epsilon=0.1,
):
    kappa_position = kappa_axis["position"]
    phi_position = phi_axis["position"]

    if type(position_start) is dict:
        kappa_start = position_start["Kappa"]
        phi_start = position_start["Phi"]
        xyz_start = get_xyz(position_start)
    else:
        kappa_start = position_start[0]
        phi_start = position_start[1]
        xyz_start = position_start[2:]

    Rk1 = get_rotation_matrix(kappa_axis, -kappa_start)
    Rp1 = get_rotation_matrix(phi_axis, -phi_start)
    Rp2 = get_rotation_matrix(phi_axis, phi_end)
    Rk2 = get_rotation_matrix(kappa_axis, kappa_end)

    if mode == 1:
        position_Rk1 = kappa_position + np.dot(Rk1, (xyz_start - kappa_position))
        if abs(phi_end - phi_start) > epsilon:
            position_Rp1 = phi_position + np.dot(Rp1, (position_Rk1 - phi_position))
            position_Rp2 = phi_position + np.dot(Rp2, (position_Rp1 - phi_position))
        else:
            position_Rp1 = position_Rk1
            position_Rp2 = position_Rp1
        position_Rk2 = kappa_position + np.dot(Rk2, (position_Rp2 - kappa_position))
    elif mode == 2:
        position_Rk1 = kappa_position - np.dot(Rk1, (kappa_position - xyz_start))
        if abs(phi_end - phi_start) > epsilon:
            position_Rp1 = phi_position - np.dot(Rp1, (phi_position - position_Rk1))
            position_Rp2 = phi_position - np.dot(Rp2, (phi_position - position_Rp1))
        else:
            position_Rp1 = position_Rk1
            position_Rp2 = position_Rp1
        position_Rk2 = kappa_position - np.dot(Rk2, (kappa_position - position_Rp2))

    if debug:
        print(f"kappa_start {kappa_start:.1f}")
        print("Rk1")
        print(Rk1)
        print(f"phi_end - phi_start: {phi_end} - {phi_start} = {phi_end-phi_start:.1f}")
        print("Rp1")
        print(Rp1)
        print("Rp2")
        print(Rp2)
        print(f"kappa_end {kappa_end:.1f}")
        print("Rk2")
        print(Rk2)
        print("position_start", xyz_start)
        print("position_Rk1  ", position_Rk1)
        print("position_Rp1  ", position_Rp1)
        print("position_Rp1  ", position_Rp2)
        print("position_Rk2  ", position_Rk2)
    return position_Rk2


def load_results(fname):
    print("load_results")
    if fname.endswith(".npy"):
        results = np.load(fname)
    elif fname.endswith(".pickle"):
        results = pickle.load(open(fname, "rb"))
    else:
        print("results format not recognized (not .npy nor .pickle), please check.")
    return results


def get_results_filename(directory, pattern="*_*_0_zoom_5_results.pickle"):
    base_pattern = os.path.join(directory, pattern)
    rfname = base_pattern.replace("*_", "").replace(".pickle", ".npy")
    return rfname
    
def get_raw_results(directory, pattern="*_*_zoom_5_results.pickle", keys=["Kappa", "Phi", "AlignmentZ", "AlignmentY", "CentringX", "CentringY"]):
    rfname = get_results_filename(directory, pattern)
    print("in get_raw_results")
    if os.path.isfile(rfname):
        results = load_results(rfname)
    else:
        print("results not yet present")
        raw_results = glob.glob(os.path.join(directory, pattern))
        print("raw_results files that conform", raw_results)
        result_vectors = []
        for rr in raw_results:
            r = pickle.load(open(rr, "rb"))
            vector = get_vector_from_position(r["result_position"], keys=keys)
            result_vectors.append(vector)
        results = np.array(result_vectors)
        np.save(rfname, results) 
    return results
    
def main(kd=kappa_direction, kp=kappa_position, pd=phi_direction, pp=phi_position):

    import argparse
    import random

    parser = argparse.ArgumentParser()

    # parser.add_option("-r", "--results", default="MK3/mkc.pickle", type=str)
    parser.add_argument(
        "-r",
        "--results",
        default="./examples/minikappa_calibration/2025-07-25_zoom_5.npy",
        type=str,
        help="results",
    )
    parser.add_argument('-p', '--pattern', default="*_zoom_5_eager_results.pickle", type=str, help="pattern")
    parser.add_argument(
        "-a", "--along_axis", default="all", type=str, help="along_axis"
    )

    args = parser.parse_args()

    if os.path.isdir(args.results):
        mkc = get_raw_results(args.results, pattern=args.pattern)
    elif args.results.endswith(".npy"):
        mkc = load_results(args.results)
    else:
        sys.exit("the results argument is not a directory nor a .npy, please check")

    print("mkc", mkc)
    mkc = clean_entries(mkc)

    name_pattern = args.results.replace(".npy", "")
    explore(mkc, name_pattern, along_axis=args.along_axis)

    # 0  : [0.0686, 0.1674, 0.0075, 0.0003, 0.4132, -0.9048] #[0.3468, -0.5826, 1.1433, -0.1746, 0.5361, -0.8553]
    # 225: [-0.094, -0.8087, 2.1235, 0.0622, 0.4187, -0.9116]
    # 135: [-0.1181, -0.521, 2.3723, 0.0638, 0.3022, -0.958]
    # 360: [0.3428, -0.3523, 0.6635, -0.2551, 0.589, -0.834]
    # 45 : [0.6437, -0.4393, 1.6068, -0.2708, 0.3573, -0.9166]
    # 270: [-0.0411, -1.164, 2.583, 0.0364, 0.4577, -0.8932]
    # 180: [0.3892, -0.5775, 1.4617, -0.1523, 0.4344, -0.8679]
    # 90 : [0.2678, -0.7335, 3.2127, -0.0523, 0.2886, -0.961]
    # 315: [0.0722, 0.1673, 0.0032, 0.0043, 0.4289, -0.9147]

    # 45 [ 0.7944  0.5303 -0.8619  0.4097  0.5467 -0.7087]
    # 90 [-0.0062  0.3777 -0.9276  0.0732  0.1124  0.1778]
    # initial_parameters = [-0.0062,  0.3777, -0.9276,  0.0732,  0.1124,  0.1778]
    # initial_parameters = [random.random() for k in range(6)]
    # initial_parameters = [0.34, -0.58, 0, 0, 0.4344, -0.9135]


if __name__ == "__main__":
    main()
