#!/usr/bin/env python

import os
import sys
import numpy as np
import pickle
import pylab
import glob
import seaborn as sns
sns.set(color_codes=True)

from matplotlib import rc


# https://stackoverflow.com/questions/70438972/latex-error-file-type1cm-sty-not-found
# yum install texlive-type1cm
# https://github.com/matplotlib/matplotlib/issues/16911
# yum install texlive-cm-super
# https://tex.stackexchange.com/questions/75166/error-in-tex-live-font-not-loadable-metric-tfm-file-not-found
# yum install texlive-collection-fontsrecommended
# yum install texlive-collection-latexrecommended
rc("font", **{"family": "serif", "serif": ["Palatino"]})
rc("text", usetex=True)

from useful_routines import (
    get_vector_from_position,
    get_pickled_file,
)

try:
    import lmfit
except:
    lmfit = None
    from scipy.optimize import minimize

""""
https://patorjk.com/software/taag/#p=display&f=Standard&t=MiniKappa&x=none&v=4&h=4&w=80&we=false
  __  __ _       _ _  __
 |  \/  (_)_ __ (_) |/ /__ _ _ __  _ __   __ _
 | |\/| | | '_ \| | ' // _` | '_ \| '_ \ / _` |
 | |  | | | | | | | . \ (_| | |_) | |_) | (_| |
 |_|  |_|_|_| |_|_|_|\_\__,_| .__/| .__/ \__,_|
                            |_|   |_|

https://www.arinax.com/mini-kappa-goniometer-head/
length: 120 mm
diameter: 131 mm
alpha: 24 deg
mass: 280 g

height of the nozzle: 66.77 mm
kappa arc radius: 37.9 mm
"""





#https://askubuntu.com/questions/697171/how-to-select-all-in-terminator

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
# kd1 = -0.91729918# (init = -0.9133)
# kd2 = -9.4534e-04# (init = 0.0023)
# kd3 =  0.39802965# (init = 0.4027)
# kp1 =  5.69867168# (init = 0.9472)
# kp2 = -0.10352071# (init = 0.067)
# kp3 = -2.15049712# (init = -0.2456)
# pd1 =  0.99602090# (init = 1)
# pd2 =  0.02097216# (init = -0.0158)
# pd3 =  0.10287612# (init = 0.0116)
# pp1 =  12.0310533# (init = -0.1466)
# pp2 =  0.11541553# (init = 0.0023)
# pp3 =  1.79116004# (init =  0.2708)

# combined 4 separate calibrations 2025-12-03
alpha = np.deg2rad(24.)
nozzle_height = 66.77
standard_pin = 21.16 - 2.31 #(18.85)
kappa_arc_radius = 37.9


kd1= -np.cos(alpha) #-0.94094450 #(init = -0.9172992)
kd2=  0. #(init = -0.00094534)
kd3= +np.sin(alpha) #0.32077670 #(init = 0.3980297)
#kp1=  0.02745424 #-0.02557607 #-0.23954886 #-0.2181 #+0.0392 #+0.1163 #-4.3662 #-(nozzle_height + standard_pin) #-93.7428 #(init = 5.698672)
#kp2=  0.85092692 #0.20107827 #0.15697340 #+0.2920 #(init = -0.1035207)
#kp3= -0.08524468 #0.72475444 #0.74200077 #+0.4945 #(init = -2.150497)
# refined
#kp1=  0.0296
#kp2=  0.8474
#kp3= -0.0867
# from mitegen 50 um tip after chaning Y offset
kp1= -0.008679
kp2=  0.860044
kp3= -0.152869

pd1= +1. #0.98306555 #(init = 0.9960209)
pd2=  0. #0.19796048 #(init = 0.02097216)
pd3=  0. #0.09274237 #(init = 0.1028761)
pp1=  0. #-1.7200 #(init = 12.03105)
# from standard calibration
#pp2=  0.78354001 #0.12338922 #0.04722776 #+0.2171 #-0.0523 #(init = 0.1154155)
#pp3= -0.01939916 #0.79697896 #0.7920446642048858 #+0.5446 #(init = 1.79116)
# from direct phi calibration
#pp2=  0.7778
#pp3= -0.0228
# from refined calibration
#pp2=  0.7801 # 0.7778
#pp3= -0.0205 #-0.0228
# from mitegen 50 um tip after chaning Y offset
pp2=  0.766138
pp3= -0.018995
# alignment offset 1 -0.06658
# alignment Y offset 2 0.1386

kappa_direction_optimize = False
phi_direction_optimize = False
kappa_position_optimize = True
phi_position_optimize = True

kp2_optimize = True
pp2_optimize = True
pp1_optimize = False

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


def position_error(parameters, observations, along_axis, C=0.0005):

    kappa_direction, kappa_position, phi_direction, phi_position = get_kdkppdpp(
        parameters, along_axis
    )

    kappa_axis = get_axis(kappa_direction, kappa_position)
    phi_axis = get_axis(phi_direction, phi_position)

    starts_obs = observations[:,-5:]
    kappas_obs = observations[:, 0]
    phis_obs = observations[:, 1]
    xyz_obs = observations[:, [2, 3, 4]]

    xyz_model = np.array(
        [
            get_position(start, kappa, phi, kappa_axis, phi_axis)
            for start, kappa, phi in zip(starts_obs, kappas_obs, phis_obs)
        ]
    )

    # error = np.sum(np.linalg.norm(xyz_model - xyz_obs, axis=1), axis=0) / len(xyz_obs)
    error = np.linalg.norm(xyz_model - xyz_obs, axis=1)  # + 0.0005 * np.linalg.norm(kappa_position) + 0.0005*np.linalg.norm(phi_position)
    error = np.hstack(
        [
            error,
            C * np.linalg.norm(phi_position),
            C * np.linalg.norm(kappa_position),
        ]
    )
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


def show_all(mkc, figsize=(12, 9)):
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
    method="nelder",
    parameters=[],
    along_axis="",
    angle="",
    ay_index=0,
    cx_index=1,
    cy_index=2,
    figsize=(12, 9),
    vertical_start=0.975,
    vertical_step=0.025,
    parameter_horizontal_position=0.01,
    error_horizontal_position=0.5,
):

    print("name_pattern", os.path.basename(name_pattern))
    pylab.figure(figsize=figsize)
    pylab.title(f"calibration: {os.path.basename(name_pattern)}, optimization method: {method}")

    pylab.plot(observations[:, ay_index], "bo", label="ay experiment")
    pylab.plot(observations[:, cx_index], "ro", label="cx experiment")
    pylab.plot(observations[:, cy_index], "go", label="cy experiment")

    pylab.plot(model[:, ay_index], "co", label="ay model")
    pylab.plot(model[:, cx_index], "mo", label="cx model")
    pylab.plot(model[:, cy_index], "ko", label="cy model")

    if parameters not in [None, []]:
        ax = pylab.gca()
        for k, (an, a) in enumerate(zip(["kappa_direction", "kappa_position", "phi_direction", "phi_position"], get_kdkppdpp(parameters))):
            print(k, an, a)
            pars = [round(item, 4) for item in a]

            ax.text(parameter_horizontal_position, vertical_start - k*vertical_step, f"{an}: {pars}", transform=ax.transAxes)

    error = np.abs(observations - model)
    exper = round(np.mean(np.linalg.norm(error, axis=1)), 4)
    meder = [round(item, 4) for item in np.median(error, axis=0)]
    meaer = [round(item, 4) for item in np.mean(error, axis=0)]
    #https://fr.overleaf.com/learn/latex/Bold%2C_italics_and_underlining#Bold_text
    ax.text(error_horizontal_position, vertical_start, r"\textbf{errors in mm}", transform=ax.transAxes)
    for k, (et, er) in enumerate(zip(["3d error", "median", "mean"], [exper, meder, meaer])):
        ax.text(error_horizontal_position, vertical_start - (k+1)*vertical_step, f"{et}: {er}", transform=ax.transAxes)
    
    pylab.legend(loc=3)

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
    method="nelder",
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
            mkc_work, initial_parameters, fr, er, name_pattern=name_pattern, along_axis=along_axis, angle=angle, method=method,
        )

    try:
        report_fit_and_error(fr, er, unique)
    except:
        import traceback
        traceback.print_exc()
        
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
    ka_start_index=-5,
    phi_start_index=-4,
    ay_start_index=-3,
    cx_start_index=-2,
    cy_start_index=-1,
    kd=kappa_direction,
    kp=kappa_position,
    pd=phi_direction,
    pp=phi_position,
    plot=True,
    library="lmfit",
    method="nelder",
    kappa_direction_optimize=kappa_direction_optimize,
    kappa_position_optimize=kappa_position_optimize,
    phi_direction_optimize=phi_direction_optimize,
    phi_position_optimize=phi_position_optimize,
    kp2_optimize=kp2_optimize,
    pp2_optimize=pp2_optimize,
    pp1_optimize=pp1_optimize,
):

    observations = mkc[:, [ka_index, ph_index, ay_index, cx_index, cy_index, ka_start_index, phi_start_index, ay_start_index, cx_start_index, cy_start_index]]
    xyz_obs = mkc[:, [ay_index, cx_index, cy_index]]
    start_obs = mkc[:, [ka_start_index, phi_start_index, ay_start_index, cx_start_index, cy_start_index]]
    ka_obs = mkc[:, ka_index]
    ph_obs = mkc[:, ph_index]

    if library == "lmfit" and lmfit is not None:
        if along_axis == "phi":
            kappa_direction_optimize = True
            kappa_position_optimize = True
            phi_direction_optimize = False
            phi_position_optimize = False

        elif along_axis == "kappa":
            kappa_direction_optimize = False
            kappa_position_optimize = False
            phi_direction_optimize = True
            phi_position_optimize = True

        initial_parameters = lmfit.Parameters()
        initial_parameters.add_many(
            ("kd1", kd[0], kappa_direction_optimize, -1.0, 1.0, None, None),
            ("kd2", kd[1], kappa_direction_optimize, -1.0, 1.0, None, None),
            ("kd3", kd[2], kappa_direction_optimize, -1.0, 1.0, None, None),
            ("kp1", kp[0], kappa_position_optimize, None, None, None, None),
            ("kp2", kp[1], kappa_position_optimize and kp2_optimize, None, None),
            ("kp3", kp[2], kappa_position_optimize, None, None, None, None),
            ("pd1", pd[0], phi_direction_optimize, -1.0, 1.0, None, None),
            ("pd2", pd[1], phi_direction_optimize, -1.0, 1.0, None, None),
            ("pd3", pd[2], phi_direction_optimize, -1.0, 1.0, None, None),
            ("pp1", pp[0], phi_position_optimize and pp1_optimize, None, None, None, None),
            ("pp2", pp[1], phi_position_optimize and pp2_optimize, None, None, None, None),
            ("pp3", pp[2], phi_position_optimize, None, None, None, None),
        )
        fit = lmfit.minimize(
            position_error,
            initial_parameters,
            args=(observations, along_axis),
            method=method,
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
            args=(observations, along_axis),
            method=method,
        )
        parameters = fit.x

    kappa_direction, kappa_position, phi_direction, phi_position = get_kdkppdpp(
        parameters, along_axis=along_axis
    )

    kappa_axis = get_axis(kappa_direction, kappa_position)
    phi_axis = get_axis(phi_direction, phi_position)

    xyz_model = np.array(
        [
            get_position(start, kappa, phi, kappa_axis, phi_axis)
            for start, kappa, phi in zip(start_obs, ka_obs, ph_obs)
        ]
    )

    fr.append(get_kdkppdpp(parameters))
    error = xyz_model - xyz_obs
    _er = np.mean(np.abs(error), axis=0)
    er.append(_er)

    plot_observations_and_model(
        xyz_obs,
        xyz_model,
        parameters=parameters,
        method=method,
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
        results = get_pickled_file(fname)
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
            r = get_pickled_file(rr)
            vector = get_vector_from_position(r["result_position"], keys=keys)
            result_vectors.append(vector)
        results = np.array(result_vectors)
        np.save(rfname, results) 
    return results


def get_only_the_most_recent_of_given_kappa_and_phi(list_of_parameters_pickles):
    
    only_recent = {}
    for p in list_of_parameters_pickles:
        pr = get_pickled_file(p)
        rr = get_pickled_file(p.replace("parameters", "clicks"))
        timestamp = pr["timestamp"]
        position = rr["reference_position"]
        kappa_phi = position["Kappa"], position["Phi"]
        if kappa_phi not in only_recent or timestamp > only_recent[kappa_phi][-1]:
            only_recent[kappa_phi] = [p, timestamp]
    
    results = [only_recent[kappa_phi][0] for kappa_phi in only_recent]
    return results
    
    
def clean_manual_results(directory, pattern="*_parameters.pickle", keys=["Kappa", "Phi", "AlignmentZ", "AlignmentY", "CentringX", "CentringY"], last=True):
    
    ps = glob.glob(os.path.join(directory, pattern))
    if last:
        ps = get_only_the_most_recent_of_given_kappa_and_phi(ps)
    
    results = []
    for p in ps:
        rr = get_pickled_file(p.replace("parameters", "clicks"))
        resp = rr["result_position"]
        results.append(
            get_vector_from_position(rr["result_position"], keys=keys)
        )
    
    results = np.array(results)
    
    np.save(os.path.join(directory, "results_cleaned.npy"), results)
    
def clean_entries(
    entries,
    sort=True,
    extend=True,
    start_index=0,
    ka_index=ka_index,
    ph_index=ph_index,
    ay_index=ay_index,
    cx_index=cx_index,
    cy_index=cy_index,
):
    entries = list(entries[np.apply_along_axis(np.any, 1, np.isnan(entries)) == False])
    entries.sort(key=lambda x: (x[ka_index], x[ph_index]))
    entries = [list(entry) for entry in entries]
    if extend:
        print('entries[start_index]', np.array(entries[start_index]))
        start=entries[start_index]
        start_position = [start[i] for i in [ka_index, ph_index, ay_index, cx_index, cy_index]]
        print('start_position', np.array(start_position))
        print('entries[0]', np.array(entries[0]))
        entries = [entry + start_position for entry in entries]

    entries = np.array(entries)
    return entries

def get_dataset(list_of_parameters_pickles, keys=["Kappa", "Phi", "AlignmentZ", "AlignmentY", "CentringX", "CentringY"]):
    """
    dataset:
    reference_position [kappa, phi, az, ay, cx, cy], result_position [kappa, phi, az, ay, cx, cy], center [v, h], calibration [cv, ch], number_of_clicks, omegas [o1, o2, ..., oN], vertical_clicks [v1, v2, ..., vN], horizontal_clicks [h1, h2, ..., hN]
    
    """
    
    dataset = []
    for p in list_of_parameters_pickles:
        pr = get_pickled_file(p)
        rr = get_pickled_file(p.replace("parameters", "clicks"))
        reference_position = get_vector_from_position(rr["reference_position"], keys=keys)
        result_position = get_vector_from_position(rr["result_position"], keys=keys[2:])
        center = np.array([pr["beam_position_vertical"], pr["beam_position_horizontal"]])
        calibration = pr["calibration"]
        vertical_clicks = np.array(pr["vertical_clicks"])
        horizontal_clicks = np.array(pr["horizontal_clicks"])
        omegas = np.array(pr["omega_clicks"])
        number_of_clicks = len(omegas)
        assert len(vertical_clicks) == number_of_clicks and len(horizontal_clicks) == number_of_clicks
        item = np.hstack([reference_position, result_position, center, calibration, [number_of_clicks], omegas, vertical_clicks, horizontal_clicks])
        dataset.append(item)
        
    dataset = np.array(dataset)
    
    return dataset
        
def main(kd=kappa_direction, kp=kappa_position, pd=phi_direction, pp=phi_position):

    import argparse
    import random

    parser = argparse.ArgumentParser(
        #https://stackoverflow.com/questions/12151306/argparse-way-to-include-default-values-in-help
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

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
    parser.add_argument('-m', '--method', default="nelder", type=str, help="minimize method")
    parser.add_argument('-n', '--name_pattern', default="calibration", type=str, help="name pattern")
    args = parser.parse_args()

    print(args)

    results = args.results.split(" ")
    print("results", results)
    mkc = np.array([])
    for r in results:
        if os.path.isdir(r):
            _mkc = get_raw_results(r, pattern=args.pattern)
        elif r.endswith(".npy"):
            _mkc = load_results(r)
        else:
            sys.exit("the results argument is not a directory nor a .npy, please check")
        _mkc = clean_entries(_mkc)
        mkc = np.vstack([mkc, _mkc]) if mkc.size > 0 else _mkc

    print("mkc.shape", mkc.shape)

    if len(results) > 1:
        name_pattern = args.name_pattern
    else:
        name_pattern = args.results.replace(".npy", "")

    print("name_pattern", name_pattern)
    explore(mkc, name_pattern, along_axis=args.along_axis, method=args.method)

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
def explore_kappa_and_phi(
    kappas=[0, 45, 90, 135],
    phis=[0, 90, 180, 225, 315, 360],
    skip_zero=True,
):
    for p in phis:
        for k in kappas:
            print(f"Setting Kappa to {k} and Phi to {p}")
            if k == 0 and p == 0 and skip_zero:
                print(f"Kappa {k} and Phi {p} already determined, moving on ...")
                continue
            g.set_kappa_phi_position(k, p, simple=False)
            d = "No"
            while d not in ["", "Yes", "yes", "y", "Y"]:
                d = input("May I continue? [Yes/no] ")
                
def misalign(motor_names=["CentringX", "CentringY"], scale=0.002):
    position = g.get_aligned_position(motor_names=motor_names)
    for mn in motor_names:
        position[mn] += (np.random.random() -1 ) * scale
    g.set_position(position)
    
if __name__ == "__main__":
    main()
