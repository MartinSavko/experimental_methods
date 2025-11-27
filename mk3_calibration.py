#!/usr/bin/env python

import numpy as np
import pickle
import itertools
import pylab
import os
from scipy.optimize import minimize

kappa_direction = [-0.91,  0.2865, 0.1193]
kappa_position = [0.448,  -0.1045, -0.0552]
phi_direction = [1., 0., 0.]
phi_position = [0., 0., 0.]

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
    #dn = direction/np.linalg.norm(direction)
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
        kappa_axis, phi_axis, kappa, phi, 0.5 * (t1 + t2), new_kappa, new_phi
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
    
    kappa_direction, kappa_position, phi_direction, phi_position = get_kdkppdpp(parameters, along_axis)
    
    kappa_axis = get_axis(kappa_direction, kappa_position)
    phi_axis = get_axis(phi_direction, phi_position)
    
    kappas_obs = observations[:, 0]
    phis_obs = observations[:, 1]
    xyz_obs = observations[:, [2, 3, 4]]
    
    xyz_model = np.array(
        [
            get_position(kappa_axis, phi_axis, position_start, kappa, phi)
            for kappa, phi in zip(kappas_obs, phis_obs)
        ]
    )

    #error = np.sum(np.linalg.norm((model - observation) ** 2)) / len(observation)
    #error = np.mean(np.linalg.norm((model - observation)))
    #error = np.sum(np.sum(( model - observation ) ** 2, axis=1), axis=0)
    error = np.sum((xyz_model-xyz_obs)**2)
    return error


def get_kdkppdpp(parameters, along_axis=None, kd=kappa_direction, kp=kappa_position, pd=phi_direction, pp=phi_position):
    if along_axis is None:
        kappa_direction, kappa_position, phi_direction, phi_position = [parameters[k:k+3] for k in range(len(parameters)//3)]
    elif along_axis == "kappa":
        phi_direction, phi_position = [parameters[k:k+3] for k in range(len(parameters)//3)]
        kappa_direction, kappa_position = kd, kp
        #phi_direction = pd.copy()
    elif along_axis == "phi":
        kappa_direction, kappa_position = [parameters[k:k+3] for k in range(len(parameters)//3)]
        phi_direction, phi_position = pd, pp
    return kappa_direction, kappa_position, phi_direction, phi_position

def clean_entries(entries, sort=True):
    entries = list(entries[np.apply_along_axis(np.any, 1, np.isnan(entries))==False])
    entries.sort(key=lambda x: (x[0], x[1],))
    entries = np.array(entries)
    return entries

def show_all(mkc, figsize=(16, 9)):
    pylab.figure(figsize=figsize)
    pylab.plot(mkc[:, az_index], 'b-', label="az")
    pylab.plot(mkc[:, ay_index], 'b-', label="ay")
    pylab.plot(mkc[:, cx_index], 'b-', label="cx")
    pylab.plot(mkc[:, cy_index], 'b-', label="cy")
    pylab.show()
    
def plot_along(observations, model, along_axis, angle, name_pattern="mkc", ay_index=0, cx_index=1, cy_index=2, figsize=(16, 9)):
    
    pylab.figure(figsize=figsize)
    pylab.title(f"{along_axis.capitalize()} is {angle}")

    pylab.plot(observations[:, ay_index], "bo", label="ay experiment")
    pylab.plot(observations[:, cx_index], "ro", label="cx experiment")
    pylab.plot(observations[:, cy_index], "go", label="cy experiment")
    
    pylab.plot(model[:, ay_index], "b-", label="ay model")
    pylab.plot(model[:, cx_index], "r-", label="cx model")
    pylab.plot(model[:, cy_index], "g-", label="cy model")
    
    pylab.legend()
    pylab.savefig(f'{name_pattern}_{along_axis}_{angle}.png')
        
    
def explore_along_axis(mkc, name_pattern, figsize=(16, 9), axis_order=[ka_index, ph_index, ay_index, cx_index, cy_index], along_axis="kappa", kd=kappa_direction, kp=kappa_position, pd=phi_direction, pp=phi_position ):
    if along_axis == "kappa":
        unique = list(set(mkc[:, ka_index]))
        initial_parameters = pd + pp
    elif along_axis == "phi":
        unique = list(set(mkc[:, ph_index]))
        initial_parameters = kd + kp
    
    unique.sort()
    print("along_axis", along_axis)
    print("initial_parameters", initial_parameters)
    
    er = []
    fr = []
    for angle in unique:
        if along_axis == "kappa":
            mkc_work = mkc[mkc[:, ka_index] == angle]
        elif along_axis == "phi":
            mkc_work = mkc[mkc[:, ph_index] == angle]
            
        observations = mkc_work[:, axis_order]
        position_start = mkc_work[0, axis_order]
        
        xyz_obs = observations[:, [2, 3, 4]]
        ka_obs = observations[:, 0]
        ph_obs = observations[:, 1]
        fit = minimize(
            position_error, initial_parameters, args=(position_start, observations, along_axis), method="Nelder-Mead",
        )
        parameters = fit.x
        fr.append(parameters)
        
        kappa_direction, kappa_position, phi_direction, phi_position = get_kdkppdpp(parameters, along_axis=along_axis)
        
        kappa_axis = get_axis(kappa_direction, kappa_position)
        phi_axis = get_axis(phi_direction, phi_position)

        xyz_model = np.array(
            [
                get_position(kappa_axis, phi_axis, position_start, kappa, phi)
                for kappa, phi in zip(ka_obs, ph_obs)
            ]
        )

        error = xyz_model - xyz_obs
        _er = np.mean(np.abs(error), axis=0)
        er.append(_er)

        plot_along(xyz_obs, xyz_model, along_axis, angle)
    
    fr = np.array(fr)
    fr = np.round(fr, 4)
    print("fit results:")
    for a, r, e in zip(unique, fr, er):
        print(f"{a}: {np.round(r,4)} {np.round(e,4)}")
    er = np.array(er)
    er = np.round(er, 4)
    #print("errors")
    #print(er)
    print("errors stats:")
    print("median =", np.round(np.median(er, axis=0), 4))
    print("mean =", np.round(np.mean(er, axis=0), 4))
    print("std =", np.round(np.std(er, axis=0), 4))
    
    print("parameters stats:")
    print("median =", np.round(np.median(fr, axis=0), 3))
    print("mean =", np.round(np.mean(fr, axis=0), 3))
    print("std =", np.round(np.std(fr, axis=0), 3))
    
    pylab.show()

def get_position(kappa_axis, phi_axis, position_start, kappa_end, phi_end):
    kappa_position = kappa_axis["position"]
    phi_position = phi_axis["position"]

    kappa_start = position_start[0]
    phi_start = position_start[1]
    xyz_start = position_start[2:]
    
    Rk1 = get_rotation_matrix(kappa_axis, -kappa_start)
    Rp = get_rotation_matrix(phi_axis, phi_end - phi_start)
    Rk2 = get_rotation_matrix(kappa_axis, kappa_end)
    
    position = kappa_position + np.dot(Rk1, (xyz_start - kappa_position))
    position = phi_position + np.dot(Rp, (position - phi_position))
    position = kappa_position + np.dot(Rk2, (position - kappa_position))
    return position
    
def load_results(fname):
    if fname.endswith(".npy"):
        results = np.load(fname)
    elif fname.endswith(".pickle"):
        results = pickle.load(open(fname, "rb"))
    else:
        print("results format not recognized (not .npy nor .pickle), please check.")
    return results

def main(
    kd=kappa_direction,
    kp=kappa_position,
    pd=phi_direction,
    pp=phi_position,
):
    
    import argparse
    import random

    parser = argparse.ArgumentParser()

    #parser.add_option("-r", "--results", default="MK3/mkc.pickle", type=str)
    parser.add_argument("-r", "--results", default="./examples/minikappa_calibration/2025-07-25_zoom_5.npy", type=str)
    
    args = parser.parse_args()

    mkc = load_results(args.results)
    mkc = clean_entries(mkc)
    
    name_pattern = args.results.replace(".npy", "")
    explore_along_axis(mkc, name_pattern, along_axis="phi")
    
    # 0  : [0.0686, 0.1674, 0.0075, 0.0003, 0.4132, -0.9048] #[0.3468, -0.5826, 1.1433, -0.1746, 0.5361, -0.8553]
    # 225: [-0.094, -0.8087, 2.1235, 0.0622, 0.4187, -0.9116]
    # 135: [-0.1181, -0.521, 2.3723, 0.0638, 0.3022, -0.958]
    # 360: [0.3428, -0.3523, 0.6635, -0.2551, 0.589, -0.834]
    # 45 : [0.6437, -0.4393, 1.6068, -0.2708, 0.3573, -0.9166]
    # 270: [-0.0411, -1.164, 2.583, 0.0364, 0.4577, -0.8932]
    # 180: [0.3892, -0.5775, 1.4617, -0.1523, 0.4344, -0.8679]
    # 90 : [0.2678, -0.7335, 3.2127, -0.0523, 0.2886, -0.961]
    # 315: [0.0722, 0.1673, 0.0032, 0.0043, 0.4289, -0.9147] 
    
    #45 [ 0.7944  0.5303 -0.8619  0.4097  0.5467 -0.7087]
    #90 [-0.0062  0.3777 -0.9276  0.0732  0.1124  0.1778]
    #initial_parameters = [-0.0062,  0.3777, -0.9276,  0.0732,  0.1124,  0.1778]
    #initial_parameters = [random.random() for k in range(6)]
    #initial_parameters = [0.34, -0.58, 0, 0, 0.4344, -0.9135]
    
if __name__ == "__main__":
    main()
