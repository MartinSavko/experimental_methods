#!/usr/bin/env python

import numpy as np
import pickle
import itertools
import pylab
import os

from scipy.optimize import minimize

kappa_direction = [0., 0.4067, -0.9086]
kappa_position = [0.0721,  0.079,  0.192]
phi_direction = [0., 0., 1.]
phi_position = [-0.363, -0.106, -0.174]

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

    shift = get_shift(
        kappa_axis, phi_axis, kappa, phi, 0.5 * (t1 + t2), new_kappa, new_phi
    )

    align_vector = new_kappa, new_phi, shift

    return align_vector

    # test 1
    #Rk1 = get_rotation_matrix(kappa_axis, -kappa1)
    #Rk2 = get_rotation_matrix(kappa_axis, kappa2)
    ###Rk = get_rotation_matrix(kappa_axis, kappa2 - kappa1)
    #Rp1 = get_rotation_matrix(phi_axis, -phi1)
    #Rp2 = get_rotation_matrix(phi_axis, phi2)
    
    #a = np.dot(Rk1, (x - tk))
    #b = np.dot(Rk2, a)
    #c = tk + b
    
    #d = np.dot(Rp1, (x - tp))
    #e = np.dot(Rp2, d)
    #f = tp + e
    #shift = f
    
    # test 2
    #k1 = tk + np.dot(Rk1, (x - tk))
    #kp1 = tp + np.dot(Rp1, (k1 - tp))
    #k2 = tk + np.dot(Rk2, (kp1 - tk))
    #kp2 = tp + np.dot(Rp2, (k2 - tp))
    #shift = kp2
    
 # test 3
    #Rk1 = get_rotation_matrix(kappa_axis, -kappa1)
    #Rp1 = get_rotation_matrix(phi_axis, -phi1)
    #Rk2 = get_rotation_matrix(kappa_axis, kappa2)
    #Rp2 = get_rotation_matrix(phi_axis, phi2)

    #x = tk + np.dot(Rk1, (x-tk))
    #x = -tp + np.dot(Rp1, (tp-x))
    #x = -tp + np.dot(Rp2, (tp-x))
    #x = tk + np.dot(Rk2, (x-tk))
    
    #shift = x

    # test 4
    #Rk = get_rotation_matrix(kappa_axis, kappa2-kappa1)
    #Rp = get_rotation_matrix(phi_axis, phi2-phi1)
    
    #a = tk + np.dot(Rk, (x-tk))
    #b = tp + np.dot(Rp, (a-tp))
    #shift = b
    #shift = tk + np.dot(Rk2, (b - tk))

def get_shift(kappa_axis, phi_axis, k0, p0, x0, k2, p2):
    tk = kappa_axis["position"]
    tp = phi_axis["position"]

    Rk2 = get_rotation_matrix(kappa_axis, k2)
    Rk1 = get_rotation_matrix(kappa_axis, -k0)
    Rp = get_rotation_matrix(phi_axis, p2 - p0)
    
    x = tk - np.dot(Rk1, (tk - x0))
    #x = tp - np.dot(Rp, (tp - x))
    x = tk - np.dot(Rk2, (tk - x))
    return x

def shift_error(parameters, k0, p0, x0, kappa, phi, observation):
    
    #kappa_direction, kappa_position, phi_direction, phi_position = get_kdkppdpp(parameters)
    #kappa_position, phi_position = get_kdkppdpp(parameters)
    kappa_direction, kappa_position = get_kdkppdpp(parameters)
    kappa_axis = get_axis(kappa_direction, kappa_position)
    phi_axis = get_axis(phi_direction, phi_position)
    
    model = np.array(
        [
            get_shift(kappa_axis, phi_axis, k0, p0, x0, k, p)
            for k, p in zip(kappa, phi)
        ]
    )

    #error = np.sum(np.linalg.norm((model - observation) ** 2)) / len(observation)
    #error = np.mean(np.linalg.norm((model - observation)))
    #error = np.sum(np.sum(( model - observation ) ** 2, axis=1), axis=0)
    error = np.sum((model-observation)**2)
    return error

ka_index = 0
ph_index = 1
az_index = 2
ay_index = 3
cx_index = 4
cy_index = 5

initial_parameters = [
    #-0.9165,  0.0021,  0.4005, 
    #-0.9135,  0.    ,  0.4067, # kappa_direction, thoretical value for [180-24, 90, 90-24]
    -0.594,  0.070,  0.436, # kappa_position
    #1.0411,  1.6131, -0.9217,
    #0., 0., -1, # theoretical value for [90, 90, 180]
    #0, 0, -1, # phi_direction
    -0.363, -0.106, -0.174, # phi_position
    ]

def get_kdkppdpp(parameters):
    #kappa_direction = parameters[:3]
    #kappa_position = parameters[3:6]
    #phi_direction = parameters[6:9]
    #phi_position = parameters[9:]
    kappa_direction = parameters[:3]
    kappa_position = parameters[-3:]
    #phi_position = parameters[-3:]
    #return kappa_position, phi_position
    #return kappa_direction, kappa_position, phi_direction, phi_position
    return kappa_direction, kappa_position

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

    if args.results.endswith(".npy"):
        mkc = np.load(args.results)
    elif args.results.endswith(".pickle"):
        mkc = pickle.load(open(args.results, "rb"))
    else:
        print("results format not recognized (not .npy nor .pickle), please check.")
    print('mkc.shape', mkc.shape)
    _mkc = list(mkc[np.apply_along_axis(np.any, 1, np.isnan(mkc))==False])
    _mkc.sort(key=lambda x: (x[0], x[1],))
    _mkc = np.array(_mkc)
    print('_mkc.shape without nan', _mkc.shape)
    phiss = list(set(_mkc[:, ph_index]))
    print(f"phis {phiss}")
    #phi_positions = []
    #kappa_positions = []
    #for ph in list(range(0, 361, 45)):
        #mkc_work = _mkc[_mkc[:, ph_index] == float(ph)]
    #for ka in list(range(0, 241, 15)):
        #mkc_work = _mkc[_mkc[:, ka_index] == float(ka)]
    fr = []
    for sphi in phiss:
        #sphi = phiss[0]
        mkc_work = _mkc[_mkc[:, ph_index] == sphi]
        #x0 = mkc[0, [3, 4, 1]]
        print("sphi", sphi)
        kappas = mkc_work[:, ka_index]
        phis = mkc_work[:, ph_index]
        observation = mkc_work[:, [cx_index, cy_index, ay_index]]
        #print("mkc[:10]")
        #print(mkc_work[:10])
        
        x0 = mkc_work[0, [ka_index, ph_index, cx_index, cy_index, ay_index]]
        print("x0", x0)
        k0, p0 = x0[:2]
        x0 = x0[2:]
        print("kappa0, phi0, x0", k0, p0, x0)
        
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
        
        #print(f"kd: {kd}, kp: {kp}")
        initial_parameters = kd + kp
        print("initial_parameters:", initial_parameters)
        b=0.025
        fit = minimize(
            #shift_error, initial_parameters, args=(x0, kappas, phis, observation), bounds=[(p-b, p+b) for p in initial_parameters], method="Nelder-Mead",
            shift_error, initial_parameters, args=(k0, p0, x0, kappas, phis, observation), method="Nelder-Mead",
        )

        #print(fit)
        parameters = fit.x
        parameters = np.round(parameters, 4)
        print("fit results:")
        print(parameters)
        fr.append(parameters)
        
        #kappa_direction, kappa_position, phi_direction, phi_position = get_kdkppdpp(parameters)
        #kappa_position, phi_position = get_kdkppdpp(parameters)
        kappa_direction, kappa_position = get_kdkppdpp(parameters)
        print("kappa_direction=", kappa_direction)
        print("kappa_position=", kappa_position)
        print("phi_direction=", phi_direction)
        print("phi_position=", phi_position)
        
        #kappa_positions.append(kappa_position)
        #phi_positions.append(phi_position)
        
        kappa_axis = get_axis(kappa_direction, kappa_position)
        phi_axis = get_axis(phi_direction, phi_position)

        shifts = np.array(
            [
                get_shift(kappa_axis, phi_axis, k0, p0, x0, kappa, phi)
                for kappa, phi in zip(kappas, phis)
            ]
        )

        #kappas_model = np.arange(0, 240, 16)
        #phis_model = np.linspace(0, 360, 45)
        #shifts_model = np.array(
            #[
                #get_shift(kappa_axis, phi_axis, x0, kappa, phi)
                #for kappa, phi in zip(kappas_model, phis_model)
            #]
        #)

        print("model errors")
        print("ay, cx, cy")
        print(np.mean(np.abs(shifts - observation), axis=0))
        print("standard deviations")
        print(np.std(shifts - observation, axis=0))
        print()
        #pylab.figure(figsize=(16, 9))
        #pylab.title(f"Phi is {sphi}")
        ##pylab.title(os.path.basename(args.results.replace(".pickle", "")))
        #pylab.plot(shifts[:, 2], "b-", label="ay model")
        #pylab.plot(shifts[:, 0], "r-", label="cx model")
        #pylab.plot(shifts[:, 1], "g-", label="cy model")

        #pylab.plot(mkc_work[:, ay_index], "bo", label="ay experiment")
        #pylab.plot(mkc_work[:, cx_index], "ro", label="cx experiment")
        #pylab.plot(mkc_work[:, cy_index], "go", label="cy experiment")
        
        #pylab.legend()
        #pylab.savefig(f'{args.results.replace(".pickle", ".png").replace(".npy", ".png").replace(".png", f"_phi_{sphi}.png")}')
        #pylab.savefig(f'{args.results.replace(".pickle", ".png").replace(".npy", ".png")}')
    
    fr = np.array(fr)
    fr = np.round(fr, 4)
    print("fit results:")
    print(fr)
    print("parameters stats:")
    print("median:", np.round(np.median(fr, axis=0), 4))
    print("mean:", np.round(np.mean(fr, axis=0), 4))
    print("std:", np.round(np.std(fr, axis=0), 4))
    pylab.figure(figsize=(16,9))
    pylab.title("parameters distribution")
    for k, p in enumerate(fr):
        pylab.plot(p, 'o', label=f"{k}")
    pylab.legend()
    pylab.savefig(f'{args.results.replace(".pickle", ".png").replace(".npy", ".png").replace(".png", f"_kappa_parameters_distribution.png")}')
    print()
    
    for k, p in enumerate(fr):
        pylab.figure(figsize=(16,9))
        pylab.title(f"parameter {k} distribution")
        pylab.hist(p, bins=25)
        pylab.savefig(f'{args.results.replace(".pickle", ".png").replace(".npy", ".png").replace(".png", f"_{k}_parameter_histogram.png")}')
    
    pylab.show()

    #print("kappa_positions", np.median(kappa_positions, axis=0))
    #print(np.array(kappa_positions))
    
    #pylab.figure()
    #pylab.title("kappa_positions")
    #pylab.plot(kappa_positions)
    
    #print("phi_positions", np.median(phi_positions, axis=0))
    #pylab.figure()
    #pylab.plot(phi_positions)
    #pylab.title("phi_positions")
    #print(np.array(phi_positions))
    #pylab.show()
    
if __name__ == "__main__":
    main()
