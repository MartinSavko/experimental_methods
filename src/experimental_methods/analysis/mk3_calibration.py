#!/usr/bin/env python

import numpy as np
import pickle
import itertools
import pylab
import os

from scipy.optimize import minimize

kappa_direction = np.array([-0.9135,  0.    ,  0.4067])
phi_direction = np.array([0., 0., -1])
    
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


def get_shift(kappa_axis, phi_axis, x, kappa2, phi2):
    tk = kappa_axis["position"]
    tp = phi_axis["position"]

    kappa1, phi1 = x[:2]
    x = x[2:]
    
    #Rk1 = get_rotation_matrix(kappa_axis, -kappa1)
    #Rk2 = get_rotation_matrix(kappa_axis, kappa2)
    ##Rk = get_rotation_matrix(kappa_axis, kappa2 - kappa1)
    #Rp1 = get_rotation_matrix(phi_axis, -phi1)
    #Rp2 = get_rotation_matrix(phi_axis, phi2)
    
    #a = np.dot(Rk1, (x - tk))
    #b = np.dot(Rk2, a)
    #c = tk + b
    
    #d = np.dot(Rp1, (x - tp))
    #e = np.dot(Rp2, d)
    #f = tp + e
    
    #k1 = tk + np.dot(Rk1, (x - tk))
    #kp1 = tp + np.dot(Rp1, (k1 - tp))
    #k2 = tk + np.dot(Rk2, (kp1 - tk))
    #kp2 = tp + np.dot(Rp2, (k2 - tp))
    
    #shift = f
    #Rk2 = get_rotation_matrix(kappa_axis, kappa2)
    #Rk1 = get_rotation_matrix(kappa_axis, -kappa1)
    #Rp = get_rotation_matrix(phi_axis, phi2 - phi1)

    #a = tk - np.dot(Rk1, (tk - x))
    #b = tp - np.dot(Rp, (tp - a))

    #shift = tk - np.dot(Rk2, (tk - b))

    
    Rk1 = get_rotation_matrix(kappa_axis, -kappa1)
    Rp1 = get_rotation_matrix(phi_axis, -phi1)
    Rk2 = get_rotation_matrix(kappa_axis, kappa2)
    Rp2 = get_rotation_matrix(phi_axis, phi2)

    x = tk + np.dot(Rk1, (x-tk))
    x = -tp + np.dot(Rp1, (tp-x))
    x = -tp + np.dot(Rp2, (tp-x))
    x = tk + np.dot(Rk2, (x-tk))
    shift = x

    #Rk = get_rotation_matrix(kappa_axis, kappa2-kappa1)
    #Rp = get_rotation_matrix(phi_axis, phi2-phi1)
    
    #a = tk + np.dot(Rk, (x-tk))
    #b = tp + np.dot(Rp, (a-tp))
    #shift = b
    #shift = tk + np.dot(Rk2, (b - tk))

    return shift


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


def shift_error(parameters, x0, kappa, phi, observation):
    # kappa_direction = np.array(list(parameters[: 2]) + [-0.913545])
    # kappa_position = parameters[2: 5]
    # phi_direction = parameters[5: 8]
    # phi_position = parameters[8:]

    kappa_position = parameters[:3]
    phi_position = parameters[3:]
    #kappa_direction = parameters[:3]  # [0.29636375,  0.29377944, -0.913545]
    #kappa_position = parameters[3:6]
    #phi_direction = parameters[6:9]  # [0, 0, -1]
    #phi_position = parameters[9:] #parameters[9:]
    
    #kappa_direction /= np.linalg.norm(kappa_direction)
    #phi_direction /= np.linalg.norm(phi_direction)

    kappa_axis = get_axis(kappa_direction, kappa_position)
    phi_axis = get_axis(-phi_direction, phi_position)

    #ka_start, ph_start = x0[:2]
    #x0 = x0[2:]
    
    model = np.array(
        [
            get_shift(kappa_axis, phi_axis, x0, k, p)
            for k, p in zip(kappa, phi)
        ]
    )

    #error = np.sum(np.linalg.norm((model - observation) ** 2)) / len(observation)
    #error = np.mean(np.linalg.norm((model - observation)))
    error = np.sum(np.sum(( model - observation ) ** 2, axis=1), axis=0)
    
    return error

ka_index = 0
ph_index = 1
az_index = 2
ay_index = 3
cx_index = 4
cy_index = 5

initial_parameters = [
    #-0.9165,  0.0021,  0.4005, 
    #-0.9135,  0.    ,  0.4067, # thoretical value for [180-24, 90, 90-24]
    -0.594,  0.070,  0.436, # kappa_position
    #1.0411,  1.6131, -0.9217,
    #0., 0., -1, # theoretical value for [90, 90, 180]
    #0, 0, 1,
    -0.363, -0.106, -0.174, # phi_position
    ]

def main():
    import optparse
    import random

    parser = optparse.OptionParser()

    parser.add_option("-r", "--results", default="MK3/mkc.pickle", type=str)

    options, args = parser.parse_args()


    mkc = np.load(options.results)
    print('mkc.shape', mkc.shape)
    _mkc = mkc[np.apply_along_axis(np.any, 1, np.isnan(mkc))==False]
    print('mkc.shape without nan', mkc.shape)
    #mkc = pickle.load(open(options.results, "rb"))
    #mkc = list(mkc)
    print(f"phis {set(mkc[:, ph_index])}")
    phi_positions = []
    kappa_positions = []
    #for ph in list(range(0, 361, 45)):
        #mkc_work = _mkc[_mkc[:, ph_index] == float(ph)]
    for ka in list(range(0, 241, 15)):
        mkc_work = _mkc[_mkc[:, ka_index] == float(ka)]
    #mkc_work = _mkc
        #x0 = mkc[0, [3, 4, 1]]
        
        kappas = mkc_work[:, ka_index]
        phis = mkc_work[:, ph_index]
        observation = mkc_work[:, [ay_index, cx_index, cy_index]]
        print("mkc[:10]")
        print(mkc[:10])
        x0 = mkc_work[0, [ka_index, ph_index, ay_index, cx_index, cy_index]]

        # initial_parameters = [random.random() for k in range(12)]
        b=0.025
        fit = minimize(
            shift_error, initial_parameters, args=(x0, kappas, phis, observation), bounds=[(p-b, p+b) for p in initial_parameters], method="Nelder-Mead",
        )

        print(fit)
        parameters = fit.x
        print("fit results")
        print(list(parameters))

        kappa_position = parameters[:3]
        #kappa_position = parameters[3:6]
        #phi_direction = parameters[6:9]
        #phi_direction = np.array([0, 1, 0])
        phi_position = parameters[3:] #[9:]
        print("kappa_direction=%s" % str(list(kappa_direction)))
        print("kappa_position=%s" % str(list(kappa_position)))
        print("phi_direction=%s" % str(list(phi_direction)))
        print("phi_position=%s" % str(list(phi_position)))
        
        kappa_positions.append(kappa_position)
        phi_positions.append(phi_position)
        
        kappa_axis = get_axis(kappa_direction, kappa_position)
        phi_axis = get_axis(phi_direction, phi_position)

        shifts = np.array(
            [
                get_shift(kappa_axis, phi_axis, x0, kappa, phi)
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

        pylab.figure(figsize=(16, 9))
        #pylab.title(f"Phi is {ph}")
        pylab.title(os.path.basename(options.results.replace(".pickle", "")))
        pylab.plot(shifts[:, 0], "-", label="ay model")
        pylab.plot(shifts[:, 1], "-", label="cx model")
        pylab.plot(shifts[:, 2], "-", label="cy model")

        pylab.plot(mkc_work[:, ay_index], "o", label="ay experiment")
        pylab.plot(mkc_work[:, cx_index], "o", label="cx experiment")
        pylab.plot(mkc_work[:, cy_index], "o", label="cy experiment")
        
        pylab.legend()
        pylab.savefig(f'{options.results.replace(".pickle", ".png").replace(".npy", ".png").replace(".png", "_phi_{ph}.png")}')
        #pylab.show()

    print("kappa_positions", np.median(kappa_positions, axis=0))
    print(np.array(kappa_positions))
    
    pylab.figure()
    pylab.title("kappa_positions")
    pylab.plot(kappa_positions)
    
    print("phi_positions", np.median(phi_positions, axis=0))
    pylab.figure()
    pylab.plot(phi_positions)
    pylab.title("phi_positions")
    print(np.array(phi_positions))
    pylab.show()
    
if __name__ == "__main__":
    main()
