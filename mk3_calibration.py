#!/usr/bin/env python

import numpy as np
import pickle
import itertools
import pylab
import os

from scipy.optimize import minimize

def get_rotation_matrix(axis, angle):
    rads = np.radians(angle)
    cosa = np.cos(rads)
    sina = np.sin(rads)
    I = np.diag([1]*3)
    rotation_matrix = I * cosa + axis['mT'] * (1-cosa) + axis['mC'] * sina
    return rotation_matrix

def get_axis(direction, position):
    axis = {}
    d = np.array(direction)
    p = np.array(position)
    axis['direction'] = d
    axis['position'] = p
    axis['mT'] = get_mT(direction)
    axis['mC'] = get_mC(direction)
    return axis

def get_mC(direction):
    mC = np.array([[ 0.0, -direction[2], direction[1]],
                   [ direction[2], 0.0, -direction[0]],
                   [-direction[1], direction[0], 0.0]])

    return mC

def get_mT(direction):
    mT = np.outer(direction, direction)
    
    return mT
    
def get_shift(kappa_axis, phi_axis, kappa1, phi1, x, kappa2, phi2):
    tk = kappa_axis['position']
    tp = phi_axis['position']
    
    Rk2 = get_rotation_matrix(kappa_axis, kappa2)
    Rk1 = get_rotation_matrix(kappa_axis, -kappa1)
    Rp = get_rotation_matrix(phi_axis, phi2-phi1)
    
    a = tk - np.dot(Rk1, (tk-x))
    b = tp - np.dot(Rp, (tp-a))
    
    shift = tk - np.dot(Rk2, (tk-b))
    
    return shift

def get_align_vector(t1, t2, kappa, phi, kappa_axis, phi_axis, align_direction):
    t1 = np.array(t1)
    t2 = np.array(t2)
    x = t1 - t2
    Rk = get_rotation_matrix(kappa_axis, -kappa)
    Rp = get_rotation_matrix(phi_axis, -phi)
    x = np.dot(Rp, np.dot(Rk, x))/np.linalg.norm(x)
    c = np.dot(phi_axis['direction'], x)
    if c < 0.:
        c = -c
        x = -x
    cos2a = pow(np.dot(kappa_axis[direction], align_direction), 2)
    
    d = (c - cos2a)/(1 - cos2a)
    
    if abs(d) > 1.:
        new_kappa = 180.
    else:
        new_kappa = np.degrees(np.arccos(d))
    
    Rk = get_rotation_matrix(kappa_axis, new_kappa)
    pp = np.dot(Rk, phi_axis['direction'])
    xp = np.dot(Rk, x)
    d1 = align_direction - c*pp
    d2 = xp - c*pp
    
    new_phi = np.degrees(np.arccos(np.dot(d1, d2)/np.linalg.norm(d1)/np.linalg.norm(d2)))
    
    newaxis = {}
    newaxis['mT'] = get_mT(pp)
    newaxis['mC'] = get_mC(pp)
    
    Rp = get_rotation_matrix(newaxis, new_phi)
    d = np.abs(np.dot(align_direction, np.dot(Rp, xp)))
    check = np.abs(np.dot(align_direction, np.dot(xp, Rp)))
                   
    if check > d:
        new_phi = -new_phi
        
    shift = get_shift(kappa_axis, phi_axis, kappa, phi, 0.5*(t1 + t2), new_kappa, new_phi)
    
    align_vector = new_kappa, new_phi, shift
    
    return align_vector
    
def shift_error(parameters, x0, kappa, phi, observation):
    #kappa_direction = np.array(list(parameters[: 2]) + [-0.913545])
    #kappa_position = parameters[2: 5]
    #phi_direction = parameters[5: 8]
    #phi_position = parameters[8:]
    
    kappa_direction = parameters[:3] #[0.29636375,  0.29377944, -0.913545]
    kappa_position = parameters[3: 6]
    phi_direction =  parameters[6: 9] #[0, 0, -1]
    phi_position = parameters[9:]
    
    kappa_axis = get_axis(kappa_direction, kappa_position)
    phi_axis = get_axis(phi_direction, phi_position)
    
    model = np.array([get_shift(kappa_axis, phi_axis, 0., 0., x0, k, p) for k, p in zip(kappa, phi)])
    
    error = np.sum((model-observation)**2)
    
    return error
    
        
def main():

    import optparse
    import random
    
    parser = optparse.OptionParser()
    
    parser.add_option('-r', '--results', default='MK3/mkc.pickle', type=str)
    
    options, args = parser.parse_args()
    
    #kappa_axis = get_axis([0.282543,  0.2925819, -0.913545], [-0.499986, -0.2591313,  0.484796])
    kappa_axis = get_axis([0.28579375, 0.29825935, -0.91069766], [0.06054262, -0.17344149, -0.39538791])
    #phi_axis = get_axis([0, 0, -1], [-0.316151, -0.039378, 0.4179955])
    phi_axis = get_axis([0, 0, -1], [0.22284277, -0.03217082, -2.03321131])
    align_direction = np.array([0, 0, -1])
    
    # x = [cx, cy, ay]
    #print('get_shift(kappa, phi, 0, 0., np.array([0.1, 0.5, 0.3]), 15., 25.)')
    #print(get_shift(kappa, phi, 0, 0., np.array([0.1, 0.5, 0.3]), 15., 25.))
    
    
    mkc = pickle.load(open(options.results, 'rb'))
    mkc = list(mkc)
    mkc.sort(key=lambda x: (x[-2], x[-1]))
    mkc = np.array(mkc)
    observation = mkc[:, [3, 4, 1]]
    kappas = mkc[:, -2]
    phis = mkc[:, -1]
    
    x0 =  mkc[0, [3, 4, 1]]
     
    #initial_parameters = [-0.30655466, -0.3570731, 0.52893628, -0.0942107, 0.15449601, 0.36023525]
    #initial_parameters = [0.29636375,  0.29377944, -0.499986, -0.2591313,  0.484796, 0, 0, -1, -0.316151, -0.039378, 0.4179955]
    
    initial_parameters = [0.282543,  0.2925819, -0.913545, -0.499986, -0.2591313,  0.484796, 0, 0, -1, -0.316151, -0.039378, 0.4179955]
    initial_parameters = [0.28579375, 0.29825935, -0.91069766,
                          0.06054262, -0.17344149, -0.39538791,
                          #0.04804586, -0.00507483, -1.01343307,
                          0, 0, -1,
                          0.22284277, -0.03217082, -2.03321131]
    #initial_parameters = [random.random() for k in range(12)]
    #initial_parameters = [ 0.31377822,  0.30374482, -0.913545, -0.47170958, -0.66398839,  0.37447831, 0, 0, -1, -0.02065501, -0.15673108,  0.40641754]
    #initial_parameters =[random.random() for k in range(11)] # [-0.30655466, -0.3570731, 0.52893628, -0.0942107, 0.15449601, 0.36023525]
    
    fit = minimize(shift_error, initial_parameters, args=(x0, kappas, phis, observation))
    
    parameters = fit.x
    print('fit results')
    print(list(parameters))
    #parameters = np.array([ 0.29636375,  0.29377944, -0.90992064, -0.30655466, -0.3570731, 0.52893628, 0.03149443, 0.03216924, -0.99469729, -0.01467116, -0.08069945, 0.46818622])
    
    #kappa_direction = np.array(list(parameters[: 2]) + [-0.913545])
    #kappa_position = parameters[2: 5]
    #phi_direction = parameters[5: 8]
    #phi_position = parameters[8:]
    
    kappa_direction = parameters[: 3]
    kappa_position = parameters[3: 6]
    phi_direction = parameters[6: 9]
    phi_position = parameters[9:]
    print('kappa_direction=%s' % str(list(kappa_direction)))
    print('kappa_position=%s' % str(list(kappa_position)))
    print('phi_direction=%s' % str(list(phi_direction)))
    print('phi_position=%s' % str(list(phi_position)))
    kappa_axis = get_axis(kappa_direction, kappa_position)
    phi_axis = get_axis(phi_direction, phi_position)
    
    #kp = list(zip(kappas, phis))
    #kp.sort()
    
    shifts = np.array([get_shift(kappa_axis, phi_axis, 0., 0., x0, kappa, phi) for kappa, phi in zip(kappas, phis)])
    
    kappas_model = np.linspace(0, 240, 49)
    phis_model = np.linspace(0, 360, 73)
    shifts_model = np.array([get_shift(kappa_axis, phi_axis, 0., 0., x0, kappa, phi) for kappa, phi in zip(kappas_model, phis_model)])
    
    print('model errors')
    print('cx, cy, ay')
    print(np.mean(np.abs(shifts - observation), axis=0))
    print('standard deviations')
    print(np.std(shifts - observation, axis=0))
    
    pylab.figure(figsize=(16, 9))
    pylab.plot(shifts[:,0], 'o-', label='cx model')
    pylab.plot(shifts[:,1], 'o-', label='cy model')
    pylab.plot(shifts[:,2], 'o-', label='ay model')
    
    pylab.plot(mkc[:, 3], 'o', label='cx experiment')
    pylab.plot(mkc[:, 4], 'o', label='cy experiment')
    pylab.plot(mkc[:, 1], 'o', label='ay experiment')
    
    pylab.title(os.path.basename(options.results.replace('.pickle', '')))
    
    pylab.legend()
    pylab.savefig(options.results.replace('.pickle', '.png'))
    pylab.show()
    
    
    
    
if __name__ == '__main__':
    main()
