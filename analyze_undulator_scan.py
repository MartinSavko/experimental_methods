#!/usr/bin/env python

import pickle
import pylab
import numpy as np
import peakutils
import os
from numpy import sqrt, exp
from scipy.constants import eV, h, c, angstrom, kilo, degree, elementary_charge as q, elementary_charge, electron_mass, speed_of_light, pi, Planck
from scipy.optimize import leastsq, minimize
from scipy.special import yn, jv, jn
from scipy.spatial import distance_matrix
from scipy.signal import medfilt
import glob

def transmission(params, e):
    t = 0
    for k, p in enumerate(params):
        t += p*e**(k)
    return t   
   
def transmission_Si_12650(thickness, attenuation_length=267.310):
    return np.exp(-thickness/attenuation_length)
    
def residual(params, energy, data):
    model = transmission(params, energy)
    return abs(model - data)

def get_params(datafile='/927bis/ccd/gitRepos/flux/xray9507_Si_125um.dat'): #xray5184.dat
    data = open(datafile).read().split('\n')[2:-1]
    dat = [map(float, item.split()) for item in data]
    da = np.array(dat)
    eys, transmissions = da[:,0], da[:,1]
    results = leastsq(residual, [0]*10, args=(eys, transmissions))
    params = results[0]     
    return params
    
def responsivity(ey, params):
    return 0.98 * (1-transmission(params, ey))/3.65
    
def get_flux(current, ey, params):
    current /= amplification
    return current / (responsivity(ey, params) * q * ey)

def get_theta_from_wavelength(wavelength, units_theta=degree, units_energy=kilo*eV, units_wavelength=angstrom, d=3.1347507142511746):
    theta = np.arcsin((angstrom/units_wavelength)*wavelength/(2*d)) / units_theta
    return theta
    
def get_wavelength_from_theta(theta, units_theta=degree, units_energy=kilo*eV, units_wavelength=angstrom, d=3.1347507142511746):
    wavelength = 2*d*np.sin(units_theta*theta)
    return wavelength
    
def get_energy_from_theta(theta, units_theta=degree, units_energy=kilo*eV, units_wavelength=angstrom):
    wavelength = get_wavelength_from_theta(theta, units_wavelength=units_wavelength, units_theta=units_theta)
    energy = get_energy_from_wavelength(wavelength, units_energy=units_energy, units_wavelength=units_wavelength)
    return energy
    
def get_theta_from_energy(energy, units_theta=degree, units_energy=kilo*eV, units_wavelength=angstrom):
    wavelength = get_wavelength_from_energy(energy, units_energy=units_energy, units_wavelength=units_wavelength)
    theta = get_theta_from_wavelength(wavelength, units_theta=units_theta, units_energy=units_energy, units_wavelength=units_wavelength)
    return theta
    
def get_wavelength_from_energy(energy, units_energy=kilo*eV, units_wavelength=angstrom):
    wavelength = h*c/(units_wavelength*units_energy)/energy
    return wavelength
    
def get_energy_from_wavelength(wavelength, units_energy=kilo*eV, units_wavelength=angstrom):
    energy = h*c/(units_wavelength*units_energy)/wavelength
    return energy
  
def F(K, n):
    # K *= 1.5
    # k = (n * K) / (1. + K ** 2 / 2)
    chi = n / (1. + 0.5 * K ** 2)
    Y = 0.25 * (K ** 2) * chi
    return (chi ** 2) * (K ** 2) * (jv((n + 1) / 2., Y) - jv((n - 1) / 2., Y)) ** 2

def central_cone_flux(K, E=2.75, I=0.5, N=80):
    return 2.86e14 * N * I * (K ** 2) / (1 + K ** 2)


def angular_flux_density(K, n, E=2.75, I=0.5, N=80):
    return 1.74e14 * (N ** 2) * (E ** 2) * I * F(K, n)


def get_lambda_harmonic(lambda_peak, n, N=80, detune=False):
    if detune:
        detune_parameter = 1 - 1. / (n * N)
    else:
        detune_parameter = 1
    return lambda_peak * detune_parameter


def undulator_peak_energy(gap, n, k0=2.72898056, k1=-3.83864548, k2=0.60969562, N=80, detune=False):
    if detune:
        detune_parameter = 1 - 2. / (n * N)
    else:
        detune_parameter = 1
    return undulator_harmonic_energy(gap, n, k0=k0, k1=k1, k2=k2) * detune_parameter


def undulator_magnetic_field(gap, k0=2.72898056, k1=-3.83864548, k2=0.60969562, period_length=24.):
    x = gap / period_length
    return k0 * exp(k1 * x + k2 * x ** 2)


def undulator_strength(B, period_length=24.):
    k = 1e-3 * elementary_charge / (electron_mass * speed_of_light * 2 * pi)
    return k * period_length * B
    # return 0.0934 * period_length * B


def undulator_magnetic_field_from_K(K, period_length=24.):
    k = 1e-3 * elementary_charge / (electron_mass * speed_of_light * 2 * pi)
    return K / ( k * period_length)


def undulator_strength_from_peak_position(peak_energy, n, electron_energy=2.75, period_length=24.0):
    #return 9.5 * n * electron_energy ** 2 / ((1 +  K ** 2 / 2.) * period_length)
    return sqrt(2 * 9.5 * n * electron_energy ** 2 / (period_length * peak_energy) - 2)

    
def undulator_harmonic_energy(gap, n, k0=2.72898056, k1=-3.83864548, k2=0.60969562, period_length=24.0, electron_energy=2.75):
    k = 1e-3 * elementary_charge / (electron_mass * speed_of_light * 2 * pi)
    B = undulator_magnetic_field(
        gap, k0=k0, k1=k1, k2=k2, period_length=period_length)
    return 1000 * 9.5 * n * electron_energy ** 2 / (period_length * (1 + undulator_strength(B) ** 2 / 2))


def undulator_peak_intensity(gap, n, k0=2.72898056, k1=-3.83864548, k2=0.60969562, period_length=24., N=80):
    k = 1e-3 * elementary_charge / (electron_mass * speed_of_light * 2 * pi)
    B = undulator_magnetic_field(gap, k0=k0, k1=k1, k2=k2)
    K = undulator_strength(B)
    return angular_flux_density(K, n)

directory = '/nfs/ruche/proxima2a-spool/2017_Run4/2017-09-05/com-proxima2a/RAW_DATA/Commissioning/undulator/full_beam'
template = 'gap_8.7'
fast_shutter_chronos_uncertainty = 0.1
amplification = 1e4 

params = get_params()

files = glob.glob('%s/*_results.pickle' % directory)

templates = [os.path.basename(filename).replace('_results.pickle', '') for filename in files]
data = []
for template in templates:
    print 'template', template
    
    parameters = pickle.load(open(os.path.join(directory, '%s_parameters.pickle' % template)))
    results = pickle.load(open(os.path.join(directory, '%s_results.pickle' % template)))
    gap = parameters['undulator_gap_encoder_position'] 
    if gap > 23 :
        continue
    #if abs(gap-8.3) > 0.1:
        #continue
    diode = results['calibrated_diode']['observations']
    diode = np.array(diode)

    diode_chronos = diode[:, 0]
    diode_current = diode[:, 1]

    actuator = results['actuator']['observations']
    actuator = np.array(actuator)

    actuator_chronos = actuator[:, 0]
    actuator_position = actuator[:, 1]

    fast_shutter = results['fast_shutter']['observations']
    fast_shutter = np.array(fast_shutter)

    fast_shutter_chronos = fast_shutter[:, 0]
    fast_shutter_state = fast_shutter[:, 1]

    start_end_indices = peakutils.indexes(np.abs(np.gradient(fast_shutter_state)))

    start_chronos, end_chronos = fast_shutter_chronos[start_end_indices]

    #dark_current = np.vstack([diode_current[diode_chronos < start_chronos - fast_shutter_chronos_uncertainty], diode_current[diode_chronos > end_chronos + fast_shutter_chronos_uncertainty]]).mean()
    print diode_current.shape
    print diode_chronos.shape
    dark_current = diode_current[diode_chronos < start_chronos - fast_shutter_chronos_uncertainty].mean()
    diode_current -= dark_current

    actuator_scan_indices = np.logical_and(actuator_chronos > start_chronos + fast_shutter_chronos_uncertainty * 5, actuator_chronos < end_chronos - fast_shutter_chronos_uncertainty * 5)
    actuator_scan_chronos = actuator_chronos[actuator_scan_indices]
    actuator_scan_position = actuator_position[actuator_scan_indices]

    position_chronos_fit = np.polyfit(actuator_scan_chronos, actuator_scan_position, 1)

    position_linear_predictor = np.poly1d(position_chronos_fit)

    diode_scan_indices = np.logical_and(diode_chronos > start_chronos + fast_shutter_chronos_uncertainty * 5, diode_chronos < end_chronos - fast_shutter_chronos_uncertainty * 5)
    diode_scan_chronos = diode_chronos[diode_scan_indices]
    diode_scan_current = diode_current[diode_scan_indices]

    thetas = position_linear_predictor(diode_scan_chronos)
    energies = get_energy_from_theta(thetas, units_energy=eV)

    flux = get_flux(diode_scan_current, energies, params)

    if energies[0] > energies[-1]:
        energies = energies[::-1]
        flux = flux[::-1]

    filtered_flux = medfilt(flux, 5)
    peaks = peakutils.indexes(filtered_flux, min_dist=55, thres=0.012)
    #peaks = peakutils.indexes(flux, min_dist=1, thres=0.02)

    harmonics = np.arange(1, 21)

    theoretic_harmonic_energies = undulator_peak_energy(gap, harmonics, detune=False)
    print 'theory'
    print theoretic_harmonic_energies
    print 'detected peaks'
    print energies[peaks][::-1]

    print 'distance_matrix'
    thr = [(t, 0) for t in theoretic_harmonic_energies]
    ep = [(e, 0) for e in energies[peaks][::-1]]
    fluxes = flux[peaks][::-1] 

    print 'theory'
    print thr
    print 'detected peaks'
    print ep
    dm = distance_matrix(thr ,  ep )
    print dm.shape
    print np.arange(1, 21)
    print dm.argmin(axis=1)
    print dm.min(axis=1)
    minimums = dm.argmin(axis=0)
    print minimums
    print dm.min(axis=1)

    matches = np.where(dm<210)

    print 'ep with criteria'
    ep2 = energies[peaks][::-1]
    ep_matched = ep2[matches[1]]
    print 'harmonics with criteria'
    thr_matched = theoretic_harmonic_energies[matches[0]]
    fluxes_matched = fluxes[matches[1]]

    #peak_half_width = 45.

    #from scipy.ndimage import center_of_mass
    #for l, e in enumerate(ep_matched):
        #print 'starting peak position refinement, l, e', l, e
        #shift = peak_half_width
        #k = 0 
        #while shift >= 5: 
            #k+=1
            #indices = np.logical_and(energies<e+peak_half_width, energies>e-peak_half_width)
            #less_then = np.logical_and(energies<e, energies>e-peak_half_width)
            #more_then = np.logical_and(energies>e, energies<e-peak_half_width)
            #print 'sum(indices) initial', sum(indices)
            #if sum(less_then) > sum(more_then):
                #diff = sum(less_then) - sum(more_then)
                #print 'diff', diff
                #valid = np.where(indices == True)[0]
                #print 'valid', valid
                #indices[valid[:diff]] = False
            #elif sum(less_then) < sum(more_then):
                #diff = -sum(less_then) + sum(more_then)
                #print 'diff', diff
                #valid = np.where(indices == True)[0]
                #print 'valid', valid
                #indices[valid[-diff:]] = False
                
            #print 'sum(indices)', sum(indices)
            #xp = energies[indices]
            #print 'energies', xp
            #fp = flux[indices]
            #print 'fluxes', fp
            #x = np.linspace(xp[0], xp[-1], 101)
            #f = np.interp(x, xp, fp)
            #com = center_of_mass(f)
            #print 'com', com
            #new_e = x[int(round(com[0]))]
            #print new_e
            #shift = e - new_e
            #print 'k, shift', k, shift
            #e = new_e
            #print 'new_e', new_e
        #ep_matched[l] = new_e
        #print

    matched = np.array(zip(matches[0]+1, thr_matched, ep_matched, np.abs(thr_matched - ep_matched), fluxes_matched))
    print matched
        
    #pylab.vlines(undulator_peak_energy(gap, np.arange(1, 21), detune=True), 0, 1.1*flux.max(), color='cyan', label='theoretic harmonic peak positions')
    #pylab.vlines(theoretic_harmonic_energies, 0, 1.1*flux.max(), color='magenta', label='theoretic harmonic positions')
    pylab.figure()
    for k, thr, ep, diff, flx in matched:
        data.append([gap, int(k), ep, flx])
        pylab.annotate(s='%d' % k, xy=(ep, flx), xytext=(ep+ 150, 1.1*flx), arrowprops=dict(arrowstyle='->', connectionstyle="arc3"))
        
    pylab.plot(energies, flux, label='flux')
    #pylab.plot(energies, filtered_flux, label='filtered_flux')
    #pylab.plot(energies[peaks], flux[peaks], 'rx', mew=2, label='peaks')
    pylab.plot(matched[:,2], matched[:,-1], 'rx', mew=2, label='harmonics')
    pylab.xlabel('energy [eV]')
    pylab.ylabel('flux [ph/s]')
    pylab.legend()
    pylab.title('Energy scan, %s mm, undulator U24 Proxima 2A, SOLEIL' % template.replace('_', ' '))
            
f = open('data_2017-09-06.pickle', 'w')
pickle.dump(np.array(data), f)
f.close()

pylab.show()


#data.append([gap, n, position, maximum_flux])
