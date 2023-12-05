#!/usr/bin/env python

import optparse
import pickle
import pylab
import numpy as np
import matplotlib.pyplot as plt
from numpy import exp, sqrt
from scipy.constants import elementary_charge, electron_mass, speed_of_light, pi, Planck
from scipy.special import yn, jv, jn
from scipy.optimize import minimize
import glob
import pandas as pd

import re

import seaborn as sns
sns.set(color_codes=True)

from matplotlib import rc
rc('font', **{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

from plot_scans import get_gap, get_slit_opening, get_ring_current


def residual(x, data_matrix):
    k0, k1, k2 = x
    data = pickle.load(open(data_matrix))
    theory = []
    for gap, n in data[:, 0:2]:
        theory.append(undulator_peak_energy(gap, n, k0=k0, k1=k1, k2=k2))
    theory = np.array(theory)
    experiment = data[:, 2]
    return 1. / (2 * len(theory)) * np.sum((experiment - theory) ** 2)

def residual2(x, gaps, bs):
    k0, k1, k2 = x
    theory = undulator_magnetic_field(gaps, k0, k1, k2)
    return 1. / 2 * (len(theory)) * np.sum((bs - theory) ** 2)
    
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
        detune_parameter = 1 - 1 / (n * N)
    else:
        detune_parameter = 1
    return lambda_peak * detune_parameter


def undulator_peak_energy(gap, n, k0=2.72142696, k1=-3.83321034, k2=0.6322541, N=80, detune=False):
    if detune:
        detune_parameter = 1 - 1 / (n * N)
    else:
        detune_parameter = 1
    return undulator_harmonic_energy(gap, n, k0=k0, k1=k1, k2=k2) * detune_parameter


def undulator_magnetic_field(gap, k0=2.72142696, k1=-3.83321034, k2=0.6322541, period_length=24.):
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

    
def undulator_harmonic_energy(gap, n, k0=2.72142696, k1=-3.83321034, k2=0.6322541, period_length=24.0, electron_energy=2.75):
    k = 1e-3 * elementary_charge / (electron_mass * speed_of_light * 2 * pi)
    B = undulator_magnetic_field(
        gap, k0=k0, k1=k1, k2=k2, period_length=period_length)
    return 1000 * 9.5 * n * electron_energy ** 2 / (period_length * (1 + undulator_strength(B) ** 2 / 2))


def undulator_peak_intensity(gap, n, k0=2.72142696, k1=-3.83321034, k2=0.6322541, period_length=24., N=80):
    k = 1e-3 * elementary_charge / (electron_mass * speed_of_light * 2 * pi)
    B = undulator_magnetic_field(gap, k0=k0, k1=k1, k2=k2)
    K = undulator_strength(B)
    return angular_flux_density(K, n)


def get_flux_vs_energy(filename):
    r = np.array(pickle.load(open(filename)))
    energy = r[:, 0]
    flux = r[:, 2]
    xbpm1 = r[:, 4]
    return energy, flux, xbpm1


def get_experimental_peaks(theory, exp_energy, exp_flux, gap, data):
    peak_positions = []
    #if gap < 8.1:
        #lim = 75
    #elif gap > 8.1 and gap < 8.5:
        #lim = 105
    #elif gap > 8.3 and gap < 10.3:
        #lim = 255
    #elif gap > 10.5 and gap < 11.9:
        #lim = 155
    #else:
        #lim = 205
    lim = 75
    total_max = exp_flux.max()
    for harmonic, n in theory[::3]:

        if harmonic > 4000. and harmonic < 20001:  # .max():
            # print('harmonic')
            # print(harmonic)
            maximum_flux = exp_flux[(exp_energy > harmonic - lim) & (exp_energy < harmonic + lim)].max()
            # print('maximum_flux', maximum_flux)
            me = exp_energy[(exp_energy > harmonic - lim) & (exp_energy < harmonic + lim)]
            fe = exp_flux[(exp_energy > harmonic - lim) & (exp_energy < harmonic + lim)]
            # print('me', me)
            # print('fe', fe)
            maximum_energy = me[np.where(np.abs(maximum_flux - fe) < 1)]
            # print('maximum_energy', maximum_energy)

            energies = me  # exp_energy[np.where(np.abs(exp_energy - maximum_energy) < 75)]
            weights = fe  # exp_flux[np.where(np.abs(exp_energy - maximum_energy) < 75)]
            position = np.average(energies, axis=0, weights=weights)
            if maximum_flux > 0.02 * total_max:
                peak_positions.append([position, harmonic])
                data.append([gap, n, position, maximum_flux])

    return peak_positions


def fit(data_matrix, method='powell'):
    x0 = 2.73096921, -3.84082989,  0.60382274
    result = minimize(residual, x0, args=(data_matrix,), method=method)
    print(result)
    return result

xkcd_colors_that_i_like = ["pale purple", "coral", "moss green", "windows blue", "amber", "greyish", "faded green", "dusty purple", "crimson", "custard", "orangeish", "dusk blue", "ugly purple", "carmine", "faded blue", "dark aquamarine", "cool grey", "faded blue"]

from sklearn.linear_model import LinearRegression


def plot(data_matrix):
    
    colormap = plt.cm.gist_ncar

    data = pickle.load(open(data_matrix))
    harmonics = list(set(map(int, data[:, 1])))
    
    harmonics.sort()
    #k0, k1, k2 = 2.71504025, -3.80924779,  0.55774775
    #k0, k1, k2 = 2.73081755, -3.84032394,  0.60509254
    k0, k1, k2 =  2.73096921, -3.84082989,  0.60382274
    #k0, k1, k2 = 2.7289875 , -3.8387636 ,  0.60923686
    pylab.figure(figsize=(16, 9))
    #plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, len(harmonics))])
    #plt.gca().set_color_cycle(sns.color_palette("cubehelix", len(harmonics)))
    #plt.gca().set_color_cycle(sns.diverging_palette(255, 133, l=60, n=len(harmonics), center="dark"))
    #plt.gca().set_color_cycle(sns.color_palette("hls", len(harmonics)))
    #plt.gca().set_color_cycle(sns.hls_palette(len(harmonics), l=.33, s=.9))
    #plt.gca().set_color_cycle(sns.color_palette("Set1", len(harmonics)))
    plt.gca().set_color_cycle(sns.xkcd_palette(xkcd_colors_that_i_like))
    print('sns.xkcd_palette(xkcd_colors_that_i_like)')
    print(sns.xkcd_palette(xkcd_colors_that_i_like))
    intercepts = []
    coeffs = []
    ns = []
    max_gap = 7.8
    for n in harmonics:
        # gap = []
        # energy = []
        selection = list(data[data[:, 1] == n])
        selection.sort(key=lambda x: x[2])
        selection = np.array(selection)
        gaps = selection[:, 0]
        if gaps.max() > max_gap:
            max_gap = gaps.max()
        energies = selection[:, 2]
        ens = energies
        X = np.vstack([ens/1.e3, gaps]).T
        np.savetxt('GAP_ENERGY_HARMONIC%s.txt' % n, X, fmt='%6.3f', delimiter=' ', header='%d\n%d\nENERGY  GAP' % X.shape[::-1], comments='')
        
        modeled_energies = undulator_peak_energy(gaps, n, k0=k0, k1=k1, k2=k2)
        X_model = np.vstack([modeled_energies, gaps]).T
        np.savetxt('fit_GAP_ENERGY_HARMONIC%s.txt' % n, X_model, fmt='%6.3f', delimiter=' ', header='%d\n%d\nENERGY  GAP' % X_model.shape[::-1], comments='')
        
        #lm = LinearRegression()
        #lm.fit(np.array([ens, ens**2]).T, gaps)
        #ens_fit = np.linspace(ens[0], ens[-1], 50)
        #gap_fit = lm.predict(np.array([ens_fit, ens_fit**2]).T)
        #print('%s score' % n, lm.score(np.array([ens, ens**2]).T, gaps))
        #print('%s intercept' % n, lm.intercept_ )
        #print('%s coeff' % n, lm.coef_)
        #intercepts.append(lm.intercept_)
        #coeffs.append(lm.coef_* 1e3)
        #ns.append(n)
        #X_fit = X = np.vstack([ens_fit/1.e3, gap_fit]).T
        #np.savetxt('fit_GAP_ENERGY_HARMONIC%s.txt' % n, X_fit, fmt='%6.3f', delimiter=' ', header='%d\n%d\nENERGY  GAP' % X_fit.shape[::-1], comments='')
                                  
        #Bs = undulator_magnetic_field(gaps, n, 2.72898056, -3.83864548,  0.60969562)
        #Ks = undulator_strength(Bs)
        
        #print('n', n)
        #print('gaps')
        #print(gaps)
        #print('energies')
        #print(energies)
        pylab.plot(energies, gaps, 'o-', color=sns.xkcd_rgb[xkcd_colors_that_i_like[n-min(harmonics)]], label='%d' % n)
        pylab.plot(modeled_energies, gaps, 'v-', color=sns.xkcd_rgb[xkcd_colors_that_i_like[n-min(harmonics)]])
        #pylab.plot(ens_fit, gap_fit,'d--', color=sns.xkcd_rgb[xkcd_colors_that_i_like[n-min(harmonics)]])
    pylab.title('Proxima 2A U24 undulator harmonic peak positions as function of gap and energy', fontsize=22)
    pylab.xlabel('energy [eV]', fontsize=18)
    pylab.ylabel('gap [mm]', fontsize=18)
    pylab.ylim([7., max_gap + 0.5])
    pylab.grid(True)
    pylab.legend(loc='best', fontsize=16)
    ax = pylab.gca()
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(16)
    pylab.savefig('U24_harmonic_peak_positions_gap_vs_energy.png')
   
    #pylab.figure(figsize=(16, 9))
    #pylab.plot(ns[:-1], coeffs[:-1], 'o-', label='coefs')
    #pylab.plot(ns[:-1], intercepts[:-1], 'd-', label='intercepts')
    #pylab.legend(loc='best', fontsize=16)
    #pylab.xlabel('harmonic number', fontsize=18)
    #pylab.ylabel('regression parameters', fontsize=18)
    #pylab.title('Linear regression parameters (gap vs. energy) as function of harmonic number', fontsize=22)
    #pylab.savefig('linear_regression_paramters_as_function_of_harmonic_number.png')
    pylab.figure(figsize=(16, 9))
    #plt.gca().set_color_cycle(sns.diverging_palette(255, 133, l=60, n=len(harmonics), center="dark"))
    #plt.gca().set_color_cycle(sns.color_palette("cubehelix", len(harmonics)))
    #plt.gca().set_color_cycle(sns.color_palette("hls", len(harmonics)))
    #plt.gca().set_color_cycle(sns.hls_palette(len(harmonics), l=.33, s=.9))
    #plt.gca().set_color_cycle(sns.color_palette("Set1", n_colors=len(harmonics)))
    plt.gca().set_color_cycle(sns.xkcd_palette(xkcd_colors_that_i_like))
    for n in harmonics:
        selection = list(data[data[:, 1] == n])
        selection.sort(key=lambda x: x[2])
        selection = np.array(selection)
        energies = selection[:, 2]
        fluxes = selection[:, 3]
        #print('n', n)
        #print('energies')
        #print(energies)
        #print('fluxes')
        #print(fluxes)
        pylab.plot(energies, fluxes, 'o-', color=sns.xkcd_rgb[xkcd_colors_that_i_like[n-min(harmonics)]], label='%d' % n)
    pylab.title(
        'Proxima 2A U24 undulator tuning curves', fontsize=22)
    pylab.xlabel('energy [eV]', fontsize=18)
    pylab.ylabel('flux [ph/s]', fontsize=18)
    # pylab.ylim([7., 12.5])
    pylab.grid(True)
    pylab.legend(loc='best', fontsize=16)
    ax = pylab.gca()
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(16)
    pylab.savefig('U24_tuning_curves_all_harmonics.png')
    
    pylab.figure(figsize=(16, 9))
    #plt.gca().set_color_cycle(sns.diverging_palette(255, 133, l=60, n=len(harmonics), center="dark"))
    #plt.gca().set_color_cycle(sns.color_palette("cubehelix", len(harmonics)))
    #plt.gca().set_color_cycle(sns.color_palette("hls", len(harmonics)))
    odd_harmonics = [n for n in harmonics if n % 2 == 1]
    #plt.gca().set_color_cycle(sns.hls_palette(len(odd_harmonics), l=.33, s=.9))
    #plt.gca().set_color_cycle(sns.color_palette("Set1", n_colors=len(odd_harmonics)))
    plt.gca().set_color_cycle(sns.xkcd_palette(xkcd_colors_that_i_like))
    for n in odd_harmonics:
        selection = list(data[data[:, 1] == n])
        selection.sort(key=lambda x: x[2])
        selection = np.array(selection)
        energies = selection[:, 2]
        fluxes = selection[:, 3]
        #print('n', n)
        #print('energies')
        #print(energies)
        #print('fluxes')
        #print(fluxes)
        pylab.plot(energies, fluxes, 'o-', color=sns.xkcd_rgb[xkcd_colors_that_i_like[n-min(harmonics)]], label='%d' % n)
    pylab.title(
        'Proxima 2A U24 undulator tuning curves, odd harmonics', fontsize=22)
    pylab.xlabel('energy [eV]', fontsize=18)
    pylab.ylabel('flux [ph/s]', fontsize=18)
    # pylab.ylim([7., 12.5])
    pylab.grid(True)
    pylab.legend(loc='best', fontsize=16)
    ax = pylab.gca()
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(16)
    pylab.savefig('U24_tuning_curves_odd_harmonics.png')
    
    pylab.figure(figsize=(16, 9))
    #plt.gca().set_color_cycle(sns.diverging_palette(255, 133, l=60, n=len(harmonics), center="dark"))
    #plt.gca().set_color_cycle(sns.color_palette("cubehelix", len(harmonics)))
    #plt.gca().set_color_cycle(sns.color_palette("hls", len(harmonics)))
    even_harmonics = [n for n in harmonics if n % 2 == 0]
    #plt.gca().set_color_cycle(sns.hls_palette(len(even_harmonics), l=.33, s=.9))
    #plt.gca().set_color_cycle(sns.color_palette("Set1", n_colors=len(even_harmonics)))
    plt.gca().set_color_cycle(sns.xkcd_palette(xkcd_colors_that_i_like))
    for n in even_harmonics:
        selection = list(data[data[:, 1] == n])
        selection.sort(key=lambda x: x[2])
        selection = np.array(selection)
        energies = selection[:, 2]
        fluxes = selection[:, 3]
        #print('n', n)
        #print('energies')
        #print(energies)
        #print('fluxes')
        #print(fluxes)
        pylab.plot(energies, fluxes, 'o-', color=sns.xkcd_rgb[xkcd_colors_that_i_like[n-min(harmonics)]], label='%d' % n)
    pylab.title(
        'Proxima 2A U24 undulator tuning curves, even harmonics', fontsize=22)
    pylab.xlabel('energy [eV]', fontsize=18)
    pylab.ylabel('flux [ph/s]', fontsize=18)
    # pylab.ylim([7., 12.5])
    pylab.grid(True)
    pylab.legend(loc='best', fontsize=16)
    ax = pylab.gca()
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(16)
    pylab.savefig('U24_tuning_curves_even_harmonics.png')
    
    pylab.figure(figsize=(16, 9))
    #plt.gca().set_color_cycle(sns.color_palette("cubehelix", len(harmonics)))
    #plt.gca().set_color_cycle(sns.diverging_palette(255, 133, l=60, n=len(harmonics), center="dark"))
    #plt.gca().set_color_cycle(sns.color_palette("hls", len(harmonics)))
    #plt.gca().set_color_cycle(sns.hls_palette(len(harmonics), l=.33, s=.9))
    #plt.gca().set_color_cycle(sns.color_palette("Set1", n_colors=len(harmonics)))
    plt.gca().set_color_cycle(sns.xkcd_palette(xkcd_colors_that_i_like))
    for n in harmonics:
        selection = list(data[data[:, 1] == n])
        selection.sort(key=lambda x: x[2])
        selection = np.array(selection)
        energies = selection[:, 2]
        fluxes = selection[:, 3]
        gaps = selection[:, 0]
        #print('n', n)
        #print('gaps')
        #print(gaps)
        #print('energies')
        #print(energies)
        #print('fluxes')
        #print(fluxes)
        pylab.plot(gaps, fluxes, 'o-', color=sns.xkcd_rgb[xkcd_colors_that_i_like[n-min(harmonics)]], label='%d' % n)
    pylab.title(
        'Proxima 2A U24 undulator flux vs. gap', fontsize=22)
    pylab.xlabel('gap [mm]', fontsize=18)
    pylab.ylabel('flux [ph/s]', fontsize=18)
    # pylab.ylim([7., 12.5])
    pylab.grid(True)
    pylab.legend(loc='best', fontsize=16)
    ax = pylab.gca()
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(16)
    pylab.savefig('U24_flux_vs_gap_all_harmonics.png')
    
    pylab.figure(figsize=(16, 9))
    #plt.gca().set_color_cycle(sns.color_palette("cubehelix", len(harmonics)))
    #plt.gca().set_color_cycle(sns.diverging_palette(255, 133, l=60, n=len(harmonics), center="dark"))
    #plt.gca().set_color_cycle(sns.color_palette("hls", len(harmonics)))
    #plt.gca().set_color_cycle(sns.hls_palette(len(harmonics), l=.33, s=.9))
    #plt.gca().set_color_cycle(sns.color_palette("Set1", n_colors=len(harmonics)))
    plt.gca().set_color_cycle(sns.xkcd_palette(xkcd_colors_that_i_like))
    for n in harmonics: # [4, 5, 7, 8, 10, 11, 12]:
        selection = list(data[data[:, 1] == n])
        selection.sort(key=lambda x: x[2])
        selection = np.array(selection)
        energies = selection[:, 2]
        fluxes = selection[:, 3]
        gaps = selection[:, 0]
        Bs = undulator_magnetic_field(gaps, k0, k1, k2) #, k0=3.8, k1=-4.47,  k2=1.83)
        Ks = undulator_strength(Bs)
        #print('gaps')
        #print(gaps)
        #print('Bs')
        #print(Bs)
        #print('Ks')
        #print(Ks)
        theoric_fluxes = angular_flux_density(Ks, n, N=80)
        #theoric_fluxes = central_cone_flux(Ks)
        pylab.plot(energies, theoric_fluxes, 'o-', color=sns.xkcd_rgb[xkcd_colors_that_i_like[n-min(harmonics)]], label='%d' % n)
    pylab.title(
        'Proxima 2A U24 theoretic undulator tuning curves', fontsize=22)
    pylab.xlabel('energy [eV]', fontsize=18)
    pylab.ylabel('flux [ph/s]', fontsize=18)
    # pylab.ylim([7., 12.5])
    pylab.grid(True)
    pylab.legend(loc='best', fontsize=16)
    ax = pylab.gca()
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(16)
    pylab.savefig('U24_theoretic_tuning_curves_all_harmonics.png')
    
    pylab.figure(figsize=(16, 9))
    #plt.gca().set_color_cycle(sns.hls_palette(len(data), l=.33, s=.9))
    plt.gca().set_color_cycle(sns.color_palette("Set1", n_colors=len(data)))
    sns.set_color_codes()
    gaps = list(set(data[:, 0]))
    gaps.sort()
    gaps = np.array(gaps)
    #bs1 = undulator_magnetic_field(gaps)
    #bs2 = undulator_magnetic_field(gaps, 3.81815378, -5.53642081,  2.71692885)
    #bs3 = undulator_magnetic_field(gaps, 3.33, -5.47, 1.8)
    bs = []
    gs = []
    ks = []
    for k, result in enumerate(data):
        gap, n, energy, flux = result
        gs.append(gap)
        #energy *= (1 + 2/(n*80))
        K = undulator_strength_from_peak_position(energy/1e3, n)
        ks.append(K)
        B = undulator_magnetic_field_from_K(K)
        if k == len(data)-1:
            pylab.plot(gap, B, 'bo', label='experiment')
        else:
            pylab.plot(gap, B, 'bo')
        #pylab.plot(gap, K, 'ro')
        bs.append(B)
    #x0 = [3.3, -5.47,  1.8]
    #print(residual((3.33, -5.47, 1.8)))
    
    gs = np.array(gs)
    bs = np.array(bs)
    ks = np.array(ks)
    data = list(zip(gs, bs))
    data.sort(key=lambda x: x[0])
    data = np.array(data)
    d = pd.DataFrame()
    #data['order'] = np.arange(len(gs))
    d['gap'] = data[:,0]
    d['B'] = data[:,1] + 0.3
    #print('data')
    #print(data)
    #sns.tsplot(data=d, time='gap', value='B', legend='sns.tsplot') #, time="gap", unit="B", legend='sns.tsplot')
    #ax = sns.tsplot(data=   
    #res = minimize(residual2, x0, args=(gs, bs), method='trust-ncg')
    #print(res)
    # nelder-mead 2.71502275, -3.80920849,  0.55769274, error=0.061607012818753311
    # powell 2.72898056, -3.83864548, 0.60969562, error= 0.061607014855154206
    # L-BFGS-B  2.71503557, -3.80922939,  0.55771415, error=0.06160701240774303
    # TNC 3.19156135, -4.60888915,  1.52785286, error = 0.11672641054304266
    # COBYLA 3.60484283, -5.23766814,  2.32703523, error = 0.2112670192219884
    # SLSQP 2.71504711, -3.80925158,  0.55774223, error= 0.061607012467943152

    #2.71504025, -3.80924779,  0.55774775
    bs3 = undulator_magnetic_field(gaps, k0, k1, k2)
    

    ##pylab.plot(gaps, bs2, 'r-')
    ##pylab.plot(gaps, bs3, 'c^-')
    pylab.xlabel('gap [mm]', fontsize=18)
    h = pylab.ylabel('B [T]', fontsize=18, labelpad=35)
    h.set_rotation(0)
    
    pylab.title('Proxima 2A U24 undulator peak magnetic field and strength as function of gap', fontsize=22)
    pylab.grid(True)
    
    ax = pylab.gca()
    ax.text(0.83, 0.8, '\# data points = %d' % len(data), color='b', fontsize=18, transform=ax.transAxes)
    ax.text(0.05, 0.15, 'model function: $B(x) = k_{0} \\exp(k_{1} x  + k_{2} x^{2}); x = \\frac{gap}{\lambda_{u}}$', fontsize=20, color='green', transform=ax.transAxes)
    ax.text(0.05, 0.08, 'fit parameters: k_{0} = %6.3f, k_{1} = %6.3f, k_{2} = %6.3f' % (k0, k1, k2), fontsize=20, color='green', transform=ax.transAxes)
    
    #ax.text(0.83, 0.45, '$K = \\frac{eB\lambda_{u}}{m_{e}c2\\pi}$', fontsize=20, transform=ax.transAxes) 
    #ax.text(0.35, 0.65, '$x = \\frac{gap}{\lambda_{u}}$', fontsize=16, transform=ax.transAxes)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(16)
    
    ax_k = ax.twinx()
    ax_k.plot(gs, ks, 'bo')
    h = ax_k.set_ylabel('$K = \\frac{eB\lambda_{u}}{m_{e}c2\\pi}$', fontsize=18, labelpad=40)
    h.set_rotation(0)
    ax_k.grid(False)
    for label in (ax_k.get_yticklabels()):
        label.set_fontsize(16)
    
    ax.plot(gaps, bs3, 'gv-', label='fit')
    ax.legend(loc='best', fontsize=18)
    #pylab.xlim([7.5, 12.3])
    pylab.savefig('B_and_K_vs_gap.png')
    #pylab.figure()
    #ax = sns.tsplot(data=d, time='gap', value='B')
    ##ax.legend()
    
    pylab.show()

def main():
    parser = optparse.OptionParser()
    parser.add_option('-d', '--data_matrix', default='data_0.1x0.1mm_450mA.pkl', type=str, help='Data matrix file')
    parser.add_option('-f', '--fit', action='store_true', help='Perform a fit and print out the fitted parameters')
    parser.add_option('-p', '--plot', action='store_true', help='Show the results and generate figures') 
    options, args = parser.parse_args()
    print(options, args)
    if options.fit:
        fit(options.data_matrix)
    else:
        plot(options.data_matrix)


if __name__ == '__main__':
    main()
