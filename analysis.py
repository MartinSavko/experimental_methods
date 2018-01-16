#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Analysis modules
'''

import gevent
from gevent.monkey import patch_all
patch_all()

import traceback
import logging
import time
import itertools
import os
import pickle
import numpy as np
import pylab
import glob
try:
    import pandas as pd
except ImportError:
    print 'Can not import pandas'

try:
    from analyze_undulator_scan import get_energy_from_theta, get_flux, undulator_peak_energy, undulator_magnetic_field, undulator_strength, angular_flux_density, angular_flux_density, undulator_magnetic_field_from_K, undulator_strength_from_peak_position
except:
    print traceback.print_exc()
    
from scipy.constants import eV, h, c, angstrom, kilo, degree, elementary_charge as q, elementary_charge, electron_mass, speed_of_light, pi, Planck
from scipy.spatial import distance_matrix
from scipy.interpolate import interp1d
from scipy.ndimage import center_of_mass
from scipy.signal import medfilt
try:
    from scipy.optimize import minimize
except ImportError:
    print 'Can not import scipy.optimize.minimize'

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
#from matplotlib import rc
#rc('font', **{'family':'serif','serif':['Palatino']})
#rc('text', usetex=True)

from motor import tango_motor

class scan_analysis:
    
    def __init__(self, parameters_filename, fast_shutter_chronos_uncertainty=0.1, monitor='calibrated_diode', display=False):
        self.parameters_filename = parameters_filename
        self.fast_shutter_chronos_uncertainty = fast_shutter_chronos_uncertainty
        self.monitor = monitor
        self.display = display
        
    def get_parameters(self):
        return pickle.load(open(self.parameters_filename))
    
    def get_results(self):
        return pickle.load(open(self.parameters_filename.replace('_parameters', '_results')))
    
    def save_results(self, results):
        f= open(self.parameters_filename.replace('_parameters', '_results'), 'w')
        pickle.dump(results, f)
        f.close()
        
    def get_fast_shutter_open_close_times(self, fast_shutter_chronos, fast_shutter_state):
        maximas = np.where(np.abs(np.gradient(fast_shutter_state)) == 0.5)[0]
        indices = [maximas[i] for i in range(1, len(maximas)-1) if abs(maximas[i]-maximas[i-1]) == 1 or abs(maximas[i+1] - maximas[i]) == 1]
        open_time, close_time =  fast_shutter_chronos[indices]
        return open_time, close_time
        
    def get_scan_indices(self, observation_chronos, start_chronos, end_chronos, uncertainty):
        indices = np.logical_and(observation_chronos > start_chronos + uncertainty, observation_chronos < end_chronos - uncertainty)
        return indices
        
    def get_position_chronos_predictor(self, actuator_scan_chronos, actuator_scan_position):
        '''assuming constant actuator speed'''
        position_chronos_fit = np.polyfit(actuator_scan_chronos, actuator_scan_position, 1)
        position_chronos_predictor = np.poly1d(position_chronos_fit)
        return position_chronos_predictor
    
    def get_observations(self, results, monitor_name):
        observations = np.array(results[monitor_name]['observations'])
        chronos = observations[:, 0]
        points = observations[:, 1]
        return chronos, points
    
    def from_number_sequence_to_character_sequence(self, number_sequence, separator=';'):
        character_sequence = ''
        number_strings = [str(n) for n in number_sequence]
        return separator.join(number_strings)

    def merge_two_overlapping_character_sequences(self, seq1, seq2, alignment_length=1000, separator=';'):
        start = seq1.index(seq2[:alignment_length])
        nvalues_seq1 = seq2.count(separator) - seq2[start:].count(separator)
        return seq1[:start] + seq2,  nvalues_seq1
        
    def from_character_sequence_to_number_sequence(self, character_sequence, separator=';'):
        return map(float, character_sequence.split(';'))
        
    def merge_two_overlapping_number_sequences(self, r1, r2, alignment_length=1000, separator=';'):
        c1 = self.from_number_sequence_to_character_sequence(r1)
        c2 = self.from_number_sequence_to_character_sequence(r2)
        c, start = self.merge_two_overlapping_character_sequences(c1, c2, alignment_length)
        r = self.from_character_sequence_to_number_sequence(c)
        return r, start
    
    def find_overlap(self, r1, r2, alignment_length=1000, separator=';'):
        c1 = self.from_number_sequence_to_character_sequence(r1)
        c2 = self.from_number_sequence_to_character_sequence(r2)
        start = c1.index(c2[:alignment_length])
        start = c2.count(separator) - c2[start:].count(separator)
        return start
        
class slit_scan_analysis(scan_analysis):
        
    def get_hflux(self, current):
        hflux = abs(current - 0.5)
        hflux -= 0.5*hflux.mean()
        hflux *= -1
        hflux[hflux<0] = 0
        return hflux
    
    def get_min_max(self, current, bins=100):
        histogram = np.histogram(current, bins=bins)
        min_boundary = histogram[1][1]
        max_boundary = histogram[1][-2]
        
        plateau_bas = current[current < min_boundary].mean()
        plateau_haut = current[current > max_boundary].mean()
       
        return plateau_bas, plateau_haut
    
    def get_normalized_current(self, current):
        plateau_bas, plateau_haut = self.get_min_max(current)
        normalized_current = (current - plateau_bas)/(plateau_haut - plateau_bas)
        return normalized_current
    
    def get_x_of_half_max(self, current, positions):
        normalized_current = self.get_normalized_current(current)
        hflux = self.get_hflux(normalized_current)
        return positions[np.argmax(hflux)]
        
    def analyze(self, display=False):
        parameters = self.get_parameters()
        results = self.get_results()
        
        for lame_name in results.keys():
            actuator_chronos, actuator_position = self.get_observations(results[lame_name], 'actuator')
            fast_shutter_chronos, fast_shutter_state = self.get_observations(results[lame_name], 'fast_shutter')
            diode_chronos, diode_current = self.get_observations(results[lame_name], self.monitor)
        
            
            start_chronos, end_chronos = self.get_fast_shutter_open_close_times(fast_shutter_chronos, fast_shutter_state)
            
            dark_current_indices = np.logical_or(diode_chronos < start_chronos - self.fast_shutter_chronos_uncertainty, diode_chronos > end_chronos + self.fast_shutter_chronos_uncertainty)
            dark_current = diode_current[dark_current_indices].mean()
            diode_current -= dark_current
            
            actuator_scan_indices = self.get_scan_indices(actuator_chronos, start_chronos, end_chronos, self.fast_shutter_chronos_uncertainty)
            actuator_scan_chronos = actuator_chronos[actuator_scan_indices]
            actuator_scan_position = actuator_position[actuator_scan_indices]
            
            position_chronos_predictor = self.get_position_chronos_predictor(actuator_scan_chronos, actuator_scan_position)
            
            diode_scan_indices = self.get_scan_indices(diode_chronos, start_chronos, end_chronos, self.fast_shutter_chronos_uncertainty)
            diode_scan_chronos = diode_chronos[diode_scan_indices]
            self.diode_scan_current = diode_current[diode_scan_indices]
            
            self.diode_scan_positions = position_chronos_predictor(diode_scan_chronos)
            
            normalized_current = self.get_normalized_current(self.diode_scan_current)
            
            pylab.figure()
            
            pylab.plot(self.diode_scan_positions, normalized_current)
            hflux = self.get_hflux(normalized_current)
            pylab.plot(self.diode_scan_positions, hflux)
            
            if parameters['slits'] in [1, 2]:
                mid_point = self.get_x_of_half_max(normalized_current, self.diode_scan_positions)
                print lame_name, mid_point
                pylab.vlines(mid_point, 0, 1)
                results[lame_name]['mid_point'] = mid_point
                results[lame_name]['offset'] = mid_point
            else:
                nc_mean = len(normalized_current)/2
                mid_point1 = self.get_x_of_half_max(normalized_current[:nc_mean], self.diode_scan_positions[:nc_mean])
                mid_point2 = self.get_x_of_half_max(normalized_current[nc_mean:], self.diode_scan_positions[nc_mean:])
                print lame_name, mid_point1, mid_point2, 'offset (2+1)/2.', (mid_point1 + mid_point2)/2.
                pylab.vlines([mid_point1, mid_point2], 0, 1)
                results[lame_name]['mid_point1'] = mid_point1
                results[lame_name]['mid_point2'] = mid_point2
                results[lame_name]['offset'] = (mid_point1 + mid_point2)/2.
                
            pylab.xlabel('position [mm]')
            pylab.ylabel('current [mA] *1e-4')
            pylab.title('Edge scan, %s' % lame_name)
            pylab.grid(True)
        
        self.save_results(results)
        if display == True:
            pylab.show()
        
    def conclude(self):
        
        results = self.get_results()
        parameters = self.get_parameters()
        
        for lame_name in results.keys():
            lame = tango_motor(lame_name)
            print lame_name, 'current offset', lame.device.offset
            print lame_name, 'decreasing offset by', results[lame_name]['offset']
            lame.device.offset -= results[lame_name]['offset']
  

class undulator_scan_analysis(scan_analysis):
    
    '''Analyze single energy scan at fixed undulator gap. Find undulator peaks position and intensity and assign the harmonics number.'''
    
    def __init__(self, 
                 parameters_filename, 
                 fast_shutter_chronos_uncertainty=0.1, 
                 monitor='calibrated_diode',
                 maximum_peak_theoretic_error=210,
                 peak_half_width = 50.,
                 display=False):
        
        scan_analysis.__init__(self,
                               parameters_filename,
                               fast_shutter_chronos_uncertainty=fast_shutter_chronos_uncertainty,
                               monitor=monitor,
                               display=display)
        
        self.maximum_peak_theoretic_error = maximum_peak_theoretic_error
        self.peak_half_width = peak_half_width
                               
    def get_peaks(self, energies, flux, width=50):
        try:
            import peakutils
            peaks = peakutils.indexes(flux, min_dist=55, thres=0.012)
        except ImportError:
            from scipy.signal import find_peaks_cwt
            l = len(energies)
            eend = energies[-l/100]
            n = sum(np.logical_and(eend + width/2 > energies, eend-width/2 < energies))
            print 'width of peak in number of points n', n
            peaks = find_peaks_cwt(flux, np.array([n]))
        return peaks
        
    def get_diode_chronos_and_current(self, results, monitor):
        calibrated_diode = results[monitor]['observations']
        if type(calibrated_diode[0][1]) == float:
            diode_chronos, diode_current = self.get_observations(results, monitor)
        else:
            diode_chronos = [calibrated_diode[0][0]]
            diode_current = calibrated_diode[0][1]
            starts = [len(diode_current)-1]
            for k, observation in enumerate(calibrated_diode[1:]):
                start = self.find_overlap(calibrated_diode[k][1], observation[1])
                #print 'start', start
                starts.append(starts[-1] + start)
                diode_current = np.hstack((diode_current, observation[1][-start:]))
                diode_chronos.append(observation[0])
       
        diode_current = np.array(diode_current)
        chronos_values_indices = np.array(starts)
        diode_chronos = np.array(diode_chronos)
     
        chronos_index_fit = np.polyfit(chronos_values_indices, diode_chronos, 1)
        chronos_based_on_index_predictor = np.poly1d(chronos_index_fit)
        
        #pylab.figure()
        
        #pylab.plot(chronos_values_indices, diode_chronos, 'bo', label='record')
        #pylab.plot(chronos_based_on_index_predictor(np.arange(0, len(diode_current))), 'b-', label='fit')
            
        diode_chronos = chronos_based_on_index_predictor(np.arange(0, len(diode_current)))
        
        #print 'chronos_values_indices'
        #print chronos_values_indices
        #pylab.figure()
        #pylab.plot(diode_chronos, diode_current)
        #pylab.xlabel('chronos [s]')
        #pylab.ylabel('current [mA 1e-4]')
        #pylab.show()

        return diode_chronos, diode_current
    
    def analyze(self):
        parameters = self.get_parameters()
        results = self.get_results()
        
        actuator_chronos, actuator_position = self.get_observations(results, 'actuator')
        fast_shutter_chronos, fast_shutter_state = self.get_observations(results, 'fast_shutter')
        
        start_chronos, end_chronos = self.get_fast_shutter_open_close_times(fast_shutter_chronos, fast_shutter_state)
        
        actuator_scan_indices = self.get_scan_indices(actuator_chronos, start_chronos, end_chronos, self.fast_shutter_chronos_uncertainty)
        actuator_scan_chronos = actuator_chronos[actuator_scan_indices]
        actuator_scan_position = actuator_position[actuator_scan_indices]
        
        position_chronos_predictor = self.get_position_chronos_predictor(actuator_scan_chronos, actuator_scan_position)
        
        diode_chronos, diode_current = self.get_diode_chronos_and_current(results, self.monitor)
                
        dark_current_indices = diode_chronos > (end_chronos + self.fast_shutter_chronos_uncertainty)
        dark_current = diode_current[dark_current_indices].mean()
        
        diode_current -= dark_current
        
        diode_scan_indices = self.get_scan_indices(diode_chronos, start_chronos, end_chronos, self.fast_shutter_chronos_uncertainty)
        diode_scan_chronos = diode_chronos[diode_scan_indices]
        
        self.diode_scan_current = diode_current[diode_scan_indices]        
        self.diode_scan_positions = position_chronos_predictor(diode_scan_chronos)
        self.diode_scan_energies = get_energy_from_theta(self.diode_scan_positions, units_energy=eV, units_theta=degree)
        self.diode_scan_flux = get_flux(self.diode_scan_current, self.diode_scan_energies)
        
        #pylab.figure()
        #pylab.plot(self.diode_scan_energies, self.diode_scan_flux)
        #pylab.xlabel('energy [eV]')
        #pylab.ylabel('flux [ph/s]')
        #pylab.grid(True)
        #pylab.show()
        
        if self.diode_scan_energies[0] > self.diode_scan_energies[-1]:
            self.diode_scan_energies = self.diode_scan_energies[::-1]
            self.diode_scan_flux = self.diode_scan_flux[::-1]
            
        gap = parameters['undulator_gap_encoder_position']
        
        harmonics = np.arange(1, 21)

        theoretic_harmonic_energies = undulator_peak_energy(gap, harmonics, detune=False)
        
        peaks = self.get_peaks(self.diode_scan_energies, medfilt(self.diode_scan_flux, 27))
        
        print 'peaks'
        print peaks
        
        thr = [(t, 0) for t in theoretic_harmonic_energies]
        ep = [(e, 0) for e in self.diode_scan_energies[peaks]]
        fluxes = self.diode_scan_flux[peaks]
      
        dm = distance_matrix(thr, ep)
        
        minimums = dm.argmin(axis=0)
        
        matches = np.where(dm<self.maximum_peak_theoretic_error)
        
        matches_0 = []
        matches_1 = []
        for harmonic_number_minus_one in set(matches[0]):
            indices = np.where(harmonic_number_minus_one == matches[0])
            closest = dm[matches[0][indices], matches[1][indices]].argmin()
            matches_0.append(matches[0][indices[0][closest]])
            matches_1.append(matches[1][indices[0][closest]])
              
        matches = (np.array(matches_0), np.array(matches_1))

        ep_matched = self.diode_scan_energies[peaks][matches[1]]
        thr_matched = theoretic_harmonic_energies[matches[0]]
        fluxes_matched = fluxes[matches[1]]
        
        peaks_from_maxima = []
        peaks_from_com = []

        pylab.figure(figsize=(16, 9))
        
        for e in ep_matched:
            indices = np.logical_and(self.diode_scan_energies > e - self.peak_half_width, 
                                     self.diode_scan_energies < e + self.peak_half_width)
            
            relevant_energies = self.diode_scan_energies[indices]
            relevant_fluxes = self.diode_scan_flux[indices]
            
            maximum = relevant_energies[relevant_fluxes.argmax()]
            
            peaks_from_maxima.append(maximum)
            
            indices = np.logical_and(self.diode_scan_energies >= maximum - self.peak_half_width, 
                                     self.diode_scan_energies <= maximum + self.peak_half_width)
            
            relevant_energies = self.diode_scan_energies[indices]
            relevant_fluxes = self.diode_scan_flux[indices]
            
            left_min = relevant_fluxes[:len(relevant_fluxes)/2].min()
            right_min = relevant_fluxes[len(relevant_fluxes)/2:].min()
            
            #print 'left_min', left_min
            #print 'right_min', right_min
            
            indices = relevant_fluxes >= max([left_min, right_min])
            #print 'indices'
            #print indices
            if sum(indices)>1:
                relevant_energies = relevant_energies[indices]
                relevant_fluxes = relevant_fluxes[indices]
                #print 'e'
                #print e
                #print 're'
                #print relevant_energies
                #print 'rf'
                #print relevant_fluxes
                pylab.plot(relevant_energies, relevant_fluxes, 'o-', lw=4, color='blue')
            else:
                pylab.plot(relevant_energies, relevant_fluxes, 'o-', lw=4, color='green')
                
            flux_vs_energy = interp1d(relevant_energies, relevant_fluxes)
            resp = np.linspace(relevant_energies[0], relevant_energies[-1], 1000)
            
            flux = flux_vs_energy(resp)
            
            com = center_of_mass(flux)
            
            try:
                peaks_from_com.append(resp[com])
            except:
                pass
            
        peaks_from_maxima = np.array(peaks_from_maxima)
        
        matched = np.array(zip(matches[0]+1, thr_matched, peaks_from_maxima, np.abs(thr_matched - peaks_from_maxima ), fluxes_matched))
        
        self.peaks = []
        for k, thr, ep, diff, flx in matched:
            self.peaks.append([gap, int(k), ep, flx])
            pylab.annotate(s='%d' % k, xy=(ep, flx), xytext=(ep+ 150, 1.1*flx), arrowprops=dict(arrowstyle='->', connectionstyle="arc3"))
            
        pylab.plot(self.diode_scan_energies, self.diode_scan_flux, label='experiment')
        pylab.vlines(theoretic_harmonic_energies, 0, self.diode_scan_flux.max(), color='cyan', label='theory')
        pylab.vlines(self.diode_scan_energies[peaks], 0, self.diode_scan_flux.max(), color='magenta', label='find_peaks_cwt')
        pylab.vlines(peaks_from_maxima, 0, self.diode_scan_flux.max(), color='black', label='maxima peaks')
        pylab.vlines(peaks_from_com, 0, self.diode_scan_flux.max(), color='orange', label='com peaks')
        
        pylab.xlabel('energy [eV]')
        pylab.ylabel('flux [ph/s]')
        pylab.title('Energy scan, Proxima 2A, undulator gap %.2f' % gap)
        pylab.grid(True)
        pylab.legend(loc='best')
        pylab.savefig(self.parameters_filename.replace('_parameters.pickle', '_analysis_plot.png'))
        if self.display:
            pylab.show()
            
        results['peaks'] = self.peaks
        results['flux'] = self.diode_scan_flux
        results['energy'] = self.diode_scan_energies
        results['gap'] = gap
        
        self.save_results(results)
        
    def conclude(self):
        pass


class undulator_peaks_analysis:
    
    def __init__(self, 
                 directory='/nfs/ruche/proxima2a-spool/2017_Run4/2017-09-07/com-proxima2a/RAW_DATA/Commissioning/undulator/full_beam',
                 minimum_intensity=1e8,
                 k0=2.73096921,
                 k1=-3.84082989,
                 k2=0.60382274,
                 display=False):
        
        self.directory = directory
        self.minimum_intensity = minimum_intensity
        self.peaks = None
        self.k0 = k0
        self.k1 = k1
        self.k2 = k2
        self.display = display
        self.energies_on_grid = None
        self.fluxes_on_grid = None
        self.gaps = None
        
    def get_peaks(self):
        peaks = []
        for result in glob.glob(os.path.join(self.directory, '*_results.pickle')):
            peaks += pickle.load(open(result))['peaks']
        peaks = np.array(peaks)
        peaks = peaks[peaks[:, 3] > self.minimum_intensity]
        return peaks
    
    def get_scans(self):
        scans = {}
        for result in glob.glob(os.path.join(self.directory, '*_results.pickle')):
            r = pickle.load(open(result))
            flux = r['flux']
            energy = r['energy']
            gap = r['gap']
            scans[gap] = {'energy': energy, 'flux': flux}
        return scans
    
    def get_harmonics(self):
        if self.peaks is None:
           self.peaks = self.get_peaks()
        harmonics = list(set(map(int, self.peaks[:, 1])))
        harmonics.sort()
        return harmonics
    
    def residual(self, x, peaks):
        k0, k1, k2 = x
        model = []
        for gap, harmonics in peaks[:, 0:2]:
            model.append(undulator_peak_energy(gap, harmonics, k0=k0, k1=k1, k2=k2))
        model = np.array(model)
        experiment = peaks[:, 2]
        diff = experiment - model
        return np.sum(diff**2)/(2*len(model)) #np.dot(diff, diff)/(2*len(model))
            
    def fit(self, method='nelder-mead'):
        if self.peaks is None:
            peaks = self.get_peaks()
        else:
            peaks = self.peaks
        x0 = self.k0, self.k1, self.k2
        self.fit_result = minimize(self.residual, x0, args=(peaks,), method=method)
        return self.fit_result
    
    def generate_undulator_tables(self):
        harmonics = self.get_harmonics()
        try:
            k0, k1, k2 = self.fit_result.x
        except:
            self.fit()
            k0, k1, k2 = self.fit_result.x
        
        pylab.figure(figsize=(16, 9))
        
        for n in harmonics:
            selection = self.get_selection(n)
            gaps = selection[:, 0]
            energies = selection[:, 2]
            X = np.vstack([energies/1e3, gaps]).T
            np.savetxt('GAP_ENERGY_HARMONICS%d.txt' % n, X, fmt='%6.3f', delimiter=' ', header='%d\n%d\nENERGY  GAP' % X.shape[::-1], comments='')
            modeled_energies = undulator_peak_energy(gaps, n, k0=k0, k1=k1, k2=k2)
            X_model = np.vstack([modeled_energies, gaps]).T
            np.savetxt('fit_GAP_ENERGY_HARMONIC%d.txt' % n, X_model, fmt='%6.3f', delimiter=' ', header='%d\n%d\nENERGY  GAP' % X_model.shape[::-1], comments='')
            
            pylab.plot(energies, gaps, 'o-', label='%d' % n)
            pylab.plot(modeled_energies, gaps, 'v-')
        
        pylab.title('Proxima 2A U24 undulator harmonic peak positions as function of gap and energy', fontsize=22)
        pylab.xlabel('energy [eV]', fontsize=18)
        pylab.ylabel('gap [mm]', fontsize=18)
        pylab.ylim([7., self.peaks[:, 0].max() + 0.5])
        pylab.grid(True)
        pylab.legend(loc='best', fontsize=16)
        ax = pylab.gca()
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(16)
        pylab.savefig('U24_harmonic_peak_positions_gap_vs_energy.png')
        
        if self.display:
            pylab.show()
       
    def plot_tuning_curves(self, which='all'):
        
        harmonics = self.get_harmonics()
        
        pylab.figure(figsize=(16, 9))
            
        if which == 'even':
            harmonics = [n for n in harmonics if n%2==0]
        if which == 'odd':
            harmonics = [n for n in harmonics if n%2==1]
       
        for n in harmonics:
            selection = self.get_selection(n)
            gaps = selection[:, 0]
            energies = selection[:, 2]
            fluxes = selection[:, 3]
            pylab.plot(energies, fluxes, 'o-', label='%d' % n)

        pylab.title('Proxima 2A U24 undulator tuning curves, %s harmonics' % which, fontsize=22)
        pylab.xlabel('energy [eV]', fontsize=18)
        pylab.ylabel('flux [ph/s]', fontsize=18)
        pylab.grid(True)
        pylab.legend(loc='best', fontsize=16)
        ax = pylab.gca()
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(16)
        pylab.savefig('U24_tuning_curves_%s_harmonics.png' % which)
            
        if self.display:
            pylab.show()
    
    def plot_flux_vs_gap(self):
        harmonics = self.get_harmonics()
        
        pylab.figure(figsize=(16, 9))
        for n in harmonics:
            selection = self.get_selection(n)
            
            gaps = selection[:, 0]
            energies = selection[:, 2]
            fluxes = selection[:, 3]
            pylab.plot(gaps, fluxes, 'o-', label='%d' % n)
        pylab.title(
        'Proxima 2A U24 undulator flux vs. gap', fontsize=22)
        pylab.xlabel('gap [mm]', fontsize=18)
        pylab.ylabel('flux [ph/s]', fontsize=18)
        pylab.grid(True)
        pylab.legend(loc='best', fontsize=16)
        ax = pylab.gca()
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(16)
        pylab.savefig('U24_flux_vs_gap_all_harmonics.png')
        
        if self.display:
            pylab.show()
    
    def get_selection(self, n):
        if self.peaks is None:
            self.get_peaks()
        selection = list(self.peaks[self.peaks[:, 1] == n])
        selection.sort(key=lambda x: x[2])
        selection = np.array(selection)
        return selection
    
    def plot_theoretic_tuning_curves(self):
        
        harmonics = self.get_harmonics()
        
        pylab.figure(figsize=(16, 9))
        
        try:
            k0, k1, k2 = self.fit_result.x
        except:
            self.fit()
            k0, k1, k2 = self.fit_result.x
        
        for n in harmonics:
            selection = self.get_selection(n)
            gaps = selection[:, 0]
            energies = selection[:, 2]
            fluxes = selection[:, 3]
        
            Bs = undulator_magnetic_field(gaps, k0=k0, k1=k1, k2=k2)
            Ks = undulator_strength(Bs)
            theoric_fluxes = angular_flux_density(Ks, n, N=80)
            
            pylab.plot(energies, theoric_fluxes, 'o-', label='%d' % n)
            
        pylab.title('Proxima 2A U24 theoretic undulator tuning curves', fontsize=22)
        pylab.xlabel('energy [eV]', fontsize=18)
        pylab.ylabel('flux [ph/s]', fontsize=18)
        pylab.grid(True)
        pylab.legend(loc='best', fontsize=16)
        ax = pylab.gca()
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontsize(16)
        pylab.savefig('U24_theoretic_tuning_curves_all_harmonics.png')
        if self.display:
            pylab.show()
    
    def plot_magnetic_field(self):
        
        pylab.figure(figsize=(16, 9))
        
        gaps = list(set(self.get_peaks()[:,0]))
        gaps.sort()
        gaps = np.array(gaps)
        bs = []
        gs = []
        ks = []
        
        self.peaks = self.get_peaks()
        
        for k, result in enumerate(self.peaks):
            gap, n, energy, flux = result
            gs.append(gap)
            #energy *= (1 + 2/(n*80))
            K = undulator_strength_from_peak_position(energy/1e3, n)
            ks.append(K)
            B = undulator_magnetic_field_from_K(K)
            if k == len(self.peaks)-1:
                pylab.plot(gap, B, 'bo', label='experiment')
            else:
                pylab.plot(gap, B, 'bo')
            bs.append(B)
    
        gs = np.array(gs)
        bs = np.array(bs)
        ks = np.array(ks)
        data = zip(gs, bs)
        data.sort(key=lambda x: x[0])
        data = np.array(data)
        d = pd.DataFrame()
        d['gap'] = data[:,0]
        d['B'] = data[:,1] + 0.3
        
        try:
            k0, k1, k2 = self.fit_result.x
        except:
            self.fit()
            k0, k1, k2 = self.fit_result.x
        
        
        bs3 = undulator_magnetic_field(gaps, k0, k1, k2)
        
        pylab.xlabel('gap [mm]', fontsize=18)
        h = pylab.ylabel('B [T]', fontsize=18, labelpad=35)
        h.set_rotation(0)
        
        pylab.title('Proxima 2A U24 undulator peak magnetic field and strength as function of gap', fontsize=22)
        pylab.grid(True)
        
        ax = pylab.gca()
        ax.text(0.83, 0.8, '\# data points = %d' % len(data), color='b', fontsize=18, transform=ax.transAxes)
        ax.text(0.05, 0.15, 'model function: $B(x) = k_{0} \\exp(k_{1} x  + k_{2} x^{2}); x = \\frac{gap}{\lambda_{u}}$', fontsize=20, color='green', transform=ax.transAxes)
        ax.text(0.05, 0.08, 'fit parameters: k_{0} = %6.3f, k_{1} = %6.3f, k_{2} = %6.3f' % (k0, k1, k2), fontsize=20, color='green', transform=ax.transAxes)
        
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
        pylab.savefig('B_and_K_vs_gap.png')
        if self.display:
            pylab.show()

    def get_data_on_grid(self, start=5200, end=18800, npoints=13600):
        energies_on_grid = np.linspace(start, end, npoints)
        scans = self.get_scans()
        fluxes_on_grid = []
        gaps = scans.keys()
        gaps.sort()
        for gap in gaps:
            flux_vs_energy = interp1d(scans[gap]['energy'], scans[gap]['flux'], kind='slinear', bounds_error=False)
            flux_on_grid = flux_vs_energy(energies_on_grid)
            fluxes_on_grid.append(flux_on_grid)
        
        self.energies_on_grid = energies_on_grid
        self.fluxes_on_grid = np.array(fluxes_on_grid)
        self.gaps = gaps
        
        return energies_on_grid, np.array(fluxes_on_grid), gaps
        
    def plot_monochromator_glitches(self):
        if any([item is None for item in [self.energies_on_grid, self.fluxes_on_grid, self.gaps]]):
            self.get_data_on_grid()
        energies_on_grid, fluxes_on_grid, gaps = self.energies_on_grid, self.fluxes_on_grid, self.gaps
        
        pylab.figure(figsize=(16, 9))
        pylab.title('Proxima 2A monochromator glitches')
        pylab.xlabel('energy [eV]')
        pylab.ylabel('relative strength')
        pylab.xlim(energies_on_grid[0], energies_on_grid[-1])
        #print 'fluxes_on_grid.shape', fluxes_on_grid.shape
        #summed_fluxes = fluxes_on_grid.sum(axis=0)
        #print 'summed_fluxes.shape', summed_fluxes.shape
        #filtered_fluxes = medfilt(summed_fluxes, 11)
        #print 'filtered_fluxes.shape', filtered_fluxes.shape
        filtered_fluxes = medfilt(fluxes_on_grid, np.array([1, 11]))
        glitches = (filtered_fluxes - fluxes_on_grid)/fluxes_on_grid
        #glitches = (filtered_fluxes - summed_fluxes)/summed_fluxes
        pylab.plot(energies_on_grid, glitches.sum(axis=0))
        #pylab.plot(energies_on_grid, glitches)
        pylab.legend()
        pylab.grid(True)
        pylab.savefig('monochromator_glitches.png')
        
    def plot_flux_vs_energy(self):
        if any([item is None for item in [self.energies_on_grid, self.fluxes_on_grid, self.gaps]]):
            self.get_data_on_grid()
        energies_on_grid, fluxes_on_grid, gaps = self.energies_on_grid, self.fluxes_on_grid, self.gaps
        
        pylab.figure(figsize=(16, 9))
        pylab.plot(energies_on_grid, fluxes_on_grid.T)
        pylab.xlabel('energy [eV]')
        pylab.ylabel('flux [ph/s]')
        pylab.title('Photon flux vs energy and undulator gap')
        pylab.xlim(energies_on_grid[0], energies_on_grid[-1])
        legend = ['%.2f' % gap for gap in gaps]
        pylab.legend(legend)
        pylab.grid(True)
        pylab.savefig('photon_flux_vs_energy.png')
        
    def plot_flux_envelope_vs_energy(self):
        if any([item is None for item in [self.energies_on_grid, self.fluxes_on_grid, self.gaps]]):
            self.get_data_on_grid()
        
        energies_on_grid, fluxes_on_grid, gaps = self.energies_on_grid, self.fluxes_on_grid, self.gaps
        
        pylab.figure(figsize=(16, 9))
        pylab.title('Photon flux envelope vs energy')
        pylab.xlabel('energy [eV]')
        pylab.ylabel('flux [ph/s]')
        pylab.xlim(energies_on_grid[0], energies_on_grid[-1])
        pylab.plot(energies_on_grid, fluxes_on_grid.max(axis=0))
        pylab.legend()
        pylab.grid(True)
        pylab.savefig('photon_flux_envelope_vs_energy.png')
        
    def plot_flux_vs_energy_and_undulator_gap(self):
        if any([item is None for item in [self.energies_on_grid, self.fluxes_on_grid, self.gaps]]):
            self.get_data_on_grid()
        energies_on_grid, fluxes_on_grid, gaps = self.energies_on_grid, self.fluxes_on_grid, self.gaps
        
        ens, gaps = np.meshgrid(energies_on_grid, gaps)
        fig = plt.figure(figsize=(16, 9))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_wireframe(ens, gaps, fluxes_on_grid, rstride=75, cstride=75)
        ax.set_xlabel('energy [eV]', fontsize=18)
        ax.set_ylabel('gap [mm]', fontsize=18)
        ax.set_zlabel('flux [ph/s]', fontsize=18)
        plt.title('Photon flux vs energy and undulator gap', fontsize=22)
        ax.view_init(54, -111)
        plt.grid(True)
        plt.savefig('photon_flux_vs_energy_and_undulator_gap.png')
        
    def plot_scans_3d(self):
        '''Plot flux vs energy and gap'''

        if any([item is None for item in [self.energies_on_grid, self.fluxes_on_grid, self.gaps]]):
            self.get_data_on_grid()
        
        self.plot_flux_vs_energy()
        
        self.plot_flux_envelope_vs_energy()
        
        self.plot_monochromator_glitches()
        
        self.plot_flux_vs_energy_and_undulator_gap()
            
        if self.display:
            pylab.show()
           
           
def USA():
    
    import optparse
    
    parser = optparse.OptionParser()
    
    parser.add_option('-f', '--filename', type=str, default='/nfs/ruche/proxima2a-spool/2017_Run4/2017-09-07/com-proxima2a/RAW_DATA/Commissioning/undulator/full_beam/gap_7.952_parameters.pickle', help='Filename of pickled experiment parameters dictionary')
    parser.add_option('-m', '--monitor', type=str, default='calibrated_diode', help='Which monitor signal to use in the analysis')
    parser.add_option('-D', '--display', action='store_true', help='Show results')
    parser.add_option('-E', '--maximum_peak_theoretic_error', type=float, default=210., help='Maximum distance between detected and predicted peak')
    parser.add_option('-p', '--peak_half_width', type=float, default=50., help='Peak half width')
    parser.add_option('-s', '--fast_shutter_chronos_uncertainty', type=float, default=0.1, help='Fast shutter time uncertainty')
    
    options, args = parser.parse_args()
    
    usa = undulator_scan_analysis(options.filename, 
                                  monitor=options.monitor, 
                                  display=options.display,
                                  fast_shutter_chronos_uncertainty=options.fast_shutter_chronos_uncertainty,
                                  peak_half_width=options.peak_half_width,
                                  maximum_peak_theoretic_error=options.maximum_peak_theoretic_error)
    
    
    usa.analyze()

def UPA():
    
    upa = undulator_peaks_analysis(minimum_intensity=1.e8, display=True)
    #peaks =  upa.get_peaks()
    #print 'shape', peaks.shape

    #print 'fit results'
    #print upa.fit()
    #print upa.fit_result.x
    #upa.generate_undulator_tables()
    for which in ['all', 'odd', 'even']:
        upa.plot_tuning_curves(which=which)
    
    upa.plot_flux_vs_gap()
    #upa.plot_theoretic_tuning_curves()
    upa.plot_magnetic_field()
    upa.plot_scans_3d()
    
if __name__ == '__main__':
    #pass
    UPA()
    #USA()
  
