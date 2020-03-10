#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
helical scan
'''
import traceback
import logging
import time
import pickle
import os

import sys

import numpy as np

from omega_scan import omega_scan
from copy import deepcopy

from beam_align import beam_align

class inverse_scan(omega_scan):
    
    actuator_names = ['Omega']
    
    specific_parameter_fields = [{'name': 'interleave_range', 'type': '', 'description': ''},
                                 {'name': 'raster', 'type': '', 'description': ''},
                                 {'name': 'nrepeats', 'type': '', 'description': ''},
                                 {'name': 'start_position', 'type': '', 'description': ''},
                                 {'name': 'end_position', 'type': '', 'description': ''},
                                 {'name': 'all_positions', 'type': '', 'description': ''}]

    def __init__(self, 
                 name_pattern='test_$id', 
                 directory='/tmp', 
                 scan_range=180, 
                 scan_exposure_time=18, 
                 scan_start_angle=0, 
                 angle_per_frame=0.1, 
                 image_nr_start=1,
                 interleave_range=5,
                 nrepeats=1,
                 npositions=1,
                 raster=False,
                 position=None,
                 kappa=None,
                 phi=None,
                 photon_energy=None,
                 resolution=None,
                 detector_distance=None,
                 detector_vertical=None,
                 detector_horizontal=None,
                 transmission=None,
                 flux=None,
                 snapshot=False,
                 nimages_per_file=None,
                 zoom=None,
                 diagnostic=None,
                 analysis=None,
                 simulation=None,
                 parent=None):
        
        if hasattr(self, 'parameter_fields'):
            self.parameter_fields += inverse_scan.specific_parameter_fields
        else:
            self.parameter_fields = inverse_scan.specific_parameter_fields[:]
            
        if position != None:
            self.raw_position = eval(position)
            if len(self.raw_position) == 2:
                position = self.raw_position[0]
                self.start_position = self.raw_position[0]
                self.end_position = self.raw_position[-1] 
        else:
            self.raw_position = None
            self.start_position = None
            self.end_position = None
            
        omega_scan.__init__(self,
                            name_pattern=name_pattern, 
                            directory=directory,
                            scan_range=scan_range,
                            scan_start_angle=scan_start_angle,
                            scan_exposure_time=scan_exposure_time,
                            angle_per_frame=angle_per_frame, 
                            image_nr_start=image_nr_start,
                            position=position,
                            kappa=kappa,
                            phi=phi,
                            photon_energy=photon_energy,
                            resolution=resolution,
                            detector_distance=detector_distance,
                            detector_vertical=detector_vertical,
                            detector_horizontal=detector_horizontal,
                            transmission=transmission,
                            flux=flux,
                            snapshot=snapshot,
                            nimages_per_file=nimages_per_file,
                            diagnostic=diagnostic,
                            analysis=analysis,
                            simulation=simulation,
                            parent=parent)
        
        self.nrepeats = nrepeats
        self.raster = raster
        self.interleave_range = float(interleave_range)
        
        self.total_expected_exposure_time = self.scan_exposure_time * 2
        self.total_expected_wedges = int(self.scan_range/self.interleave_range) * 2
        
        self.npositions = npositions
                
        self.parameter_fields = self.parameter_fields.union(inverse_scan.specific_parameter_fields)
        
        self.last_beamcheck = -np.inf
        self.beamcheck_period = 1800.
        self.nbeamcheck = 0
        

    def get_nimages_per_file(self):
        return int(self.interleave_range/self.angle_per_frame)
    
    
    def get_nimages(self):
        return int(self.interleave_range/self.angle_per_frame)
    
    
    def get_ntrigger(self):
        return int(self.scan_range/self.interleave_range) * 2 * self.nrepeats * self.npositions
    
    
    def get_frame_time(self):
        return self.scan_exposure_time/self.get_nimages()
    
    
    def get_wedges(self):
        
        if self.raw_position > 1:
            
            direct = np.array(list(np.arange(self.scan_start_angle, self.scan_start_angle+self.scan_range, self.interleave_range)))
            inverse = direct + 180.
            wedges = np.vstack((direct, inverse)).T
            sweeps = wedges.ravel()
            
            all_positions = self.get_all_positions()
            
            wedges_positions = []
            for k, repeat in enumerate(range(self.nrepeats)):
                for l, position in enumerate(all_positions[::pow(-1, k)]):
                    for sweep in sweeps:
                        wedges_positions.append([sweep + l*self.interleave_range] + list(position))
            
            return wedges_positions
                        
        else:
            direct = np.array(list(np.arange(self.scan_start_angle, self.scan_start_angle+self.scan_range, self.interleave_range))*self.nrepeats)
            inverse = direct + 180.
            wedges = np.vstack((direct, inverse)).T
        
        if self.raster == True:
            if np.__version__ >= '1.8.3':
                #functionality present probably already earlier
                wedges[1::2, ::] = wedges[1::2, ::-1]
            else:
                raster = []
                for k, line in enumerate(wedges):
                    if k%2 == 1:
                        line = line[::-1]
                    raster.append(line)
                wedges = np.array(raster)
        return wedges.ravel()
    

    def get_position_vector(self, position):
        position_vector = []
        for key in sorted(position.keys()):
            position_vector.append(position[key])
        return np.array(position_vector) 
        

    def get_position_dictionary_from_position_vector(self, position_vector):
        sorted_keys = sorted(self.start_position.keys())
        position_dictionary = {}
        
        for k, key in enumerate(sorted_keys):
            position_dictionary[key] = position_vector[k]
        return position_dictionary
        

    def get_all_positions(self):
        start_vector = self.get_position_vector(self.start_position)
        end_vector = self.get_position_vector(self.end_position)
        
        all_positions = []
        for start, end in zip(start_vector, end_vector):
            all_positions.append(np.linspace(start, end, self.npositions))
        return np.array(all_positions).T
    
        
    def run(self):
        self._start = time.time()
        
        self.wedges = self.get_wedges()
        
        print 'len(self.wedges)', len(self.wedges)

        scan_range = self.interleave_range
        scan_exposure_time = self.interleave_range * self.scan_exposure_time / self.scan_range 
        
        self.md2_tasks_info = []
        
        for scan_start_angle in self.wedges:
            current_time = time.time()
            if current_time - self.last_beamcheck > self.beamcheck_period:
                print 'current_time', current_time
                print 'self.last_beamcheck', self.last_beamcheck
                print 'self.beamcheck_period', self.beamcheck_period
                self.nbeamcheck += 1
                ba = beam_align('%s_beam_check_%d' % (self.name_pattern, self.nbeamcheck),
                                self.directory,
                                photon_energy=self.photon_energy)
                ba.execute()
                self.last_beamcheck = current_time
                self.set_transmission(self.transmission)
                
            if type(scan_start_angle) == list:
                scan_start_angle, position_vector = scan_start_angle[0], scan_start_angle[1:]
                position = self.get_position_dictionary_from_position_vector(position_vector)
                self.goniometer.set_position(position)
            
            task_id = self.goniometer.omega_scan(scan_start_angle, scan_range, scan_exposure_time, wait=True)
            self.md2_tasks_info.append(self.goniometer.get_task_info(task_id))
            

        
        
def main():
    import optparse
        
    parser = optparse.OptionParser()
    parser.add_option('-n', '--name_pattern', default='inverse_test_raster_$id', type=str, help='Prefix default=%default')
    parser.add_option('-d', '--directory', default='/nfs/data/default', type=str, help='Destination directory default=%default')
    parser.add_option('-r', '--scan_range', default=180, type=float, help='Scan range [deg]')
    parser.add_option('-e', '--scan_exposure_time', default=18, type=float, help='Scan exposure time [s]')
    parser.add_option('-s', '--scan_start_angle', default=0, type=float, help='Scan start angle [deg]')
    parser.add_option('-a', '--angle_per_frame', default=0.1, type=float, help='Angle per frame [deg]')
    parser.add_option('-f', '--image_nr_start', default=1, type=int, help='Start image number [int]')
    parser.add_option('-I', '--interleave_range', default=10, type=float, help='Interleave range [deg]')
    parser.add_option('-i', '--position', default=None, type=str, help='Gonio alignment start position [dict], or positions 2x[dict]')
    parser.add_option('-p', '--photon_energy', default=None, type=float, help='Photon energy ')
    parser.add_option('-t', '--detector_distance', default=None, type=float, help='Detector distance')
    parser.add_option('-Z', '--detector_vertical', default=None, type=float, help='Detector vertical position')
    parser.add_option('-X', '--detector_horizontal', default=None, type=float, help='Detector horizontal position')
    parser.add_option('-o', '--resolution', default=None, type=float, help='Resolution [Angstroem]')
    parser.add_option('-x', '--flux', default=None, type=float, help='Flux [ph/s]')
    parser.add_option('-m', '--transmission', default=None, type=float, help='Transmission. Number in range between 0 and 1.')
    parser.add_option('-A', '--analysis', action='store_true', help='If set will perform automatic analysis.')
    parser.add_option('-D', '--diagnostic', action='store_true', help='If set will record diagnostic information.')
    parser.add_option('-S', '--simulation', action='store_true', help='If set will simulate most of the beamline (except detector and goniometer).')
    parser.add_option('-R', '--raster', action='store_true', help='If set collect in raster mode.')
    parser.add_option('-N', '--nrepeats', default=1, type=int, help='Allows to specify number of repeats of the experiment')
    parser.add_option('-L', '--npositions', default=1, type=int, help='Allows to specify number of positions between the two specified positions. Used only if more than one position specified.')
    parser.add_option('-K', '--kappa', default=0., type=float, help='Kappa axis position')
    parser.add_option('-P', '--phi', default=0., type=float, help='Phi axis position')
    
    options, args = parser.parse_args()
    
    print 'options', options
    print 'args', args
    
    invs = inverse_scan(**vars(options))
    
    filename = '%s_parameters.pickle' % invs.get_template()
    
    if not os.path.isfile(filename):
        invs.execute()
    elif options.analysis == True:
        invs.analyze()
    
def test():
    scan_range = 180
    scan_exposure_time = 18.
    scan_start_angle = 0
    angle_per_frame = 0.1
    interleave_range = 5
    s = inverse_scan(scan_range=scan_range, scan_exposure_time=scan_exposure_time, scan_start_angle=scan_start_angle, angle_per_frame=angle_per_frame, interleave_range=interleave_range)
    
if __name__ == '__main__':
    main()
    