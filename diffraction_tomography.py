#!/usr/bin/env python
# -*- coding: utf-8 -*-

import traceback
import logging
import time
import numpy as np
import copy
import os
import subprocess
import pickle
import re
import pylab
import sys
import gevent 

from diffraction_experiment import diffraction_experiment
from goniometer import goniometer
from area import area
import scipy.ndimage as nd
from scipy.optimize import minimize

class diffraction_tomography(diffraction_experiment):
    specific_parameter_fields = [{'name': 'scan_start_angles', 'type': 'list', 'description': ''},
                                 {'name': 'vertical_range', 'type': 'bool', 'description': ''},
                                 {'name': 'vertical_step_size', 'type': 'bool', 'description': ''},
                                 {'name': 'reference_position', 'type': 'dict', 'description': ''}]
    
    def __init__(self,
                 name_pattern='excenter_$id',
                 directory='/nfs/data2/excenter',
                 treatment_directory='/dev/shm',
                 scan_start_angles='[0, 90, 180, 225, 315]',
                 vertical_range=0.1,
                 horizontal_range=0,
                 scan_range=0.01,
                 vertical_step_size=0.002,
                 frame_time=0.005,
                 transmission=None,
                 position=None,
                 photon_energy=None,
                 resolution=None,
                 diagnostic=True,
                 analysis=True,
                 conclusion=True,
                 display=True,
                 method='xds',
                 dont_move_motors=False,
                 parent=None,
                 beware_of_top_up=False,
                 beware_of_download=False,
                 generate_cbf=True,
                 generate_h5=False):
    
        if hasattr(self, 'parameter_fields'):
            self.parameter_fields += diffraction_tomography.specific_parameter_fields
        else:
            self.parameter_fields = diffraction_tomography.specific_parameter_fields[:]
            
        diffraction_experiment.__init__(self, 
                                        name_pattern, 
                                        directory,
                                        transmission=transmission,
                                        photon_energy=photon_energy,
                                        resolution=resolution,
                                        diagnostic=diagnostic,
                                        analysis=analysis,
                                        conclusion=conclusion,
                                        parent=parent,
                                        beware_of_top_up=beware_of_top_up,
                                        beware_of_download=beware_of_download,
                                        generate_cbf=generate_cbf,
                                        generate_h5=generate_h5)
        
        self.description = 'X-ray diffraction tomgraphy, Proxima 2A, SOLEIL, %s' % time.ctime(self.timestamp)
        self.display = display
        
        self.scan_start_angles = eval(scan_start_angles)
        self.vertical_range = vertical_range
        self.horizontal_range = horizontal_range
        self.vertical_step_size = vertical_step_size
        self.frame_time = frame_time
        self.nimages = int(vertical_range/vertical_step_size)
        self.scan_exposure_time = self.frame_time * self.nimages
        self.scan_range = scan_range
        self.ntrigger = len(self.scan_start_angles)
        self.number_of_rows = int(vertical_range/vertical_step_size)
        self.number_of_columns = 1
        
        print('number_of_rows', self.number_of_rows)
        print('number_of_columns', self.number_of_columns)
        print('motor_speed', self.vertical_range/(self.number_of_rows * self.frame_time))
        print('scan_range', scan_range)
        
        if position == None:
            self.reference_position = self.goniometer.get_aligned_position()
        else:
            self.reference_position = position
            
        self.horizontal_center = self.reference_position['AlignmentY']
        self.nimages_per_file = self.number_of_rows
        self.scan_start_angle = self.scan_start_angles[0]
        self.angle_per_frame = self.scan_range/self.nimages
        self.image_nr_start = 1
        self.treatment_directory = treatment_directory
        self.format_dictionary = {'directory': self.directory, 'name_pattern': self.name_pattern, 'treatment_directory': self.treatment_directory}
        
        self.line_scan_time = self.frame_time * self.number_of_rows
        self.total_expected_exposure_time = self.line_scan_time * self.ntrigger
        self.total_expected_wedges = self.ntrigger
        self.overlap = 0.
        
    def get_overlap(self):
        return self.overlap
    
    def get_helical_lines(self):
        #start, stop, scan_start_angle, scan_range, scan_exposure_time
        helical_lines = []
        for scan_start_angle in self.scan_start_angles:
            position = copy.copy(self.reference_position)
            position_start = copy.copy(self.reference_position)
            position_stop = copy.copy(self.reference_position)
            position['Omega'] = scan_start_angle
            position_start['Omega'] = scan_start_angle
            position_stop['Omega'] = scan_start_angle
            focus_center, vertical_center = self.goniometer.get_focus_and_vertical_from_position(position=position)
            
            a = area(self.vertical_range, self.horizontal_range, self.number_of_rows, self.number_of_columns, vertical_center, self.horizontal_center)
            grid, points = a.get_grid_and_points()
            jumps = a.get_jump_sequence(grid.T)
            collect_sequence = a.get_linearized_point_jumps(jumps, points)
            for start, stop in collect_sequence:
                x_start, y_start = self.goniometer.get_x_and_y(focus_center, start[0], scan_start_angle)
                x_stop, y_stop = self.goniometer.get_x_and_y(focus_center, stop[0], scan_start_angle)
                position_start['CentringX'] = x_start
                position_start['CentringY'] = y_start
                position_stop['CentringX'] = x_stop
                position_stop['CentringY'] = y_stop
                helical_lines.append([position_start, position_stop, scan_start_angle, self.scan_range, self.scan_exposure_time])
        return helical_lines
    
    def get_reference_position(self):
        if os.path.isfile(self.get_parameters_filename()):
            self.reference_position = self.load_parameters_from_file()['reference_position']
        return self.reference_position
            
    def run(self):
        self._start = time.time()
        
        self.md2_task_info = []
        for helical_line in self.get_helical_lines():            
            start, stop, scan_start_angle, scan_range, scan_exposure_time = helical_line
            task_id = self.goniometer.helical_scan(start, stop, scan_start_angle, scan_range, scan_exposure_time)
            self.md2_task_info.append(self.goniometer.get_task_info(task_id))

    def analyze(self, method='xds'):
        if method=='dozor':
            self.run_dozor(blocking=True)
        elif method=='xds':
            self.run_xds()
        elif method=='dials':
            self.run_dials()

    
    def get_results(self, method='xds'):
        self.logger.info('get_results, method %s' % method)
        if method == 'dozor':
            results = self.get_dozor_results()[:, 2]
            print('results', results.shape, results[:10])
        elif method == 'dials':
            results = self.get_dials_results()
        elif method == 'xds':
            results = self.get_xds_results()
        else:
            results = []
        return results
    
    def get_result_position(self, threshold=0.25, min_spots=5, alignmenty_direction=-1., alignmentz_direction=1., centringx_direction=-1., centringy_direction=1., method='xds', geometric_center=True):
        self.logger.info('get_result_position')
        
        parameters = self.get_parameters()
        results = self.get_results(method=method)
        
        nimages = parameters['nimages']
        angles = parameters['scan_start_angles']
        
        vertical_displacements = []
        #beam_position = 0.5 * (self.nimages-1.)
        #print('beam_position', beam_position)
        
        for k in range(parameters['ntrigger']):
            line = results[k*nimages: (k+1)*nimages]
            line[line<min_spots] = 0
            line[line<=line.max()*threshold] = 0
            if geometric_center:
                line[line>0] = 1
            y = nd.center_of_mass(line)[0]
            print('center_of_mass', y)            
            #y -= beam_position
            #print('position in steps', y)
            #y *= parameters['vertical_step_size']
            #print('shift in mm', y)
            vertical_displacements.append(y)
        
        angles_radians = np.radians(parameters['scan_start_angles'])
        print('vertical_displacements', vertical_displacements)
        vertical_displacements = np.array(vertical_displacements)
        #vertical_displacements *= 1e3
        initial_parameters = [np.mean(vertical_displacements), np.std(vertical_displacements), np.random.random()]
        print('initial_parameters', initial_parameters)
        
        fit_y = minimize(self.goniometer.circle_model_residual, 
                         initial_parameters, 
                         method='nelder-mead', 
                         args=(angles_radians, vertical_displacements))
        print('fit_y', fit_y)
        
        c, r, alpha = fit_y.x
        omega_axis_position = c 
        print('omega_axis_position', omega_axis_position)
        omega_axis_shift = omega_axis_position - 0.5*nimages
        print('estimated omega_axis_shift in px', omega_axis_shift)
        print('estimated omega_axis_shift in mm', omega_axis_shift * parameters['vertical_step_size']) 
        
        #c *= parameters['vertical_step_size'] 
        c = omega_axis_shift * parameters['vertical_step_size']
        r *= parameters['vertical_step_size']
        v = {'c': c, 'r': r, 'alpha': alpha}
        print('c, r, alpha', c, r, alpha)
        d_sampx = centringx_direction * r * np.sin(alpha)
        d_sampy = centringy_direction * r * np.cos(alpha)
        #d_y = alignmenty_direction * horizontal_center
        d_z = alignmentz_direction * c
        
        move_vector_dictionary = {'AlignmentZ': d_z,
                                  #'AlignmentY': d_y,
                                  'CentringX': d_sampx,
                                  'CentringY': d_sampy}
        
        print('move_vector', move_vector_dictionary)
        
        result_position = {}
        reference_position = self.get_reference_position()
        for motor in reference_position:
            result_position[motor] = reference_position[motor]
            if motor in move_vector_dictionary:
                result_position[motor] += move_vector_dictionary[motor]
        print('reference_position', reference_position)
        print('result_position', result_position)
        return result_position
    
    def run_shape_reconstruction(self):
        os.system('/nfs/data2/Martin/Research/tomography/shape_from_diffraction_tomography.py -d %s -n %s -D &' % (self.directory, self.name_pattern))
        
    def conclude(self, method='xds', move_motors=True):
        self.logger.info('conclude')
        self.run_shape_reconstruction()
        result_position = self.get_result_position(method=method)
        if move_motors:
            self.logger.info('moving motors')
            self.goniometer.set_position(result_position)
            self.goniometer.save_position()
            #os.system('excenter_finished_dialog.py &')
        
def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-n', '--name_pattern', default='excenter_$id', type=str, help='Prefix')
    parser.add_argument('-d', '--directory', default='/nfs/data2/excenter', type=str, help='Destination directory')
    parser.add_argument('-a', '--scan_start_angles', default='[0, 90, 180, 225, 315]', type=str, help='angles')
    parser.add_argument('-y', '--vertical_range', default=0.1, type=float, help='vertical range')
    parser.add_argument('-f', '--frame_time', default=0.005, type=float, help='frame time')
    parser.add_argument('-A', '--analysis', action='store_true', help='If set will perform automatic analysis.')
    parser.add_argument('-C', '--conclusion', action='store_true', help='If set will move the motors upon analysis.')
    parser.add_argument('-D', '--diagnostic', action='store_true', help='If set will record diagnostic information.')
    parser.add_argument('-m', '--method', type=str, default='xds', help='analysis method')
    parser.add_argument('-S', '--dont_move_motors', action='store_true', help='Do not move after conclusion')
    parser.add_argument('-5', '--generate_h5', action='store_false', help='Do not generate h5 files')
    options = parser.parse_args()
    
    print('options', options)
    print('vars(options)', vars(options))

    experiment = diffraction_tomography(**vars(options))
    print('get_parameters_filename', experiment.get_parameters_filename())
    if not os.path.isfile(experiment.get_parameters_filename()):
        experiment.execute()
    elif options.analysis == True:
        experiment.analyze(method=options.method)
        if options.conclusion == True:
            experiment.conclude(method=options.method, move_motors=not options.dont_move_motors)
    
if __name__ == '__main__':
    main()
