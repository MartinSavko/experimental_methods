#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
optical alignement procedure

'''
import os
import sys
import time
import zmq
import redis
import logging
import gevent
import traceback
import pickle
import h5py
import pylab
import copy
from skimage.io import imread
import open3d as o3d
import numpy as np
from scipy.optimize import leastsq, minimize
import scipy.ndimage as ndi
from math import cos, sin, sqrt, radians, atan, asin, acos, pi, degrees

from experiment import experiment
from camera import camera
from film import film
from cats import cats
from optical_path_report import select_better_model, create_mosaic

from goniometer import (
    goniometer,
    get_points_in_goniometer_frame, 
    get_voxel_calibration, 
    get_position_from_vector, 
    get_vector_from_position)

from shape_from_history import (
    get_reconstruction, 
    get_predictions, 
    get_notion_string, 
    principal_axes)

def get_initial_parameters(aspect, name=None):

    c = np.mean(aspect)
    try:
        r = 0.5 * (max(aspect) - min(aspect))
    except:
        traceback.print_exc()
        print('name', name)
        print(aspect)
        try:
            r = np.std(aspect)/np.sin(np.pi/4)
        except:
            r = 0.
    alpha = np.random.rand()*np.pi

    return c, r, alpha


def get_bbox_patch(aoi_bbox, linewidth=2, edgecolor='green', facecolor='none'):

    r, c, h, w, area = aoi_bbox[1:]
    C, R = int(c-w/2), int(r-h/2)
    W, H = int(w), int(h)
    aoi_bbox_patch = pylab.Rectangle((C, R), W, H, linewidth=linewidth, edgecolor=edgecolor, facecolor=facecolor)

    return aoi_bbox_patch


def get_click_patch(click, radius=5, color='green'):

    click_patch = pylab.Circle(click[::-1], radius=radius, color=color)

    return click_patch


def annotate_image(image, description, alpha=1):
    
    pylab.figure(1, figsize=(16, 9))
    pylab.axis('off')
    pylab.grid(False)
    ax = pylab.gca()
    
    pylab.imshow(image)
    #pylab.imshow(description['hierarchical_mask'], alpha=alpha)
    try:
        aoi_bbox_patch = get_bbox_patch(description['aoi_bbox_px'])
        most_likely_click_patch = get_click_patch(description['most_likely_click_px'])
        extreme_patch = get_click_patch(description['extreme_px'], color='blue')
        end_patch = get_click_patch(description['end_likely_px'], color='red')
        start_patch = get_click_patch(description['start_likely_px'], color='red')
        start_possible_patch = get_click_patch(description['start_possible_px'], color='magenta')

        ax.add_patch(aoi_bbox_patch)
        ax.add_patch(most_likely_click_patch)
        ax.add_patch(extreme_patch)
        ax.add_patch(end_patch)
        ax.add_patch(start_patch)
        ax.add_patch(start_possible_patch)
        pylab.show()
    except:
        traceback.print_exc()


def annotate_alignment(results, figsize=(8, 6)):
    fits = results['fits']
    test_angles = np.linspace(0, 2*np.pi, 360)
    k = 0
    for aspect in ['extreme_shift_mm_verticals', 'extreme_shift_mm_from_position_verticals', 'aoi_bbox_mm_heights', 'extreme_hypotetical_shift_mm_from_position_verticals', 'crystal_bbox_mm_verticals']:
    #'extreme_shift_mm_from_position_horizontals', 'aoi_bbox_mm_heights', 'aoi_bbox_mm_widths', 'crystal_bbox_mm_verticals', 'crystal_bbox_mm_areas']:
        k+=1
        if ('verticals' in aspect or 'horizontals' in aspect) and 'bbox' not in aspect:
            likely_model = goniometer().circle_model(test_angles, *fits['results'][aspect]['fit_circle'].x)
        else:
            likely_model = goniometer().projection_model(test_angles, *fits['results'][aspect]['fit_projection'].x)
        #best_model = fits['results'][aspect]['best_model'](test_angles, *fits['results'][aspect]['fit'].x)
        
        experiment_angles = fits['angles'][aspect]
        experiment_data = fits['aspects'][aspect]
        
        pylab.figure(k, figsize=figsize)
        pylab.title(aspect)
        pylab.plot(np.rad2deg(experiment_angles)%360, experiment_data, 'o', color='red', label='experiment')
        pylab.plot(np.rad2deg(test_angles)%360, likely_model, color='green', label='model')
        pylab.xlabel('Omega [deg]')
        if aspect == 'aoi_bbox_mm_heights':
            c_p, r_p, alpha_p = fits['results'][aspect]['fit'].x
            omega_max = results['omega_max']
            omega_min = results['omega_min']
            pylab.vlines([omega_max], c_p-r_p, c_p+r_p, color='magenta', label='omega_max')
            pylab.vlines([omega_min], c_p-r_p, c_p+r_p, color='orange', label='omega_min')
        pylab.legend()
    pylab.show()


class optical_alignment(experiment):
    
    specific_parameter_fields = [{'name': 'position', 'type': '', 'description': ''},
                                 {'name': 'n_angles', 'type': '', 'description': ''},
                                 {'name': 'angles', 'type': '', 'description': ''},
                                 {'name': 'zoom', 'type': '', 'description': ''},
                                 {'name': 'kappa', 'type': '', 'description': ''},
                                 {'name': 'phi', 'type': '', 'description': ''},
                                 {'name': 'calibration', 'type': '', 'description': ''},
                                 {'name': 'beam_position_vertical', 'type': '', 'description': ''},
                                 {'name': 'beam_position_horizontal', 'type': '', 'description': ''},
                                 {'name': 'frontlight', 'type': 'bool', 'description': ''},
                                 {'name': 'backlight', 'type': 'bool', 'description': ''},
                                 {'name': 'generate_report', 'type': '', 'description': ''},
                                 {'name': 'default_background', 'type': '', 'description': ''},
                                 {'name': 'save_raw_background', 'type': 'bool', 'description': ''},
                                 {'name': 'save_history', 'type': 'bool', 'description': ''},
                                 {'name': 'rightmost', 'type': 'bool', 'description': ''},
                                 {'name': 'film_step', 'type': 'bool', 'description': ''},
                                 {'name': 'verbose', 'type': 'bool', 'description': ''}]
    
    def __init__(self,
                 name_pattern,
                 directory,
                 angles=[0, 90, 225, 315],
                 scan_start_angle=None, #0
                 scan_range=None, #360,
                 n_angles=25,
                 zoom=None,
                 kappa=None,
                 phi=None,
                 position=None,
                 frontlight=False,
                 backlight=True,
                 phiy_direction=-1.,
                 phiz_direction=1.,
                 centringx_direction=-1.,
                 analysis=None,
                 conclusion=None,
                 generate_report=None,
                 default_background=False,
                 save_raw_background=False,
                 save_history=False,
                 rightmost=False,
                 move_zoom=False,
                 film_step=-120.,
                 size_of_target=0.050,
                 verbose=False,
                 parent=None,
                 debug=False):
        
        if hasattr(self, 'parameter_fields'):
            self.parameter_fields += optical_alignment.specific_parameter_fields
        else:
            self.parameter_fields = optical_alignment.specific_parameter_fields
            
        experiment.__init__(self,
                            name_pattern,
                            directory,
                            analysis=analysis,
                            conclusion=conclusion)
        
        self.description = 'Optical alignment, Proxima 2A, SOLEIL, %s' % time.ctime(self.timestamp)
        self.camera = camera()
        self.cats = cats()
        self.goniometer = goniometer()
        
        self.n_angles = n_angles
        if type(angles) == str:
            self.angles = eval(angles)
        elif type(angles) in [list, tuple]:
            self.angles = angles
        elif self.n_angles != None:
            self.angles = np.linspace(0, 360, self.n_angles + 1)[:-1]
        self.scan_range = scan_range

        if zoom is None:
            zoom = self.camera.get_zoom()
        
        self.zoom = zoom
        self.kappa = kappa
        self.phi = phi
        
        self.position = self.goniometer.check_position(position)
        self.frontlight = frontlight
        self.backlight = backlight
        self.phiy_direction = phiy_direction
        self.phiz_direction = phiz_direction
        self.centringx_direction = centringx_direction
        
        if generate_report != True:
           generate_report = False 
        self.generate_report = generate_report
        self.default_background = default_background
        self.save_raw_background = save_raw_background
        self.save_history = save_history
        self.rightmost = rightmost
        self.move_zoom = move_zoom
        self.film_step = film_step
        self.size_of_target = size_of_target
        self.verbose = verbose
        self.parent = parent
        self.debug = debug
        
        self.foreground_string = get_notion_string('foreground')
        self.crystal_loop_string = get_notion_string(['crystal', 'loop'])
        self.possible_string = get_notion_string(['crystal', 'loop', 'stem'])
        self.last_optical_alignment_results_key = 'last_optical_alignment_results'
        self.redis = redis.StrictRedis()
    
    def get_kappa(self):
        
        if self.kappa is None:
            self.kappa = self.goniometer.get_kappa_position()
        
        return self.kappa
        
    
    def get_phi(self):
        
        if self.phi is None:
            self.phi = self.goniometer.get_phi_position()
        
        return self.phi
        
    
    def get_position(self):
        
        if self.position is None:
            if os.path.isfile(self.get_parameters_filename()):
                position = self.get_parameters()['position']
            else:
                position = self.goniometer.get_aligned_position()
        else:
            position = self.position
        
        return position
        
    
    def get_beam_position_vertical(self):
        
        if os.path.isfile(self.get_parameters_filename()):
            beam_position_vertical = self.get_parameters()['beam_position_vertical']
        else:
            beam_position_vertical = self.beam_position_vertical
        
        return beam_position_vertical

    
    def get_beam_position_horizontal(self):
        
        if os.path.isfile(self.get_parameters_filename()):
            beam_position_horizontal = self.get_parameters()['beam_position_horizontal']
        else:
            beam_position_horizontal = self.beam_position_horizontal
        
        return beam_position_horizontal

    
    def get_calibration(self):
        
        if os.path.isfile(self.get_parameters_filename()):
            calibration = self.get_parameters()['calibration']
        else:
            calibration = self.camera.get_calibration()
        
        return calibration

    
    def get_zoom(self):
        
        if os.path.isfile(self.get_parameters_filename()):
            zoom = self.get_parameters()['zoom']
        else:
            zoom = self.camera.get_zoom()
        
        return zoom
    
    def check_previous_results(self):
        try:
            last_results = pickle.loads(self.redis.get(self.last_optical_alignment_results_key))
            print('last_results present')
            print(last_results)
            if str(last_results['mounted_sample_id']) == str(self.cats.get_mounted_sample_id()):
                print('mounted_sample_id is the same as the previous one, will try to make use of it')
                self.goniometer.set_position(last_results['result_position'])
        except:
            traceback.print_exc()
            print('last_results not available')
    
    def prepare(self):
        _start = time.time()
        self.check_directory(self.directory)
        #self.check_previous_results()
        
        position = self.get_position()
        if self.kappa != None:
            if abs(self.goniometer.get_kappa_position() - self.kappa) > 0.05:
                self.goniometer.set_kappa_position(self.kappa)
            position['Kappa'] = self.kappa
        if self.phi != None:
            if abs(self.goniometer.get_phi_position() - self.phi) > 0.05:
                self.goniometer.set_phi_position(self.phi)
            position['Phi'] = self.phi
        if self.get_zoom() != self.zoom:
            self.camera.set_zoom(self.zoom)
        
        self.logger.info('about to set position %s' % str(position))
        self.goniometer.set_position(position)
        self.beam_position_vertical = self.camera.get_beam_position_vertical()
        self.beam_position_horizontal = self.camera.get_beam_position_horizontal()
        self.calibration = self.get_calibration()
        if self.backlight:
            self.goniometer.insert_backlight()
        else:
            self.goniometer.extract_backlight()
        if self.frontlight:
            self.goniometer.insert_frontlight()
        else:
            self.goniometer.extract_frontlight()
        self.logger.info('prepare took %.3f seconds' % (time.time() - _start))

    
    def is_passive_advisable(self, description, threshold=0.25):
        
        return description['extreme_distance'] < threshold
    
    
    def _save_history(self, start=None, end=None):
        
        if start is None:
            start = self.start_run_time
        if end is None:
            end = self.end_run_time
        
        save_history_command = "history_saver.py -s %.2f -e %.2f -d %s -n %s &" % (
            start,
            end,
            self.directory,
            self.name_pattern)
        self.logger.info("save_history_command: %s" % save_history_command)
        os.system(save_history_command)
        
        
    def analyse_single_view(self, image=None, reference_position=None, debug=False, image_name=None, save=True):
        _start = time.time()
        
        if image is None:
            image = self.camera.get_image()
        if save:
            if image_name is None:
                image_name = '%s_%s.jpg' % (self.get_template(), time.asctime().replace(' ', '_'))
            self.camera.save_image(image_name, image=image, color=True)
        
        request_arguments = {}
        request_arguments['to_predict'] = image
        request_arguments['raw_predictions'] = False
        request_arguments['description'] = ['foreground', 'crystal', 'loop_inside', 'loop', ['crystal', 'loop'], ['crystal', 'loop', 'stem']]
        analysis = get_predictions(request_arguments)
        description = analysis['descriptions'][0]

        self.logger.info('analysis took %.2f seconds' % (time.time() - _start))

        return description
        
    
    def _align_eagerly(self, sign=-1, step=0.25, debug=None):
        
        self.start_run_time = time.time()
        self.eagerly = True
        if debug is None:
            debug = self.debug
            
        zoom = self.get_zoom()
        calibration = self.get_calibration()
        center = self.camera.get_beam_position()
        
        descriptions = []
        omega_start = self.goniometer.get_omega_position()
        
        for omega in self.get_angles():
            angle = omega_start + omega
            self.goniometer.set_omega_position(angle)
            reference_position = self.goniometer.get_aligned_position()
            image = self.camera.get_image()
            description = self.analyse_single_view(image=image, debug=debug)
            description = self.add_calibrated_data(description, center, calibration, reference_position, angle=angle)
            
            descriptions.append(description)
            
            most_likely_click = description['most_likely_click']

            if most_likely_click[0] == -1:
                reference_position['AlignmentY'] += -1 * sign * step
                self.goniometer.set_position(reference_position)
                continue

            aligned_position = description['most_likely_click_aligned_position']
            self.goniometer.set_position(aligned_position)
            
            if debug:
                self.logger.info('most_likely_click %s (fractional %s) ' % (description['most_likely_click_px'], description['most_likely_click']))
                self.logger.info('most_likely_click_shift_mm_from_position %s' % description['most_likely_click_shift_mm_from_position'])
                self.logger.info('aoi_bbox %s ' % str(description['aoi_bbox']))
                self.logger.info('aoi_bbox_px %s ' % str(description['aoi_bbox_px']))
                self.logger.info('aoi_bbox_mm %s ' % str(description['aoi_bbox_mm']))
                self.logger.info('aoi_bbox_area %s ' % description['aoi_bbox_mm'][5])
                self.logger.info('extreme %s' % description['extreme'])
                self.logger.info('extreme_px %s' % description['extreme_px'])
                self.logger.info('start point %s (possible %s)' % (description['start_likely_px'], description['start_possible_px']))
                self.logger.info('end point %s' % description['end_likely_px'])
                self.logger.info('click shift %s mm' %  description['most_likely_click_shift_mm'])
                self.logger.info('extreme_shift %s mm' % description['extreme_shift_mm'])
                self.logger.info('extreme distance from beam %.3f mm' % description['extreme_distance_mm'])
                annotate_image(image, description)
        self.end_run_time = time.time()
        return descriptions
            
    
    def _align_carefully(self):
        
        _start = time.time()
        self.eagerly = False
        zoom = self.get_zoom()
        calibration = self.get_calibration()
        center = self.camera.get_beam_position()
        reference_position = self.goniometer.get_aligned_position()
        
        history = self.camera.get_history(self.history_start, self.history_end)
        self.images = history[1]
        self.state_vectors = history[2]
        self.timestamps = history[0]
        
        request_arguments = {}
        request_arguments['to_predict'] = self.images
        request_arguments['raw_predictions'] = False
        request_arguments['description'] = ['foreground', 'crystal', 'loop_inside', 'loop', ['crystal', 'loop'], ['crystal', 'loop', 'stem']]
        analysis = get_predictions(request_arguments)
        
        descriptions = []
        for description, angle in zip(analysis['descriptions'], self.state_vectors[:, 0]):
            description = self.add_calibrated_data(description, center, calibration, reference_position, angle=angle)
            descriptions.append(description)
            
        self.logger.info('analysis took %.2f seconds' % (time.time() - _start))
        
        return descriptions
    
    
    def get_fits(self, descriptions, minimize_method='nelder-mead', keys=['most_likely_click', 'extreme', 'start_likely', 'end_likely', 'start_possible', 'aoi_bbox_mm', 'crystal_bbox_mm'], subkeys=['', 'px', 'shift_mm', 'shift_mm_from_position'], point_subkeys=['verticals', 'horizontals'], bbox_subkeys=['verticals', 'horizontals', 'heights', 'widths', 'areas'], fits=None):
        
        description_keys = {}
        extended_keys = []
        
        for key in keys:
            if 'bbox' not in key:
                for subkey in subkeys:
                    description_key = '%s_%s' % (key, subkey) if subkey != '' else key
                    for k, subsubkey in enumerate(point_subkeys):
                        extended_key = '%s_%s' % (description_key, subsubkey)
                        extended_keys.append(extended_key)
                        description_keys[extended_key] = {'description_key': description_key, 'index': k}
            else:
                description_key = key
                for k, subkey in enumerate(bbox_subkeys):
                    extended_key = '%s_%s' % (description_key, subkey)
                    extended_keys.append(extended_key)
                    description_keys[extended_key] = {'description_key': description_key, 'index': k+1}
        
        fit_start = time.time()
        if fits is None:
            fits = {'angles': {}, 'aspects': {}, 'results': {}}
        
        for key in extended_keys:
            fits['angles'][key] = []
            fits['aspects'][key] = []
                    
        for description in descriptions:
            angle = description['angle']
            for key in extended_keys:
                description_key = description_keys[key]['description_key']
                i = description_keys[key]['index']
                fits['angles'][key].append(np.deg2rad(angle))
                fits['aspects'][key].append(description[description_key][i])
        
        for aspect in fits['aspects']:
            try:
                fits['results'][aspect] = self.fit_aspect(fits['angles'][aspect], fits['aspects'][aspect], aspect_name=aspect, minimize_method=minimize_method)
            except:
                print('could not make sense of aspect %s' % aspect)
                logging.info('could not make sense of aspect %s' % aspect)
                
                print(traceback.print_exc())
                logging.info(traceback.format_exc())
                fits['results'][aspect] = None
                
        fit_end = time.time()
        self.logger.info('Fits took %.3f seconds' % (fit_end - fit_start))
        
        return fits
    
    
    def get_omega_max_and_min(self, fits):
    
        try:
            c_p, r_p, alpha_p = fits['results']['aoi_bbox_mm_heights']['fit_projection'].x
            self.logger.info('c_p, r_p, alpha_p %s' % str((c_p, r_p, alpha_p)))
            omega_max = alpha_p/2
            omega_min = alpha_p/2 + np.pi/2
            if r_p<0:
                omega_max, omega_min = omega_min, omega_max
        except:
            omega_max = fits['angles']['aoi_bbox_mm_heights'][np.argmax(fits['aspects']['aoi_bbox_mm_heights'])]
            omega_min = omega_max + np.pi/2
            self.logger.info(traceback.format_exc())
        
        omega_max = np.rad2deg(divmod(omega_max, np.pi)[1])
        omega_min = np.rad2deg(divmod(omega_min, np.pi)[1])
        
        return omega_max, omega_min
        
        
    def get_omega_max_omega_min_height_max_height_min_and_width(self, fits):
        
        try:
            c_p, r_p, alpha_p = fits['results']['aoi_bbox_mm_heights']['fit_projection'].x
            self.logger.info('c_p, r_p, alpha_p %s' % str((c_p, r_p, alpha_p)))
            height_max = c_p + np.abs(r_p)
            height_min = c_p - np.abs(r_p)
            width = fits['results']['aoi_bbox_mm_heights']['fit_projection'].x[0]
            omega_max = alpha_p/2
            omega_min = alpha_p/2 + np.pi/2
            if r_p<0:
                omega_max, omega_min = omega_min, omega_max
        except:
            omega_max = fits['angles']['aoi_bbox_mm_heights'][np.argmax(fits['aspects']['aoi_bbox_mm_heights'])]
            omega_min = omega_max + np.pi/2
            height_max = max(fits['aspects']['aoi_bbox_mm_heights'])
            height_min = min(fits['aspects']['aoi_bbox_mm_heights'])
            width = np.median(fits['aspects']['aoi_bbox_mm_widths'])
            self.logger.info(traceback.format_exc())
        
        omega_max = np.rad2deg(divmod(omega_max, np.pi)[1])
        omega_min = np.rad2deg(divmod(omega_min, np.pi)[1])
        
        return omega_max, omega_min, height_max, height_min, width
    
    
    def get_optimum_zoom(self, height=None, width=None, margin_factor=1.5):

        if height is None:
            height = self.results['height_max_mm']
        if width is None:
            width = self.results['width_mm']
        
        raster = np.array([height, width]) * margin_factor
        self.logger.info('raster %s' % str(raster))
        
        view_shape = self.camera.get_calibration() * np.array(self.camera.shape[:2])
        current_zoom = self.get_zoom()
        magnifications = self.camera.magnifications
        possible_increase =  np.min(view_shape/raster) * magnifications[current_zoom-1] / magnifications - 1
        
        try: 
            optimum_zoom = np.argmin(possible_increase[possible_increase>=0]) + 1
        except:
            optimum_zoom = current_zoom
        
        if self.verbose:
            self.logger.info('possible increase %s' % str(possible_increase))
            self.logger.info('optimumx zoom: -z %d' % (optimum_zoom))
        
        return optimum_zoom
    
    
    def make_sense_of_descriptions(self, descriptions=None, reference_position=None, eagerly=None, debug=None):
        
        _start = time.time()
        if descriptions is None:
            descriptions = self.descriptions
            
        if reference_position is None:
            reference_position = self.goniometer.get_aligned_position()
        
        if eagerly is None:
            eagerly = self.eagerly
            
        if debug is None:
            debug = self.debug
        
        self.logger.info('reference_position %s' % reference_position)
        
        results = {'reference_position': reference_position, 'mounted_sample_id': self.cats.get_mounted_sample_id()}
        
        fits = self.get_fits(descriptions)
        
        #omega_max, omega_min = self.get_omega_max_and_min(fits)
        omega_max, omega_min, height_max, height_min, width = self.get_omega_max_omega_min_height_max_height_min_and_width(fits)
        
        reference_position['Omega'] = omega_max
        results['omega_max'] = omega_max
        results['omega_min'] = omega_min
        results['height_max_mm'] = height_max
        results['height_min_mm'] = height_min
        results['width_mm'] = width
        results['optimum_zoom'] = self.get_optimum_zoom(height_max, width)
        results['calibration'] = self.get_calibration()
        results['original_image_shape'] = self.camera.shape
        
        if eagerly:
            result_position = copy.copy(reference_position)
            result_position['AlignmentY'] = np.median([d['most_likely_click_aligned_position']['AlignmentY'] for d in descriptions[1:]])
        else:
            fit_vertical = fits['results']['most_likely_click_shift_mm_verticals']['fit_circle']
            fit_horizontal = fits['results']['most_likely_click_shift_mm_horizontals']['fit_circle']
            result_position = self.goniometer.get_aligned_position_from_fit_and_reference(fit_vertical, fit_horizontal, reference_position)
            result_position['AlignmentZ'] = fits['results']['extreme_shift_mm_verticals']['fit_circle'].x[0]
            
        for description in descriptions:
            description = self.add_hypotetical_data(description, result_position)
            
        #results['fits'] = fits
        hypotetical_fits = self.get_fits(descriptions, keys=['most_likely_click', 'extreme', 'start_likely', 'end_likely', 'start_possible'], subkeys=['hypotetical_shift_mm_from_position'], point_subkeys=['verticals', 'horizontals'], fits=fits)
        
        if eagerly:
            aligned_positions = self.get_aligned_positions(hypotetical_fits, reference_position, result_position=result_position, eagerly=eagerly)
        else:
            aligned_positions = self.get_aligned_positions(hypotetical_fits, reference_position)
            result_position = aligned_positions['most_likely_click']
            volume = self.get_volume(descriptions, results, fits)

        results['result_position'] = result_position
        results['aligned_positions'] = aligned_positions
        
        self.logger.info('resulting AlignmentZ %.3f' % result_position['AlignmentZ'])
        self.logger.info('resulting Omega max %.3f' % result_position['Omega'])
        self.logger.info('aoi max height %.3f' % results['height_max_mm'])
        self.logger.info('aoi width %.3f' % results['width_mm'])
        
        _end = time.time()
        self.logger.info('making sense of %d views took %.2f seconds' % (len(descriptions), _end-_start))
        
        if debug:
            
            try:
                fit_vertical_shift = fits['results']['extreme_shift_mm_verticals']['fit_circle']
                fit_vertical = fits['results']['extreme_shift_mm_from_position_verticals']['fit_circle']
                fit_vertical_hypotetical_shift = fits['results']['extreme_hypotetical_shift_mm_from_position_verticals']['fit_circle']
                fit_horizontal_shift = fits['results']['extreme_shift_mm_horizontals']['fit_circle']
                fit_horizontal = fits['results']['extreme_shift_mm_from_position_horizontals']['fit_circle']
                fit_horizontal_hypotetical_shift = fits['results']['extreme_hypotetical_shift_mm_from_position_horizontals']['fit_circle']

                self.logger.info('vertical from shift mm %s' % str(fit_vertical_shift.x))
                self.logger.info('vertical from positions %s' % str(fit_vertical.x))
                self.logger.info('vertical from hypotetical %s' % str(fit_vertical_hypotetical_shift.x))
                self.logger.info('horizontal from shift %s' % str(fit_horizontal_shift.x))
                self.logger.info('horizontal from position %s' % str(fit_horizontal.x))
                self.logger.info('horizontal from hypotetical %s' % str(fit_horizontal_hypotetical_shift.x))
            except:
                self.logger.info(traceback.format_exc())
            annotate_alignment(results)
        
        self.results = results
    
        return self.results
    
    
    def get_aligned_positions(self, fits, reference_position, points=['extreme', 'start_likely', 'end_likely', 'start_possible', 'most_likely_click'], result_position=None, eagerly=False):
        
        aligned_positions = {}
        for point in points:
            vertical_key = '%s_shift_mm_verticals' % point
            horizontal_key = '%s_shift_mm_horizontals' % point
            fit_vertical = fits['results'][vertical_key]['fit_circle']
            fit_horizontal = fits['results'][horizontal_key]['fit_circle']
            aligned_position = self.goniometer.get_aligned_position_from_fit_and_reference(fit_vertical, fit_horizontal, reference_position)
            aligned_positions[point] = aligned_position
        
        for point in points:
            vertical_key = '%s_hypotetical_shift_mm_from_position_verticals' % point
            horizontal_key = '%s_hypotetical_shift_mm_from_position_horizontals' % point
            fit_vertical = fits['results'][vertical_key]['fit_circle']
            fit_horizontal = fits['results'][horizontal_key]['fit_circle']
            aligned_position = self.goniometer.get_aligned_position_from_fit_and_reference(fit_vertical, fit_horizontal, reference_position)
            aligned_positions['%s_hypotetical' % point] = aligned_position
            
        if eagerly:
            print('result_position', result_position)
            print('most_likely_click', aligned_positions['most_likely_click'])
                  
            shift = get_vector_from_position(result_position) - get_vector_from_position(aligned_positions['most_likely_click'])
            print('shift', shift)
            for point in points:
                shifted_point = get_position_from_vector(get_vector_from_position(aligned_positions[point]) + shift)
                aligned_positions[point] = shifted_point
            
        return aligned_positions
        
    def get_projections(self, descriptions, notion='foreground'):
        notion_string = get_notion_string(notion)
        detector_rows, detector_cols = descriptions[0][notion_string]['notion_mask'].shape
        
        number_of_projections = len(descriptions)
        
        projections = np.zeros((detector_cols, number_of_projections, detector_rows))
        valid_angles = []
        valid_index = 0
        print('%d projections' % len(descriptions))
        for description in descriptions:
            if description['present']:
                projections[:, valid_index, :] = description[notion_string]['notion_mask'].T
                valid_angles.append(np.deg2rad(description['angle']))
                valid_index += 1
        projections = projections[:, :valid_index, :]
        
        return projections, valid_angles
    
    def get_volume(self, descriptions, results, fits, notion='foreground', detector_col_spacing=1, detector_row_spacing=1):
        
        _total_start = time.time()
        reference_position = results['reference_position']
        voxel_calibration = get_voxel_calibration(*results['calibration'])
        original_image_shape = results['original_image_shape']
    
        projections, valid_angles = self.get_projections(descriptions, notion=notion)

        ps = projections.shape
        detector_rows, detector_cols = ps[2], ps[0]
        
        center_of_mass = ndi.center_of_mass(projections)
        print('projections shape, center_of_mass', projections.shape, center_of_mass, projections.max(), projections.mean())

        px_axis_vertical_position = fits['results']['extreme_px_verticals']['fit'].x[0] * detector_rows /original_image_shape[0]
        print('estimated axis vertical position %.3f' % px_axis_vertical_position)
        vertical_correction = detector_rows/2 - px_axis_vertical_position
        
        _start = time.time()
        request = { 'projections': projections,
                    'angles': valid_angles,
                    'detector_rows': detector_rows,
                    'detector_cols': detector_cols,
                    'detector_col_spacing': detector_col_spacing,
                    'detector_row_spacing': detector_row_spacing,
                    'vertical_correction': vertical_correction}
        
        reconstruction = get_reconstruction(request, verbose=True)
        _end = time.time()
        print('reconstruction done in %.3f seconds (%.4f from start)' % (_end-_start, _end-_total_start))
        
        _start = time.time()
        print('reconstruction', reconstruction.shape, reconstruction.max(), reconstruction.mean())
        objectpoints = np.argwhere(reconstruction>0.95*reconstruction.max()) 
        print('#objectpoints', len(objectpoints))
        print('objectpoints.shape', objectpoints.shape)
        pcd_px = self.save_points_as_pcd(objectpoints, '%s_px.pcd' % self.get_template())
        
        cols_ratio = original_image_shape[1]/detector_cols
        rows_ratio = original_image_shape[0]/detector_rows
        
        voxel_calibration[0] *= cols_ratio
        voxel_calibration[1:] *= rows_ratio
        
        origin = get_vector_from_position(reference_position, keys=['CentringX', 'CentringY', 'AlignmentY'])
        print('origin', origin)
        center = np.array([detector_cols/2, detector_rows, detector_rows])
        print('center', center)

        objectpoints_mm = get_points_in_goniometer_frame(objectpoints, voxel_calibration, origin[:3], center=center, directions=np.array([-1, -1, 1]))
        
        print('objectpoints_mm median')
        print(np.median(objectpoints_mm, axis=0))
        
        pca3d = principal_axes(objectpoints_mm, verbose=True) # inertia, eigenvalues, eigenvectors, center
        
        pcd_mm = self.save_points_as_pcd(objectpoints_mm, '%s_mm.pcd' % self.get_template())
        
        _end = time.time()
        print('3d coordinates calculated in %.3f seconds (%.4f from start)' % (_end-_start, _end-_total_start))

        return pcd_mm

    def save_points_as_pcd(self, points, filename):
        _start = time.time()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.estimate_normals()
        o3d.io.write_point_cloud(filename, pcd)
        print('point cloud %s save took %.4f seconds' % (filename, time.time() - _start))
        return pcd 
    
    def add_calibrated_data(self, description, center, calibration, reference_position, angle=None):
        original_shape = description['original_shape']
        description['original_shape'] = original_shape
        prediction_shape = description['prediction_shape']
        scale = original_shape/prediction_shape
        description['scale'] = scale
        
        if angle is None:
            omega = reference_position['Omega']
        else:
            omega = angle
            reference_position['Omega'] = omega
        
        description['angle'] = omega
        description['reference_position'] = reference_position
        
        for bbox_key in ['aoi_bbox', 'crystal_bbox']:
            bbox_px = list(description['aoi_bbox'][:])
            bbox_px[1] *= original_shape[0]
            bbox_px[2] *= original_shape[1]
            bbox_px[3] *= original_shape[0]
            bbox_px[4] *= original_shape[1]
            bbox_px.append(bbox_px[3] * bbox_px[4])
            description['%s_px' % bbox_key] = bbox_px
            
            bbox_mm = list(bbox_px[:])
            bbox_mm[1] = (bbox_px[1] - center[0]) * calibration[0]
            bbox_mm[2] = (bbox_px[2] - center[1]) * calibration[1]
            bbox_mm[3] *= calibration[0]
            bbox_mm[4] *= calibration[1]
            bbox_mm[5] = bbox_mm[3] * bbox_mm[4]
            description['%s_mm' % bbox_key] = bbox_mm
        
        def add_point_calibrated_data(description, key, reference_position):
            px = description[key]
            shift_mm = [np.nan, np.nan]
            distance = np.nan
            aligned_position = None
            shift_mm_from_position = [np.nan, np.nan]
            if description[key][0] >= 0:
                px *= original_shape
                shift_mm = (px - center) * calibration
                distance = np.linalg.norm(shift_mm)
                aligned_position = self.goniometer.get_aligned_position_from_reference_position_and_shift(reference_position, shift_mm[1], shift_mm[0])
                shift_mm_from_position = self.goniometer.get_vertical_and_horizontal_shift_between_two_positions(aligned_position, reference_position)
            
            description['%s_px' % key] = px
            description['%s_shift_mm' % key] = shift_mm
            description['%s_distance_mm' % key] = distance
            description['%s_aligned_position' % key] = aligned_position
            description['%s_shift_mm_from_position' % key] = shift_mm_from_position
            
            return description
        
        for key in ['most_likely_click', 'extreme', 'end_likely', 'start_likely', 'start_possible']:
            description = add_point_calibrated_data(description, key, reference_position)
                
        return description
    
    
    def add_hypotetical_data(self, description, hypotetical_reference_position):
        
        for key in ['most_likely_click', 'extreme', 'end_likely', 'start_likely', 'start_possible']:
            aligned_position = description['%s_aligned_position' % key]
            try:
                shift_mm_from_position = self.goniometer.get_vertical_and_horizontal_shift_between_two_positions(aligned_position, hypotetical_reference_position)
            except:
                print('problem in add_hypotetical_data')
                print('aligned_position', aligned_position)
                print('reference_position', hypotetical_reference_position)
                shift_mm_from_position = np.array([np.nan, np.nan])
            description['%s_hypotetical_shift_mm_from_position' % key] = shift_mm_from_position
        
        return description
        
    
    def fit_aspect(self, angles, aspect,aspect_name=None, minimize_method='nelder-mead', debug=False):
        
        initial_parameters = get_initial_parameters(aspect, name=aspect_name)
                                                    
        fit_circle = minimize(self.goniometer.circle_model_residual, 
                                initial_parameters, 
                                method=minimize_method, 
                                args=(angles, aspect))
        
        fit_projection = minimize(self.goniometer.projection_model_residual,
                                initial_parameters, 
                                method=minimize_method, 
                                args=(angles, aspect))
        
        if debug:
            print('aspect', aspect, 'initial_parameters', initial_parameters, 'optimized_parameters circle, projection', fit_circle.x, fit_projection.x)
        
        fit, k = select_better_model(fit_circle, fit_projection)
        
        result = {'fit_circle': fit_circle, 
                'fit_projection': fit_projection, 
                'fit': fit, 
                'k': k}
        
        if k == 1:
            result['best_model'] = self.goniometer.circle_model
        else:
            result['best_model'] = self.goniometer.projection_model
            
        return result

    
    def run(self):
        
        _start = time.time()
        angles = self.get_angles()
        if len(angles) > 10 or self.scan_range != None:
            f = film(self.name_pattern, self.directory, scan_range=self.scan_range)
            self.history_start, self.history_end = f.run(step=self.film_step)
            descriptions = self._align_carefully()
        else:
            descriptions = self._align_eagerly()
        
        self.descriptions = descriptions    
        self.logger.info('run took %.3f seconds' % (time.time() - _start))
        
    
    def save_optical_history(self):
        self._save_history()
        
        
    def clean(self):
        super().clean()
        
        
    def analyze(self):
        
        self.make_sense_of_descriptions()
        
        
    def conclude(self):
        
        if self.rightmost:
            result_position = self.results['aligned_positions']['extreme']
        else:
            result_position = self.results['result_position']
        self.goniometer.set_position(result_position)
                             
        if self.move_zoom == True:
            self.camera.set_zoom(self.results['optimum_zoom'])
            #result_position['Zoom'] = self.camera.zoom_motor_positions[self.analysis_results['max_raster_parameters'][-1]]
            #self.camera.set_zoom(self.analysis_results['max_raster_parameters'][-1])
        self.goniometer.save_position()
        self.redis.set(self.last_optical_alignment_results_key, pickle.dumps(self.results))
        self.conclusion_end_time = time.time()
      
    
def main():
    import optparse
    
    parser = optparse.OptionParser()
    parser.add_option('-n', '--name_pattern', default="autocenter_%s_%s" % (os.getuid(), time.asctime().replace(" ", "_")), type=str, help='Prefix default=%default')
    parser.add_option('-d', '--directory', default="%s/manual_optical_alignment" % os.getenv("HOME"), type=str, help='Destination directory default=%default')
    parser.add_option('-g', '--n_angles', default=24, type=int, help='Number of equidistant angles to collect at. Takes precedence over angles parameter if specified')
    parser.add_option('-a', '--angles', default='(0, 0, 90, 180, 225, 315)', type=str, help='Specific angles to collect at')
    parser.add_option('-f', '--frontlight', action='store_true', help='Use frontlight')
    parser.add_option('-b', '--backlight', action='store_true', help='Use backlight')
    parser.add_option('-r', '--scan_range', default=360, type=float, help='Range of angles')
    parser.add_option('-z', '--zoom', default=None, type=int, help='Zoom')
    parser.add_option('-p', '--position', default=None, type=str, help='Position')
    parser.add_option('-K', '--kappa', default=None, type=float, help='Kappa orientation')
    parser.add_option('-P', '--phi', default=None, type=float, help='Phi orientation')
    parser.add_option('-A', '--analysis', action='store_true', help='If set will perform automatic analysis.')
    parser.add_option('-C', '--conclusion', action='store_true', help='If set will move the motors upon analysis.')
    parser.add_option('-R', '--generate_report', action='store_true', help='If set will generate report.')
    parser.add_option('-B', '--default_background', action='store_true', help='If set will try to use lookup background image.')
    parser.add_option('--rightmost', action='store_true', help='Go for rightmost point.')
    parser.add_option('--move_zoom', action='store_true', help='If set will change zoom to the one corresponding to the biggest still containing the whole loop.')
    parser.add_option('--save_history', action='store_true', help='If set will save raw images.')
    parser.add_option('-F', '--film_step', default=-120., type=float, help='Film step')
    parser.add_option('-S', '--size_of_target', default=0.05, type=float, help='Size of target at the end of the sample (e.g. loop)')
    
    options, args = parser.parse_args()
    
    print('options', options)
    print('args', args)
    
    if options.scan_range <= 0.:
        options.scan_range = None
        eagerly = True
    else:
        eagerly = False
    oa = optical_alignment(**vars(options))
    
    filename = '%s_parameters.pickle' % oa.get_template()
    
    print('filename %s' % filename)
    
    if not os.path.isfile(filename):
        print('filename %s not found executing' % filename)
        oa.execute()
        #oa.analyse_single_view(debug=True)
        #desc_filename = '/tmp/descriptions.pickle'
        
        #descriptions = oa.run()
        #descriptions = oa._align_eagerly(debug=True)
        #descriptions = oa._align_carefully()
        
        #if os.path.isfile(desc_filename):
            #descriptions = pickle.load(open(desc_filename, 'rb'))
        #else:
            #descriptions = oa._align_eagerly(debug=False)
            #f = open(desc_filename, 'wb')
            #pickle.dump(descriptions, f)
            #f.close()
        #oa.make_sense_of_descriptions(descriptions, eagerly=eagerly, debug=True)
    elif options.analysis == True:
        oa.analyze()
        if options.conclusion == True:
            oa.conclude()
        

if __name__ == '__main__':
    main()

