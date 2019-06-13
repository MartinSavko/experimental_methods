#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
optical alignement procedure
'''
import gevent

import traceback
import logging
import time
import pickle
import os
import h5py

from experiment import experiment
from goniometer import goniometer
from camera import camera
from film import film

from skimage.io import imread

try:
    import lucid2
except ImportError:
    pass

import numpy as np
try:
    from scipy.optimize import leastsq, minimize
except ImportError:
    pass

from math import cos, sin, sqrt, radians, atan, asin, acos, pi, degrees
from optical_path_report import optical_path_analysis

def projection_model(angles, c, r, alpha):
    return c + r*np.cos(2*angles - alpha)
    
def circle_model(angles, c, r, alpha):
    return c + r*np.cos(angles - alpha)
    
class optical_alignment(experiment):
    
    specific_parameter_fields = [{'name': 'position', 'type': '', 'description': ''},
                                 {'name': 'n_angles', 'type': '', 'description': ''},
                                 {'name': 'angles', 'type': '', 'description': ''},
                                 {'name': 'zoom', 'type': '', 'description': ''},
                                 {'name': 'kappa', 'type': '', 'description': ''},
                                 {'name': 'phi', 'type': '', 'description': ''},
                                 {'name': 'image_template', 'type': '', 'description': ''}, 
                                 {'name': 'calibration', 'type': '', 'description': ''},
                                 {'name': 'beam_position_vertical', 'type': '', 'description': ''},
                                 {'name': 'beam_position_horizontal', 'type': '', 'description': ''},
                                 {'name': 'frontlight', 'type': 'bool', 'description': ''},
                                 {'name': 'backlight', 'type': 'bool', 'description': ''},
                                 {'name': 'background', 'type': '', 'description': ''},
                                 {'name': 'generate_report', 'type': '', 'description': ''},
                                 {'name': 'default_background', 'type': '', 'description': ''},
                                 {'name': 'save_raw_background', 'type': 'bool', 'description': ''},
                                 {'name': 'save_raw_images', 'type': 'bool', 'description': ''},
                                 {'name': 'rightmost', 'type': 'bool', 'description': ''},
                                 {'name': 'film_step', 'type': 'bool', 'description': ''},
                                 {'name': 'verbose', 'type': 'bool', 'description': ''}]
    
    def __init__(self,
                 name_pattern,
                 directory,
                 n_angles=25,
                 angles=[0, 90, 180, 270],
                 scan_start_angle=0,
                 scan_range=360,
                 zoom=None,
                 kappa=None,
                 phi=None,
                 position=None,
                 frontlight=None,
                 backlight=None,
                 background=True,
                 phiy_direction=-1.,
                 phiz_direction=1.,
                 centringx_direction=-1.,
                 analysis=None,
                 conclusion=None,
                 generate_report=None,
                 default_background=False,
                 save_raw_background=False,
                 save_raw_images=False,
                 rightmost=False,
                 move_zoom=False,
                 film_step=-60.,
                 size_of_target=0.050,
                 verbose=False,
                 parent=None):
        
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
        self.goniometer = goniometer()
        
        self.n_angles = n_angles
        if type(angles) == str:
            self.angles = eval(angles)
        if self.n_angles != None:
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
        self.background = background
        self.phiy_direction = phiy_direction
        self.phiz_direction = phiz_direction
        self.centringx_direction = centringx_direction
        
        self.image_template = os.path.join(self.directory, self.name_pattern) + '_%.2f.png'
        self.results_lucid = None
        if generate_report != True:
           generate_report = False 
        self.generate_report = generate_report
        self.default_background = default_background
        self.save_raw_background = save_raw_background
        self.save_raw_images = save_raw_images
        self.rightmost = rightmost
        self.move_zoom = move_zoom
        self.film_step = film_step
        self.size_of_target = size_of_target
        self.verbose = verbose
        self.parent = parent

    def get_kappa(self):
        if self.kappa is None:
            self.kappa = self.goniometer.get_kappa_position()
        return self.kappa
        
  
    def get_phi(self):
        if self.phi is None:
            self.phi = self.goniometer.get_phi_position()
        return self.phi
        

    def get_position(self):
        if self.position == None:
            if os.path.isfile(self.get_parameters_filename()):
                position = pickle.load(open(self.get_parameters_filename()))['position']
            else:
                position = self.goniometer.get_aligned_position()
        else:
            position = self.position
        return position
        

    def get_beam_position_vertical(self):
        if os.path.isfile(self.get_parameters_filename()):
            beam_position_vertical = pickle.load(open(self.get_parameters_filename()))['beam_position_vertical']
        else:
            beam_position_vertical = self.beam_position_vertical
        return beam_position_vertical
    

    def get_beam_position_horizontal(self):
        if os.path.isfile(self.get_parameters_filename()):
            beam_position_horizontal = pickle.load(open(self.get_parameters_filename()))['beam_position_horizontal']
        else:
            beam_position_horizontal = self.beam_position_horizontal
        return beam_position_horizontal
    

    def get_calibration(self):
        if os.path.isfile(self.get_parameters_filename()):
            calibration = pickle.load(open(self.get_parameters_filename()))['calibration']
        else:
            calibration = self.calibration
        return calibration


    def get_background_image(self):
        background_filename = self.get_background_filename()
        if hasattr(self, 'background_image'):
            background_image = self.background_image
        elif os.path.isfile(background_filename):
            if '.png' in background_filename:
                background_image = imread(background_filename)
            elif '.pickle' in background_filename:
                background_image = pickle.load(open(background_filename))
            else:
                background_image = h5py.File(self.get_images_filename())['background'].value
        else:
            alignmentzposition = self.goniometer.get_alignmentz_position()
            self.goniometer.set_position({'AlignmentZ': alignmentzposition + 2})
            background_image = self.camera.get_rgbimage() #self.save_optical_image(background=True)
            self.goniometer.set_position({'AlignmentZ': alignmentzposition})

        return background_image
    
    
    def save_optical_image(self, angle=0, background=False):
        if background == True:
            imagename = os.path.join(self.directory, self.name_pattern) + '_background.png'
        else:
            imagename = self.image_template % angle

        imagename, image, image_id = self.camera.save_image(imagename)

        return imagename, image, image_id
    
    
    def prepare(self):
        _start = time.time()
        self.check_directory(self.directory)
        if self.kappa != None:
            if abs(self.goniometer.get_kappa_position() - self.kappa) > 0.05:
                self.goniometer.set_kappa_position(self.kappa)
        
        if self.phi != None:
            if abs(self.goniometer.get_phi_position() - self.phi) > 0.05:
                self.goniometer.set_phi_position(self.phi)
        self.camera.set_zoom(self.zoom)
        self.goniometer.set_position(self.get_position())
        self.beam_position_vertical = self.camera.get_beam_position_vertical()
        self.beam_position_horizontal = self.camera.get_beam_position_horizontal()
        self.calibration = self.camera.calibrations[int(self.zoom)] #self.camera.get_calibration()
        self.goniometer.insert_backlight()
        self.goniometer.set_omega_position(0.2)
        if self.frontlight:
            self.goniometer.insert_frontlight()
        else:
            self.goniometer.extract_frontlight()
        if self.background == True:
            self.background_image = self.get_background_image()
        print 'prepare took %.3f seconds' % (time.time() - _start)

    def run(self):
        _start = time.time()
        self.results = []
        self.images = []
        angles = self.get_angles()
        if len(angles) > 10 or self.scan_range != None:
            f = film(self.name_pattern, self.directory, scan_range=self.scan_range)
            f.run(step=self.film_step)
            self.images = []
            measured_angles = np.array([item[1] for item in f.images])
            added = []
            for angle in self.angles:
                closest = np.argmin(np.abs(angle - measured_angles))
                
                if f.images[closest][0] not in added:
                    self.images.append(f.images[closest])
                    added.append(f.images[closest][0])
                    
            #self.images = f.images
        else:
            for angle in self.get_angles():
                print 'angle %.2f' % angle
                self.goniometer.set_omega_position(angle)
                imagename, image, image_id = self.save_optical_image(angle)
                self.images.append([image_id, angle, image])
                try:
                    info, x, y = lucid2.find_loop(imagename)
                    print '%s %d %d' % (info, x, y)
                    if info == 'Coord':
                        self.results.append([angle, x, y])
                except:
                    print traceback.print_exc()
            
        print 'run took %.3f seconds' % (time.time() - _start)
        
    def get_images_filename(self):
        template = self.get_template()
        images_filename = '%s_images.h5' % template
        return images_filename

    def get_images(self):
        images_filename = self.get_images_filename()
        if not hasattr(self, 'images'):
            m = h5py.File(images_filename)
            raw_images = m['images'].value
            images = [item.mean(axis=-1) for item in raw_images]
        else:
            images = [item[2].mean(axis=-1) for item in self.images]
        return images
    

    def get_omegas(self):
        images_filename = self.get_images_filename()
        if not hasattr(self, 'images'):
            m = h5py.File(images_filename)
            omegas = m['omegas']
        else:
            omegas = [item[1] for item in self.images]
            
        return omegas

    def get_size_of_target(self):
        return self.size_of_target
    
    def analyze(self):
        _start = time.time()
        images = self.get_images()
        omegas = self.get_omegas()
        calibration = self.get_calibration()
        background_image = self.get_background_image()
        if background_image.shape[-1] == 3:
            background_image = background_image.mean(axis=-1)
            
        _end = time.time()
        print 'loading of images took %.3f' % (_end - _start)
        
        fits = optical_path_analysis(images, omegas, calibration, background_image=background_image, smoothing_factor=self.get_size_of_target(), template=self.get_template(), generate_report=self.generate_report)
        
        ((c, r, alpha), k) = fits['centroid_vertical'][0].x, fits['centroid_vertical'][1]
        
        ((c_rightmost, r_rightmost, alpha_rightmost), k_rightmost) = fits['rightmost_vertical'][0].x, fits['rightmost_vertical'][1]
        
        ((c_horizontal, r_horizontal, alpha_horizontal), k_horizontal) = fits['centroid_horizontal'][0].x, fits['centroid_horizontal'][1]
        
        ((c_horizontal_rightmost, r_horizontal_rightmost, alpha_horizontal_rightmost), k_horizontal_rightmost) = fits['rightmost_horizontal'][0].x, fits['rightmost_horizontal'][1]
        
        ((c_width, r_width, alpha_width), k_width) = fits['width'][0].x, fits['width'][1]
        ((c_height, r_height, alpha_height), k_height) = fits['height'][0].x, fits['height'][1]
                
        height_max = c_height + abs(r_height)
        height_min = c_height - abs(r_height)
        if self.verbose:
            print 'height_max', height_max
            print 'height_min', height_min
            print 'k_height', k_height
            
        test_angles = np.radians(np.linspace(0,360,1000))
        angle_height_extreme = alpha_height/k_height
        
        if k_height == 1:
            test_heights = circle_model(test_angles, c_height, r_height, alpha_height)
            special_height = circle_model(angle_height_extreme, c_height, r_height, alpha_height)
        else:
            test_heights = projection_model(test_angles, c_height, r_height, alpha_height)
            special_height = projection_model(angle_height_extreme, c_height, r_height, alpha_height)
        if abs(special_height - test_heights.max()) < abs(special_height - test_heights.min()):
            angle_height_max = np.degrees(angle_height_extreme)
        else:
            angle_height_max = np.degrees(angle_height_extreme) + 90.

        angle_height_min = angle_height_max + 90.
        if self.verbose:
            print 'angle_height_max', angle_height_max
            print 'angle_height_min', angle_height_min
            
        width_max = c_width + abs(r_width)
        width_min = c_width - abs(r_width)
        if self.verbose:
            print 'width_max', width_max
            print 'width_min', width_min
            
        if self.rightmost == True:
            c, r, alpha, k = c_rightmost, r_rightmost, alpha_rightmost, k_rightmost
            c_horizontal, r_horizontal, alpha_horizontal, k_horizontal = c_horizontal_rightmost, r_horizontal_rightmost, alpha_horizontal_rightmost, k_horizontal_rightmost
            
        d_sampx = self.centringx_direction * r * sin(alpha)
        d_sampy = r * cos(alpha)
        if self.verbose:
            print 'd_sampx', d_sampx
            print 'd_sampy', d_sampy
            
        vertical_center = c
        horizontal_center = c_horizontal
        if self.verbose:
            print 'vertical_center', vertical_center
            print 'horizontal_center', horizontal_center
        
        beam_position_vertical = self.get_beam_position_vertical()
        beam_position_horizontal = self.get_beam_position_horizontal()
        if self.verbose:
            print 'beam_center_vertical', beam_position_vertical 
            print 'beam_center_horizontal', beam_position_horizontal
        
        d_y = self.phiy_direction * (horizontal_center - beam_position_horizontal)
        d_z = self.phiz_direction * (vertical_center - beam_position_vertical)
        if self.verbose:
            print 'd_y', d_y
            print 'd_z', d_z
            
        self.move_vector = (d_sampx, d_sampy, d_y, d_z)
        self.height = c_height
        self.height_max = height_max
        self.height_min = height_min
        self.height_max_mm = height_max * self.get_calibration()[0]
        self.height_min_mm = height_min * self.get_calibration()[0]
        self.width = c_width
        self.width_mm = self.width * self.get_calibration()[1]
        self.width_max = width_max
        self.width_max_mm = self.width_max * self.get_calibration()[1]
        self.width_min = width_min
        self.width_min_mm = self.width_min * self.get_calibration()[1]
        self.angle_height_max = angle_height_max
        self.angle_height_min = angle_height_min
        self.result_position = self.get_result_position()
        
        self.analysis_results = {'move_vector': self.move_vector,
                                 'height': self.height,
                                 'height_max': self.height_max,
                                 'height_min': self.height_min,
                                 'height_max_mm': self.height_max_mm,
                                 'height_min_mm': self.height_min_mm,
                                 'width': self.width,
                                 'width_mm': self.width_mm,
                                 'width_min': self.width_min,
                                 'width_min_mm': self.width_min_mm,
                                 'width_max': self.width_max,
                                 'width_max_mm': self.width_max_mm,
                                 'angle_height_max': self.angle_height_max,
                                 'angle_height_min': self.angle_height_min,
                                 'fits': fits,
                                 'result_position': self.result_position}
                                 
        max_raster_parameters = self.get_max_raster_parameters()
        min_raster_parameters = self.get_min_raster_parameters()
        
        self.analysis_results['max_raster_parameters'] = max_raster_parameters
        self.analysis_results['min_raster_parameters'] = min_raster_parameters
        if self.verbose:
            print self.analysis_results
        print 'analysis took %.3f seconds' % (time.time() - _start)
        return self.analysis_results
        

    def residual(self, varse, angles, data):
        c, r, alpha = varse
        model = c + r*np.sin(alpha + angles + pi/2)
        return data - model
    

    def old_analyze(self, initial_parameters=[10, 100, 0]):
        
        data_points = np.array(self.results)
        try:
            angles = data_points[:, 0]
        except:
            print traceback.print_exc()
            print 'Spurious analysis. Is there any loop mounted?'
            self.move_vector = None
            return
        x = data_points[:, 1]
        y = data_points[:, 2]
        
        print 'x', x
        print 'y', y
        
        angles_radians = np.radians(angles)
        
        fit = leastsq(self.residual, initial_parameters, args=(angles_radians, y))
        print 'fit'
        print fit
        
        c, r, alpha = fit[0]
        
        d_sampx = -r * sin(alpha)
        d_sampy = r * cos(alpha)
        
        print 'd_sampx', d_sampx
        print 'd_sampy', d_sampy
        
        horizontal_center = x.mean()
        vertical_center = c
        
        print 'horizontal_center', horizontal_center
        print 'vertical_center', vertical_center
        
        print 'beam_center_horizontal', self.beam_position_horizontal
        print 'beam_center_vertical', self.beam_position_vertical
        
        d_y = self.phiy_direction * (horizontal_center - self.beam_position_horizontal)
        d_z = self.phiz_direction * (vertical_center - self.beam_position_vertical)
        
        print 'd_y', d_y
        print 'd_z', d_z
        
        self.move_vector = (d_sampx, d_sampy, d_y, d_z)
            
        return self.move_vector
    

    def get_move_vector(self):
        return self.move_vector
    
 
    def get_move_vector_mm(self):
        d_sampx, d_sampy, d_y, d_z = self.get_move_vector()
        pixels_per_mm_y = 1./self.get_calibration()[1]
        pixels_per_mm_z = 1./self.get_calibration()[0]
        d_sampx /= pixels_per_mm_z
        d_sampy /= pixels_per_mm_y
        d_y /= pixels_per_mm_y
        d_z /= pixels_per_mm_z
        move_vector_mm = (d_sampx, d_sampy, d_y, d_z)
        return move_vector_mm
    

        
    def save_results(self):
        results = {}
        for key in self.analysis_results:
            results[key] = self.analysis_results[key]
        
        if self.results_lucid != None:
            results['lucid2'] = self.results_lucid
        results['move_vector'] = self.get_move_vector()
        results['move_vector_mm'] = self.get_move_vector_mm()
        
        f = open(self.get_results_filename(), 'w')
        pickle.dump(results, f)
        f.close()
        
    def save_images(self):
        m = h5py.File(self.get_images_filename())
        m.create_dataset('images', data=np.array([item[2] for item in self.images]), compression='lzf', dtype=np.uint8)
        m.create_dataset('omegas', data=np.array([item[1] for item in self.images]), compression='lzf')
        m.create_dataset('background', data=self.background_image, compression='lzf', dtype=np.uint8)
        m.close()

    def get_background_filename(self):
        if self.default_background:
            default_background_file = '/nfs/data/Martin/Research/minikappa_calibration/%s_background.pickle' % self.get_zoom()
            if os.path.isfile(default_background_file):
                return default_background_file
        else:
            return '%s_background.png' % self.get_template()
        

    def save_background(self):
        background_filename = self.get_background_filename()
        f = open(background_filename, 'w')
        pickle.dump(self.background_image, f)
        f.close()
        

    def clean(self):
        _start = time.time()
        if self.save_raw_images:
            self.save_images()
        if self.verbose:
            print 'save_images() took %.3f seconds' % (time.time() - _start)
        #if self.background and not self.default_background and self.save_raw_background:
            #self.save_background()
        self.collect_parameters()
        self.save_parameters()
        self.save_log()
        print 'clean took %.3f seconds' % (time.time() - _start)

    def get_max_raster_parameters(self, margin_factor=1.3):
        x = self.analysis_results['width_max']
        y = self.analysis_results['height_max']
        width = self.analysis_results['width_max_mm']
        height = self.analysis_results['height_max_mm']
        angle = self.analysis_results['angle_height_max']
        
        raster = np.array([y, x]) * margin_factor
        camera_shape = self.camera.shape[:2]
        possible_increase =  np.min(camera_shape/raster) * self.camera.magnifications[self.get_zoom()-1] / self.camera.magnifications - 1
        
        try: 
            zoom = np.argmin(possible_increase[possible_increase>=0]) + 1
        except:
            zoom = self.get_zoom()
        
        if self.verbose:
            print 'max raster possible increase', possible_increase
            print 'max_raster_parameters: -y %.3f -x %.3f -a %.2f -z %d' % ( height, width, angle, zoom)
        
        return width, height, angle, zoom
        

    def get_min_raster_parameters(self, margin_factor=1.3):
        x = self.analysis_results['width_max']
        y = self.analysis_results['height_min']
        width = self.analysis_results['width_max_mm']
        height = self.analysis_results['height_min_mm']
        angle = self.analysis_results['angle_height_min']
        
        raster = np.array([y, x]) * margin_factor
        camera_shape = self.camera.shape[:2]
        possible_increase =  np.min(camera_shape/raster) * self.camera.magnifications[self.get_zoom()-1] / self.camera.magnifications - 1
        
        try:
            zoom = np.argmin(possible_increase[possible_increase>=0]) + 1
        except:
            zoom = self.get_zoom()
        
        if self.verbose:
            print 'min raster possible increase', possible_increase
            print 'min_raster_parameters: -y %.3f -x %.3f -a %.2f -z %d ' % ( height, width, angle, zoom)
        
        return width, height, angle, zoom
    

    def get_result_position(self, reference_position=None, move_vector_mm=None):
        result_position = {}
        if reference_position == None:
            reference_position = self.get_position()
        if move_vector_mm == None:
            move_vector_mm = self.get_move_vector_mm()
            
        d_sampx, d_sampy, d_y, d_z = move_vector_mm
        if self.verbose:
            print 'd_sampx, d_sampy, d_y, d_z in mm', d_sampx, d_sampy, d_y, d_z
        move_vector_dictionary = {'AlignmentZ': d_z,
                                  'AlignmentY': d_y,
                                  'CentringX': d_sampx,
                                  'CentringY': d_sampy}
        
        for motor in reference_position:
            result_position[motor] = reference_position[motor]
            if motor in move_vector_dictionary:
                result_position[motor] += move_vector_dictionary[motor]
        
        result_position['Kappa'] = self.get_kappa()
        result_position['Phi'] = self.get_phi()
        
        motors = result_position.keys()
        motors.sort()
        if self.verbose:
            print 'reference_position', [(motor, reference_position[motor]) for motor in motors]
            print 'result_position', [(motor, result_position[motor]) for motor in motors]
        
        return result_position
        

    def conclude(self):
        result_position = self.get_result_position()
        
        result_position['Omega'] = self.analysis_results['angle_height_max']
        if self.move_zoom == True:
            result_position['Zoom'] = self.camera.zoom_motor_positions[self.analysis_results['max_raster_parameters'][-1]]
        
        self.goniometer.set_position(result_position)
        self.goniometer.save_position()
        
        if self.move_zoom == True:
            self.camera.set_zoom(self.analysis_results['max_raster_parameters'][-1])
        

def main():
    import optparse
    
    parser = optparse.OptionParser()
    parser.add_option('-n', '--name_pattern', default='test_$id', type=str, help='Prefix default=%default')
    parser.add_option('-d', '--directory', default='/nfs/data/default', type=str, help='Destination directory default=%default')
    parser.add_option('-g', '--n_angles', default=24, type=int, help='Number of equidistant angles to collect at. Takes precedence over angles parameter if specified')
    parser.add_option('-a', '--angles', default='(0, 90, 180, 270)', type=str, help='Specific angles to collect at')
    parser.add_option('-r', '--scan_range', default=360, type=float, help='Range of angles')
    parser.add_option('-z', '--zoom', default=None, type=int, help='Zoom')
    parser.add_option('-p', '--position', default=None, type=str, help='Position')
    parser.add_option('-K', '--kappa', default=None, type=float, help='Kappa orientation')
    parser.add_option('-P', '--phi', default=None, type=float, help='Phi orientation')
    parser.add_option('-A', '--analysis', action='store_true', help='If set will perform automatic analysis.')
    parser.add_option('-C', '--conclusion', action='store_true', help='If set will move the motors upon analysis.')
    parser.add_option('-R', '--generate_report', action='store_true', help='If set will generate report.')
    parser.add_option('-B', '--default_background', action='store_true', help='If set will try to use lookup background image.')
    parser.add_option('--rightmost', action='store_true', help='If set will try to use lookup background image.')
    parser.add_option('--move_zoom', action='store_true', help='If set will change zoom to the one corresponding to the biggest still containing the whole loop.')
    parser.add_option('--save_raw_images', action='store_true', help='If set will save raw images.')
    parser.add_option('-F', '--film_step', default=-60., type=float, help='Film step')
    parser.add_option('-S', '--size_of_target', default=0.05, type=float, help='Size of target at the end of the sample (e.g. loop)')
    
    options, args = parser.parse_args()
    
    print 'options', options
    print 'args', args
    
    oa = optical_alignment(**vars(options))
    
    filename = '%s_parameters.pickle' % oa.get_template()
    
    print 'filename %s' % filename
    
    if not os.path.isfile(filename):
        print 'filename %s not found executing' % filename
        oa.execute()
    elif options.analysis == True:
        oa.analyze()
        if options.conclusion == True:
            oa.conclude()
        

if __name__ == '__main__':
    main()
            
        
