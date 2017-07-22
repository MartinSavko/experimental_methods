#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
from numpy import sin, arcsin, cos, pi
from scipy.optimize import minimize

from goniometer import goniometer
from camera import camera


class sample_alignment(experiment):

    def __init__(self,
                 name_pattern,
                 directory,
                 step=36,
                 orientations=[],
                 analysis=True,
                 conclusion=True):

        experiment.__init__(self,
                            name_pattern,
                            directory,
                            analysis=analysis,
                            conclusion=conclusion)
        
        self.goniometer = goniometer()
        self.camera = camera()
        
        self.step = step
        self.orientations = orientations
        self.observations = []
        self.observe = False
        
    def run(self):
        self.aligned_position = self.goniometer.get_aligned_position()
        
        while self.observe == True:
            point = self.get_keypoint()
            orientation = self.goniometer.get_orientation()
            image = self.camera.get_image()
            rgbimage = self.camera.get_rgbimage()
            self.observations.append([orientation, point, image, rgbimage])
            goniometer.set_omega_relative_position(step)
            
    def analyze(self):
        # fit models
        experimental_data = np.array([(item[0], item[1][0], item[1][1]) for item in self.observations])
        
        orientations = radians(experimental_data[:,0])
        x = experimental_data[:,2]
        y = experimental_data[:,1]
        
        x_fit = x.mean()
        
        circle_fit = minimize(circle_residual, x0=[1024./2, 0, 0], args=(orientations, y))
        c_c, r_c, alpha_c = circle_fit[0]
        
        bounds=((None, None), (None, None), (None, None), (None, None), (None, None), (1, 2.5), (0., 2*pi))
        slab_on_circle_fit = minimize(slab_on_circle_residual, x0=[c_circle, r_circle, alpha_circle, 0, 0, 1, 0], args=(orientations, y), bounds=bounds)
        c_soc, r_soc, alpha_soc, thickness, depth, index_of_refraction, normal = slab_on_circle_fit[0]
        
        
        
    def conclude(self):
        # move motors
    
    def circle_residual(c, r, alpha, orientations, y):
        diff = y - circle(orientations, c, r, alpha)
        return np.dot(diff, diff)/(len(diff))
    
    def slab_on_circle_residual(c, r, alpha, thickness, depth, index_of_refraction, normal, orientations, y):
        diff = y - slab_on_circle(orientations, c, r, alpha, thickness, depth, index_of_refraction, normal)
        return np.dot(diff, diff)/(len(diff))
    
    def circle(self, omega, c, r, alpha):
        return c + r*sin(omega - alpha)
        
    def slab_on_circle(self, omega, c, r, alpha, thickness, depth, index_of_refraction, normal):
        return self.circle(omega, c, r, alpha) + self.slab_shift(omega, thickness, depth, index_of_refraction, normal)
    
    def slab_shift(self, omega, thickness, depth, index_of_refraction, normal):
        '''
        depth >= 0
        2.5 >= index_of_refraction >= 1
        thickness >= depth
        '''
        if depth < 0:
            depth = 0
        
        if depth > thickness:
            depth = thickness
            
        if index_of_refraction < 1:
            index_of_refraction = 1
        elif index_of_refraction > 2:
            index_of_refraction = 2
        
        n = index_of_refraction
        i = omega - normal
        i = i % (2*pi)
        if -pi/2 <= i < pi/2:
            signum = +1
        else:
            signum = -1
            depth = thickness - depth
            
        beta = arcsin(sin(i)/n) 
        h = depth/cos(beta)
        shift = h * sin(signum*i - beta)
        
        return shift
        
    def save_parameters(self):
        self.parameters = {}
        
        self.parameters['timestamp'] = self.timestamp
        self.parameters['name_pattern'] = self.name_pattern
        self.parameters['directory'] = self.directory
        self.parameters['position'] = self.position
        self.parameters['nimages'] = self.get_nimages()
        self.parameters['orientations'] = self.orientations
        self.parameters['step'] = self.step
        self.parameters['duration'] = self.end_time - self.start_time
        self.parameters['start_time'] = self.start_time
        self.parameters['end_time'] = self.end_time
        self.parameters['md2_task_info'] = self.md2_task_info
        self.parameters['camera_zoom'] = self.camera.get_zoom()
        self.parameters['camera_calibration_horizontal'] = self.camera.get_horizontal_calibration()
        self.parameters['camera_calibration_vertical'] = self.camera.get_vertical_calibration()
        self.parameters['beam_position_vertical'] = self.camera.md2.beampositionvertical
        self.parameters['beam_position_horizontal'] = self.camera.md2.beampositionhorizontal
        
        self.parameters['images'] = self.images
        self.parameters['rgb_images'] = self.rgb_images
        
        f = open(os.path.join(self.directory, '%s_%s_parameters.pickle' % (self.name_pattern, self.__module__)), 'w')
        pickle.dump(self.parameters, f)
        f.close()
    
        