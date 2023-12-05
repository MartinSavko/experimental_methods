#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import gevent
import pickle
import copy
import logging
import zmq
import os

import scipy.ndimage as nd
try:
    from scipy.misc import imsave
except ImportError:
    from skimage.io import imsave
import numpy as np

from goniometer import goniometer
from camera import camera
from instrument import instrument

from xray_experiment import xray_experiment
from beam_position_controller import get_bpc

class beam_align(xray_experiment):

    specific_paramter_fields = [{'name': 'zoom', 'type': '', 'description': ''},
                                {'name': 'camera_exposure_time', 'type': '', 'description': ''},
                                {'name': 'camera_gain', 'type': '', 'description': ''},
                                {'name': 'target', 'type': '', 'description': ''},
                                {'name': 'images',  'type': '', 'description': ''}]

    calibration = np.array([  4.90304723e-05/2.,   8.92797940e-05/2.])
    
    def __init__(self,
                 name_pattern,
                 directory,
                 photon_energy=None,
                 transmission=100,
                 #camera_exposure_time=0.002, #Multi Bunch
                 #camera_exposure_time=0.005, # 8 Bunch
                 #camera_exposure_time=0.050, #Single bunch
                 camera_exposure_time=0.005, # 17 keV
                 camera_gain=0,
                 zoom=10,
                 horizontal_target=800,
                 vertical_target=600,
                 horizontal_convergence_criterium=5e-4, # 0.5 micrometer
                 vertical_convergence_criterium=5e-4, # 0.5 micrometer
                 move_to_data_collection_phase=False,
                 analysis=True,
                 conclusion=True,
                 diagnostic=True,
                 parent=None,
                 panda=False):
        
        if hasattr(self, 'parameter_fields'):
            self.parameter_fields += beam_align.specific_parameter_fields
        else:
            self.parameter_fields = beam_align.specific_parameter_fields
            
        xray_experiment.__init__(self,
                                 name_pattern,
                                 directory,
                                 photon_energy=photon_energy,
                                 transmission=transmission,
                                 diagnostic=diagnostic,
                                 parent=parent,
                                 panda=panda)
        
        self.description = 'Beam alignment, Proxima 2A, SOLEIL, %s' % time.ctime(self.timestamp)
        
        if self.parent != None:
            logging.getLogger('user_level_log').info(self.description)
        self.instrument = instrument()
        
        self.camera_exposure_time = camera_exposure_time
        self.camera_gain = camera_gain
        self.zoom = zoom
        self.target = np.array([vertical_target, horizontal_target])
        shape = self.camera.get_shape()
        calib = self.camera.calibrations[zoom]
        self.vertical_convergence_criterium = vertical_convergence_criterium / calib[0] / shape[0]
        self.horizontal_convergence_criterium = horizontal_convergence_criterium / calib[1] / shape[1]
        self.convergence_criteria = np.array([vertical_convergence_criterium, horizontal_convergence_criterium])
        print('convergence_criteria', self.vertical_convergence_criterium , self.horizontal_convergence_criterium)
        self.results = {}
        
        self.move_to_data_collection_phase = move_to_data_collection_phase
        self.no_beam_message = 'Beam does not seem to interact with the scintillator (the flux might be too low).\nPlease check that the scintillator is inserted, shutters are/can open and the undulator gap is in tune with the monochromator'
        
        self.vbpc = get_bpc(monitor='cam', actuator='vertical_trans')
        self.hbpc = get_bpc(monitor='cam', actuator='horizontal_trans')
        
        self.total_expected_exposure_time = 5.
        self.total_expected_wedges = 1.
        
    def get_motor_positions(self):
        return np.array([device.get_position() for device in [self.vbpc.output_device, self.hbpc.output_device]])
    
    def show_the_beam(self):
        self.camera.set_exposure(self.camera_exposure_time)
        self.camera.set_gain(self.camera_gain)
        self.camera.set_zoom(self.zoom, wait=True, adjust_zoom=False)
        
        if self.goniometer.get_current_phase() != 'BeamLocation':
            self.goniometer.set_beam_location_phase(wait=True)
        #self.energy_motor.turn_off()
        self.goniometer.set_frontlightlevel(0)
        self.fast_shutter.open()
        
    def prepare(self):
        self.check_camera()
        self.check_hbpc()
        self.check_vbpc()
        self.protective_cover.insert()
        self.position_before = self.goniometer.get_aligned_position()
        self.zoom_before = self.camera.get_zoom()
        print('self.position_before', self.position_before)
        self.check_directory(self.directory)
        self.safety_shutter.open()
        if self.simulation != True:
            try:
                self.safety_shutter.open()
                if self.frontend_shutter.closed():
                    self.frontend_shutter.open()
            except:
                self.logger.info(traceback.print_exc())
                traceback.print_exc()
        self.set_photon_energy(self.photon_energy)
        self.set_transmission(self.transmission)
        self.camera.set_exposure(self.camera_exposure_time)
        self.camera.set_gain(self.camera_gain)
        self.camera.set_zoom(self.zoom, wait=True, adjust_zoom=False)
        
        if self.goniometer.get_current_phase() != 'BeamLocation':
            self.goniometer.set_beam_location_phase(wait=True)
        self.energy_motor.turn_off()
        self.goniometer.set_frontlightlevel(0)
        k = 0
        while not self.light_and_dark_are_different() and k <= 7:
            k+=1
            print('light_and_dark_are_different came out False')
            print('will try to restart camera: attempt  %d' % k)
            self.check_camera()
            gevent.sleep(3)
        
    def get_shift_from_target(self, image, threshold=0.8):
        denoised_image = copy.copy(image)
        denoised_image[image<image.max()*threshold] = 0
        return self.get_center_of_mass(denoised_image) - self.target
    

    def get_center_of_mass(self, image):
        return nd.center_of_mass(image)
        

    def get_images(self):
        return self.images


    def save_image(self, image, order):
        imsave('%s_%d.png' % (self.get_template(), order), image.astype(np.uint8))
        

    def get_image(self, sleep_time=1.):
        self.goniometer.wait()
        self.fast_shutter.open()
        _start = time.time()
        #print('time', _start)
        #print('fast shutter is open', self.fast_shutter.isopen())
        gevent.sleep(sleep_time)
        image = self.camera.get_image(color=False)
        #gevent.sleep(sleep_time/2.)
        #print('open duration', time.time() - _start)
        #print('fast shutter is open', self.fast_shutter.isopen())
        #self.fast_shutter.close()
        return image
        
    def get_dark_image(self):
        self.fast_shutter.close()
        self.goniometer.set_frontlightlevel(0)
        gevent.sleep(0.5)
        dark_image = self.camera.get_image(color=False)
        return dark_image
    
    def suspicious_dark_image(self, threshold=57):
        dark_image = self.get_dark_image()
        if np.sum(dark_image > 100) > threshold:
            return True
        return False
    
    def light_and_dark_are_different(self):
        light = self.get_image()
        dark = self.get_dark_image()
        return not np.allclose(light, dark)
    

    def beam_in_image(self, image, threshold_max=0.9, threshold_count=200):
        print('len(image[image>%f*image.max()]) > %f' % (threshold_max, threshold_count), len(image[image>threshold_max*image.max()]))
        return len(image[image>threshold_max*image.max()]) > threshold_count
        

    def run(self):
        self.no_beam = False
        k = 0
        image = self.get_image()
        self.save_image(image, k)
        
        if not self.beam_in_image(image) or self.suspicious_dark_image():
            if self.parent != None:
                logging.getLogger('user_level_log').info(self.no_beam_message)
            else:
                print(self.no_beam_message)
            self.no_beam = True
            return 
        
        shift = self.get_shift_from_target(image)
        print('shift', shift)
        self.initial_pixel_shift = shift
        self.initial_pixel_shift_mm = shift * self.camera.calibrations[self.zoom]
        current_positions = self.get_motor_positions()
        self.initial_mirror_positions = current_positions
        self.results['initial_mirror_positions'] = self.initial_mirror_positions
        self.results['initial_pixel_shift'] = self.initial_pixel_shift
        self.results['initial_pixel_shift_mm'] = self.initial_pixel_shift_mm
        print('current_positions', current_positions)
        self.results[k] = {'shift': shift, 'current_positions': current_positions}
        self.vbpc.set_on(True)
        self.hbpc.set_on(True)
        start = time.time()
        while ((abs(self.vbpc.get_pe()) > self.vertical_convergence_criterium or abs(self.hbpc.get_pe()) > self.horizontal_convergence_criterium) and k < 15) or (time.time() - start < 15) :
            k+=1
            print('k', k)
            image = self.get_image()
            self.save_image(image, k)
            if not self.beam_in_image(image):
                print(self.no_beam_message)
                self.no_beam = True
                break
            shift = self.get_shift_from_target(image)
            print('shift', shift)
            print('vertical and horizontal error', self.vbpc.get_pe(), self.hbpc.get_pe())
            self.results[k] = {'shift': shift, 'current_positions': current_positions}
        self.number_of_iterations = k  
        self.results['number_of_iterations'] = self.number_of_iterations 
        self.final_mirror_position = self.get_motor_positions()
        self.results['final_mirror_positions'] = self.final_mirror_position
        self.final_pixel_shift = shift
        self.results['final_pixel_shift'] = self.final_pixel_shift
        self.final_pixel_shift_mm = shift * self.camera.calibrations[self.zoom]
        self.results['final_pixel_shift_mm'] = self.final_pixel_shift_mm

    def clean(self):
        self.vbpc.set_on(False)
        self.hbpc.set_on(False)
        self.fast_shutter.close()
        super().clean()
        self.camera.set_exposure(0.05)
        self.save_results()
        if self.parent != None:
            logging.getLogger('user_level_log').info('Beam alignment finished')
        else:
            print('Alignment finished, setting goniometer to data collection phase')
        if self.move_to_data_collection_phase:
            self.goniometer.set_data_collection_phase(wait=True)
            self.goniometer.set_position(self.position_before, wait=True)
            self.camera.set_zoom(self.zoom_before, wait=True, adjust_zoom=False)
        self.goniometer.wait()
        self.goniometer.set_frontlightlevel(55)
        
def main():
    import optparse
    
    parser = optparse.OptionParser()
    
    parser.add_option('-n', '--name_pattern', default="%s_%s" % (os.getuid(), time.asctime().replace(" ", "_")), type=str, help='Prefix default=%default')
    parser.add_option('-d', '--directory', default="%s/beam_align" % os.getenv("HOME"), type=str, help='Destination directory default=%default')
    parser.add_option('-m', '--move_to_data_collection_phase', action='store_true', help='Extract the scintillator after at the end.')
    parser.add_option('-t', '--transmission', default=100.0, type=float, help='transmission [percent]')
    parser.add_option('-e', '--camera_exposure_time', default=0.005, type=float, help='camera_exposure_time [s]')
    parser.add_option('-p', '--photon_energy', default=None, type=float, help='photon energy [eV]')
    #parser.add_option('-D', '--diagnostic', action='store_true', help='record diagnostics')
    parser.add_option('-P', '--panda', action='store_true', help='fast shutter controlled by pandabox')
    options, args = parser.parse_args()
    print('options', options)
    
    ba = beam_align(**vars(options))
    ba.execute()


if __name__ == '__main__':
    main()
        
