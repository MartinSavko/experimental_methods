#!/usr/bin/env python
# -*- coding: utf-8 -*-

from xray_experiment import xray_experiment

from goniometer import goniometer
from camera import camera
from instrument import instrument

import scipy.ndimage as nd
from scipy.misc import imsave
import numpy as np
import time
import gevent
import pickle
import copy
import logging

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
                 camera_exposure_time=0.002,
                 camera_gain=8,
                 zoom=10,
                 horizontal_target=680,
                 vertical_target=512,
                 horizontal_convergence_criterium=2,
                 vertical_convergence_criterium=1,
                 mirror_default_step=0.0005,
                 move_to_data_collection_phase=False,
                 analysis=True,
                 conclusion=True,
                 diagnostic=None,
                 parent=None):
        
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
                                 parent=parent)
        
        self.description = 'Beam alignment, Proxima 2A, SOLEIL, %s' % time.ctime(self.timestamp)
        
        if self.parent != None:
            logging.getLogger('user_level_log').info(self.description)
        self.instrument = instrument()
        
        self.camera_exposure_time = camera_exposure_time
        self.camera_gain = camera_gain
        self.zoom = zoom
        self.target = np.array([vertical_target, horizontal_target])
        self.vertical_convergence_criterium = vertical_convergence_criterium
        self.horizontal_convergence_criterium = horizontal_convergence_criterium
        self.convergence_criteria = np.array([vertical_convergence_criterium, horizontal_convergence_criterium])
        self.mirror_default_step = mirror_default_step
        self.mirror_steps = np.array([self.mirror_default_step, self.mirror_default_step])
        self.results = {}
        
        self.vertical_motor = self.instrument.vfm.rx
        self.horizontal_motor = self.instrument.hfm.rz
        
        self.move_to_data_collection_phase = move_to_data_collection_phase
        self.no_beam_message = 'Beam does not seem to interact with the scintillator (the flux might be too low).\nPlease check that the scintillator is inserted, shutters are/can open and the undulator gap is in tune with the monochromator'
        
        
    def get_motor_positions(self):
        return np.array([self.vertical_motor.position, self.horizontal_motor.position])
        

    def wait_for_motor(self, motor, sleep_time=0.1):
        while motor.state().name != 'STANDBY':
            gevent.sleep(sleep_time)
            if motor.state().name == 'ALARM':
                print 'This would block otherwise'
                try:
                    motor.write_attribute('position', motor.read_attribute('position').w_value)
                except:
                    pass
            

    def set_motor_positions(self, position):
        self.vertical_motor.position, self.horizontal_motor.position = position[0], position[1]
        self.wait_for_motor(self.vertical_motor)
        self.wait_for_motor(self.horizontal_motor)
        

    def prepare(self):
        self.check_directory(self.directory)
        self.safety_shutter.open()
        self.set_photon_energy(self.photon_energy)
        self.set_transmission(self.transmission)
        self.camera.set_exposure(self.camera_exposure_time)
        self.camera.set_gain(self.camera_gain)
        self.goniometer.set_zoom(self.zoom)
        self.goniometer.set_beam_location_phase()
        self.energy_motor.turn_off()
            

    def get_shift_from_target(self, image, threshold=0.8):
        denoised_image = copy.copy(image)
        denoised_image[image<image.max()*threshold] = 0
        return self.get_center_of_mass(denoised_image) - self.target
    

    def get_center_of_mass(self, image):
        return nd.center_of_mass(image)
        

    def get_images(self):
        return self.images


    def save_image(self, image, order):
        imsave('%s_%d.png' % (self.get_template(), order), image)
        

    def get_image(self):
        self.goniometer.wait()
        self.fast_shutter.open()
        _start = time.time()
        print 'time', _start
        print 'fast shutter is open', self.fast_shutter.isopen()
        time.sleep(0.5)
        image = self.camera.get_image(color=False)
        time.sleep(0.5)
        print 'open duration', time.time() - _start
        print 'fast shutter is open', self.fast_shutter.isopen()
        self.fast_shutter.close()
        return image
        

    def convergence_criterium_met(self, shift):
        return np.all(self.convergence_criteria > np.abs(shift))
            

    def move_mirrors_towards_target(self, shift):
        current_positions = self.get_motor_positions()
        print 'current_positions', current_positions
        delta = - np.sign(shift) * (self.convergence_criteria < np.abs(shift)) * self.mirror_steps
        print 'delta', delta
        new_positions = current_positions + delta
        print 'new_positions', new_positions
        self.set_motor_positions(new_positions)
        self.previous_shift_sign = np.sign(shift)
        return delta, current_positions, new_positions
    

    def beam_in_image(self, image, threshold_max=0.9, threshold_count=200):
        print 'len(image[image>%f*image.max()]) > %f' % (threshold_max, threshold_count), len(image[image>threshold_max*image.max()])
        return len(image[image>threshold_max*image.max()]) > threshold_count
        

    def run(self):
        self.no_beam = False
        k = 0
        image = self.get_image()
        self.save_image(image, k)
        
        if not self.beam_in_image(image):
            if self.parent != None:
                logging.getLogger('user_level_log').info(self.no_beam_message)
            else:
                print self.no_beam_message
            self.no_beam = True
            return 
        
        shift = self.get_shift_from_target(image)
        print 'shift', shift
        self.initial_pixel_shift = shift
        current_positions = self.get_motor_positions()
        self.initial_mirror_positions = current_positions
        self.results['initial_mirror_positions'] = self.initial_mirror_positions
        self.results['initial_pixesl_shift'] = self.initial_pixel_shift
        print 'current_positions', current_positions
        delta = - np.sign(shift) * (self.convergence_criteria < np.abs(shift)) * self.calibration * np.abs(shift)
        print 'delta', delta
        new_positions = current_positions + delta
        print 'new_positions', new_positions
        self.set_motor_positions(new_positions)
        self.results[k] = {'shift': shift, 'delta': delta, 'current_positions': current_positions, 'new_positions': new_positions}
        while not self.convergence_criterium_met(shift) and k < 15:
            delta, current_positions, new_positions = self.move_mirrors_towards_target(shift)
            k+=1
            print 'k', k
            image = self.get_image()
            self.save_image(image, k)
            if not self.beam_in_image(image):
                print self.no_beam_message
                self.no_beam = True
                break
            shift = self.get_shift_from_target(image)
            if np.all(np.sign(shift) == self.previous_shift_sign) == False:
                self.mirror_steps
                self.mirror_steps[np.sign(shift) != self.previous_shift_sign] /= 2
            print 'shift', shift
            self.results[k] = {'shift': shift, 'delta': delta, 'current_positions': current_positions, 'new_positions': new_positions}
            print
        self.number_of_iterations = k  
        self.results['number_of_iterations'] = self.number_of_iterations 
        self.final_mirror_position = self.get_motor_positions()
        self.results['final_mirror_positions'] = self.final_mirror_position
        self.final_pixel_shift = shift
        self.results['final_pixel_shift'] = self.final_pixel_shift
        
        
    def save_results(self):
        f = open('%s_results.pck' % self.get_template(), 'w')
        pickle.dump(self.results, f)
        f.close()
        

    def clean(self):
        self.camera.set_exposure(0.05)
        self.save_results()
        if self.parent != None:
            logging.getLogger('user_level_log').info('Beam alignment finished')
        else:
            print 'Alignment finished, setting goniometer to data collection phase'
        if self.move_to_data_collection_phase:
            self.goniometer.set_data_collection_phase()
        

def main():
    import optparse
    
    parser = optparse.OptionParser()
    
    parser.add_option('-n', '--name_pattern', default='beam_alignment', type=str, help='Prefix default=%default')
    parser.add_option('-d', '--directory', default='/nfs/data/default', type=str, help='Destination directory default=%default')
    parser.add_option('-m', '--move_to_data_collection_phase', action='store_true', help='Extract the scintillator after at the end.')
    parser.add_option('-p', '--photon_energy', default=None, type=float, help='photon energy [eV]')
    
    options, args = parser.parse_args()
    print 'options', options
    
    ba = beam_align(**vars(options))
    ba.execute()


if __name__ == '__main__':
    main()
        
