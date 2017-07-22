#!/usr/bin/env python
# -*- coding: utf-8 -*-

import traceback
import logging
import time
import pickle
import os

from xray_experiment import xray_experiment
from monitor import xray_camera as detector

class tomography(xray_experiment):
    
    def __init__(self,
                 name_pattern,
                 directory,
                 scan_range=360, 
                 scan_exposure_time=72, 
                 scan_start_angle=0, 
                 angle_per_frame=0.2, 
                 image_nr_start=1,
                 position=None,
                 photon_energy=None,
                 resolution=None,
                 detector_distance=137,
                 detector_vertical=19.,
                 detector_horizontal=21.3,
                 transmission=None,
                 flux=None,
                 ntrigger=1,
                 snapshot=False,
                 zoom=None,
                 analysis=None):
                     
        xray_experiment.__init__(self, 
                                name_pattern, 
                                directory,
                                position=position,
                                photon_energy=photon_energy,
                                resolution=resolution,
                                detector_distance=detector_distance,
                                detector_vertical=detector_vertical,
                                detector_horizontal=detector_horizontal,
                                transmission=transmission,
                                flux=flux,
                                ntrigger=ntrigger,
                                snapshot=snapshot,
                                zoom=zoom,
                                analysis=analysis)
         
        # Scan parameters
        self.scan_range = float(scan_range)
        self.scan_exposure_time = float(scan_exposure_time)
        self.scan_start_angle = float(scan_start_angle)
        self.angle_per_frame = float(angle_per_frame)
        self.image_nr_start = int(image_nr_start)
        self.position = self.goniometer.check_position(position)
        print 'self.position'
        print self.position
        self.detector = detector()
        
        self.images = None
        self.background = None

    def program_detector(self):
        self.detector.stop()
        self.detector.set_latency_time(0*1e3)
        self.detector.set_integration_time(self.get_frame_time()*1e3)
    
    def get_scan_speed(self):
        '''get scan speed'''
        return self.scan_range/self.scan_exposure_time
    
    def get_nimages(self):
        nimages = int(self.scan_range/self.angle_per_frame)
        if abs(nimages*self.angle_per_frame - self.scan_range) > self.angle_per_frame/0.5:
            nimages += 1
        return nimages
                
    def get_frame_time(self):
        '''get frame time'''
        return self.scan_exposure_time/self.get_nimages()

    def get_position(self):
        '''get position '''
        if self.position is None:
            return self.goniometer.get_position()
        else:
            return self.position

    def set_position(self, position=None):
        '''set position'''
        if position is None:
            self.position = self.goniometer.get_position()
        else:
            self.position = position
            self.goniometer.set_position(self.position)
            self.goniometer.wait()
        self.goniometer.save_position()
    
    def prepare(self):
        _start = time.time()
        self.check_directory(self.directory)
        if self.snapshot == True:
            self.detector.set_safe_distance()
            print 'taking image'
            self.camera.set_exposure(0.05)
            self.camera.set_zoom(self.zoom)
            self.goniometer.insert_backlight()
            self.goniometer.extract_frontlight()
            self.goniometer.set_position(self.reference_position)
            self.goniometer.wait()
            self.image = self.camera.get_image()
            self.rgbimage = self.camera.get_rgbimage()
                    
        print 'set motors'
        self.goniometer.set_data_collection_phase(wait=False)
        self.set_photon_energy(self.photon_energy, wait=False)
        self.set_transmission(self.transmission)
        self.detector.insert()
        self.goniometer.extract_frontlight()
        self.program_detector()
        if self.safety_shutter.closed():
            self.safety_shutter.open()

        # wait for all motors to finish movements
        #print 'wait for motors to reach destinations'
        #if self.energy_moved > 0:
            #self.energy_motor.wait()
        #if self.detector_ts_moved != 0:
            #self.detector.position.ts.wait()
        #if self.detector_ts_moved != 0:
            #self.detector.position.tx.wait()
        #if self.detector_tz_moved != 0:
            #self.detector.position.tz.wait()
        
        if self.position != None:
            self.goniometer.set_position(self.position)
        else:
            self.position = self.goniometer.get_position()
            
        self.goniometer.wait()
        if self.scan_start_angle is None:
            self.scan_start_angle = self.reference_position['Omega']
        self.goniometer.set_omega_position(self.scan_start_angle)

        if self.goniometer.backlight_is_on():
            self.goniometer.remove_backlight()
        
        self.write_destination_namepattern(self.directory, self.name_pattern)
        self.energy_motor.turn_off()
        
        self.observations = []
        self.background = []
        print 'tomography prepare took %s' % (time.time()-_start)

    def get_point(self, new_image_id, start_time):
        chronos = time.time() - start_time
        position = self.goniometer.get_omega_position()
        image = self.detector.get_image()
        return [position, new_image_id, chronos, image]
                    
    def get_background(self):
        print 'get_background'
        self.position['AlignmentY'] -= 1.
        self.goniometer.set_position(self.position)
        last_image = None

        self.background_start_time = time.time()
        
        task_id = self.goniometer.omega_scan(self.scan_start_angle, self.scan_range/10, self.scan_exposure_time/10, wait=False)
        
        while self.goniometer.is_task_running(task_id):
            new_image_id = self.detector.get_current_image_id()
            if new_image_id != last_image:
                last_image = new_image_id
                self.background.append(self.get_point(new_image_id, self.background_start_time))
        self.background_end_time = time.time()
        self.background_md2_task_info = self.goniometer.get_task_info(task_id)
        self.position['AlignmentY'] += 1.
        self.goniometer.set_position(self.position)

    def run(self):
        print 'tomography running'
        self.detector.start()
        self.get_background()
        last_image = None

        self.scan_start_time = time.time()
        
        task_id = self.goniometer.omega_scan(self.scan_start_angle, self.scan_range, self.scan_exposure_time, wait=False)
        
        while self.goniometer.is_task_running(task_id):
            new_image_id = self.detector.get_current_image_id()
            if new_image_id != last_image:
                last_image = new_image_id
                self.observations.append(self.get_point(new_image_id, self.scan_start_time))
        self.md2_task_info = self.goniometer.get_task_info(task_id)
        self.scan_end_time = time.time()
    
    def save_results(self):
        f = open(os.path.join(self.directory, '%s_tomography.pickle' % self.name_pattern), 'w')
        pickle.dump({'observations': self.observations, 'background': self.background}, f)
        f.close()
        
    def save_parameters(self):
        self.parameters = {}
        
        self.parameters['timestamp'] = self.timestamp
        self.parameters['name_pattern'] = self.name_pattern
        self.parameters['directory'] = self.directory
        self.parameters['scan_range'] = self.scan_range
        self.parameters['scan_exposure_time'] = self.scan_exposure_time
        self.parameters['scan_start_angle'] = self.scan_start_angle
        self.parameters['image_nr_start'] = self.image_nr_start
        self.parameters['frame_time'] = self.get_frame_time()
        self.parameters['position'] = self.position
        self.parameters['nimages'] = len(self.observations)
        self.parameters['nimages_background'] = len(self.background)
        self.parameters['background_md2_task_info'] = self.background_md2_task_info
        self.parameters['md2_task_info'] = self.md2_task_info
        self.parameters['scan_duration'] = self.scan_end_time - self.scan_start_time
        self.parameters['scan_start_time'] = self.scan_start_time
        self.parameters['scan_end_time'] = self.scan_end_time
        self.parameters['duration'] = self.end_time - self.start_time
        self.parameters['start_time'] = self.start_time
        self.parameters['end_time'] = self.end_time
        self.parameters['photon_energy'] = self.photon_energy
        self.parameters['wavelength'] = self.wavelength
        self.parameters['transmission'] = self.transmission
        self.parameters['detector_ts_intention'] = self.detector_distance
        self.parameters['detector_tz_intention'] = self.detector_vertical
        self.parameters['detector_tx_intention'] = self.detector_horizontal
        self.parameters['detector_ts'] = self.detector.distance_motor.get_position()
        self.parameters['detector_tz'] = self.detector.stage_vertical_motor.get_position()
        self.parameters['detector_tx'] = self.detector.stage_horizontal_motor.get_position()
        self.parameters['focus'] = self.detector.focus_motor.get_position()
        self.parameters['camera_vertical_motor'] = self.detector.vertical_motor.get_position()
        
        if self.snapshot == True:
            self.parameters['camera_zoom'] = self.camera.get_zoom()
            self.parameters['camera_calibration_horizontal'] = self.camera.get_horizontal_calibration()
            self.parameters['camera_calibration_vertical'] = self.camera.get_vertical_calibration()
            self.parameters['beam_position_vertical'] = self.camera.md2.beampositionvertical
            self.parameters['beam_position_horizontal'] = self.camera.md2.beampositionhorizontal
            self.parameters['image'] = self.image
            self.parameters['rgb_image'] = self.rgbimage.reshape((self.image.shape[0], self.image.shape[1], 3))
            scipy.misc.imsave(os.path.join(self.directory, '%s_optical_bw.png' % self.name_pattern), self.image)
            scipy.misc.imsave(os.path.join(self.directory, '%s_optical_rgb.png' % self.name_pattern), self.rgbimage.reshape((self.image.shape[0], self.image.shape[1], 3)))
        
        f = open(os.path.join(self.directory, '%s_parameters.pickle' % self.name_pattern), 'w')
        pickle.dump(self.parameters, f)
        f.close()
    
def main():
    import optparse
    
    parser = optparse.OptionParser()
    parser.add_option('-n', '--name_pattern', default='test_$id', type=str, help='Prefix default=%default')
    parser.add_option('-d', '--directory', default='/nfs/data/default', type=str, help='Destination directory default=%default')
    parser.add_option('-r', '--scan_range', default=360, type=float, help='Scan range [deg]')
    parser.add_option('-e', '--scan_exposure_time', default=180, type=float, help='Scan exposure time [s]')
    parser.add_option('-s', '--scan_start_angle', default=0, type=float, help='Scan start angle [deg]')
    parser.add_option('-a', '--angle_per_frame', default=0.5, type=float, help='Angle per frame [deg]')
    parser.add_option('-f', '--image_nr_start', default=1, type=int, help='Start image number [int]')
    parser.add_option('-i', '--position', default=None, type=str, help='Gonio alignment position [dict]')
    parser.add_option('-p', '--photon_energy', default=None, type=float, help='Photon energy ')
    parser.add_option('-t', '--detector_distance', default=None, type=float, help='Detector distance')
    parser.add_option('-x', '--flux', default=None, type=float, help='Flux [ph/s]')
    parser.add_option('-m', '--transmission', default=None, type=float, help='Transmission. Number in range between 0 and 1.')
    
    options, args = parser.parse_args()
    print 'options', options
    t = tomography(**vars(options))
    t.execute()
    
if __name__ == '__main__':
    main()
    