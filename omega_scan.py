#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
single position oscillation scan
'''
import gevent
from gevent.monkey import patch_all
patch_all()

import traceback
import logging
import time
import pickle
import os

from diffraction_experiment import diffraction_experiment
from monitor import xbpm

class omega_scan(diffraction_experiment):
    ''' Will execute single continuous omega scan '''
    
    actuator_names = ['Omega']
    
    def __init__(self, 
                 name_pattern, 
                 directory, 
                 scan_range=180, 
                 scan_exposure_time=18, 
                 scan_start_angle=0, 
                 angle_per_frame=0.1, 
                 image_nr_start=1,
                 position=None, 
                 photon_energy=None,
                 resolution=None,
                 detector_distance=None,
                 detector_vertical=None,
                 detector_horizontal=None,
                 transmission=None,
                 flux=None,
                 snapshot=False,
                 ntrigger=1,
                 nimages_per_file=100,
                 zoom=None,
                 diagnostic=None,
                 analysis=None,
                 simulation=None):
        
        diffraction_experiment.__init__(self, 
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
                                        snapshot=snapshot,
                                        ntrigger=ntrigger,
                                        zoom=zoom,
                                        diagnostic=diagnostic,
                                        analysis=analysis,
                                        simulation=simulation)

        # Scan parameters
        self.scan_range = float(scan_range)
        self.scan_exposure_time = float(scan_exposure_time)
        self.scan_start_angle = float(scan_start_angle) % 360
        self.angle_per_frame = float(angle_per_frame)
        self.image_nr_start = int(image_nr_start)
        self.position = self.goniometer.check_position(position)

        self.ntrigger = ntrigger
        self.nimages_per_file = nimages_per_file
        self.total_expected_exposure_time = self.scan_exposure_time
        self.total_expected_wedges = 1
        
    def get_nimages(self, epsilon=1e-3):
        nimages = int(self.scan_range/self.angle_per_frame)
        if abs(nimages*self.angle_per_frame - self.scan_range) > epsilon:
            nimages += 1
        return nimages
    
    def get_nimages_per_file(self):
        return self.nimages_per_file
    
    def get_dpf(self):
        '''get degrees per frame'''
        return self.angle_per_frame
    
    def get_fps(self):
        '''get frames per second'''
        return self.get_nimages()/self.scan_exposure_time
    
    def get_dps(self):
        '''get degrees per second'''
        return self.get_scan_speed()
    
    def get_scan_speed(self):
        '''get scan speed'''
        return self.scan_range/self.scan_exposure_time
    
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

    def run(self, wait=True):
        '''execute omega scan.'''
        
        self._start = time.time()
        
        task_id = self.goniometer.omega_scan(self.scan_start_angle, self.scan_range, self.scan_exposure_time, wait=wait)

        self.md2_task_info = self.goniometer.get_task_info(task_id)
        
    def analyze(self):
        xdsme_process_line = 'ssh process1 "cd {directory:s}; xdsme -i "LIB=/nfs/data/plugin.so" ../{name_pattern:s}_master.h5" > {name_pattern:s}_xdsme.log &'.format(**{'directory': os.path.join(self.directory, 'process'), 'name_pattern': os.path.basename(self.name_pattern)})
        print 'xdsme process_line', process_line
        os.system(xdsme_process_line)
        
        autoPROC_process_line = 'ssh process1 "cd {directory:s}; mkdir autoPROC; cd autoPROC; process -nthread 72 -h5 ../../{name_pattern:s}_master.h5" > ../{name_pattern:s}_autoPROC.log &'.format(**{'directory': os.path.join(self.directory, 'process'), 'name_pattern': os.path.basename(self.name_pattern)})
        print 'autoPROC process_line', process_line
        os.system(xdsme_process_line)
        
        xia2_dials_process_line = 'ssh process1 "cd {directory:s}; mkdir xia2; cd xia2; xia2 pipeline=dials dials.fast_mode=True nproc=72 ../../{name_pattern:s}_master.h5" > ../{name_pattern:s}_xia2.log &'.format(**{'directory': os.path.join(self.directory, 'process'), 'name_pattern': os.path.basename(self.name_pattern)})
        print 'xia2_dials process_line', process_line
        os.system(xia2_dials_process_line)
        
    def save_parameters(self):
        self.parameters = {}
        
        self.parameters['timestamp'] = self.timestamp
        self.parameters['name_pattern'] = self.name_pattern
        self.parameters['directory'] = self.directory
        self.parameters['scan_range'] = self.scan_range
        self.parameters['scan_exposure_time'] = self.scan_exposure_time
        self.parameters['scan_start_angle'] = self.scan_start_angle
        self.parameters['angle_per_frame'] = self.angle_per_frame
        self.parameters['image_nr_start'] = self.image_nr_start
        self.parameters['frame_time'] = self.get_frame_time()
        self.parameters['position'] = self.position
        self.parameters['nimages'] = self.get_nimages()
        self.parameters['duration'] = self.end_time - self.start_time
        self.parameters['start_time'] = self.start_time
        self.parameters['end_time'] = self.end_time
        self.parameters['md2_task_info'] = self.md2_task_info
        self.parameters['photon_energy'] = self.photon_energy
        self.parameters['wavelength'] = self.wavelength
        self.parameters['transmission'] = self.transmission
        self.parameters['detector_ts_intention'] = self.detector_distance
        self.parameters['detector_tz_intention'] = self.detector_vertical
        self.parameters['detector_tx_intention'] = self.detector_horizontal
        if self.simulation != True:
            self.parameters['detector_ts'] = self.get_detector_distance()
            self.parameters['detector_tz'] = self.get_detector_vertical_position()
            self.parameters['detector_tx'] = self.get_detector_horizontal_position()
        self.parameters['beam_center_x'] = self.beam_center_x
        self.parameters['beam_center_y'] = self.beam_center_y
        self.parameters['resolution'] = self.resolution
        self.parameters['analysis'] = self.analysis
        self.parameters['diagnostic'] = self.diagnostic
        self.parameters['simulation'] = self.simulation
        self.parameters['total_expected_exposure_time'] = self.total_expected_exposure_time
        
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
    parser.add_option('-r', '--scan_range', default=45, type=float, help='Scan range [deg]')
    parser.add_option('-e', '--scan_exposure_time', default=4.5, type=float, help='Scan exposure time [s]')
    parser.add_option('-s', '--scan_start_angle', default=0, type=float, help='Scan start angle [deg]')
    parser.add_option('-a', '--angle_per_frame', default=0.1, type=float, help='Angle per frame [deg]')
    parser.add_option('-f', '--image_nr_start', default=1, type=int, help='Start image number [int]')
    parser.add_option('-i', '--position', default=None, type=str, help='Gonio alignment position [dict]')
    parser.add_option('-p', '--photon_energy', default=None, type=float, help='Photon energy ')
    parser.add_option('-t', '--detector_distance', default=None, type=float, help='Detector distance')
    parser.add_option('-o', '--resolution', default=None, type=float, help='Resolution [Angstroem]')
    parser.add_option('-x', '--flux', default=None, type=float, help='Flux [ph/s]')
    parser.add_option('-m', '--transmission', default=None, type=float, help='Transmission. Number in range between 0 and 1.')
    parser.add_option('-A', '--analysis', action='store_true', help='If set will perform automatic analysis.')
    parser.add_option('-D', '--diagnostic', action='store_true', help='If set will record diagnostic information.')
    parser.add_option('-S', '--simulation', action='store_true', help='If set will record diagnostic information.')
    
    options, args = parser.parse_args()
    print 'options', options
    s = omega_scan(**vars(options))
    s.execute()
    
def test():
    scan_range = 180
    scan_exposure_time = 18.
    scan_start_angle = 0
    angle_per_frame = 0.1
    
    s = omega_scan(scan_range=scan_range, scan_exposure_time=scan_exposure_time, scan_start_angle=scan_start_angle, angle_per_frame=angle_per_frame)
    
if __name__ == '__main__':
    main()