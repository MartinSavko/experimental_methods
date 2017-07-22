#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
helical scan
'''
import traceback
import logging
import time

from omega_scan import omega_scan, check_position

class helical_scan(omega_scan):
    def __init__(self, 
                 name_pattern='test_$id', 
                 directory='/tmp', 
                 scan_range=180, 
                 scan_exposure_time=18, 
                 scan_start_angle=0, 
                 angle_per_frame=0.1, 
                 image_nr_start=1,
                 position_start=None,
                 position_end=None,
                 photon_energy=None,
                 resolution=None,
                 detector_distance=None,
                 detector_vertical=None,
                 detector_horizontal=None
                 transmission=None,
                 flux=None,
                 analysis=None): 
        
        omega_scan.__init__(self,
                            name_pattern=name_pattern, 
                            directory=directory,
                            scan_range=scan_range, 
                            scan_exposure_time=scan_exposure_time, 
                            scan_start_angle=scan_start_angle,
                            angle_per_frame=angle_per_frame, 
                            image_nr_start=image_nr_start,
                            photon_energy=photon_energy,
                            resolution=resolution,
                            detector_distance=detector_distance,
                            detector_vertical=detector_vertical,
                            detector_horizontal=detector_horizontal,
                            transmission=transmission,
                            flux=flux,
                            analysis=analysis)
            
        self.position_start = check_position(position_start)
        self.position_end = check_position(position_end)

    def run(self):
        task_id = self.goniometer.helical_scan(self.position_start, self.position_end, self.scan_start_angle, self.scan_range, self.scan_exposure_time)
        self.md2_task_info = self.goniometer.get_task_info(task_id)
        
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
        self.parameters['position_start'] = self.position_start
        self.parameters['position_end'] = self.position_end
        self.parameters['nimages'] = self.get_nimages()
        self.parameters['camera_zoom'] = self.camera.get_zoom()
        self.parameters['duration'] = self.end_time - self.start_time
        self.parameters['start_time'] = self.start_time
        self.parameters['end_time'] = self.end_time
        self.parameters['md2_task_info'] = self.md2_task_info
        self.parameters['photon_energy'] = self.photon_energy
        self.parameters['transmission'] = self.transmission
        self.parameters['detector_ts_intention'] = self.detector_distance
        self.parameters['detector_tz_intention'] = self.detector_vertical
        self.parameters['detector_tx_intention'] = self.detector_horizontal
        self.parameters['detector_ts'] = self.get_detector_distance()
        self.parameters['detector_tz'] = self.get_detector_vertical_position()
        self.parameters['detector_tx'] = self.get_detector_horizontal_position()
        self.parameters['beam_center_x'] = self.beam_center_x
        self.parameters['beam_center_y'] = self.beam_center_y
        self.parameters['resolution'] = self.resolution
        
        if self.snapshot == True:
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
    
    parser.add_option('-r', '--scan_range', default=180, type=float, help='Scan range [deg]')
    parser.add_option('-e', '--scan_exposure_time', default=18, type=float, help='Scan exposure time [s]')
    parser.add_option('-s', '--scan_start_angle', default=0, type=float, help='Scan start angle [deg]')
    parser.add_option('-a', '--angle_per_frame', default=0.1, type=float, help='Angle per frame [deg]')
    parser.add_option('-f', '--image_nr_start', default=1, type=int, help='Start image number [int]')
    parser.add_option('-S', '--position_start', default=None, type=str, help='Gonio alignment start position [dict]')
    parser.add_option('-E', '--position_end', default=None, type=str, help='Gonio alignment end position [dict]')
    parser.add_option('-p', '--photon_energy', default=None, type=float, help='Photon energy ')
    parser.add_option('-t', '--detector_distance', default=None, type=float, help='Detector distance')
    parser.add_option('-o', '--resolution', default=None, type=float, help='Resolution [Angstroem]')
    parser.add_option('-x', '--flux', default=None, type=float, help='Flux [ph/s]')
    parser.add_option('-m', '--transmission', default=None, type=float, help='Transmission. Number in range between 0 and 1.')
    
    options, args = parser.parse_args()
    print 'options', options
    hs = helical_scan(**vars(options))
    hs.execute()

if __name__ == '__main__':
    main()