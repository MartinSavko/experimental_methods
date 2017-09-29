#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
The raster object allows to define and carry out a collection of series of diffraction still images on a grid specified over a rectangular area.
'''
import gevent
from gevent.monkey import patch_all
patch_all()

import time
import copy
import pickle
import scipy.misc
import numpy
import os

from diffraction_experiment import diffraction_experiment
from area import area

class raster(diffraction_experiment):
    
    actuator_names = ['Omega', 'AlignmentX', 'AlignmentY', 'AlignmentZ', 'CentringX', 'CentringY']
    
    def __init__(self,
                name_pattern,
                directory,
                vertical_range,
                horizontal_range,
                number_of_rows,
                number_of_columns,
                frame_time=0.025,
                scan_start_angle=None,
                scan_range=0.01,
                image_nr_start=1,
                position=None, 
                photon_energy=None,
                resolution=None,
                detector_distance=None,
                detector_vertical=None,
                detector_horizontal=None,
                transmission=None,
                flux=None,
                scan_axis='vertical', # 'horizontal' or 'vertical'
                direction_inversion=True,
                zoom=None, # by default use the current zoom
                snapshot=True,
                diagnostic=None,
                analysis=None,
                simulation=None,
                conclusion=True):
        
        diffraction_experiment.__init__(self, 
                                        name_pattern, 
                                        directory,
                                        photon_energy=photon_energy,
                                        resolution=resolution,
                                        detector_distance=detector_distance,
                                        detector_vertical=detector_vertical,
                                        detector_horizontal=detector_horizontal,
                                        transmission=transmission,
                                        flux=flux,
                                        snapshot=snapshot,
                                        diagnostic=diagnostic,
                                        analysis=analysis,
                                        simulation=simulation,
                                        conclusion=conclusion)
        

        self.vertical_range = vertical_range
        self.horizontal_range = horizontal_range
        self.shape = numpy.array((number_of_rows, number_of_columns))
        self.number_of_rows = number_of_rows
        self.number_of_columns = number_of_columns
        self.frame_time = frame_time
        if scan_start_angle == None:
            self.scan_start_angle = self.goniometer.get_omega_position()
        else:
            self.scan_start_angle = scan_start_angle
        self.scan_range = scan_range
        self.image_nr_start = image_nr_start
        position = self.goniometer.check_position(position)
        if position == None:
            self.reference_position = self.goniometer.get_aligned_position()
        else:
            self.reference_position = position
        self.scan_axis = scan_axis
        self.direction_inversion = direction_inversion
        self.zoom = zoom
        
        vertical_center = self.reference_position['AlignmentZ']
        horizontal_center = self.reference_position['AlignmentY']
        
        self.area = area(vertical_range, horizontal_range, number_of_rows, number_of_columns, vertical_center, horizontal_center)
        grid, points = self.area.get_grid_and_points()
        
        if self.scan_axis == 'horizontal':
            self.line_scan_time = self.frame_time * self.number_of_columns
            self.angle_per_frame = self.scan_range / self.number_of_columns
            self.ntrigger = self.number_of_columns
            self.nimages = self.number_of_rows
            if self.direction_inversion == True:
                raster_grid = self.area.get_horizontal_raster(grid)
                jumps = self.area.get_jump_sequence(raster_grid)
            else:
                jumps = self.area.get_jump_sequence(grid)
        else:
            self.line_scan_time = self.frame_time * self.number_of_rows
            self.angle_per_frame = self.scan_range / self.number_of_rows
            self.ntrigger = self.number_of_rows
            self.nimages = self.number_of_columns
            if self.direction_inversion == True:
                raster_grid = self.area.get_vertical_raster(grid)
                jumps = self.area.get_jump_sequence(raster_grid.T)
            else:
                jumps = self.area.get_jump_sequence(grid.T)
        
        self.grid = grid
        self.points = points
        self.jumps = jumps
        self.linearized_point_jumps = self.area.get_linearized_point_jumps(jumps, points)
    
        self.total_expected_exposure_time = self.line_scan_time * self.ntrigger
        self.total_expected_wedges = self.ntrigger
        
    def get_step_sizes(self):
        step_sizes = numpy.array((self.vertical_range, self.horizontal_range)) / numpy.array((self.shape))
        return step_sizes
    
    def get_nimages_per_file(self):
        return self.nimages
    
    def get_nimages(self):
        return self.nimages 
    
    def get_ntrigger(self):
        return self.ntrigger
    
    def get_frame_time(self):
        return self.frame_time
    
    def run(self, gonio_moving_wait_time=0.5, check_wait_time=0.1):
        self._start = time.time()
        
        self.md2_task_info = []
        
        self.goniometer.wait()
        start_angle = '%6.4f' % self.scan_start_angle
        scan_range = '%6.4f' % self.scan_range
        exposure_time = '%6.4f' % self.line_scan_time
        print 'at the start position'
        for start, stop in self.linearized_point_jumps:
            start_position = {'AlignmentZ': start[0], 'AlignmentY': start[1]}
            self.goniometer.set_position(start_position, motor_names=['AlignmentY', 'AlignmentZ'])
            start_z, start_y = map(str, start)
            stop_z, stop_y = map(str, stop)
            start_cx = '%6.4f' % self.reference_position['CentringX']
            start_cy = '%6.4f' % self.reference_position['CentringY']
            stop_cx = '%6.4f' % self.reference_position['CentringX']
            stop_cy = '%6.4f' % self.reference_position['CentringY']
            parameters = [start_angle, scan_range, exposure_time, start_y, start_z, start_cx, start_cy, stop_y, stop_z, stop_cx, stop_cy]
            print 'helical scan parameters'
            print parameters
            self.goniometer.wait()
            tried = 0
            while tried < 3:
                tried += 1
                try:
                    task_id = self.goniometer.start_scan_4d_ex(parameters)
                    break
                except:
                    print 'not possible to start the scan. Is the MD2 still moving or have you specified the range in mm rather then microns ?'
                    gevent.sleep(gonio_moving_wait_time)
            while self.goniometer.is_task_running(task_id):
                gevent.sleep(check_wait_time)
            self.md2_task_info.append(self.goniometer.get_task_info(task_id))
            
        self.goniometer.set_position(self.reference_position)
        self.goniometer.wait()
    
    def clean(self):
        self.detector.disarm()
        self.goniometer.set_position(self.reference_position)
        self.save_parameters()
        self.save_results()
        self.save_log()
        if self.diagnostic == True:
            self.save_diagnostic()
            
    def analyze(self):
        spot_find_line = 'ssh process1 "source /usr/local/dials-v1-3-3/dials_env.sh; cd %s ; echo $(pwd); dials.find_spots shoebox=False per_image_statistics=True spotfinder.filter.ice_rings.filter=True nproc=80 ../%s_master.h5"' % (self.process_directory, self.name_pattern)
        os.system(spot_find_line)
        
    def save_parameters(self):
        self.parameters = {}
        
        self.parameters['timestamp'] = self.timestamp
        self.parameters['name_pattern'] = self.name_pattern
        self.parameters['directory'] = self.directory
        self.parameters['scan_start_angle'] = self.scan_start_angle
        self.parameters['frame_time'] = self.frame_time
        self.parameters['scan_range'] = self.scan_range
        self.parameters['image_nr_start'] = self.image_nr_start
        self.parameters['vertical_step_size'], self.parameters['horizontal_step_size'] = self.get_step_sizes()
        self.parameters['reference_position'] = self.reference_position
        self.parameters['vertical_range'] = self.vertical_range
        self.parameters['horizontal_range'] = self.horizontal_range
        self.parameters['beam_position_vertical'] = self.camera.md2.beampositionvertical
        self.parameters['beam_position_horizontal'] = self.camera.md2.beampositionhorizontal
        self.parameters['number_of_rows'] = self.number_of_rows
        self.parameters['number_of_columns'] = self.number_of_columns
        self.parameters['direction_inversion'] = self.direction_inversion
        self.parameters['scan_axis'] = self.scan_axis
        self.parameters['grid'] = self.grid
        self.parameters['points'] = self.points
        self.parameters['jumps'] = self.jumps
        self.parameters['linearized_point_jumps'] = self.linearized_point_jumps
        self.parameters['nimages'] = self.nimages
        self.parameters['ntrigger'] = self.ntrigger
        self.parameters['nframes'] = self.number_of_rows * self.number_of_columns
        self.parameters['shape'] = self.shape
        self.parameters['image'] = self.image
        self.parameters['rgb_image'] = self.rgbimage.reshape((self.image.shape[0], self.image.shape[1], 3))
        self.parameters['camera_calibration_horizontal'] = self.camera.get_horizontal_calibration()
        self.parameters['camera_calibration_vertical'] = self.camera.get_vertical_calibration()
        self.parameters['camera_zoom'] = self.camera.get_zoom()
        self.parameters['duration'] = self.end_time - self.start_time
        self.parameters['start_time'] = self.start_time
        self.parameters['end_time'] = self.end_time
        self.parameters['photon_energy'] = self.photon_energy
        self.parameters['transmission'] = self.transmission
        self.parameters['detector_distance'] = self.detector_distance
        self.parameters['resolution'] = self.resolution
        self.parameters['analysis'] = self.analysis
        self.parameters['diagnostic'] = self.diagnostic
        self.parameters['simulation'] = self.simulation
        self.parameters['total_expected_exposure_time'] = self.total_expected_exposure_time
        self.parameters['md2_task_info'] = self.md2_task_info
        
        scipy.misc.imsave(os.path.join(self.directory, '%s_optical_bw.png' % self.name_pattern), self.image)
        scipy.misc.imsave(os.path.join(self.directory, '%s_optical_rgb.png' % self.name_pattern), self.rgbimage.reshape((self.image.shape[0], self.image.shape[1], 3)))
        
        f = open(os.path.join(self.directory, '%s_parameters.pickle' % self.name_pattern), 'w')
        pickle.dump(self.parameters, f)
        f.close()

def main():
    import optparse
        
    parser = optparse.OptionParser()
    parser.add_option('-n', '--name_pattern', default='raster_test_$id', type=str, help='Prefix default=%default')
    parser.add_option('-d', '--directory', default='/nfs/data/default', type=str, help='Destination directory default=%default')
    
    parser.add_option('-y', '--vertical_range', default=0.1, type=float, help='Scan range [deg] per helical line (-> 0)')
    parser.add_option('-x', '--horizontal_range', default=0.2, type=float, help='Scan range [deg] per helical line (-> 0)')
    parser.add_option('-r', '--number_of_rows', default=10, type=int, help='Scan range [deg] per helical line (-> 0)')
    parser.add_option('-c', '--number_of_columns', default=15, type=int, help='Scan range [deg] per helical line (-> 0)')
    parser.add_option('-a', '--scan_start_angle', default=None, type=float, help='Scan start angle [deg]')
    parser.add_option('-i', '--position', default=None, type=str, help='Gonio alignment position [dict]')
    
    parser.add_option('-s', '--scan_range', default=0.01, type=float, help='Scan range [deg] per helical line (-> 0)')
    parser.add_option('-e', '--frame_time', default=0.1, type=float, help='Exposure time per image [s]')
    parser.add_option('-f', '--image_nr_start', default=1, type=int, help='Start image number [int]')
    parser.add_option('-p', '--photon_energy', default=None, type=float, help='Photon energy ')
    parser.add_option('-t', '--detector_distance', default=None, type=float, help='Detector distance')
    parser.add_option('-o', '--resolution', default=None, type=float, help='Resolution [Angstroem]')
    parser.add_option('-X', '--flux', default=None, type=float, help='Flux [ph/s]')
    parser.add_option('-m', '--transmission', default=None, type=float, help='Transmission. Number in range between 0 and 1.')
    parser.add_option('-A', '--analysis', action='store_true', help='If set will perform automatic analysis.')
    parser.add_option('-D', '--diagnostic', action='store_true', help='If set will record diagnostic information.')
    parser.add_option('-S', '--simulation', action='store_true', help='If set will record diagnostic information.')
    
    options, args = parser.parse_args()
    print 'options', options
    r = raster(**vars(options))
    r.execute()
    
if __name__ == '__main__':
    main()
