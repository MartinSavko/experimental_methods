#!/usr/bin/env python

'''
The raster object allows to define and carry out a collection of series of diffraction still images on a grid specified over a rectasngular area.
'''

import itertools
import time
import copy
import pickle
import scipy.misc
import scipy.ndimage
from math import sin, radians

from detector import detector
from goniometer import goniometer
from camera import camera
from protective_cover import protective_cover
from beam_center import beam_center

class raster(experiment):
    def __init__(self,
                 vertical_range,
                 horizontal_range,
                 number_of_rows,
                 number_of_columns,
                 scan_exposure_time,
                 scan_start_angle=None,
                 scan_range=0.01,
                 image_nr_start=1,
                 scan_axis='horizontal', # 'horizontal' or 'vertical'
                 direction_inversion=True,
                 method='md2', # possible methods: "md2", "helical"
                 zoom=None, # by default use the current zoom
                 name_pattern='grid_$id',
                 directory='/nfs/ruchebis/spool/2016_Run3/orphaned_collects'): 
        
        self.goniometer = goniometer()
        self.detector = detector()
        self.camera = camera()
        self.guillotine = protective_cover()
        self.beam_center = beam_center()
        
        self.scan_axis = scan_axis
        self.method = method
        self.vertical_range = vertical_range
        self.horizontal_range = horizontal_range
        self.shape = numpy.array((number_of_rows, number_of_columns))
        self.number_of_rows = number_of_rows
        self.number_of_columns = number_of_columns
        
        self.frame_time = scan_exposure_time
        self.count_time = self.frame_time - self.detector.get_detector_readout_time()
        
        self.scan_start_angle = scan_start_angle
        self.scan_range = scan_range
        
        if self.scan_axis == 'horizontal':
            self.line_scan_time = self.frame_time * self.number_of_columns
            self.angle_per_frame = self.scan_range / self.number_of_columns
        else:
            self.line_scan_time = self.frame_time * self.number_of_rows
            self.angle_per_frame = self.scan_range / self.number_of_rows
        
        self.image_nr_start = image_nr_start
        
        self.direction_inversion = direction_inversion
        
        self.name_pattern = name_pattern
        self.directory = directory
        
        self.method = method
        self.zoom = zoom
        super(self, experiment).__init__()

    def save_parameters(self):
        parameters = {}
        
        parameters['timestamp'] = self.timestamp
        parameters['angle'] = self.scan_start_angle
        parameters['vertical_step_size'], parameters['horizontal_step_size'] = self.get_step_sizes()
        parameters['reference_position'] = self.reference_position
        parameters['vertical_range'] = self.vertical_range
        parameters['horizontal_range'] = self.horizontal_range
        parameters['beam_position_vertical'] = self.camera.md2.beampositionvertical
        parameters['beam_position_horizontal'] = self.camera.md2.beampositionhorizontal
        parameters['number_of_rows'] = self.number_of_rows
        parameters['number_of_columns'] = self.number_of_columns
        parameters['direction_inversion'] = self.direction_inversion
        parameters['method'] = self.method
        parameters['scan_axis'] = self.scan_axis
        parameters['grid'] = self.grid
        parameters['cell_positions'] = self.cell_positions 
        parameters['name_pattern'] = self.name_pattern
        parameters['directory'] = self.directory
        parameters['nimages'] = self.number_of_rows * self.number_of_columns
        parameters['shape'] = self.shape
        parameters['indexes'] = self.indexes
        parameters['image'] = self.image
        parameters['rgb_image'] = self.rgbimage.reshape((493, 659, 3))
        parameters['camera_calibration_horizontal'] = self.camera.get_horizontal_calibration()
        parameters['camera_calibration_vertical'] = self.camera.get_vertical_calibration()
        parameters['camera_zoom'] = self.camera.get_zoom()
        
        scipy.misc.imsave(os.path.join(self.directory, '%s_optical_bw.png' % self.name_pattern), self.image)
        scipy.misc.imsave(os.path.join(self.directory, '%s_optical_rgb.png' % self.name_pattern), self.rgbimage.reshape((493, 659, 3)))
        
        f = open(os.path.join(self.directory, '%s_parameters.pickle' % self.name_pattern), 'w')
        pickle.dump(parameters, f)
        f.close()
        
    def get_step_sizes(self):
        step_sizes = numpy.array((self.vertical_range, self.horizontal_range)) / numpy.array((self.shape))
        return step_sizes
        
    def shift(self, vertical_shift, horizontal_shift):
        s = numpy.array([[1., 0.,    vertical_shift], 
                         [0., 1.,  horizontal_shift], 
                         [0., 0.,               1.]])
        return s

    def scale(self, vertical_scale, horizontal_scale):
        s = numpy.diag([vertical_scale, horizontal_scale, 1.])
        return s
        
    def get_cell_positions(self):
        
        step_sizes = self.get_step_sizes()
        
        self.reference_position = self.goniometer.get_position()
        
        zy_reference_position = numpy.array((self.reference_position['AlignmentZ'], self.reference_position['AlignmentY']))
        
        positions = itertools.product(range(self.number_of_rows), range(self.number_of_columns), [1])
        
        points = numpy.array([numpy.array(position) for position in positions])
        
        self.indexes = numpy.reshape(points, numpy.hstack((self.shape, 3)))
        
        center_of_mass = -numpy.array(scipy.ndimage.center_of_mass(numpy.ones(self.shape)))

        points = numpy.dot(self.shift(*center_of_mass), points.T).T
        points = numpy.dot(self.scale(*step_sizes), points.T).T
        points = numpy.dot(self.shift(*zy_reference_position), points.T).T

        #points = points[:,[0,1]]
        
        grid = numpy.reshape(points, numpy.hstack((self.shape, 3)))
        
        self.grid = grid
        
        return grid
    
    def invert(self, cell_positions):
        cell_positions_inverted = cell_positions[:,::-1,:]
        cell_raster = numpy.zeros(cell_positions.shape)
        for k in range(len(cell_positions)):
            if k%2 == 1:
                cell_raster[k] = cell_positions_inverted[k]
            else:
                cell_raster[k] = cell_positions[k]
        return cell_raster

    def program_detector(self):
        if self.detector.get_compression() != 'bslz4':
            self.detector.set_compression('bslz4')
        if self.scan_axis == 'vertical':
            self.detector.set_nimages_per_file(self.number_of_rows)
            self.detector.set_ntrigger(self.number_of_columns)
            self.detector.set_nimages(self.number_of_rows)
        else:
            self.detector.set_nimages_per_file(self.number_of_columns)
            self.detector.set_ntrigger(self.number_of_rows)
            self.detector.set_nimages(self.number_of_columns)
            
        self.detector.set_frame_time(self.frame_time)
        self.detector.set_count_time(self.count_time)
        self.detector.set_name_pattern(self.name_pattern)
        self.detector.set_omega(self.scan_start_angle)
        if self.angle_per_frame <= 0.01:
            self.detector.set_omega_increment(0)
        else:
            self.detector.set_omega_increment(self.angle_per_frame)
        self.detector.set_image_nr_start(self.image_nr_start)
        beam_center_x, beam_center_y = self.beam_center.get_beam_center()
        self.detector.set_beam_center_x(beam_center_x)
        self.detector.set_beam_center_y(beam_center_y)
        self.detector.set_detector_distance(self.beam_center.get_detector_distance() / 1000.)
        return self.detector.arm()
        
    def program_goniometer(self):
        if self.scan_start_angle is None:
            self.scan_start_angle = self.reference_position['Omega']
        self.goniometer.set_scan_start_angle(self.scan_start_angle)
        self.goniometer.set_scan_exposure_time(self.line_scan_time)
        self.goniometer.set_scan_range(self.scan_range)
        self.goniometer.set_scan_number_of_frames(1)
        
    def prepare(self):
        self.timestamp = time.asctime()
        self.goniometer.set_collect_phase()
        self.detector.clear_monitor()
        self.guillotine.extract()
        self.camera.set_zoom(self.zoom)
        self.goniometer.wait()
        self.camera.set_exposure(0.05)
        self.goniometer.wait()
        if self.scan_start_angle is None:
            self.scan_start_angle = self.reference_position['Omega']
        self.goniometer.set_omega_position(self.scan_start_angle)
        self.goniometer.insert_backlight()
        self.goniometer.insert_frontlight()
        self.goniometer.set_position(self.reference_position)
        print 'taking image'
        self.goniometer.wait()
        self.image = self.camera.get_image()
        self.rgbimage = self.camera.get_rgbimage()
        self.goniometer.remove_backlight()
        self.check_directory(os.path.join(self.directory, 'process'))
        self.write_destination_namepattern(image_path=self.directory, name_pattern=self.name_pattern)
    
    def collect(self):
        cell_positions = self.get_cell_positions()
        indexes = copy.deepcopy(self.indexes)
        self.goniometer.wait()
        self.prepare()
        self.program_goniometer()
        self.program_detector()
        
        if self.method == 'md2':
            if self.direction_inversion is True:
                cell_positions = self.invert(cell_positions)
                indexes = self.invert(indexes)
                cp = cell_positions[:, [0, -1]]
                starts, stops = cp[:, 0, :], cp[:, 1, :]
            
            cpr = cell_positions.ravel()
            cpr[2::3] = range(1, cpr.size/3 + 1)
            ir = indexes.ravel()
            ir[2::3] = range(1, cpr.size/3 + 1)
            self.cell_positions = cpr.reshape(numpy.hstack((self.shape, 3)))
            self.indexes = ir.reshape(numpy.hstack((self.shape, 3)))
            vertical_range, horizontal_range, number_of_rows, number_of_columns, direction_inversion = [str(self.vertical_range), str(self.horizontal_range), str(self.number_of_rows), str(self.number_of_columns), str(self.direction_inversion).lower()]
            
            scan_id = self.goniometer.start_raster_scan(vertical_range, horizontal_range, number_of_rows, number_of_columns, direction_inversion)
            
            while self.goniometer.is_task_running(scan_id):
                time.sleep(0.1)
        
        elif self.method == 'helical':
            
            if self.scan_axis == 'horizontal':
                if self.direction_inversion is True:
                    cell_positions = self.invert(cell_positions)
                    indexes = self.invert(indexes)
                cpr = cell_positions.ravel()
                cpr[2::3] = range(1, cpr.size/3 + 1)
                ir = indexes.ravel()
                ir[2::3] = range(1, ir.size/3 + 1)
                self.cell_positions = cpr.reshape(numpy.hstack((self.shape, 3)))
                self.indexes = ir.reshape(numpy.hstack((self.shape, 3)))
                
                cp = self.cell_positions[:, [0, -1]]
                starts, stops = cp[:, 0, :], cp[:, 1, :]
            else:
                if self.direction_inversion is True:
                    tcp = numpy.transpose(cell_positions, [1, 0, 2])
                    ti = numpy.transpose(indexes, [1, 0, 2])
                    icr = self.invert(tcp)
                    iti = self.invert(ti)
                    cell_positions = numpy.transpose(icr, [1, 0, 2])
                    indexes = numpy.transpose(iti, [1, 0, 2])
                cpr = cell_positions.ravel()
                cpr[2::3] = range(1, cpr.size/3 + 1)
                ir = indexes.ravel()
                ir[2::3] = range(1, ir.size/3 + 1)
                
                self.cell_positions = cpr.reshape(numpy.hstack((self.shape, 3)))
                self.indexes = ir.reshape(numpy.hstack((self.shape, 3)))
                
                starts, stops = self.cell_positions[[0, -1], :]
            
            print 'starts'
            print starts
            print 'stops'
            print stops
            start_position = self.reference_position.copy()
            start_position['AlignmentZ'], start_position['AlignmentY'], start_index = starts[0]
            self.goniometer.wait()
            print 'at the start position'
            for start, stop in zip(starts, stops):
                start_angle = '%6.4f' % self.scan_start_angle
                scan_range = '%6.4f' % self.scan_range
                exposure_time = '%6.4f' % self.line_scan_time
                start_z, start_y, i = map(str, start)
                stop_z, stop_y, k = map(str, stop)
                start_cx = '%6.4f' % self.reference_position['CentringX']
                start_cy = '%6.4f' % self.reference_position['CentringY']
                stop_cx = '%6.4f' % self.reference_position['CentringX']
                stop_cy = '%6.4f' % self.reference_position['CentringY']
                parameters = [start_angle, scan_range, exposure_time, start_y,start_z, start_cx, start_cy, stop_y, stop_z, stop_cx, stop_cy]
                print 'helical scan parameters'
                print parameters
                print 'start position index', i
                print 'stop position index', k
                self.goniometer.wait()
                tried = 0
                while tried < 3:
                    tried += 1
                    try:
                        scan_id = self.goniometer.start_scan_4d_ex(parameters)
                        break
                    except:
                        print 'not possible to start the scan. Is the MD2 still moving or have you specified the range in mm rather then microns ?'
                        time.sleep(0.5)
                while self.goniometer.is_task_running(scan_id):
                    time.sleep(0.1)
                print self.goniometer.get_task_info(scan_id)
        self.goniometer.wait()
        self.clean()
            
    def stop(self):
        self.goniometer.abort()
        self.detector.abort()

    def clean(self):
        self.detector.disarm() 
        self.goniometer.set_position(self.reference_position)
        self.save_parameters()