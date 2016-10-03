#!/usr/bin/env python


import sys
import time
import os
import numpy 
import PyTango
import logging
import itertools
import scipy.ndimage
import scipy.misc
import traceback
import pickle
import copy
from math import sin, radians

sys.path.insert(0,"/usr/local/dectris/python")
sys.path.insert(0,"/usr/local/dectris/albula/3.1/python")

from eigerclient import DEigerClient
from detector import detector

class goniometer(object):
    motorsNames = ['AlignmentX', 
                   'AlignmentY', 
                   'AlignmentZ',
                   'CentringX', 
                   'CentringY']
                   
    motorShortNames = ['PhiX', 'PhiY', 'PhiZ', 'SamX', 'SamY']
    mxcubeShortNames = ['phix', 'phiy', 'phiz', 'sampx', 'sampy']
    
    shortFull = dict(zip(motorShortNames, motorsNames))
    phiy_direction=-1.
    phiz_direction=1.
    
    def __init__(self):
        self.md2 = PyTango.DeviceProxy('i11-ma-cx1/ex/md2')
      
    def set_scan_start_angle(self, scan_start_angle):
        self.scan_start_angle = scan_start_angle
        self.md2.scanstartangle = scan_start_angle
    
    def get_scan_start_angle(self):
        return self.md2.scanstartangle
       
    def set_scan_range(self, scan_range):
        self.scan_range = scan_range
        self.md2.scanrange = scan_range
        
    def get_scan_range(self):
        return self.md2.scanrange
        
    def set_scan_exposure_time(self, scan_exposure_time):
        self.scan_exposure_time = scan_exposure_time
        self.md2.scanexposuretime = scan_exposure_time
    
    def get_scan_exposure_time(self):
        return self.md2.scanexposuretime
    
    def set_scan_number_of_frames(self, scan_number_of_frames):
        self.scan_number_of_frames = scan_number_of_frames
        self.md2.scannumberofframes = scan_number_of_frames
       
    def get_scan_number_of_frames(self):
        return self.md2.scannumberofframes
        
    def set_collect_phase(self):
        return self.md2.startsetphase('DataCollection')
        
    def abort(self):
        return self.md2.abort()

    def start_scan(self):
        return self.md2.startscan()
        
    def helical_scan(self, start, stop, scan_start_angle, scan_range, scan_exposure_time):
        scan_start_angle = '%6.4f' % scan_start_angle
        scan_range = '%6.4f' % scan_range
        scan_exposure_time = '%6.4f' % scan_exposure_time
        start_z = '%6.4f' % start['AlignmentZ']
        start_y = '%6.4f' % start['AlignmentY']
        stop_z = '%6.4f' % stop['AlignmentZ']
        stop_y = '%6.4f' % stop['AlignmentY']
        start_cx = '%6.4f' % start['CentringX']
        start_cy = '%6.4f' % start['CentringY']
        stop_cx = '%6.4f' % stop['CentringX']
        stop_cy = '%6.4f' % stop['CentringY']
        parameters = [scan_start_angle, scan_range, scan_exposure_time, start_y,start_z, start_cx, start_cy, stop_y, stop_z, stop_cx, stop_cy]
        print 'helical scan parameters'
        print parameters
        tried = 0
        while tried < 3:
            tried += 1
            try:
                scan_id = self.md2.startScan4DEx(parameters)
                break
            except:
                print 'not possible to start the scan. Is the MD2 still moving or have you specified the range in mm rather then microns ?'
                time.sleep(0.5)
        while self.md2.istaskrunning(scan_id):
            time.sleep(0.1)
        print self.md2.gettaskinfo(scan_id)
                    
    def start_helical_scan(self):
        return self.md2.startscan4d()
        
    def set_helical_start(self):
        return self.md2.setstartscan4d()
    
    def set_helical_stop(self):
        return self.md2.setstopscan4d()
        
    def getMotorState(self, motor_name):
        return self.md2.getMotorState(motor_name).name
        
    def getState(self):
        motors = ['Omega', 'AlignmentX', 'AlignmentY', 'AlignmentZ', 'CentringX', 'CentringY', 'ApertureHorizontal', 'ApertureVertical', 'CapillaryHorizontal', 'CapillaryVertical', 'ScintillatorHorizontal', 'ScintillatorVertical', 'Zoom']
        state = set([self.getMotorState(m) for m in motors])
        if len(state) == 1 and 'STANDBY' in state:
            return 'STANDBY'
        else:
            return 'MOVING'
            
    def wait(self, device=None):
        green_light = False
        while green_light is False:
            try:
                if device is None:
                    if self.getState() in ['MOVING', 'RUNNING']:
                        logging.info("MiniDiffPX2 wait" )
                    else:
                        green_light = True
                        return
                else:   
                    if device.state().name not in ['STANDBY']:
                        logging.info("Device %s wait" % device)
                    else:
                        green_light = True
                        return
            except:
                traceback.print_exc()
                logging.info('Problem occured in wait %s ' % device)
                logging.info(traceback.print_exc())
            time.sleep(.1)
        

    def moveToPosition(self, position={}, epsilon = 0.0002):
        print 'position %s' % position
        if position != {}:
            for motor in position:
                while abs(self.md2.read_attribute('%sPosition' % self.shortFull[motor]).value - position[motor]) > epsilon:
                    self.wait()
                    time.sleep(0.5)
                    self.md2.write_attribute('%sPosition' % self.shortFull[motor], position[motor])
                
            self.wait()
        self.wait()
        return
    
    def set_position(self, position, motors=['AlignmentX', 'AlignmentY', 'AlignmentZ', 'CentringY', 'CentringX']):
        motor_name_value_list = ['%s=%6.4f' % (motor, position[motor]) for motor in motors]
        command_string = ','.join(motor_name_value_list)
        print 'command string', command_string
        k=0
        while k < 3:
            k+=1
            try:
                return self.md2.startSimultaneousMoveMotors(command_string)
            except:
                time.sleep(1)
             
    def get_position(self):
        return dict([(m.split('=')[0], float(m.split('=')[1])) for m in self.md2.motorpositions])
        
class resolution(object):
    def __init__(self, x_pixels_in_detector=3110, y_pixels_in_detector=3269, x_pixel_size=75e-6, y_pixel_size=75e-6):
        self.distance_motor = PyTango.DeviceProxy('i11-ma-cx1/dt/dtc_ccd.1-mt_ts')
        self.wavelength_motor = PyTango.DeviceProxy('i11-ma-c03/op/mono1')
        self.x_pixel_size = x_pixel_size
        self.y_pixel_size = y_pixel_size
        self.x_pixels_in_detector = x_pixels_in_detector
        self.y_pixels_in_detector = y_pixels_in_detector
        self.bc = beam_center()
        
    def get_detector_radii(self):
        beam_center_x, beam_center_y = self.bc.get_beam_center()
        detector_size_x = self.x_pixel_size * self.x_pixels_in_detector
        detector_size_y = self.y_pixel_size * self.y_pixels_in_detector
        
        beam_center_distance_x = self.x_pixel_size * beam_center_x
        beam_center_distance_y = self.y_pixel_size * beam_center_y
        
        distances_x = numpy.array([detector_size_x - beam_center_distance_x, beam_center_distance_x])
        distances_y = numpy.array([detector_size_y - beam_center_distance_y, beam_center_distance_y])
        
        edge_distances = numpy.hstack([distances_x, distances_y])
        corner_distances = numpy.array([(x**2 + y**2)**0.5 for x in distances_x for y in distances_y])
        
        distances = numpy.hstack([edge_distances, corner_distances]) * 1000.
        return distances
        
    def get_detector_min_radius(self):
        distances = self.get_detector_radii()
        return distances.min()
        
    def get_detector_max_radius(self):
        distances = self.get_detector_radii()
        return distances.max()
        
    def get_distance(self):
        return self.distance_motor.position
        
    def get_wavelength(self):
        return self.wavelength_motor.read_attribute('lambda').value
        
    def get_resolution(self, distance=None, wavelength=None, radius=None):
        if distance is None:
            distance = self.get_distance()
        if radius is None:
            detector_radius = self.get_detector_min_radius()
        if wavelength is None:
            wavelength = self.get_wavelength()
        
        two_theta = numpy.math.atan(detector_radius/distance)
        resolution = 0.5 * wavelength / numpy.sin(0.5*two_theta)
        
        return resolution
        
    def get_resolution_from_distance(self, distance, wavelength=None):
        return self.get_resolution(distance=distance, wavelength=wavelength)
        
    def get_distance_from_resolution(self, resolution, wavelength=None):
        if wavelength is None:
            wavelength = self.get_wavelength()
        two_theta = 2*numpy.math.asin(0.5*wavelength/resolution)
        detector_radius = self.get_detector_min_radius()
        distance = detector_radius/numpy.math.tan(two_theta)
        return distance
     
class camera(object):
    def __init__(self):
        self.md2 = PyTango.DeviceProxy('i11-ma-cx1/ex/md2')
        self.prosilica = PyTango.DeviceProxy('i11-ma-cx1/ex/imag.1')
        
    def get_image(self):
        return self.prosilica.image
        
    def get_rgbimage(self):
        return self.prosilica.rgbimage.reshape((493, 659, 3))
        
    def get_zoom(self):
        return self.md2.coaxialcamerazoomvalue
    
    def set_zoom(self, value):
        if value is not None:
            value = int(value)
            self.md2.coaxialcamerazoomvalue = value
    
    def get_calibration(self):
        return numpy.array(self.md2.coaxcamscaley, self.md2.coaxcamscalex)
        
    def get_vertical_calibration(self):
        return self.md2.coaxcamscaley
        
    def get_horizontal_calibration(self):
        return self.md2.coaxcamscalex
        
class protective_cover(object):
    def __init__(self):
        self.guillotine = PyTango.DeviceProxy('i11-ma-cx1/dt/guillot-ev')
        
    def insert(self):
        self.guillotine.insert()
    
    def extract(self):
        self.guillotine.extract()
        
class raster(object):
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
                 method='md2', #'helical', # possible methods: "md2", "helical"
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
        self.goniometer.md2.OmegaPosition = scan_start_angle
        
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
        if self.goniometer.md2.backlightison == True:
            self.goniometer.md2.backlightison = False
        if self.scan_start_angle is None:
            self.scan_start_angle = self.reference_position['Omega']
        self.goniometer.set_scan_start_angle(self.scan_start_angle)
        self.goniometer.set_scan_exposure_time(self.line_scan_time)
        self.goniometer.set_scan_range(self.scan_range)
        self.goniometer.set_scan_number_of_frames(1)
        
    def prepare(self):
        self.timestamp = time.asctime()
        self.detector.check_dir(os.path.join(self.directory,'process'))
        self.goniometer.set_collect_phase()
        self.detector.clear_monitor()
        self.guillotine.extract()
        self.camera.set_zoom(self.zoom)
        self.goniometer.wait()
        self.camera.prosilica.exposure = 0.05
        self.goniometer.wait()
        tries = 0
        while abs(sin(radians(self.goniometer.md2.OmegaPosition)) - sin(radians(self.scan_start_angle))) > 0.2 and tries < 3:
            try:
                self.goniometer.wait()
                self.goniometer.md2.OmegaPosition = self.scan_start_angle
            except:
                time.sleep(0.1)
                tries += 1
                print '%s try to sent omega to %s' % (tries, self.scan_start_angle)
        #self.goniometer.md2.frontlightlevel *= 0.8
        while abs(sin(radians(self.goniometer.md2.OmegaPosition)) - sin(radians(self.scan_start_angle))) > 0.2:
            time.sleep(0.2)
            try:
                self.goniometer.md2.OmegaPosition = self.scan_start_angle
            except:
                pass
            print 'waiting for omega axis to come to %s' % self.scan_start_angle
        self.goniometer.wait()
        time.sleep(1)
        try:
            self.goniometer.md2.backlightison = True
            self.goniometer.md2.frontlightison = True
        except:
            pass
        while not self.goniometer.md2.backlightison:
            time.sleep(0.2)
            try:
                self.goniometer.md2.backlightison = True
                self.goniometer.md2.frontlightison = True
            except:
                pass
            print 'waiting for back light to come on' 
        self.goniometer.set_position(self.reference_position)
        print 'taking image'
        time.sleep(1)
        self.image = self.camera.get_image()
        self.rgbimage = self.camera.get_rgbimage()
        if not os.path.isdir(self.directory):
            os.makedirs(os.path.join(self.directory, 'process'))
        self.detector.write_destination_namepattern(image_path=self.directory, name_pattern=self.name_pattern)
        self.status = 'prepare' 
    
    def invert(self, cell_positions):
        cell_positions_inverted = cell_positions[:,::-1,:]
        cell_raster = numpy.zeros(cell_positions.shape)
        for k in range(len(cell_positions)):
            if k%2 == 1:
                cell_raster[k] = cell_positions_inverted[k]
            else:
                cell_raster[k] = cell_positions[k]
        return cell_raster
     
    def collect(self):
        cell_positions = self.get_cell_positions()
        indexes = copy.deepcopy(self.indexes)
        time.sleep(2)
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
            
            scan_id = self.goniometer.md2.startRasterScan([str(self.vertical_range), str(self.horizontal_range), str(self.number_of_rows), str(self.number_of_columns), str(self.direction_inversion).lower()])
            
            while self.goniometer.md2.istaskrunning(scan_id):
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
                        scan_id = self.goniometer.md2.startScan4DEx(parameters)
                        break
                    except:
                        print 'not possible to start the scan. Is the MD2 still moving or have you specified the range in mm rather then microns ?'
                        time.sleep(0.5)
                while self.goniometer.md2.istaskrunning(scan_id):
                    time.sleep(0.1)
                print self.goniometer.md2.gettaskinfo(scan_id)
        self.goniometer.wait()
        self.clean()
            
    def stop(self):
        self.goniometer.abort()
        self.detector.abort()

    def clean(self):
        self.detector.disarm() 
        self.goniometer.set_position(self.reference_position)
        self.save_parameters()
  
class beam_center(object):
    def __init__(self):
        self.distance_motor = PyTango.DeviceProxy('i11-ma-cx1/dt/dtc_ccd.1-mt_ts')
        self.wavelength_motor = PyTango.DeviceProxy('i11-ma-c03/op/mono1')
        self.det_mt_tx = PyTango.DeviceProxy('i11-ma-cx1/dt/dtc_ccd.1-mt_tx') #.read_attribute('position').value - 30.0
        self.det_mt_tz = PyTango.DeviceProxy('i11-ma-cx1/dt/dtc_ccd.1-mt_tz') #.read_attribute('position').value + 14.3
        self.detector = detector()
        self.pixel_size = 75e-6
        
    def get_beam_center_x(self, X):
        logging.info('beam_center_x calculation')
        theta = numpy.matrix([ 1.65113065e+03,   5.63662370e+00,   3.49706731e-03, 9.77188997e+00])
        orgy = X * theta.T
        if self.detector.get_roi_mode() == '4M':
            orgy -= 550
        return float(orgy)
    
    def get_beam_center_y(self, X):
        logging.info('beam_center_y calculation')
        theta = numpy.matrix([  1.54776707e+03,   3.65108709e-01,  -1.12769165e-01,   9.74625808e+00])
        orgx = X * theta.T
        return float(orgx)
        
    def get_beam_center(self):
        #Theta = numpy.matrix([[  1.54776707e+03,   1.65113065e+03], [  3.65108709e-01,   5.63662370e+00], [ -1.12769165e-01,   3.49706731e-03]])
        #X = numpy.matrix([1., self.wavelength_motor.read_attribute('lambda').value, self.distance_motor.position])
        #X = X.T
        #beam_center = Theta.T * X
        #beam_center_x = beam_center[0, 0]
        #beam_center_y = beam_center[1, 0]
        #beam_center_x -= 26.9
        #beam_center_y -= 5.7
        q = 0.075 #0.102592
        
        wavelength = self.wavelength_motor.read_attribute('lambda').value
        distance   = self.distance_motor.read_attribute('position').value
        tx         = self.det_mt_tx.read_attribute('position').value - 30.0
        tz         = self.det_mt_tz.read_attribute('position').value + 14.3
        logging.info('wavelength %s' % wavelength)
        logging.info('mt_ts %s' % distance)
        logging.info('mt_tx %s' % tx)
        logging.info('mt_tz %s' % tz)
        print('wavelength %s' % wavelength)
        print('mt_ts %s' % distance)
        print('mt_tx %s' % tx)
        print('mt_tz %s' % tz)
        #wavelength  = self.mono1.read_attribute('lambda').value
        #distance    = self.detector_mt_ts.read_attribute('position').value
        #tx          = self.detector_mt_tx.position
        #tz          = self.detector_mt_tz.position
        
        X = numpy.matrix([1., wavelength, distance, 0, 0 ]) #tx, tz])
        
        beam_center_y = self.get_beam_center_x(X[:, [0, 1, 2, 4]])
        beam_center_x = self.get_beam_center_y(X[:, [0, 1, 2, 3]])
        
        beam_center_x += tx / q
        beam_center_y += tz / q 
        
        beam_center_x += 0.58
        beam_center_y += -1.36
        
        #2016-09-06 adjusting table
        beam_center_x += -16.3
        beam_center_y += 2.0
        
        #2016-09-07 adjusting table
        #ORGX= 1534.19470215    ORGY= 1652.97814941
        #1544.05   1652.87
        beam_center_x += 10.15
        #beam_center_y += 2.0
        
        return beam_center_x, beam_center_y
        
    def get_detector_distance(self):
        return self.distance_motor.position
        
        
class reference_images(object):
    def __init__(self,
                 scan_range,
                 scan_exposure_time,
                 scan_start_angles, #this is an iterable
                 angle_per_frame,
                 name_pattern,
                 directory='/nfs/ruchebis/spool/2016_Run3/orphaned_collects',
                 image_nr_start=1):
                     
        self.goniometer = goniometer()
        self.detector = detector()
        self.beam_center = beam_center()
        
        scan_range = float(scan_range)
        scan_exposure_time = float(scan_exposure_time)
        
        nimages = float(scan_range)/angle_per_frame

        frame_time = scan_exposure_time/nimages
        
        self.scan_range = scan_range
        self.scan_exposure_time = scan_exposure_time
        self.scan_start_angles = scan_start_angles
        self.angle_per_frame = angle_per_frame
        
        self.nimages = int(nimages)
        self.frame_time = float(frame_time)
        self.count_time = self.frame_time - self.detector.get_detector_readout_time()
        
        self.name_pattern = name_pattern
        self.directory = directory
        self.image_nr_start = image_nr_start
        self.status = None
                    
    def program_detector(self):
        self.detector.set_ntrigger(len(self.scan_start_angles))
        if self.detector.get_compression() != 'bslz4':
            self.detector.set_compression('bslz4')
        self.detector.set_nimages_per_file(self.nimages)
        self.detector.set_nimages(self.nimages)
        self.detector.set_frame_time(self.frame_time)
        self.detector.set_count_time(self.count_time)
        self.detector.set_name_pattern(self.name_pattern)
        self.detector.set_omega(self.scan_start_angles[0])
        self.detector.set_omega_increment(self.angle_per_frame)
        self.detector.set_image_nr_start(self.image_nr_start)
        beam_center_x, beam_center_y = self.beam_center.get_beam_center()
        self.detector.set_beam_center_x(beam_center_x)
        self.detector.set_beam_center_y(beam_center_y)
        self.detector.set_detector_distance(self.beam_center.get_detector_distance() / 1000.)
        return self.detector.arm()
        
    def program_goniometer(self):
        if self.goniometer.md2.backlightison == True:
            self.goniometer.md2.backlightison = False
        self.goniometer.set_scan_range(self.scan_range)
        self.goniometer.set_scan_exposure_time(self.scan_exposure_time)
            
    def prepare(self):
        self.detector.check_dir(os.path.join(self.directory,'process'))
        self.detector.clear_monitor()
        self.detector.write_destination_namepattern(image_path=self.directory, name_pattern=self.name_pattern)
        self.status = 'prepare' 
    
    def collect(self):
        self.prepare()
        self.program_detector()
        self.program_goniometer()
        for scan_start_angle in self.scan_start_angles:
            self.goniometer.set_scan_start_angle(scan_start_angle)
            scan_id = self.goniometer.start_scan()
            while self.goniometer.md2.istaskrunning(scan_id):
                time.sleep(0.1)
        self.clean()
            
    def stop(self):
        self.goniometer.abort()
        self.detector.abort()

    def clean(self):
        self.detector.disarm() 

#class nested_helical(object):
    #def __init__(self,
                 #start,
                 #end,
                 #nested_vertical_range,
                 #total_scan_range,
                 #exposure_per_image,
                 #scan_start_angle,
                 #angle_per_frame,
                 #name_pattern,
                 #directory='/nfs/ruchebis/spool/2016_Run3/orphaned_collects',
                 #image_nr_start=1):
                     
        #self.goniometer = goniometer()
        #self.detector = detector()
        
        #self.beam_center = beam_center()
        
        #self.detector.set_trigger_mode('exts')
        
        #scan_range = float(scan_range)
        #scan_exposure_time = float(scan_exposure_time)
        
        #nimages, rest = divmod(scan_range, angle_per_frame)
        
        #if rest > 0:
            #nimages += 1
            #scan_range += rest*angle_per_frame
            #scan_exposure_time += rest*angle_per_frame/scan_range
            
        #frame_time = scan_exposure_time/nimages
        
        #self.scan_range = scan_range
        #self.scan_exposure_time = scan_exposure_time
        #self.scan_start_angle = scan_start_angle
        #self.angle_per_frame = angle_per_frame
        
        #self.nimages = int(nimages)
        #self.frame_time = float(frame_time)
        #self.count_time = self.frame_time - self.detector.get_detector_readout_time()
        
        #self.name_pattern = name_pattern
        #self.directory = directory
        #self.image_nr_start = image_nr_start
        #self.helical = helical
        #self.status = None               
                     
class sweep(object):
    
    def __init__(self,
                 scan_range,
                 scan_exposure_time,
                 scan_start_angle,
                 angle_per_frame,
                 name_pattern,
                 directory='/nfs/ruchebis/spool/2016_Run3/orphaned_collects',
                 image_nr_start=1,
                 helical=False):
        
        self.goniometer = goniometer()
        self.detector = detector()
        self.beam_center = beam_center()
        
        self.detector.set_trigger_mode('exts')
        self.detector.set_nimages_per_file(100)
        self.detector.set_ntrigger(1)
        scan_range = float(scan_range)
        scan_exposure_time = float(scan_exposure_time)
        
        nimages, rest = divmod(scan_range, angle_per_frame)
        
        if rest > 0:
            nimages += 1
            scan_range += rest*angle_per_frame
            scan_exposure_time += rest*angle_per_frame/scan_range
            
        frame_time = scan_exposure_time/nimages
        
        self.scan_range = scan_range
        self.scan_exposure_time = scan_exposure_time
        self.scan_start_angle = scan_start_angle
        self.angle_per_frame = angle_per_frame
        
        self.nimages = int(nimages)
        self.frame_time = float(frame_time)
        self.count_time = self.frame_time - self.detector.get_detector_readout_time()
        
        self.name_pattern = name_pattern
        self.directory = directory
        self.image_nr_start = image_nr_start
        self.helical = helical
        self.status = None
    
    def program_goniometer(self):
        self.goniometer.md2.backlightison = False
        self.goniometer.set_scan_start_angle(self.scan_start_angle)
        self.goniometer.set_scan_range(self.scan_range)
        self.goniometer.set_scan_exposure_time(self.scan_exposure_time)
        
    def program_detector(self, ntrigger=1):
        self.detector.set_ntrigger(ntrigger)
        if self.detector.get_compression() != 'bslz4':
            self.detector.set_compression('bslz4')
        if self.detector.get_trigger_mode() != 'exts':
            self.detector.set_trigger_mode('exts')
        self.detector.set_nimages(self.nimages)
        if self.nimages > 100:
            self.detector.set_nimages_per_file(100)
        self.detector.set_frame_time(self.frame_time)
        self.detector.set_count_time(self.count_time)
        self.detector.set_name_pattern(self.name_pattern)
        self.detector.set_omega(self.scan_start_angle)
        self.detector.set_omega_increment(self.angle_per_frame)
        self.detector.set_image_nr_start(self.image_nr_start)
        beam_center_x, beam_center_y = self.beam_center.get_beam_center()
        self.detector.set_beam_center_x(beam_center_x)
        self.detector.set_beam_center_y(beam_center_y)
        self.detector.set_detector_distance(self.beam_center.get_detector_distance() / 1000.)
        return self.detector.arm()
        
    def prepare(self):
        self.detector.check_dir(os.path.join(self.directory,'process'))
        self.detector.clear_monitor()
        self.detector.write_destination_namepattern(image_path=self.directory, name_pattern=self.name_pattern)
        self.status = 'prepare'
        
    def collect(self):
        self.prepare()
        self.program_goniometer()
        self.series_id = self.program_detector()['sequence id']
        self.status = 'collect'
        if self.helical:
            return self.goniometer.start_helical_scan()
        return self.goniometer.start_scan()
    
    def stop(self):
        self.goniometer.abort()
        self.detector.abort()

    def clean(self):
        self.detector.disarm()

    

if __name__ == '__main__':
    import optparse
    parser = optparse.OptionParser() 
    # testbed ip 62.12.151.50
    parser.add_option('-i', '--ip', default="172.19.10.26", type=str, help='IP address of the server')
    parser.add_option('-p', '--port', default=80, type=int, help='port on which to which it listens to')
    
    options, args = parser.parse_args()
     
    d = detector(host=options.ip, port=options.port)
    g = goniometer()