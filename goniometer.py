#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import gevent
import numpy as np
from math import sin, cos, atan2, radians, sqrt
import logging
import traceback
import time
import datetime
import copy
from scipy.optimize import minimize

try:
    import lmfit
    from lmfit import fit_report
except ImportError:
    logging.warning(
        "Could not lmfit minimization library, "
        + "refractive model centring will not work."
    )

try:
    if sys.version_info.major == 3:
        import tango
    else:
        import PyTango as tango
except ImportError:
    print('goniometer could not import tango')

from md2_mockup import md2_mockup


def get_voxel_calibration(vertical_step, horizontal_step):
    calibration = np.ones((3,))
    calibration[0] = horizontal_step
    calibration[1:] = vertical_step
    return calibration

def get_origin(parameters, position_key='reference_position'):
    p = parameters[position_key]
    o = np.array([p['CentringX'], p['CentringY'], p['AlignmentY'], p['AlignmentZ']])
    return o

def get_points_in_goniometer_frame(points_px, calibration, origin, center=np.array([160, 256, 256]), directions=np.array([-1,-1,1]), order=[1,2,0]):
    points_mm = ((points_px-center)*calibration*directions)[:, order] + origin
    return points_mm

def get_points_in_camera_frame(points_mm, calibration, origin, center=np.array([160, 256, 256]), directions=np.array([-1, -1, 1]), order=[1,2,0]):
    mm = points_mm - origin
    mm = mm[:, order[::-1]]
    mm *= directions
    mm /= calibration
    points_px = mm + center
    return points_px
        
def add_shift(position, shift, keys=['CentringX', 'CentringY', 'AlignmentY', 'AlignmentZ']):
    shifted_position = {}
    for k, key in enumerate(keys):
        shifted_position[key] = position[key] + shift[k]
    return shifted_position

def get_shift(position, reference, keys=['CentringX', 'CentringY', 'AlignmentY', 'AlignmentZ']):
    p = get_vector_from_position(position, keys=keys)
    r = get_vector_from_position(reference, keys=keys)
    shift = p - r
    return shift

def get_position_from_vector(v, keys=['CentringX', 'CentringY', 'AlignmentY', 'AlignmentZ', 'AlignmentX', 'Kappa', 'Phi']):
    return dict([(key, value) for key, value in zip(keys, v)])
                 
def get_vector_from_position(p, keys=['CentringX', 'CentringY', 'AlignmentY', 'AlignmentZ', 'AlignmentX', 'Kappa', 'Phi']):
    return np.array([p[key] for key in keys if key in p])

def get_distance(p1, p2, keys=['CentringX', 'CentringY']):
    return np.linalg.norm(get_vector_from_position(p1, keys=keys) - get_vector_from_position(p2, keys=keys))

def get_reduced_point(p, keys=['CentringX', 'CentringY']):
    return dict([(key, value) for key, value in p.items() if key in keys])

def copy_position(p):
    new_position = {}
    for key in p:
        new_position[key] = p[key]
    return position

def get_point_between(p1, p2, keys=['CentringX', 'CentringY', 'AlignmentY', 'AlignmentZ']):
    v1 = get_vector_from_position(p1, keys=keys)
    v2 = get_vector_from_position(p2, keys=keys)
    v = v1 + (v2-v1)/2.
    p = get_position_from_vector(v, keys=keys)
    return p

# minikappa translational offsets
def circle_model(angle, center, amplitude, phase):
    return center + amplitude*np.cos(angle - phase)

def line_and_circle_model(kappa, intercept, growth, amplitude, phase):
    return intercept + kappa*growth + amplitude * np.sin(np.radians(kappa) - phase)

def amplitude_y_model(kappa, amplitude, amplitude_residual, amplitude_residual2):
    return amplitude * np.sin(0.5*kappa) + amplitude_residual * np.sin(kappa) + amplitude_residual2 * np.sin(2*kappa)

def get_alignmentz_offset(kappa, phi):
    return 0

def get_alignmenty_offset(kappa, phi, 
                            center_center=-2.2465475, 
                            center_amplitude=0.3278655, 
                            center_phase=np.radians(269.3882546),
                            amplitude_amplitude=0.47039019, 
                            amplitude_amplitude_residual=0.01182333, 
                            amplitude_amplitude_residual2=0.00581796, 
                            phase_intercept=-4.7510392, 
                            phase_growth=0.5056157, 
                            phase_amplitude=-2.6508604, 
                            phase_phase=np.radians(14.9266433)):
    
    center = circle_model(kappa, center_center, center_amplitude, center_phase)
    amplitude = amplitude_y_model(kappa, amplitude_amplitude, amplitude_amplitude_residual, amplitude_amplitude_residual2)
    phase = line_and_circle_model(kappa, phase_intercept, phase_growth, phase_amplitude, phase_phase)
    phase = np.mod(phase, 180)
    
    alignmenty_offset = circle_model(phi, center, amplitude, np.radians(phase))

    return alignmenty_offset


def amplitude_cx_model(kappa, *params):
    kappa = np.mod(kappa, 2*np.pi)
    powers = []
    
    if type(params) == tuple and len(params) < 2:
        params = params[0]
    else:
        params = np.array(params)
        
    if len(params.shape) > 1:
        params = params[0]
        
    params = np.array(params)
    
    params = params[:-2]
    amplitude_residual, phase_residual = params[-2:]
    
    kappa = np.array(kappa)

    for k in range(len(params)):
        powers.append(kappa**k)
    
    powers = np.array(powers)
    
    return np.dot(powers.T, params) + amplitude_residual * np.sin(2*kappa - phase_residual)

def amplitude_cx_residual_model(kappa, amplitude, phase):
    return amplitude * np.sin(2*kappa - phase)


def phase_error_model(kappa, amplitude, phase, frequency):
    return amplitude * np.sin(frequency*np.radians(kappa) - phase)

def get_centringx_offset(kappa, phi, 
                            center_center=0.5955864, 
                            center_amplitude=0.7738802, 
                            center_phase=np.radians(222.1041400), 
                            amplitude_p1=0.63682813, 
                            amplitude_p2=0.02332819, 
                            amplitude_p3=-0.02999456, 
                            amplitude_p4=0.00366993, 
                            amplitude_residual=0.00592784, 
                            amplitude_phase_residual=1.82492612, 
                            phase_intercept=25.8526552, 
                            phase_growth=1.0919045, 
                            phase_amplitude=-12.4088622, 
                            phase_phase=np.radians(96.7545812), 
                            phase_error_amplitude=1.23428124, 
                            phase_error_phase=0.83821785, 
                            phase_error_frequency=2.74178863,
                            amplitude_error_amplitude=0.00918566,
                            amplitude_error_phase=4.33422268):

    amplitude_params = [amplitude_p1, amplitude_p2, amplitude_p3, amplitude_p4, amplitude_residual, amplitude_phase_residual]
    
    center = circle_model(kappa, center_center, center_amplitude, center_phase)
    amplitude = amplitude_cx_model(kappa, *amplitude_params)
    amplitude_error = amplitude_cx_residual_model(kappa, amplitude_error_amplitude, amplitude_error_phase)
    amplitude -= amplitude_error
    phase = line_and_circle_model(kappa, phase_intercept, phase_growth, phase_amplitude, phase_phase)
    phase_error = phase_error_model(kappa, phase_error_amplitude, phase_error_phase, phase_error_frequency)
    phase -= phase_error
    phase = np.mod(phase, 180)
    
    centringx_offset = circle_model(phi, center, amplitude, np.radians(phase))

    return centringx_offset
    
def amplitude_cy_model(kappa, center, amplitude, phase, amplitude_residual, phase_residual):
    return center + amplitude*np.sin(kappa - phase) + amplitude_residual*np.sin(kappa*2 - phase_residual)

def get_centringy_offset(kappa, phi,
                            center_center=0.5383092, 
                            center_amplitude=-0.7701891, 
                            center_phase=np.radians(137.6146006), 
                            amplitude_center=0.56306051, 
                            amplitude_amplitude=-0.06911649, 
                            amplitude_phase=0.77841959, 
                            amplitude_amplitude_residual=0.03132799, 
                            amplitude_phase_residual=-0.12249943, 
                            phase_intercept=146.9185176, 
                            phase_growth=0.8985232,
                            phase_amplitude=-17.5015172, 
                            phase_phase=-409.1764969, 
                            phase_error_amplitude=1.18820494, 
                            phase_error_phase=4.12663751, 
                            phase_error_frequency=3.11751387):
    
    center = circle_model(kappa, center_center, center_amplitude, center_phase) #3   
    amplitude = amplitude_cy_model(kappa, amplitude_center, amplitude_amplitude, amplitude_phase, amplitude_amplitude_residual, amplitude_phase_residual) #5
    phase = line_and_circle_model(kappa, phase_intercept, phase_growth, phase_amplitude, phase_phase) #4
    phase_error = phase_error_model(kappa, phase_error_amplitude, phase_error_phase, phase_error_frequency) #3
    phase -= phase_error
    phase = np.mod(phase, 180)
    centringy_offset = circle_model(phi, center, amplitude, np.radians(phase))
    return centringy_offset


class goniometer(object):
    motorsNames = ['AlignmentX', 
                   'AlignmentY', 
                   'AlignmentZ',
                   'CentringX', 
                   'CentringY']
                   
    motorShortNames = ['PhiX', 'PhiY', 'PhiZ', 'SamX', 'SamY']
    mxcubeShortNames = ['phix', 'phiy', 'phiz', 'sampx', 'sampy']
    
    shortFull = dict(list(zip(motorShortNames, motorsNames)))
    phiy_direction=-1.
    phiz_direction=1.
    
    motor_name_mapping = [
        ("AlignmentX", "phix"),
        ("AlignmentY", "phiy"),
        ("AlignmentZ", "phiz"),
        ("CentringX", "sampx"),
        ("CentringY", "sampy"),
        ("Omega", "phi"),
        ("Kappa", "kappa"),
        ("Phi", "kappa_phi"),
        ("beam_x", "beam_x"),
        ("beam_y", "beam_y"),
    ]
    
    #2019
    #kappa_direction=[0.29636375,  0.29377944, -0.90992064],
    #kappa_position=[-0.30655466, -0.3570731, 0.52893628],
    #phi_direction=[0.03149443, 0.03216924, -0.99469729],
    #phi_position=[-0.01467116, -0.08069945, 0.46818622],
    
    # 2023-11
    #kappa_direction=[0.2857927293557859, 0.29825779103438044, -0.9106980618759559],
    #kappa_position=[0.05983227148328028, -0.17418159369049926, -0.3931170045291165],
    #phi_direction=[0.05080273994228953, -0.006002697566495966, -1.0134508568340999],
    #phi_position=[0.21993931768594516, -0.03147698225694694, -2.0073121331928205],
    
    # 2023-12
    #kappa_direction=[0.2699214563788776, 0.2838993348700071, -1.0018455551762098],
    #kappa_position=[0.7765041345864001, 0.5561954243441296, -2.6141365642906167],
    #phi_direction=[0.017981988387908307, 0.04904500982612193, -1.1478709640779396],
    #phi_position=[0.2813575203934254, -0.030698833411830308, -2.584128291492474],
    def __init__(self, monitor_sleep_time=0.05, 
                 kappa_direction=[0.2857927293557859, 0.29825779103438044, -0.9106980618759559],
                 kappa_position=[0.05983227148328028, -0.17418159369049926, -0.3931170045291165],
                 phi_direction=[0.05080273994228953, -0.006002697566495966, -1.0134508568340999],
                 phi_position=[0.21993931768594516, -0.03147698225694694, -2.0073121331928205],             
                 align_direction=[0, 0, -1]):
        try:
            self.md2 = tango.DeviceProxy('i11-ma-cx1/ex/md2')
        except:
            from md2_mockup import md2_mockup
            self.md2 = md2_mockup()
            
        self.monitor_sleep_time = monitor_sleep_time
        self.observe = None
        self.kappa_axis = self.get_axis(kappa_direction, kappa_position)
        self.phi_axis = self.get_axis(phi_direction, phi_position)
        self.align_direction = align_direction
        #self.redis = redis.StrictRedis()
        self.observation_fields = ['chronos', 'Omega'] 
        self.observations = []
        self.centringx_direction = -1.
        self.centringy_direction = +1.
        self.alignmenty_direction = -1.
        self.alignmentz_direction = +1.
        
        self.md2_to_mxcube = dict(
            [(key, value) for key, value in self.motor_name_mapping]
        )
        self.mxcube_to_md2 = dict(
            [(value, key) for key, value in self.motor_name_mapping]
        )
        
    def set_scan_start_angle(self, scan_start_angle):
        self.md2.scanstartangle = scan_start_angle
    
    def get_scan_start_angle(self):
        return self.md2.scanstartangle
       
    def set_scan_range(self, scan_range):
        self.md2.scanrange = scan_range
        
    def get_scan_range(self):
        return self.md2.scanrange
        
    def set_scan_exposure_time(self, scan_exposure_time):
        self.md2.scanexposuretime = scan_exposure_time
    
    def get_scan_exposure_time(self):
        return self.md2.scanexposuretime
    
    def set_scan_number_of_frames(self, scan_number_of_frames):
        try:
            if self.get_scan_number_of_frames() != scan_number_of_frames:
                self.md2.scannumberofframes = scan_number_of_frames
        except:
            logging.info(traceback.format_exc())
       
    def get_scan_number_of_frames(self):
        return self.md2.scannumberofframes
        
    def set_collect_phase(self):
        return self.set_data_collection_phase()
        
    def abort(self):
        return self.md2.abort()

    def start_scan(self, number_of_attempts=3, wait=False):
        tried = 0
        while tried < number_of_attempts:
            tried += 1
            try:
                self.task_id = self.md2.startscan()
                break
            except:
                self.wait()
        if wait:
            self.wait_for_task_to_finish(self.task_id)
        return self.task_id

    def omega_scan(self, start_angle, scan_range, exposure_time, frame_number=1, number_of_passes=1, number_of_attempts=7, wait=True):
        start_angle = '%6.4f' % start_angle
        scan_range = '%6.4f' % scan_range
        exposure_time = '%6.4f' % exposure_time
        frame_number = '%d' % frame_number
        number_of_passes = '%d' % number_of_passes
        parameters = [frame_number, start_angle, scan_range, exposure_time, number_of_passes]
        tried = 0
        self.wait()
        while tried < number_of_attempts:
            tried += 1
            try:
                self.task_id = self.md2.startscanex(parameters)
                break
            except:
                self.wait()        
        if wait:
            self.wait_for_task_to_finish(self.task_id)
        return self.task_id

    def vertical_helical_scan(self, vertical_scan_length, position, scan_start_angle, scan_range, scan_exposure_time, wait=True):
        start = {}
        stop = {}
        for motor in position:
            if motor == 'AlignmentZ':
                start[motor] = position[motor] + vertical_scan_length/2.
                stop[motor] = position[motor] - vertical_scan_length/2.
            else:
                start[motor] = position[motor]
                stop[motor] = position[motor]
                
        return self.helical_scan(start, stop, scan_start_angle, scan_range, scan_exposure_time, wait=wait)
            
    def helical_scan(self, start, stop, scan_start_angle, scan_range, scan_exposure_time, number_of_attempts=7, wait=True, sleeptime=0.5):
        scan_start_angle = '%6.4f' % (scan_start_angle % 360., )
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
        parameters = [scan_start_angle, scan_range, scan_exposure_time, start_y, start_z, start_cx, start_cy, stop_y, stop_z, stop_cx, stop_cy]
       
        tried = 0
        while tried < number_of_attempts:
            tried += 1
            try:
                self.task_id = self.start_scan_4d_ex(parameters)
                break
            except:
                gevent.sleep(sleeptime)
        if wait:
            self.wait_for_task_to_finish(self.task_id)
        return self.task_id

    def start_helical_scan(self):
        return self.md2.startscan4d()
    
    def start_scan_4d_ex(self, parameters):
        return self.md2.startScan4DEx(parameters)
    
    def set_helical_start(self):
        return self.md2.setstartscan4d()
    
    def set_helical_stop(self):
        return self.md2.setstopscan4d()
        
    def start_raster_scan(self, vertical_range, horizontal_range, number_of_rows, number_of_columns, direction_inversion):
        return self.md2.startRasterScan([vertical_range, horizontal_range, number_of_rows, number_of_columns, direction_inversion])
    
    def get_motor_state(self, motor_name):
        if isinstance(self.md2, md2_mockup):
            return 'STANDBY'
        else:
            return self.md2.getMotorState(motor_name).name
      
    def get_status(self):
        try:
            return self.md2.read_attribute('Status').value
        except:
            return 'Unknown'
            
    def get_state(self):
        # This solution takes approximately 2.4 ms on average
        try:
            return self.md2.read_attribute('State').value.name
        except:
            return 'UNKNOWN'
        
        # This solution takes approximately 15.5 ms on average
        #motors = ['Omega', 'AlignmentX', 'AlignmentY', 'AlignmentZ', 'CentringX', 'CentringY', 'ApertureHorizontal', 'ApertureVertical', 'CapillaryHorizontal', 'CapillaryVertical', 'ScintillatorHorizontal', 'ScintillatorVertical', 'Zoom']
        #state = set([self.get_motor_state(m) for m in motors])
        #if len(state) == 1 and 'STANDBY' in state:
            #return 'STANDBY'
        #else:
            #return 'MOVING'
            
    def wait(self, device=None, timeout=7):
        green_light = False
        last_state = None
        last_status = None
        _start = time.time()
        while green_light is False and (time.time()-_start)<timeout:
            state = self.get_state()
            status = self.get_status()
            try:
                if device is None:
                    if state.lower() in ['moving', 'running', 'unknown']:
                        if state != last_state:
                            logging.info("MD2 wait" )
                            last_state = state
                    elif status.lower() in ['running', 'unknown', 'setting beamlocation phase', 'setting transfer phase', 'setting centring phase', 'setting data collection phase']:
                        if status != last_status:
                            logging.info("MD2 wait" )
                            last_status = status
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
            gevent.sleep(.1)
        

    def move_to_position(self, position={}, epsilon = 0.0002):
        if position != {}:
            for motor in position:
                while abs(self.md2.read_attribute('%sPosition' % self.shortFull[motor]).value - position[motor]) > epsilon:
                    self.wait()
                    gevent.sleep(0.5)
                    self.md2.write_attribute('%sPosition' % self.shortFull[motor], position[motor])
                
            self.wait()
        self.wait()
        return
    
    def get_head_type(self):
        return self.md2.headtype
    
    def has_kappa(self):
        return self.get_head_type() == 'MiniKappa'
    
    def set_position(self, position, number_of_attempts=3, wait=True, collect_auxiliary_images=False, ignored_motors=['beam_x', 'beam_y', 'kappa', 'kappa_phi']):
        
        if not self.has_kappa():
            ignored_motors += ['Phi', 'Kappa']
        motor_name_value_list = ['%s=%6.4f' % (motor, position[motor]) for motor in position if position[motor] != None and (motor not in ignored_motors)]
        command_string = ','.join(motor_name_value_list)
        
        k=0
        task_id = None
        success = False
        self.auxiliary_images = []
        
        while k < number_of_attempts and success == False:
            k+=1
            try:
                task_id = self.md2.startSimultaneousMoveMotors(command_string)
                success = True
            except:
                gevent.sleep(1)
        if wait == True and task_id != None:
            self.wait_for_task_to_finish(task_id)
        return task_id
        
    def save_aperture_and_capillary_beam_positions(self):
        self.md2.saveaperturebeamposition()
        self.md2.savecapillarybeamposition()
 
    def get_omega_position(self):
        return self.get_position()['Omega']
    
    def get_kappa_position(self):
        return self.get_position()['Kappa']
    
    def set_kappa_position(self, kappa_position):
        current_position = self.get_aligned_position()
        current_kappa = current_position['Kappa']
        current_phi = current_position['Phi']

        x = self.get_x()
        
        shift = self.get_shift(current_kappa, current_phi, x, kappa_position, current_phi)
        
        destination = copy.deepcopy(current_position)
        destination['CentringX'] = shift[0]
        destination['CentringY'] = shift[1]
        destination['AlignmentY'] = shift[2]
        #destination['AlignmentZ'] += (az_destination_offset - az_current_offset)
        destination['Kappa'] = kappa_position
        
        self.set_position(destination)
    
    def get_phi_position(self):
        return self.get_position()['Phi']
    
    def set_phi_position(self, phi_position):
        current_position = self.get_aligned_position()
        current_kappa = current_position['Kappa']
        current_phi = current_position['Phi']
        
        x = self.get_x()
        
        shift = self.get_shift(current_kappa, current_phi, x, current_kappa, phi_position)
        
        destination = copy.deepcopy(current_position)
        #destination['CentringX'] = shift[0]
        #destination['CentringY'] = shift[1]
        #destination['AlignmentY'] = shift[2]
        #destination['AlignmentZ'] += (az_destination_offset - az_current_offset)
        destination['Phi'] = phi_position

        self.set_position(destination)
    
    def set_kappa_phi_position(self, kappa_position, phi_position):
        
        current_position = self.get_aligned_position()
        current_kappa = current_position['Kappa']
        current_phi = current_position['Phi']

        x = self.get_x()
        
        shift = self.get_shift(current_kappa, current_phi, x, kappa_position, phi_position)
        
        destination = copy.deepcopy(current_position)
        destination['CentringX'] = shift[0]
        destination['CentringY'] = shift[1]
        destination['AlignmentY'] = shift[2]
        #destination['AlignmentZ'] += (az_destination_offset - az_current_offset)
        destination['Kappa'] = kappa_position
        destination['Phi'] = phi_position
        
        self.set_position(destination)
        
    def get_chi_position(self):
        return self.get_position()['Chi']
    
    
    def get_x(self):
        current_position = self.get_aligned_position()
        return [current_position[motor] for motor in ['CentringX', 'CentringY', 'AlignmentY']]
    
    def get_centringx_position(self):
        return self.get_position()['CentringX']
    
    def get_centringy_position(self):
        return self.get_position()['CentringY']
    
    def get_alignmentx_position(self):
        return self.get_position()['AlignmentX']
    
    def get_alignmenty_position(self):
        return self.get_position()['AlignmentY']
    
    def get_alignmentz_position(self):
        return self.get_position()['AlignmentZ']
    
    def get_centringtablevertical_position(self):
        return self.get_position()['CentringTableVertical']
    
    def get_centringtablefocus_position(self):
        return self.get_position()['CentringTableFocus']

    def get_zoom_position(self):
        return self.get_position()['Zoom']
    
    def get_beam_x_position(self):
        return 0
    
    def get_beam_y_position(self):
        return 0.
    
    def get_centring_x_y_tabledisplacement(self):
        x = self.get_centringx_position()
        y = self.get_centringy_position()
        return x, y, sqrt(x**2 + y**2)

    def get_omega_alpha_and_centringtabledisplacement(self):
        omega = radians(self.get_omega_position())
        x, y, centringtabledisplacement = self.get_centring_x_y_tabledisplacement()
        alpha = atan2(y, -x)
        return omega, alpha, centringtabledisplacement
        
    def get_centringtablevertical_position_abinitio(self):
        omega, alpha, centringtabledisplacement = self.get_omega_alpha_and_centringtabledisplacement()
        return sin(omega + alpha) * centringtabledisplacement
        
    def get_centringtablefocus_position_abinitio(self):
        omega, alpha, centringtabledisplacement = self.get_omega_alpha_and_centringtabledisplacement()
        return cos(omega + alpha) * centringtabledisplacement
        
    def get_centringtable_vertical_position_from_hypothetical_centringx_centringy_and_omega(self, x, y, omega):
        d = sqrt(x**2 + y**2)
        alpha = atan2(y, -x)
        omega = radians(omega)
        return sin(omega + alpha) * d
      
    def get_centringtable_focus_position_from_hypothetical_centringx_centringy_and_omega(self, x, y, omega):
        d = sqrt(x**2 + y**2)
        alpha = atan2(y, -x)
        omega = radians(omega)
        return cos(omega + alpha) * d
    
    def get_focus_and_vertical_from_position(self, position=None):
        if position is None:
            position = self.get_aligned_position()
        x = position['CentringX']
        y = position['CentringY']
        omega = position['Omega']
        focus, vertical = self.get_focus_and_vertical(x, y, omega)
        return focus, vertical
    
    def get_aligned_position_from_reference_position_and_shift(self, reference_position, horizontal_shift, vertical_shift, AlignmentZ_reference=0.0944, epsilon=1e-3):
        
        alignmentz_shift = reference_position['AlignmentZ'] - AlignmentZ_reference
        if abs(alignmentz_shift) < epsilon:
            alignmentz_shift = 0
        
        vertical_shift += alignmentz_shift
        
        #centringx_shift, centringy_shift = self.goniometer.get_x_and_y(0, vertical_shift, reference_position['Omega']) : changed by Elke
        centringx_shift, centringy_shift = self.get_x_and_y(0, vertical_shift, reference_position['Omega'])
        
        aligned_position = copy.deepcopy(reference_position)
        
        aligned_position['AlignmentZ'] -= alignmentz_shift
        aligned_position['AlignmentY'] -= horizontal_shift
        aligned_position['CentringX'] += centringx_shift
        aligned_position['CentringY'] += centringy_shift
        
        return aligned_position
        
    def get_aligned_position_from_reference_position_and_x_and_y(self, reference_position, x, y, AlignmentZ_reference=0.0944):
        horizontal_shift = x - reference_position['AlignmentY']
        vertical_shift = y - reference_position['AlignmentZ']
        
        return self.get_aligned_position_from_reference_position_and_shift(reference_position, horizontal_shift, vertical_shift, AlignmentZ_reference=AlignmentZ_reference)
        
    
    def get_x_and_y(self, focus, vertical, omega):
        omega = -radians(omega)
        R = np.array([[cos(omega), -sin(omega)], [sin(omega), cos(omega)]])
        R = np.linalg.pinv(R)
        return np.dot(R, [-focus, vertical])
    
    def get_focus_and_vertical(self, x, y, omega):
        omega = radians(omega)
        R = np.array([[cos(omega), -sin(omega)], [sin(omega), cos(omega)]])
        return np.dot(R, [-x, y])
    
    
    def get_centring_x_y_for_given_omega_and_vertical_position(self, omega, vertical_position, focus_position, C=1., l=1., nruns=10):
        from scipy.optimize import minimize
        import random
        def vertical_position_model(x, y, omega):
            d = sqrt(x**2 + y**2)
            alpha = atan2(y, -x)
            omega = radians(omega)
            return sin(omega + alpha) * d

        def focus_position_model(x, y, omega):
            d = sqrt(x**2 + y**2)
            alpha = atan2(y, -x)
            omega = radians(omega)
            return cos(omega + alpha) * d
            
        def error(varse, omega, truth_vertical, truth_focus, C=C, l=l):
            x, y = varse
            model_vertical = vertical_position_model(x, y, omega)
            model_focus = focus_position_model(x, y, omega)
            return C*(abs(truth_vertical-model_vertical) + abs(truth_focus-model_focus)) + l*(x**2 + y**2)
        
        def fit(nruns=nruns):
            results = []
            for run in range(int(nruns)):
                initial_parameters = [random.random(), random.random()]
                fit_results = minimize(error, initial_parameters, method='nelder-mead', args=(omega, vertical_position, focus_position))
                results.append(fit_results.x)
            results = np.array(results)
            return np.median(results, axis=0)
        
        x, y = fit(nruns=nruns)
        return x, y
        
    def get_analytical_centring_x_y_for_given_omega_and_vertical_position(self, omega, vertical_position, focus_position):
        omega = radians(omega)
        alpha = atan2(vertical, focus) - omega
        y_over_x = tan(alpha)
        
    def get_position(self):
        return dict([(m.split('=')[0], float(m.split('=')[1])) for m in self.md2.motorpositions])
    
    def get_aligned_position(self, motor_names=['AlignmentX', 'AlignmentY', 'AlignmentZ', 'CentringY', 'CentringX', 'Kappa', 'Phi', 'Omega']):
        return dict([(m.split('=')[0], float(m.split('=')[1])) for m in self.md2.motorpositions if m.split('=')[0] in motor_names and m.split('=')[1] != 'NaN'])
    
    def get_state_vector(self, motor_names=['Omega', 'Kappa', 'Phi', 'CentringX', 'CentringY', 'AlignmentX', 'AlignmentY', 'AlignmentZ', 'ScintillatorVertical', 'Zoom']):
        motor_positions_dictionary = self.get_motor_positions_dictionary()
        return [motor_positions_dictionary[motor_name] for motor_name in motor_names]
        #return [m.split('=')[1] for m in motor_positions if m.split('=')[0] in motor_names]

    def get_motor_positions_dictionary(self, motor_names=['Omega', 'Kappa', 'Phi', 'Chi', 'CentringX', 'CentringY', 'AlignmentX', 'AlignmentY', 'AlignmentZ', 'ApertureVertical', 'ApertureHorizontal', 'CapillaryVertical', 'CapillaryHorizontal', 'ScintillatorVertical', 'ScintillatorHorizontal', 'BeamstopX', 'BeamstopY', 'BeamstopZ', 'Zoom', 'PlateTranslation', 'CentringTableVertical', 'CentringTableFocus', 'BeamstopDistance'], logfile='/nfs/data2/log/md2_motor_positions_problem.log'):
        try:
            motor_positions_dictionary = dict([item.split('=') for item in self.md2.motorpositions])
        except:
            message = 'failure to read md2.motorpositions attribute'
            logging.info(message)
            os.system('echo "{now:s} {message:s}" >> {logfile:s}'.format(logfile=logfile, message=message, now=str(datetime.datetime.now())))
            motor_positions_dictionary = dict((motor_name, np.nan) for motor_name in motor_names)
        return motor_positions_dictionary
    
    def sample_is_loaded(self, sample_size=7, sleeptime=0.01, timeout=3, logfile='/nfs/data2/log/md2_sample_detection_problem.log'):
        _start = time.time()
        sample_is_coherent = False
        all_answers = []
        while not sample_is_coherent and (time.time()-_start<timeout):
            is_loaded_sample = []
            for k in range(sample_size):
                is_loaded_sample.append(int(self.md2.SampleIsLoaded))
                gevent.sleep(np.random.random()*sleeptime)
            median = np.median(is_loaded_sample)
            mean = np.mean(is_loaded_sample)
            sample_is_coherent = median == mean
            if not sample_is_coherent:
                message = 'gonio sample detection is not coherent, you may want to check ...'
                logging.info(message)
                print(message)
                os.system('echo "{now:s} {is_loaded_sample:s}" >> {logfile:s}'.format(logfile=logfile, is_loaded_sample=str(is_loaded_sample), now=str(datetime.datetime.now())))
                all_answers += is_loaded_sample
        if not sample_is_coherent:
            median = np.median(all_answers)
        return bool(median)
    
        
    def insert_backlight(self, sleeptime=0.1, timeout=7):
        _start = time.time()
        self.wait()
        while not self.backlight_is_on() and (time.time()-_start) < timeout:
            try:
                self.md2.backlightison = True
            except:
                gevent.sleep(sleeptime)
        
    def insert_frontlight(self, sleeptime=0.1, timeout=7):
        _start = time.time()
        print('inserting frontlight')
        self.wait()
        while not self.frontlight_is_on() and (time.time()-_start) < timeout:
            try:
                self.md2.frontlightison = True
            except:
                gevent.sleep(sleeptime)
        print('success? %s' % self.frontlight_is_on())
        
    def extract_backlight(self):
        self.remove_backlight()
        
    def remove_backlight(self, sleeptime=0.1, timeout=7):
        _start = time.time()
        while self.backlight_is_on() and (time.time()-_start) < timeout:
            try:
                self.md2.backlightison = False
            except:
                gevent.sleep(sleeptime)
    
    def extract_frontlight(self, sleeptime=0.1, timeout=7):
        _start = time.time()
        print('extracting frontlight')
        while self.frontlight_is_on() and (time.time()-_start) < timeout:
            try:
                self.md2.frontlightison = False
            except:
                gevent.sleep(sleeptime)
        print('success? %s' % (not self.frontlight_is_on()))
        
    def backlight_is_on(self):
        return self.md2.backlightison

    def frontlight_is_on(self):
        return self.md2.frontlightison

    def get_backlightlevel(self):
        return self.md2.backlightlevel
    
    def set_backlightlevel(self, level=10, number_of_attempts=7, sleeptime=0.5):
        n = 0
        while self.md2.backlightlevel != level and n <= number_of_attempts:
            n += 1
            try:
                self.md2.backlightlevel = level
            except:
                gevent.sleep(sleeptime)
                
    def get_frontlightlevel(self):
        return self.md2.frontlightlevel
    
    def set_frontlightlevel(self, level=55, number_of_attempts=7, sleeptime=0.5):
        n = 0
        while self.md2.frontlightlevel != level and n <= number_of_attempts:
            n += 1
            try:
                self.md2.frontlightlevel = level
                success = True
            except:
                gevent.sleep(sleeptime)
                
    def insert_fluorescence_detector(self):
        self.md2.fluodetectorisback = False
    
    def extract_fluorescence_detector(self):
        self.md2.fluodetectorisback = True

    def insert_cryostream(self):
        self.md2.cryoisback = False
    
    def extract_cryostream(self):
        self.md2.cryoisback = True

    def is_task_running(self, task_id):
        return self.md2.istaskrunning(task_id)

    def get_last_task_info(self):
        return self.md2.lasttaskinfo
        
    def get_time_from_string(self, timestring, format='%Y-%m-%d %H:%M:%S.%f'):
        micros = float(timestring[timestring.find('.'):])
        return time.mktime(time.strptime(timestring, format)) + micros
        
    def get_last_task_start(self):
        lasttaskinfo = self.md2.lasttaskinfo
        start = lasttaskinfo[2]
        return self.get_time_from_string(start)
        
    def get_last_task_end(self):
        lasttaskinfo = self.md2.lasttaskinfo
        end = lasttaskinfo[3]
        if end == 'null':
            return None
        return self.get_time_from_string(end)
        
    def get_last_task_duration(self):
        start = self.get_last_task_start()
        end = self.get_last_task_end()
        if end == None:
            return time.time() - start
        return end - start
        
    
    def get_task_info(self, task_id):
        return self.md2.gettaskinfo(task_id)
        
    
    def set_detector_gate_pulse_enabled(self, value=True):
        self.md2.DetectorGatePulseEnabled = value
        
    
    def set_goniometer_phase(self, phase, wait=False, number_of_attempts=7, sleeptime=0.5):
        self.wait()
        k = 0
        while k < number_of_attempts:
            try:
                self.task_id = self.md2.startsetphase(phase)
                break
            except:
                gevent.sleep(sleeptime)
                k += 1
        if wait:
            self.wait_for_task_to_finish(self.task_id)
        else:
            return self.task_id
    
    
    def set_data_collection_phase(self, wait=False):
        self.save_position()
        self.set_goniometer_phase('DataCollection', wait=wait)
            
    
    def set_transfer_phase(self, transfer_position={'AlignmentZ': 0.1017, 'AlignmentY': -1.35, 'AlignmentX': -0.0157, 'CentringX': 0.431, 'CentringY': 0.210, 'ApertureVertical': 83, 'CapillaryVertical': 0., 'Zoom': 33524.0, 'Omega': 0}, phase=False, wait=False): #, 'Kappa': 0, 'Phi': 0
        #transfer_position={'AlignmentX': -0.42292751002956075, 'AlignmentY': -1.5267995679700732,  'AlignmentZ': -0.049934926625844867, 'ApertureVertical': 82.99996634818412, 'CapillaryHorizontal': -0.7518915227941564, 'CapillaryVertical': -0.00012483891752579357, 'CentringX': 1.644736842105263e-05, 'CentringY': 1.644736842105263e-05, 'Kappa': 0.0015625125024403275, 'Phi': 0.004218750006591797, 'Zoom': 34448.0}
        self.set_position(transfer_position, wait=wait)
        if phase:
            self.set_goniometer_phase('Transfer', wait=wait)
        

    
    def set_beam_location_phase(self, wait=False):
        if self.get_current_phase() != 'BeamLocation':
            self.save_position()
        self.set_goniometer_phase('BeamLocation', wait=wait)
    
    
    def set_centring_phase(self, wait=False):
        self.set_goniometer_phase('Centring', wait=wait)

    def get_current_phase(self):
        return self.md2.currentphase
    
    def save_position(self, number_of_attempts=15, sleeptime=0.2):
        success = False
        k = 0
        while success == False and k < number_of_attempts:
            k+=1
            self.wait()
            try:
                self.md2.savecentringpositions()
                success = True
            except:
                gevent.sleep(sleeptime)

    def wait_for_task_to_finish(self, task_id, collect_auxiliary_images=False, sleeptime=0.1):
        k = 0
        self.auxiliary_images = []
        while self.is_task_running(task_id):
            if k == 0:
               logging.info('waiting for task %d to finish' % task_id)
            gevent.sleep(sleeptime)
            k += 1

    def set_omega_relative_position(self, step):
        current_position = self.get_omega_position()
        return self.set_omega_position(self, current_position+step)
        
    def set_omega_position(self, omega_position, number_of_attempts=3, wait=True):
        return self.set_position({'Omega': omega_position}, wait=wait)
        
    def set_zoom(self, zoom, wait=True):
        self.md2.coaxialcamerazoomvalue = zoom
        if wait:
            self.wait()
            
    def get_orientation(self):
        return self.get_omega_position()
    
    def set_orientation(self, orientation):
        self.set_omega_position(orientation)
    
    def check_position(self, candidate_position):
        if isinstance(candidate_position, str):
            candidate_position = eval(candidate_position)
            
        if candidate_position is not None and not isinstance(candidate_position, dict):
            candidate_position = candidate_position.strip('}{')
            positions = candidate_position.split(',')
            keyvalues = [item.strip().split(':') for item in positions]
            keyvalues = [(item[0], float(item[1])) for item in keyvalues]
            candidate_position = dict(keyvalues)
        
        if isinstance(candidate_position, dict):
            current_position = self.get_aligned_position()
            for key in candidate_position:
                if candidate_position[key] is None and key in current_position:
                    candidate_position[key] = current_position[key]
            return candidate_position
        else:
            return self.get_aligned_position()
        
    def get_point(self):
        return self.get_position()
    
    def get_observation_fields(self):
        return self.observation_fields
    
    def monitor(self, start_time, motor_names=['Omega']):
        self.observations = []
        self.observation_fields = ['chronos'] + motor_names
        
        while self.observe == True:
            chronos = time.time() - start_time
            position = self.get_position()
            point = [chronos] +  [position[motor_name] for motor_name in motor_names]
            self.observations.append(point)
            gevent.sleep(self.monitor_sleep_time)
            
    def get_observations(self):
        return self.observations
    
    def get_observation_fields(self):
        return self.observation_fields
       
    def get_points(self):
        return np.array(self.observations)[:,1]
        
    def get_chronos(self):
        return np.array(self.observations)[:,0]
    
    def circle_model(self, angles, c, r, alpha):
        return c + r*np.cos(angles - alpha)
    
    def circle_model_residual(self, varse, angles, data):
        c, r, alpha = varse
        model = self.circle_model(angles, c, r, alpha)
        return 1./(2*len(model)) * np.sum(np.sum(np.abs(data - model)**2))

    def projection_model(self, angles, c, r, alpha):
        return c + r*np.cos(np.dot(2, angles) - alpha)

    def projection_model_residual(self, varse, angles, data):
        c, r, alpha = varse
        model = self.projection_model(angles, c, r, alpha)
        return 1./(2*len(model)) * np.sum(np.sum(np.abs(data - model)**2))

    def get_rotation_matrix(self, axis, angle):
        rads = np.radians(angle)
        cosa = np.cos(rads)
        sina = np.sin(rads)
        I = np.diag([1]*3)
        rotation_matrix = I * cosa + axis['mT'] * (1-cosa) + axis['mC'] * sina
        return rotation_matrix

    def get_axis(self, direction, position):
        axis = {}
        d = np.array(direction)
        p = np.array(position)
        axis['direction'] = d
        axis['position'] = p
        axis['mT'] = self.get_mT(direction)
        axis['mC'] = self.get_mC(direction)
        return axis

    def get_mC(self, direction):
        mC = np.array([[ 0.0, -direction[2], direction[1]],
                       [ direction[2], 0.0, -direction[0]],
                       [-direction[1], direction[0], 0.0]])

        return mC

    def get_mT(self, direction):
        mT = np.outer(direction, direction)
        
        return mT
        
    def get_shift(self, kappa1, phi1, x, kappa2, phi2):
        tk = self.kappa_axis['position']
        tp = self.phi_axis['position']
        
        Rk2 = self.get_rotation_matrix(self.kappa_axis, kappa2)
        Rk1 = self.get_rotation_matrix(self.kappa_axis, -kappa1)
        Rp = self.get_rotation_matrix(self.phi_axis, phi2-phi1)
        
        a = tk - np.dot((tk-x), Rk1.T)
        b = tp - np.dot((tp-a), Rp.T)
        
        shift = tk - np.dot((tk-b), Rk2.T)
        
        return shift

    def get_align_vector(self, t1, t2, kappa, phi):
        t1 = np.array(t1)
        t2 = np.array(t2)
        x = t1 - t2
        Rk = self.get_rotation_matrix(self.kappa_axis, -kappa)
        Rp = self.get_rotation_matrix(self.phi_axis, -phi)
        x = np.dot(Rp, np.dot(Rk, x))/np.linalg.norm(x)
        c = np.dot(self.phi_axis['direction'], x)
        if c < 0.:
            c = -c
            x = -x
        cos2a = pow(np.dot(self.kappa_axis['direction'], self.align_direction), 2)
        
        d = (c - cos2a)/(1 - cos2a)
        
        if abs(d) > 1.:
            new_kappa = 180.
        else:
            new_kappa = np.degrees(np.arccos(d))
        
        Rk = self.get_rotation_matrix(self.kappa_axis, new_kappa)
        pp = np.dot(Rk, self.phi_axis['direction'])
        xp = np.dot(Rk, x)
        d1 = self.align_direction - c*pp
        d2 = xp - c*pp
        
        new_phi = np.degrees(np.arccos(np.dot(d1, d2)/np.linalg.norm(d1)/np.linalg.norm(d2)))
        
        newaxis = {}
        newaxis['mT'] = self.get_mT(pp)
        newaxis['mC'] = self.get_mC(pp)
        
        Rp = self.get_rotation_matrix(newaxis, new_phi)
        d = np.abs(np.dot(self.align_direction, np.dot(Rp, xp)))
        check = np.abs(np.dot(self.align_direction, np.dot(xp, Rp)))
                    
        if check > d:
            new_phi = -new_phi
            
        shift = self.get_shift(kappa, phi, 0.5*(t1 + t2), new_kappa, new_phi)
        
        align_vector = new_kappa, new_phi, shift
        
        return align_vector
    
    def get_points_in_goniometer_frame(self, points, calibration, origin, center=np.array([160, 256, 256]), directions=np.array([-1,1,1]), order=[1,2,0]):
        mm = ((points-center)*calibration*directions)[:, order] + origin
        return mm

 
    def get_move_vector_dictionary_from_fit(self, fit_vertical, fit_horizontal):
        c, r, alpha = fit_vertical.x
        
        centringx_direction=-1
        centringy_direction=1.
        alignmenty_direction=-1.
        alignmentz_direction=1.
        
        d_sampx = centringx_direction * r * np.sin(alpha)
        d_sampy = centringy_direction * r * np.cos(alpha)
        d_y = alignmenty_direction * fit_horizontal.x[0]
        d_z = alignmentz_direction * c

        move_vector_dictionary = {
            "AlignmentZ": d_z,
            "AlignmentY": d_y,
            "CentringX": d_sampx,
            "CentringY": d_sampy,
        }

        return move_vector_dictionary
    
    def get_aligned_position_from_fit_and_reference(self, fit_vertical, fit_horizontal, reference):
        move_vector_dictionary = self.get_move_vector_dictionary_from_fit(fit_vertical, fit_horizontal)
        aligned_position = {}
        for key in reference:
            aligned_position[key] = reference[key]
            if key in move_vector_dictionary:
                aligned_position[key] += move_vector_dictionary[key]
        return aligned_position
    
    def get_move_vector_dictionary(self, vertical_displacements, horizontal_displacements, angles, calibrations, centringx_direction=-1, centringy_direction=1., alignmenty_direction=-1., alignmentz_direction=1., centring_model='circle'):
        
        if centring_model == 'refractive':
            initial_parameters = lmfit.Parameters()
            initial_parameters.add_many(
                ("c", 0.0, True, -5e3, +5e3, None, None),
                ("r", 0.0, True, 0.0, 4e3, None, None),
                ("alpha", -np.pi / 3, True, -2 * np.pi, 2 * np.pi, None, None),
                ("front", 0.01, True, 0.0, 1.0, None, None),
                ("back", 0.005, True, 0.0, 1.0, None, None),
                ("n", 1.31, True, 1.29, 1.33, None, None),
                ("beta", 0.0, True, -2 * np.pi, +2 * np.pi, None, None),
            )

            fit_y = lmfit.minimize(
                self.refractive_model_residual,
                initial_parameters,
                method="nelder",
                args=(angles, vertical_discplacements),
            )
            self.log.info(fit_report(fit_y))
            optimal_params = fit_y.params
            v = optimal_params.valuesdict()
            c = v["c"]
            r = v["r"]
            alpha = v["alpha"]
            front = v["front"]
            back = v["back"]
            n = v["n"]
            beta = v["beta"]
            c *= 1.e-3
            r *= 1.e-3
            front *= 1.e-3
            back *= 1.e-3
            
        elif centring_model == 'circle':
            initial_parameters = [np.mean(vertical_discplacements), np.std(vertical_discplacements)/np.sin(np.pi/4), np.random.rand()*np.pi]
            fit_y = minimize(
                self.circle_model_residual,
                initial_parameters,
                method="nelder-mead",
                args=(angles, vertical_discplacements),
            )

            c, r, alpha = fit_y.x
            c *= 1.e-3
            r *= 1.e-3
            v = {"c": c, "r": r, "alpha": alpha}

        horizontal_center = np.mean(horizontal_displacements)

        d_sampx = centringx_direction * r * np.sin(alpha)
        d_sampy = centringy_direction * r * np.cos(alpha)
        d_y = alignmenty_direction * horizontal_center
        d_z = alignmentz_direction * c

        move_vector_dictionary = {
            "AlignmentZ": d_z,
            "AlignmentY": d_y,
            "CentringX": d_sampx,
            "CentringY": d_sampy,
        }

        return move_vector_dictionary
    
    def get_point_coordinates_from_position(self, position):
        horizontal_shift = position_initial['AlignmentY'] - position_final['AlignmentY']
        
    def get_aligned_position_from_reference_position_and_shift(self, reference_position, horizontal_shift, vertical_shift, AlignmentZ_reference=0.0944, epsilon=1.e-3):
        
        alignmentz_shift = reference_position['AlignmentZ'] - AlignmentZ_reference
        if abs(alignmentz_shift) < epsilon:
            alignmentz_shift = 0
        
        vertical_shift += alignmentz_shift
        
        centringx_shift, centringy_shift = self.get_x_and_y(0, vertical_shift, reference_position['Omega'])
        
        aligned_position = copy.deepcopy(reference_position)
        
        aligned_position['AlignmentZ'] -= alignmentz_shift
        aligned_position['AlignmentY'] -= horizontal_shift
        aligned_position['CentringX'] += centringx_shift
        aligned_position['CentringY'] += centringy_shift
        #a_cx = r_cx + s_cx => s_cx = a_cx - r_cx
        #a_cy = r_cy + s_cy => s_cy = a_cy - r_cy
        #a_az = r_az - s_az => s_az = r_az - a_az
        #a_ay = r_ay - s_ay => s_ay = r_ay - a_ay
        return aligned_position

    def get_vertical_and_horizontal_shift_between_two_positions(self, aligned_position, reference_position, epsilon=1.e-3):
        horizontal_shift = reference_position['AlignmentY'] - aligned_position['AlignmentY']
        alignmentz_shift = reference_position['AlignmentZ'] - aligned_position['AlignmentZ']
        centringx_shift = aligned_position['CentringX'] - reference_position['CentringX'] 
        centringy_shift = aligned_position['CentringY'] - reference_position['CentringY']
        
        focus, vertical_shift = self.get_focus_and_vertical(centringx_shift, centringy_shift, reference_position['Omega'])
        if abs(alignmentz_shift) > epsilon:
            vertical_shift -= alignmentz_shift
            
        return np.array([vertical_shift, horizontal_shift])

    def translate_from_mxcube_to_md2(self, position):
        translated_position = {}

        for key in position:
            if isinstance(key, str):
                try:
                    translated_position[self.mxcube_to_md2[key]] = position[key]
                except:
                    pass
                    #self.log.exception(traceback.format_exc())
                
            else:
                translated_position[key.actuator_name] = position[key]
        return translated_position
    
    def translate_from_md2_to_mxcube(self, position):
        translated_position = {}

        for key in position:
            translated_position[self.md2_to_mxcube[key]] = position[key]

        return translated_position
