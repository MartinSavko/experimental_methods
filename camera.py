#!/usr/bin/env python
# -*- coding: utf-8 -*-

from scipy.misc import imsave

import time
import os
import sys
import numpy as np
import PyTango
import logging
import gevent

from goniometer import goniometer
import redis
from pymba import *
from optical_path_report import optical_path_analysis

class camera(object):
    def __init__(self, 
                 camera_type='prosilica',
                 y_pixels_in_detector=1024, 
                 x_pixels_in_detector=1360,
                 channels=3,
                 default_exposure_time=0.05,
                 default_gain=8.,
                 pixel_format='RGB8Packed',
                 tango_address='i11-ma-cx1/ex/imag.1',
                 tango_beamposition_address='i11-ma-cx1/ex/md2-beamposition',
                 use_redis=True,
                 history_size_threshold=600,
                 state_difference_threshold=0.005):

        self.y_pixels_in_detector = y_pixels_in_detector
        self.x_pixels_in_detector = x_pixels_in_detector
        self.channels = channels
        self.default_exposure_time = default_exposure_time
        self.current_exposure_time = None
        self.default_gain = default_gain
        self.current_gain = None
        self.pixel_format=pixel_format
        self.goniometer = goniometer()
        self.use_redis = use_redis
        if self.use_redis == True:
            self.camera = None
            self.redis = redis.StrictRedis()
        else:
            self.camera = PyTango.DeviceProxy(tango_address)
            self.redis = None
            
        self.beamposition = PyTango.DeviceProxy(tango_beamposition_address)
        self.camera_type = camera_type
        self.shape = (y_pixels_in_detector, x_pixels_in_detector, channels)
        self.history_size_threshold = history_size_threshold
        self.state_difference_threshold = state_difference_threshold
        
        #self.focus_offsets = \
           #{1: -0.0819,
            #2: -0.0903,
            #3: -0.1020,
            #4: -0.1092,
            #5: -0.1098,
            #6: -0.1165,
            #7: -0.1185,
            #8: -0.1230,
            #9: -0.1213,
            #10: -0.1230}
        # After changing zoom 2019-03-07
        self.focus_offsets = \
           {1: -0.002,
            2: 0.010,
            3: 0.017,
            4: 0.022,
            5: 0.027,
            6: 0.021,
            7: 0.023,
            8: 0.018,
            9: 0.022,
            10: 0.022}
       
        self.zoom_motor_positions = \
           {1: 34500.0,
            2: 31165.0,
            3: 27185.0,
            4: 23205.0,
            5: 19225.0,
            6: 15245.0,
            7: 11265.0,
            8: 7285.0,
            9: 3305.0,
            10: 0.0 }
        
        self.backlight = \
           {1: 10.0,
            2: 10.0,
            3: 11.0,
            4: 13.0,
            5: 15.0,
            6: 21.0,
            7: 29.0,
            8: 41.0,
            9: 50.0,
            10: 61.0}
        
        self.frontlight = \
           {1: 10.0,
            2: 10.0,
            3: 11.0,
            4: 13.0,
            5: 15.0,
            6: 21.0,
            7: 29.0,
            8: 41.0,
            9: 50.0,
            10: 61.0}
        
        self.gain = \
           {1: 8.,
            2: 8.,
            3: 8.,
            4: 8.,
            5: 8.,
            6: 8.,
            7: 8.,
            8: 8.,
            9: 8.,
            10: 8.}
        
        # calibrations zoom X2
        #self.calibrations = \
           #{1: np.array([ 0.0008041425, 0.0008059998]),
            #2: np.array([ 0.0006467445, 0.0006472503]),
            #3: np.array([ 0.0004944528, 0.0004928871]),
            #4: np.array([ 0.0003771583, 0.0003756801]),
            #5: np.array([ 0.0002871857, 0.0002864572]),
            #6: np.array([ 0.0002194856, 0.0002190059]),
            #7: np.array([ 0.0001671063, 0.0001670309]),
            #8: np.array([ 0.0001261696, 0.0001275330]),
            #9: np.array([ 0.0000966609, 0.0000974697]),
            #10: np.array([0.0000790621, 0.0000784906])}
       
        self.calibrations = \
           {1: np.array([0.00160829, 0.001612  ]),
            2: np.array([0.00129349, 0.0012945 ]),
            3: np.array([0.00098891, 0.00098577]),
            4: np.array([0.00075432, 0.00075136]),
            5: np.array([0.00057437, 0.00057291]),
            6: np.array([0.00043897, 0.00043801]),
            7: np.array([0.00033421, 0.00033406]),
            8: np.array([0.00025234, 0.00025507]),
            9: np.array([0.00019332, 0.00019494]),
            10: np.array([0.00015812, 0.00015698])}
       
       
        self.magnifications = np.array([np.mean(self.calibrations[1]/self.calibrations[k]) for k in range(1, 11)])      
        self.master = False
        
    def get_point(self):
        return self.get_image()
        
    def get_image(self, color=True):
        if color:
            return self.get_rgbimage()
        else:
            return self.get_bwimage()
    
    def get_image_id(self):
        if self.use_redis:
            image_id = self.redis.get('last_image_id')
        else:
            image_id = self.camera.imagecounter
        return image_id
        
    def get_rgbimage(self, image_data=None):
        if self.use_redis:
            if image_data == None:
                image_data = self.redis.get('last_image_data')
            rgbimage = np.ndarray(buffer=image_data, dtype=np.uint8, shape=(1024, 1360, 3))
        else:
            rgbimage = self.camera.rgbimage.reshape((self.shape[0], self.shape[1], 3))
        return rgbimage
    
    def get_bwimage(self, image_data=None):
        rgbimage = self.get_rgbimage(image_data=image_data)
        return rgbimage.mean(axis=2)
    
    def clear_history(self):
        for item in ['history_image_timestamp', 'history_state_vector', 'history_image_data']:
            self.redis.ltrim(item, 0, -2)
            
    def get_history(self, start, end):
        self.redis.set('can_clear_history', 0)
        try:
            timestamps = np.array([float(self.redis.lindex('history_image_timestamp', i)) for i in range(self.redis.llen('history_image_timestamp'))])
            
            mask = np.logical_and(timestamps>=start, timestamps<=end)
            
            interesting_stamps = np.array([float(self.redis.lindex('history_image_timestamp', int(i))) for i in np.argwhere(mask)])
            
            interesting_images = np.array([self.get_rgbimage(image_data=self.redis.lindex('history_image_data', int(i))) for i in np.argwhere(mask)])
            
            interesting_state_vectors = np.array([self.get_state_vector_with_float_values_from_state_vector_as_single_string(self.redis.lindex('history_state_vector', int(i))) for i in np.argwhere(mask)])
            
        except:
            interesting_stamps = np.array([])
            interesting_images = np.array([])
            interesting_state_vectors = np.array([])
            
        self.redis.set('can_clear_history', 1)
        return interesting_stamps, interesting_images, interesting_state_vectors
    
    def get_image_corresponding_to_timestamp(self, timestamp):
        self.redis.set('can_clear_history', 0)
        try:
            timestamps = np.array([float(self.redis.lindex('history_image_timestamp', i)) for i in range(self.redis.llen('history_image_timestamp'))])
            
            timestamps_before = timestamps[timestamps <= timestamp]
            
            closest = np.argmin(np.abs(timestamps_before - timestamp))
            
            corresponding_image = self.get_rgbimage(image_data=self.redis.lindex('history_image_data', int(closest)))
            
            #corresponding_state_vector =  self.get_state_vector_with_float_values_from_state_vector_as_single_string(self.redis.lindex('history_state_vector', int(closest)))
            
        except:
            corresponding_image = self.get_rgbimage()
            #corresponding_state_vector = None
        
        self.redis.set('can_clear_history', 1)
    
        return corresponding_image
    
    def save_image(self, imagename, color=True):
        image_id, image = self.get_image_id(), self.get_image(color=color)
        if not os.path.isdir(os.path.dirname(imagename)):
            try:
                os.makedirs(os.path.dirname(imagename))
            except OSError:
                print 'Could not create the destination directory'
        imsave(imagename, image)
        return imagename, image, image_id
        

    def get_zoom_from_calibration(self, calibration):
        a = list([(key, value[0]) for key, value in self.calibrations.items()])
        a.sort(key=lambda x: x[0])
        a = np.array(a)
        return range(1, 11)[np.argmin(np.abs(calibration-a[:,1]))]
    
    def get_zoom(self):
        a = list(self.zoom_motor_positions.items())
        a.sort(key=lambda x: x[0])
        a = np.array(a)
        return range(1, 11)[np.argmin(np.abs(self.goniometer.md2.zoomposition-a[:,1]))]
        
    def set_zoom(self, value, wait=True):
        if value is not None:
            value = int(value)
            self.set_gain(self.gain[value])
            self.goniometer.md2.backlightlevel = self.backlight[value]
            self.goniometer.set_position({'Zoom': self.zoom_motor_positions[value], 'AlignmentX': self.focus_offsets[value]}, wait=wait)
            self.goniometer.md2.coaxialcamerazoomvalue = value
        
    def get_calibration(self):
        return np.array([self.get_vertical_calibration(), self.get_horizontal_calibration()])
        
    def get_vertical_calibration(self):
        return self.goniometer.md2.coaxcamscaley
        
    def get_horizontal_calibration(self):
        return self.goniometer.md2.coaxcamscalex

    def set_exposure(self, exposure=0.05):
        if not (exposure >= 3.e-6 and exposure<3):
            print('specified exposure time is out of the supported range (3e-6, 3)')
            return -1
        if not self.use_redis:
            self.camera.exposure = exposure
        if self.master:
            self.camera.ExposureTimeAbs = exposure * 1.e6
        self.redis.set('camera_exposure_time', exposure)
        self.current_exposure_time = exposure
        
    def get_exposure(self):
        if not self.use_redis:
            exposure = self.camera.exposure
        if self.master:
            exposure = self.camera.ExposureTimeAbs/1.e6
            print 'exposure from camera %s' % exposure
        else:
            exposure = float(self.redis.get('camera_exposure_time'))
            print 'exposure from redis %s' % exposure
        print 'final exposure %s' % exposure
        return exposure 
    
    def set_exposure_time(self, exposure_time):
        self.set_exposure(exposure_time)
    
    def get_exposure_time(self):
        return self.get_exposure()
    
    def get_gain(self):
        if not self.use_redis:
            gain = self.camera.gain
        elif self.master:
            gain = self.camera.GainRaw
        else:
            gain = float(self.redis.get('camera_gain'))
        return gain
    
    def set_gain(self, gain):
        if not (gain >= 0 and gain <=24):
            print('specified gain value out of the supported range (0, 24)')
            return -1
        if not self.use_redis:
            self.camera.gain = gain
        elif self.master:
            self.camera.GainRaw = int(gain)
        self.redis.set('camera_gain', gain)                
        self.current_gain = gain
        
    def get_beam_position(self):
        return np.array([512., 680.])
    
    def get_beam_position_vertical(self):
        return self.beamposition.read_attribute('Zoom%d_Z' % self.get_zoom()).value
    
    def get_beam_position_horizontal(self):
        return self.beamposition.read_attribute('Zoom%d_X' % self.get_zoom()).value
    
    def set_frontlightlevel(self, frontlightlevel):
        self.goniometer.md2.frontlightlevel = frontlightlevel
        
    def get_frontlightlevel(self):
        return self.goniometer.md2.frontlightlevel
    
    def set_backlightlevel(self, backlightlevel):
        self.goniometer.md2.backlightlevel = backlightlevel
        
    def get_backlightlevel(self):
        return self.goniometer.md2.backlightlevel
    
    def get_width(self):
        return self.x_pixels_in_detector
    
    def get_height(self):
        return self.y_pixels_in_detector
    
    def get_image_dimensions(self):
        return [self.get_width(), self.get_height()]
    
    def get_state_vector_with_string_values(self):
        gain = self.get_gain()
        exposure_time = self.get_exposure_time()
        return self.goniometer.get_state_vector() + ['%.2f' % gain, '%.3f' % exposure_time]
    
    def get_state_vector_with_float_values(self, state_vector_with_string_values=None):
        if state_vector_with_string_values is None:
            state_vector_with_string_values = self.get_state_vector_with_string_values()
        return np.array(map(float, state_vector_with_string_values))
    
    def get_state_vector_as_single_string(self, state_vector_with_string_values=None):
        if state_vector_with_string_values is None:
            state_vector_with_string_values = self.get_state_vector_with_string_values()
        return ','.join(state_vector_with_string_values)
        
    def get_state_vector_with_string_values_from_state_vector_as_single_string(self, state_vector_as_single_string):
        return state_vector_as_single_string.split(',')
    
    def get_state_vector_with_float_values_from_state_vector_as_single_string(self, state_vector_as_single_string):
        state_vector_with_string_values = self.get_state_vector_with_string_values_from_state_vector_as_single_string(state_vector_as_single_string)
        return self.get_state_vector_with_float_values(state_vector_with_string_values)
    
    def get_last_saved_state_vector_string(self):
        return self.redis.lindex('history_state_vector', self.redis.llen('history_state_vector') - 1)
    
    def get_minimum_angle_difference(self, delta):
        return (delta + 180.)%360. - 180.
            
    def state_vectors_are_different(self, v1, v2):
        delta = v1 - v2
        delta[0] = self.get_minimum_angle_difference(delta[0])
        delta[2] = self.get_minimum_angle_difference(delta[2])
        return np.linalg.norm(delta) > self.state_difference_threshold
    
    def get_default_background(self, zoom=None):
        if zoom is None:
           background = self.get_rgbimage(image_data=self.redis.get('background_image_data_zoom_%d' % self.get_zoom()))
        else:
           background = self.get_rgbimage(image_data=self.redis.get('background_image_data_zoom_%d' % zoom))
        return background
        
    def set_default_background(self):
        self.redis.set('background_image_data_zoom_%d' % self.get_zoom(), self.redis.get('last_image_data'))
        
    def run_camera(self):
        self.master = True
        
        vimba = Vimba()
        system = vimba.getSystem()
        vimba.startup()
        
        if system.GeVTLIsPresent:
            system.runFeatureCommand("GeVDiscoveryAllOnce")
            gevent.sleep(3)
        
        cameraIds = vimba.getCameraIds()
        print('cameraIds %s' % cameraIds)
        self.camera = vimba.getCamera('DEV_000F3102FD4E')
        self.camera.openCamera()
        self.camera.PixelFormat = self.pixel_format
        
        self.frame0 = self.camera.getFrame()    # creates a frame
        self.frame0.announceFrame()
        
        self.image_dimensions = (self.frame0.width, self.frame0.height)
        
        self.set_exposure(self.default_exposure_time)
        self.set_gain(self.default_gain)
        
        self.current_gain = self.get_gain()
        self.current_exposure_time = self.get_exposure_time()
        
        self.camera.startCapture()
        
        self.camera.runFeatureCommand("AcquisitionStart")
        
        k = 0
        last_frame_id = None
        _start = time.time()
        while self.master:
            self.frame0.waitFrameCapture()
            try:
                self.frame0.queueFrameCapture()
            except:
                print('camera: frame dropped')
                continue
            
            #img = self.frame0.getImage()
            if self.frame0._frame.frameID != last_frame_id:
                k+=1
                data = self.frame0.getBufferByteData()
                img = np.ndarray(buffer=data, 
                                 dtype=np.uint8, 
                                 shape=(self.frame0.height, self.frame0.width, self.frame0.pixel_bytes))
                
                last_image_data = img.ravel().tostring()
                last_image_timestamp = str(time.time())
                last_image_id = self.frame0._frame.frameID
                last_image_frame_timestamp =  str(self.frame0._frame.timestamp)
                self.redis.set('last_image_data', last_image_data)
                self.redis.set('last_image_timestamp', last_image_timestamp)
                self.redis.set('last_image_id', last_image_id)
                self.redis.set('last_image_frame_timestamp', last_image_frame_timestamp)
                
                current_state_vector_with_string_values = self.get_state_vector_with_string_values()
                current_state_vector_with_float_values = self.get_state_vector_with_float_values(current_state_vector_with_string_values)
                current_state_vector_as_single_string = self.get_state_vector_as_single_string(current_state_vector_with_string_values)
                
                try:
                    last_saved_state_vector_string = self.get_last_saved_state_vector_string()
                    last_saved_state_vector_with_float_values = self.get_state_vector_with_float_values_from_state_vector_as_single_string(last_saved_state_vector_string)
                except:
                    last_saved_state_vector_with_float_values = None
                
                if last_saved_state_vector_with_float_values is None or self.state_vectors_are_different(current_state_vector_with_float_values, last_saved_state_vector_with_float_values):
                    self.redis.rpush('history_image_data', last_image_data)
                    self.redis.rpush('history_image_timestamp', last_image_timestamp)
                    self.redis.rpush('history_state_vector', current_state_vector_as_single_string)
                
                current_history_size = self.redis.llen('history_image_timestamp')
                if (current_history_size > self.history_size_threshold * 1.2 and self.redis.get('can_clear_history') == '1') or current_history_size >= 2 * self.history_size_threshold:
                    for item in ['history_image_data',
                                 'history_image_timestamp',
                                 'history_state_vector']:
                        
                        self.redis.ltrim(item, self.history_size_threshold, self.redis.llen(item))

                requested_gain = float(self.redis.get('camera_gain'))
                if requested_gain != self.current_gain:
                    self.set_gain(requested_gain)
                requested_exposure_time = float(self.redis.get('camera_exposure_time'))
                if requested_exposure_time != self.get_exposure_time():
                    self.set_exposure(requested_exposure_time)
                    
            #if k%50 == 0:
                #print('camera last frame id %d fps %.1f ' % (self.frame0._frame.frameID, k/(time.time() - _start)))
                #_start = time.time()
                #k = 0
            gevent.sleep(0.01)
            
        self.camera.runFeatureCommand("AcquisitionStop")
        self.close_camera()
    
    def close_camera(self):
        self.master = False
        
        with Vimba() as vimba:
            self.camera.flushCaptureQueue()
            self.camera.endCapture()
            self.camera.revokeAllFrames()
            vimba.shutdown()
    
    def start_camera(self):
        return

    def align_from_single_image(self, generate_report=False, display=False, turn=True, dark=False):
        logging.getLogger('HWR').info('camera align_from_single_image')
        _start = time.time()
        print 'align_from_single_image start %.2f' % _start
        logging.getLogger('HWR').info('align_from_single_image start %.2f' % _start)
        reference_position = self.goniometer.get_aligned_position()
        
        print 'align_from_single_image: reference_position %s' % str(reference_position)
        logging.getLogger('HWR').info('align_from_single_image: reference_position %s' % str(reference_position))
        calibration = self.get_calibration()
        zoom = self.get_zoom()
        center = self.get_beam_position()
        
        print 'align_from_single_image: about to acquire an image and start the analysis'
        if dark == True:
            name_pattern= 'autocenter_%s_%s_dark_failed.jpg' % (os.getuid(), time.asctime().replace(' ', '_'))
        else:
            name_pattern= 'autocenter_%s_%s_bright_failed.jpg' % (os.getuid(), time.asctime().replace(' ', '_'))
        directory ='%s/manual_optical_alignment' % os.getenv('HOME')
        logging.getLogger('HWR').info('align_from_single_image: about to acquire an image and start the analysis')

        print 'align_from_single_image: saving the image %s' % name_pattern
        imagename, sample_image, image_id = self.save_image(os.path.join(directory, name_pattern), color=True)
        logging.getLogger('HWR').info('align_from_single_image: saving the image %s' % name_pattern)        

        print 'get_results %s' % name_pattern
        results = optical_path_analysis([sample_image.mean(axis=2)], [reference_position['Omega']], calibration, background_image=self.get_default_background().mean(axis=2), display=display, smoothing_factor=0.025, generate_report=generate_report, dark=dark) 
        logging.getLogger('HWR').info('align_from_single_image: results obtained')
        
        _end = time.time()
        sign = -1.
        step = 0.25
        if results == -1:
            print 'align_from_single_image: no results (sample not visible?)'
            reference_position['Omega'] += 90.
            reference_position['AlignmentY'] += -1 * sign * step
            self.goniometer.set_position(reference_position)
            print 'align_from_single_image took %.2f' % (_end - _start)
            return
        
        print 'align_from_single_image took %.2f' % (_end - _start)
        centroid = results['centroids'][0]
        rightmost = results['rightmost'][0]
        
        if (rightmost[-1] - centroid[-1])*calibration[1] > 0.25:
            y, x = rightmost[1:]
        else:
            y, x = centroid
        os.rename(imagename, imagename.replace('_failed.jpg', '_zoom_%d_y_%d_x_%d.jpg' % (zoom, y, x)))
        
        print 'centroid', centroid
        print 'rightmost', rightmost
        print 'zoom', zoom
        print 'calibration', calibration
        print 'center', center
        
        vector = (np.array([y, x]) - center)*calibration
        print 'vector', vector
        aligned_position = self.goniometer.get_aligned_position_from_reference_position_and_shift(reference_position, vector[1], vector[0])
        self.goniometer.set_position(aligned_position)
        if turn == True:
            aligned_position['Omega'] += 90.
        self.goniometer.set_position(aligned_position)
        _end = time.time()
        #return aligned_position
        
    def get_contrast(self, image=None, method='RMS', roi=None):
        if image is None:
            image = self.get_image(color=False)
        elif len(image.shape) == 3:
            image = image.mean(axis=2)
        
        #if roi != None:
            #image = 
        #image = image.astype(np.float)
        Imean = image.mean()
        if method == 'Michelson':
            Imax = image.max()
            Imin = image.min()
            contrast = (Imax - Imin)/(Imax + Imin)
        elif method == 'Weber':
            background = self.get_default_background()
            Ib = background.mean()
            contrast = (Imean-Ib)/Ib
        elif method == 'RMS':
            contrast = np.sqrt(np.mean((image - Imean)**2))
        
        return contrast 
    
        
    
if __name__ == '__main__':
    cam = camera()
    cam.run_camera()

