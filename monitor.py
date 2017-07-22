#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gevent
from gevent.monkey import patch_all
patch_all()


import PyTango
import time

import numpy as np
from motor import tango_motor
#, tango_named_positions_motor

class monitor:
    def __init__(self, name=None, integration_time=None, sleeptime=0.05):
        self.integration_time = integration_time
        self.name = name
        self.sleeptime = sleeptime
        
    def set_integration_time(self, integration_time):
        self.integration_time = integration_time
        
    def get_integration_time(self):
        return self.integration_time
            
    def start(self):
        pass
    
    def get_image(self):
        pass
    def get_spectrum(self):
        pass
    def measure(self):
        return self.get_point()
    def observe(self):
        return self.get_point()
    def read(self):
        return self.get_point()
    def readout(self):
        return self.get_point()
    def read_out(self):
        return self.get_point()
   
    def get_point(self):
        pass
    
    def arm(self):
        pass
    
    def abort(self):
        pass
    
    def stop(self):
        pass
    
    def cancel(self):
        pass
    
    def get_device_name(self):
        return
    
    def get_name(self):
        return
    
    def monitor(self, start_time):
        self.observations = []
        self.observations_fields = ['chronos', 'point']
        while self.observe == True:
            chronos = time.time() - start_time
            self.observations.append([chronos, self.get_point()])
            gevent.sleep(self.sleeptime)
            
    def get_observations(self):
        return self.observations
    
    def get_observations_fields(self):
        return self.observations_fields
        
class sai(monitor):
    def __init__(self,
                 device_name='i11-ma-c00/ca/sai.2'):
        
        self.device = PyTango.DeviceProxy(device_name)
        self.configuration_fields = ['configurationid', 'samplesnumber', 'frequency', 'integrationtime', 'stathistorybufferdepth', 'datahistorybufferdepth']
        self.channels = ['channel%d' for k in range(4)]
        self.channels_numbers = range(4)
        
    def get_configuration(self):
        configuration = {}
        for parameter in self.configuration_fields:
            configuration[parameter] = self.device.read_attribute(parameter).value
        return configuration
    
    def get_historized_channel_values(self, channel_number):
        return self.device.read_attribute('historizedchannel%d' % channel_number).value
    
    def get_historized_intensity(self):
        historized_intensity = np.zeros(self.get_stathistorybufferdepth())
        for channel_number in self.channels_numbers:
            historized_intensity += self.get_historized_channel_values(channel_number)
        return historized_intensity
        
    def get_stathistorybufferdepth(self):
        return self.device.stathistorybufferdepth
    
    def set_stathistorybufferdepth(self, size):
        self.device.stathistorybufferdepth = size
        
    def get_frequency(self):
        return self.device.frequency
    
    def set_frequency(self, frequency):
        self.device.frequency = frequency
        
    def get_integration_time(self):
        return self.device.integrationtime
    
    def set_integration_time(self, integration_time):
        self.device.integrationtime = integration_time
        
    def get_state(self):
        return self.device.state().name 
        
    def start(self):
        return self.device.Start()
        
    def stop(self):
        return self.device.Stop()
    
    def abort(self):
        return self.device.Abort()
    
    def get_point(self):
        return self.device.averagechannel0
    
    def get_device_name(self):
        return self.device.dev_name()
    
    def get_name(self):
        return self.get_device_name()
    
class Si_PIN_diode(sai):
    
    def __init__(self,
                 thickness=125e-6,
                 device_name='i11-ma-c00/ca/sai.2'):
        
        sai.__init__(self,
                     device_name=device_name)
        
        self.thickness = thickness

class xbpm(monitor):
    
    def __init__(self,
                 device_name,
                 sai='i11-ma-c00/ca/sai.1'):
        
        monitor.__init__(self)
        
        self.device = PyTango.DeviceProxy(device_name)
        self.sai = PyTango.DeviceProxy(sai)
        
    def get_point(self):
        return self.get_intensity() 
    
    def get_intensity(self):
        return self.device.intensity
    
    def get_intensity_from_sai(self):
        intensity = 0
        for k in range(4):
            intensity += self.sai.read_attribute('averagechannel%d' % k).value
        return intensity
    
    def get_intensity_history(self):
        intensity = 0
        for k in range(4):
            intensity += self.sai.read_attribute('averagechannel%d' % k).value
        return intensity
    
    def get_x(self):
        return self.device.horizontalposition
    
    def get_z(self):
        return self.device.verticalposition
    
 
class camera:
    
    def get_image(self):
        return
    def set_integration_time(self):
        pass
    def get_integration_time(self):
        return
    def get_point(self):
        return
    def get_current_image_id(self):
        return
    def start(self):
        pass
    def stop(self):
        pass
    def abort(self):
        pass
    def cancel(self):
        pass
    def arm(self):
        pass
    def disarm(self):
        pass
    def trigger(self):
        pass
    
class basler_camera(camera):
    
    def __init__(self,
                 device_name='i11-ma-cx1/dt/camx.1-vg'):
        
        self.device = PyTango.DeviceProxy(device_name)
        self.device_specific = PyTango.DeviceProxy('%s-specific' % device_name)
        self.analyzer = PyTango.DeviceProxy(device_name.replace('vg', 'analyzer'))
        
    def set_integration_time(self, integration_time):
        self.device.exposureTime = integration_time
    
    def get_integration_time(self):
        return self.device.exposureTime
        
    def set_count_time(self, count_time):
        self.set_integration_time(count_time)
        
    def set_latency_time(self, latency_time):
        self.device.latencyTime = latency_time
        
    def set_frame_time(self, frame_time):
        self.device.latencyTime = frame_time - self.get_integration_time()
    
    def get_frame_time(self):
        return self.device.latencyTime + self.get_integration_time()
        
    def get_current_image_id(self):
        return self.device.currentFrame
    
    def get_frame_rate(self):
        return self.device.frameRate, self.device_specific.frameRate
      
    def get_data_rate(self):
        return self.device_specific.dataRate
       
    def get_trigger_mode(self):
        return self.device.triggerMode
    
    def get_acquisition_mode(self):
        return self.device.acquisitionMode
        
    def get_state(self):
        return self.device.State().name
        
    def start(self):
        return self.device.start()
    
    def stop(self):
        return self.device.stop()
        
    def get_pixel_size_x(self):
        return self.analyzer.pixelsizex
        
    def get_pixel_size_y(self):
        return self.analyzer.pixelsizey
        
    def get_com_x(self):
        return self.analyzer.centroidx
        
    def get_com_y(self):
        return self.analyzer.centroidy
        
    def get_gaussian_fit_center_x(self):
        return self.analyzer.gaussianfitcenterx
        
    def get_gaussian_fit_center_y(self):
        return self.analyzer.gaussianfitcentery
        
    def get_gaussianfit_width_x(self):
        return self.analyzer.gaussianfitmajoraxifwhm
        
    def get_gaussianfit_width_y(self):
        return self.analyzer.gaussianfitminoraxifwhm
        
    def get_gaussianfit_amplitude(self):
        return self.analyzer.gaussianfitmagnitude
        
    def get_max(self):
        return self.analyzer.maxintensity
        
    def get_mean(self):
        return self.analyzer.meanintensity
        
    def get_image(self):
        return self.device.image
    
    def get_point(self):
        return self.get_image()
        
class xray_camera(basler_camera):
    
    def __init__(self,
                 insert_position=-7.67,
                 extract_position=290.0,
                 safe_distance=250.,
                 observation_distance=137.,
                 stage_horizontal_observation_position=21.3,
                 stage_vertical_observation_position=19.0,
                 device_name='i11-ma-cx1/dt/camx.1-vg',
                 vertical_motor='i11-ma-cx1/dt/camx.1-mt_tz',
                 distance_motor='i11-ma-cx1/dt/dtc_ccd.1-mt_ts',
                 stage_horizontal_motor='i11-ma-cx1/dt/dtc_ccd.1-mt_tx',
                 stage_vertical_motor='i11-ma-cx1/dt/dtc_ccd.1-mt_tz',
                 focus_motor='i11-ma-cx1/dt/camx.1-mt_foc',
                 named_positions_motor='i11-ma-cx1/dt/camx1-pos'):
                 
        basler_camera.__init__(self, device_name)
            
        self.vertical_motor = tango_motor(vertical_motor)
        self.distance_motor = tango_motor(distance_motor)
        self.stage_horizontal_motor = tango_motor(stage_horizontal_motor)
        self.stage_vertical_motor = tango_motor(stage_vertical_motor)
        self.focus_motor = tango_motor(focus_motor)
        self.named_positions_motor = tango_named_positions_motor(named_positions_motor)
        
        self.insert_position = insert_position
        self.extract_position = extract_position
        self.safe_distance = safe_distance
        self.observation_distance = observation_distance
        self.stage_horizontal_observation_position = stage_horizontal_observation_position
        self.stage_vertical_observation_position = stage_vertical_observation_position
        
    def insert(self):
        self.vertical_motor.set_position(self.insert_position, wait=True)
        self.distance_motor.set_position(self.observation_distance, wait=True)
        self.stage_horizontal_motor.set_position(self.stage_horizontal_observation_position, wait=True)
        self.stage_vertical_motor.set_position(self.stage_vertical_observation_position, wait=True)
        
    def extract(self):
        self.distance_motor.set_position(self.safe_distance, wait=True)
        self.vertical_motor.set_position(self.extract_position, wait=True)
        
    def set_safe_position(self):
        self.distance_motor.set_position(self.safe_distance, wait=True)
        