#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gevent

import PyTango
import time
import math

import numpy as np
from scipy.constants import elementary_charge as q
from scipy.optimize import leastsq

from motor import tango_motor, tango_named_positions_motor
import redis

class monitor:
    
    def __init__(self, integration_time=None, sleeptime=0.05, use_redis=False, name='monitor', history_size_threshold=1000):
        self.integration_time = integration_time
        self.sleeptime = sleeptime
        self.observe = None
        self.observations = []
        self.use_redis = use_redis
        self.name = name
        self.history_size_threshold = history_size_threshold
        if self.use_redis == True:
            self.redis = redis.StrictRedis()
            self.last_data_key = '%s_last_data' % self.name
            self.last_timestamp_key = '%s_last_timestamp' % self.name
            self.history_data_key = '%s_history_data' % self.name
            self.history_timestamp_key = '%s_history_timestamp' % self.name
            self.clear_flag_key = '%s_can_clear_history' % self.name
        else:
            self.redis = None
            self.last_data_key = None
            self.last_timestamp_key = None
            self.history_data_key = None
            self.history_timestamp_key = None
            self.clear_flag_key = None
        
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
        return self.name
    
    def monitor(self, start_time):
        self.observations = []
        self.observation_fields = ['chronos', 'point']
        self.observe = True
        while self.observe == True:
            chronos = time.time() - start_time
            point = self.get_point()
            self.observations.append([chronos, point])
            gevent.sleep(self.sleeptime)
    
    def can_clear_history(self):
        current_history_size = self.redis.llen(self.history_timestamp_key)
        return current_history_size > self.history_size_threshold * 1.2 and self.redis.get(self.clear_flag_key) == '1' or current_history_size >= 2*self.history_size_threshold
    
    def run_history(self):
        while True:
            last_point_data = self.get_point()
            last_point_timestamp = time.time()
            self.redis.set(self.last_data_key, last_point_data)
            self.redis.set(self.last_timestamp_key, last_point_timestamp)
            self.redis.rpush(self.history_data_key, last_point_data)
            self.redis.rpush(self.history_timestamp_key, last_point_timestamp)
            
            if self.can_clear_history():
                for item in [self.history_data_key, self.history_timestamp_key]:
                    self.redis.ltrim(item, self.history_size_threshold, self.redis.llen(item))
                    
            gevent.sleep(self.sleeptime)
           
    def get_history(self, start=-np.inf, end=np.inf):
        self.redis.set(self.clear_flag_key, 0)
        try:
            timestamps = np.array([float(self.redis.lindex(self.history_timestamp_key, i)) for i in range(self.redis.llen(self.history_timestamp_key))])
            
            mask = np.logical_and(timestamps>=start, timestamps<=end)
            
            interesting_stamps = np.array( [float(self.redis.lindex(self.history_timestamp_key, int(i))) for i in np.argwhere(mask)] )
            
            interesting_points = np.array([self.get_rgbimage(image_data=self.redis.lindex(self.history_data_key, int(i))) for i in np.argwhere(mask)])
            
        except:
            interesting_stamps = np.array([])
            interesting_points = np.array([])
            
        self.redis.set(self.clear_flag_key, 1)
        
        return interesting_stamps, interesting_points
    
    def get_point_corresponding_to_timestamp(self, timestamp):
        self.redis.set(self.clear_flag_key, 0)
        try:
            timestamps = np.array([float(self.redis.lindex(self.history_timestamp_key, i)) for i in range(self.redis.llen(self.history_timestamp_key))])
            
            timestamps_before = timestamps[timestamps <= timestamp]
            
            closest = np.argmin(np.abs(timestamps_before - timestamp))
            
            corresponding_point = self.get_rgbimage(image_data=self.redis.lindex(self.history_data_key, int(closest)))
                        
        except:
            corresponding_point = self.get_point()
        
        self.redis.set(self.clear_flag_key, 1)
    
        return corresponding_point
    
    def get_observations(self):
        return self.observations
    
    def get_observation_fields(self):
        return self.observation_fields
       
    def get_points(self):
        return np.array(self.observations)[:,1]
        
    def get_chronos(self):
        return np.array(self.observations)[:,0]

    def from_number_sequence_to_character_sequence(self, number_sequence, separator=';'):
        character_sequence = ''
        number_strings = [str(n) for n in number_sequence]
        return separator.join(number_strings)

    def merge_two_overlapping_character_sequences(self, seq1, seq2, alignment_length=1000, separator=';'):
        start = seq1.index(seq2[:alignment_length])
        nvalues_seq1 = seq2.count(separator) - seq2[start:].count(separator)
        return seq1[:start] + seq2,  nvalues_seq1
        
    def from_character_sequence_to_number_sequence(self, character_sequence, separator=';'):
        return map(float, character_sequence.split(';'))
        
    def merge_two_overlapping_number_sequences(self, r1, r2, alignment_length=1000, separator=';'):
        c1 = self.from_number_sequence_to_character_sequence(r1)
        c2 = self.from_number_sequence_to_character_sequence(r2)
        c, start = self.merge_two_overlapping_character_sequences(c1, c2, alignment_length)
        r = self.from_character_sequence_to_number_sequence(c)
        return r, start
    
    def find_overlap(self, r1, r2, alignment_length=1000, separator=';'):
        c1 = self.from_number_sequence_to_character_sequence(r1)
        c2 = self.from_number_sequence_to_character_sequence(r2)
        start = c1.index(c2[:alignment_length])
        start = c2.count(separator) - c2[start:].count(separator)
        return start
    
    def merge_two_overlapping_buffers(self, seq1, seq2, alignment_length=1000):
        start = seq1.index(seq2[:alignment_length])
        merged = seq1[:start] + seq2 
        return merged, (len(merged) - len(seq1))/8
    
class counter(monitor):
    
    def __init__(self,
                 device_name='i11-ma-c00/ca/cpt.2',
                 attribute_name='Ext_Eiger',
                 sleeptime=0.005):
        
        monitor.__init__(self)
        
        self.device = PyTango.DeviceProxy(device_name)
        self.attribute = PyTango.AttributeProxy('%s/%s' % (device_name, attribute_name))
    
    def stop(self):
        return self.device.stop()
    
    def start(self):
        return self.device.start()
        
    def get_point(self):
        return self.attribute.read().value
        
    def set_total_buffer_size(self, buffer_size=10000):
        self.device.totalNbPoint = buffer_size
        
    def get_frequency(self):
        return self.device.frequency
    
    def set_total_buffer_duration(self, duration):
        frequency = self.get_frequency()
        buffer_size = int(math.ceil(duration * frequency))
        self.set_total_buffer_size(buffer_size)

        
class eiger_en_out(counter):
    
    def __init__(self,
                 attribute_name='Ext_Eiger',
                 sleeptime=0.005):
        
        counter.__init__(self, attribute_name=attribute_name, sleeptime=sleeptime)
        

class fast_shutter_close(counter):
    
    def __init__(self,
                 attribute_name='Fast_Shutter_Close',
                 sleeptime=0.05):
        
        counter.__init__(self, attribute_name=attribute_name, sleeptime=sleeptime)
        
class fast_shutter_open(counter):
    
    def __init__(self,
                 attribute_name='Fast_Shutter_Open',
                 sleeptime=0.05):
        
        counter.__init__(self, attribute_name=attribute_name, sleeptime=sleeptime)

class trigger_eiger_on(counter):
    
    def __init__(self,
                 attribute_name='Trigger_Eiger_On',
                 sleeptime=0.05):
        
        counter.__init__(self, attribute_name=attribute_name, sleeptime=sleeptime)

class trigger_eiger_off(counter):
    
    def __init__(self,
                 attribute_name='Trigger_Eiger_Off',
                 sleeptime=0.05):
        
        counter.__init__(self, attribute_name=attribute_name, sleeptime=sleeptime)


class sai(monitor):
    
    def __init__(self,
                 device_name='i11-ma-c00/ca/sai.2',
                 number_of_channels=4,
                 history_size_threshold=1e6,
                 sleeptime=1.,
                 use_redis=False):
        
        monitor.__init__(self, name=device_name, history_size_threshold=1e6, sleeptime=sleeptime, use_redis=use_redis)
        
        self.device = PyTango.DeviceProxy(device_name)
        self.configuration_fields = ['configurationid', 'samplesnumber', 'frequency', 'integrationtime', 'stathistorybufferdepth', 'datahistorybufferdepth']
        self.channels = ['channel%d' for k in range(number_of_channels)]
        self.keys = [['%s_ch%d' % (key, k) for key in [self.last_timestamp_key, self.history_timestamp_key, self.history_data_key]] for k in range(number_of_channels)]
        self.number_of_channels = number_of_channels 
        self.history_sizes = np.zeros(self.number_of_channels)
        
    def run_history(self):
        for channel in range(self.number_of_channels):
            for key in self.keys[channel]:
                self.redis.set(key, 0)
        
        self.redis.set(self.clear_flag_key, 1)
        
        while True:
            for channel in range(self.number_of_channels):
                last_timestamp_key, history_timestamp_key, history_data_key = self.keys[channel]
                
                last_point_data = self.get_historized_channel_values(channel) # about 1ms
                last_timestamp = time.time()
                
                last_point_data_buffer = last_point_data.tostring() # 2.5 us
                history_data = self.redis.get(history_data_key)
                
                if history_data != '0':
                    history_data, new_values = self.merge_two_overlapping_buffers(history_data, last_point_data_buffer)
                    previous_timestamp = float(self.redis.get(last_timestamp_key))
                    history_timestamp = self.redis.get(history_timestamp_key)
                else:
                    history_data, new_values = last_point_data_buffer, len(last_point_data)
                    previous_timestamp = last_timestamp - new_values*self.get_integration_time()*1.e-3
                    history_timestamp = ''
                
                new_history_timestamps = np.linspace(last_timestamp, previous_timestamp, new_values, endpoint=False)[::-1]
                                
                history_timestamp += new_history_timestamps.tostring()
                
                self.redis.set(last_timestamp_key, last_timestamp)
                self.redis.set(history_data_key, history_data)
                self.redis.set(history_timestamp_key, history_timestamp)
                
                self.history_sizes[channel] = len(history_data)/8
            
            if np.any(self.history_sizes > 1.2*self.history_size_threshold) and self.redis.get(self.clear_flag_key) == '1' or self.history_sizes.max() > 2*self.history_size_threshold: 
                for channel in range(self.number_of_channels):
                    last_timestamp_key, history_timestamp_key, history_data_key = self.keys[channel]
                    self.redis.set(history_timestamp_key, history_timestamp[-self.history_size_threshold:])
                    self.redis.set(history_data_key, history_data[-self.history_size_threshold:])
                        
            gevent.sleep(self.sleeptime)

    def get_history(self):
        timestamps, intensities = [], []
        self.redis.set(self.clear_flag_key, 0)
        for channel in range(self.number_of_channels):
            last_timestamp_key, history_timestamp_key, history_data_key = self.keys[channel]
            channel_timestamps = np.frombuffer(self.redis.get(history_timestamp_key))
            channel_intensity = np.frombuffer(self.redis.get(history_data_key))
            timestamps.append(channel_timestamps)
            intensities.append(channel_intensity)
        self.redis.set(self.clear_flag_key, 1)
        return timestamps, intensities
    
    def get_intensity_history(self):
        timestamps, intensities = self.get_history()
        min_length = min(map(len, timestamps))
        timestamps = np.array([item[-min_length:] for item in timestamps])
        intensities = np.array([item[-min_length:] for item in intensities])
        
        timestamps = timestamps.mean(axis=0)
        intensities = np.abs(intensities).sum(axis=0)
        return timestamps, intensities
        
    def get_configuration(self):
        configuration = {}
        for parameter in self.configuration_fields:
            configuration[parameter] = self.device.read_attribute(parameter).value
        return configuration
    
    def get_historized_channel_values(self, channel_number):
        return self.device.read_attribute('historizedchannel%d' % channel_number).value
    
    def get_channel_current(self, channel_number):
        return self.device.read_attribute('averagechannel%d' % channel_number).value
        
    def get_total_current(self):
        current = 0
        for channel in range(self.number_of_channels):
            current += self.get_channel_current(channel)
        return current
        
    def get_historized_intensity(self):
        historized_intensity = np.zeros(self.get_stathistorybufferdepth())
        historized_intensity = []
        for channel_number in range(self.number_of_channels):
            historized_intensity.append(self.get_historized_channel_values(channel_number))
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
        return self.get_total_current()
    
    def get_device_name(self):
        return self.device.dev_name()
    
    def get_name(self):
        return self.get_device_name()
  

class Si_PIN_diode(sai):
    
    def __init__(self,
                 thickness=125e-6,
                 amplification=1e4,
                 device_name='i11-ma-c00/ca/sai.2',
                 named_positions_motor='i11-ma-cx1/dt/camx1-pos',
                 horizontal_motor='i11-ma-cx1/dt/dtc_ccd.1-mt_tx',
                 vertical_motor='i11-ma-cx1/dt/dtc_ccd.1-mt_tz',
                 distance_motor='i11-ma-cx1/dt/dtc_ccd.1-mt_ts'):
                 
        sai.__init__(self,
                     device_name=device_name,
                     number_of_channels=1)
        
        self.thickness = thickness
        self.amplification = amplification
        self.attenuation_length_12650 = 267.310
        self._params = None
        
        self.named_positions_motor = tango_named_positions_motor(named_positions_motor)
        self.horizontal_motor = tango_motor(horizontal_motor)
        self.vertical_motor = tango_motor(vertical_motor)
        self.distance_motor = tango_motor(distance_motor)
        
    def transmission(self, params, e):
        t = 0
        for k, p in enumerate(params):
            t += p*e**(k)
        return t 
     
    def get_flux(self, current, ey, params=None):
        self._params = self.get_params()
        #if params is None and self._params is None:
            #self._params = self.get_params()
        #else:
            #self._params = params
        current /= self.amplification
        return current / (self.responsivity(ey, self._params) * q * ey)
    
    def responsivity(self, ey, params):
        return 0.98 * (1-self.transmission(params, ey))/3.65

    def transmission_12650(self):
        return np.exp(-self.thickness/self.attenuation_length_12650)
    
    def residual(self, params, energy, data):
        model = self.transmission(params, energy)
        return abs(model - data)
    
    def get_params(self, datafile='/927bis/ccd/gitRepos/flux/xray9507_Si_125um.dat'): #xray5184.dat
        data = open(datafile).read().split('\n')[2:-1]
        dat = [map(float, item.split()) for item in data]
        da = np.array(dat)
        eys, transmissions = da[:,0], da[:,1]
        results = leastsq(self.residual, [0]*10, args=(eys, transmissions))
        params = results[0]     
        return params
        
    def get_current(self, channel_number=0):
        return self.get_channel_current(channel_number)
    
    def get_thickness(self):
        return self.thickness
    
    def get_amplification(self):
        return self.amplification

    def get_point(self):
        return self.get_current()
        #return self.get_historized_channel_values(0)
        
    def insert(self, vertical_position=25, horizontal_position=27.5, distance=180.):
        if distance < 150:
            return -1
        self.named_positions_motor.set_named_position('DIODE')
        self.horizontal_motor.set_position(horizontal_position)
        self.vertical_motor.set_position(vertical_position)
        self.distance_motor.set_position(distance)
    
    def extract(self, vertical_position=44.5, horizontal_position=20.5, distance=350.):
        if distance < 150:
            return -1
        self.distance_motor.set_position(distance)
        self.horizontal_motor.set_position(horizontal_position)
        self.vertical_motor.set_position(vertical_position)
        self.named_positions_motor.set_named_position('Extract')
    
    def isinserted(self, insert_threshold_position=280):
        return self.named_positions_motor.get_position() < insert_threshold_position
            
    def isextracted(self):
        return not self.isinserted()
     
class xbpm(monitor):
    
    def __init__(self,
                 device_name='i11-ma-c04/dt/xbpm_diode.1-base'):
        
        monitor.__init__(self)
        
        self.device = PyTango.DeviceProxy(device_name)
        sai_controller_proxy = self.device.get_property('SaiControllerProxyName')
        self.sai = sai(sai_controller_proxy['SaiControllerProxyName'][0])
        try:
            self.position = PyTango.DeviceProxy(device_name.replace('-base', '-pos'))
        except:
            self.position = None

    def get_point(self):
        #return self.get_historized_intensity()
        return self.get_intensity() 
    
    def get_intensity(self):
        return self.device.intensity
    
    def get_intensity_from_sai(self):
        return self.sai.get_total_current()
    
    def get_historized_intensity(self):
        return self.sai.get_historized_intensity()
    
    def get_x(self):
        return self.device.horizontalposition
    
    def get_z(self):
        return self.device.verticalposition
    
    def get_name(self):
        return self.device.dev_name()
    
    def insert(self):
        self.position.insert()

    def extract(self):
        self.position.extract()
    
    def is_inserted(self):
        if self.position == None:
           pass
        else:
           return self.position.isInserted

    def is_extracted(self):
        if self.position == None:
           pass
        else:
           return self.position.isExtracted


class xbpm_mockup(monitor):
    def __init__(self, device_name='i11-ma-c04/dt/xbpm_diode.1-base'):
        monitor.__init__(self)
        self.device_name = device_name
    
    def get_name(self):
        return self.device_name
       
    def is_inserted(self):
        True

    def is_extracted(self):
        False
 
class peltier(monitor):
    
    def __init__(self,
                 device_name='i11-ma-c03/op/mono1-pt100.2'):
        
        monitor.__init__(self)
        
        self.device = PyTango.DeviceProxy(device_name)
        
    def get_point(self):
        return self.get_temperature()
        
    def get_temperature(self):
        return self.device.temperature
    
    def get_name(self):
        return self.device.dev_name()
    
class thermometer(monitor):
    
    def __init__(self,
                 device_name='i11-ma-cx1/ex/tc.1'):
                 
        monitor.__init__(self)
        
        self.device = PyTango.DeviceProxy(device_name)
        
    def get_point(self):
        return self.get_temperature()
    
    def get_temperature(self):
        return self.device.temperature
        
    def get_name(self):
        return self.device.dev_name()
        
class camera(monitor):
    
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
                 device_name='i11-ma-cx1/dt/camx.1-vg',
                 sleeptime=0.004,
                 history_size_threshold=1000,
                 use_redis=False):
        
        camera.__init__(self, name=device_name, use_redis=use_redis, history_size_threshold=history_size_threshold, sleeptime=sleeptime)
        
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
        try:
            com_x = self.analyzer.centroidx
        except:
            com_x = np.nan
        return com_x
        
    def get_com_y(self):
        try:
            com_y = self.analyzer.centroidy
        except:
            com_y = np.nan
        return com_y
        
    def get_gaussian_fit_center_x(self):
        try:
            gaussian_fit_center_x = self.analyzer.gaussianfitcenterx
        except:
            gaussian_fit_center_x = np.nan
        return gaussian_fit_center_x
        
    def get_gaussian_fit_center_y(self):
        try:
            gaussian_fit_center_y = self.analyzer.gaussianfitcentery
        except:
            gaussian_fit_center_y = np.nan
        return gaussian_fit_center_y
        
    def get_gaussianfit_width_x(self):
        try:
            gaussianfit_width_x = self.analyzer.gaussianfitmajoraxisfwhm
        except:
            gaussianfit_width_x = np.nan
        return gaussianfit_width_x
    
    def get_gaussianfit_width_y(self):
        try:
            gaussianfit_width_y = self.analyzer.gaussianfitminoraxisfwhm
        except:
            gaussianfit_width_y = np.nan
        return gaussianfit_width_y
    
    def get_gaussianfit_amplitude(self):
        try:
            gaussianfit_amplitude = self.analyzer.gaussianfitmagnitude
        except:
            gaussianfit_amplitude = np.nan
        return gaussianfit_amplitude
    
    def get_max(self):
        try:
            max_intensity =  self.analyzer.maxintensity
        except:
            max_intensity = np.nan
        return max_intensity
        
    def get_mean(self):
        try:
            mean_intensity = self.analyzer.meanintensity
        except:
            mean_intensity = np.nan
        return mean_intensity
    
    def get_image(self):
        return self.device.image
    
    def get_point(self):
        return self.get_image()
        
class analyzer(basler_camera):
    
    def get_point(self):
        return [self.get_gaussian_fit_center_x(), self.get_gaussian_fit_center_y(), self.get_gaussianfit_amplitude(), self.get_gaussianfit_width_x(), self.get_gaussianfit_width_y(), self.get_max(), self.get_mean(), self.get_com_x(), self.get_com_y()]
        
                     
class xray_camera(basler_camera):
    
    def __init__(self,
                 insert_position=8.94,
                 extract_position=290.0,
                 safe_distance=250.,
                 observation_distance=132.,
                 stage_horizontal_observation_position=25.78, #21.3,
                 stage_vertical_observation_position=25.0,
                 device_name='i11-ma-cx1/dt/camx.1-vg',
                 vertical_motor='i11-ma-cx1/dt/camx.1-mt_tz',
                 distance_motor='i11-ma-cx1/dt/dtc_ccd.1-mt_ts',
                 stage_horizontal_motor='i11-ma-cx1/dt/dtc_ccd.1-mt_tx',
                 stage_vertical_motor='i11-ma-cx1/dt/dtc_ccd.1-mt_tz',
                 focus_motor='i11-ma-cx1/dt/camx.1-mt_foc',
                 named_positions_motor='i11-ma-cx1/dt/camx1-pos',
                 use_redis=False):
                 
        basler_camera.__init__(self, device_name, use_redis=use_redis)
            
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
        
