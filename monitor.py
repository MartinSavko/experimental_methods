#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import gevent
import redis
import subprocess
try:
    import tango
except:
    import PyTango as tango
import time
import math
import traceback

import numpy as np
from scipy.constants import elementary_charge as q
from scipy.optimize import leastsq
from scipy.ndimage import center_of_mass

from motor import tango_motor, tango_named_positions_motor
from camera import camera as redis_camera

from speech import speech, defer

class monitor(object): #speech):
    
    def __init__(
        self, 
        integration_time=None, 
        sleeptime=0.05, 
        use_redis=True, 
        name='monitor', 
        history_size_threshold=1000, 
        continuous_monitor_name=None,
        port=5555,
        history_size_target=10000,
        debug_frequency=100,
        framerate_window=25,
        service=None,
        verbose=None,
        server=False,
        default_save_destination="/nfs/data4/movies",
    ):
    
        self.integration_time = integration_time
        self.sleeptime = sleeptime
        self.observe = None
        self.observations = []
        self.use_redis = use_redis
        self.name = name
        self.history_size_threshold = int(history_size_threshold)
        self.continuous_monitor_name = continuous_monitor_name
        self.observation_fields = None, None
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
        
        #speech.__init__(
            #self,
            #port=port,
            #service=service,
            #verbose=verbose,
            #server=server,
            #history_size_target=history_size_target,
            #debug_frequency=debug_frequency,
            #framerate_window=framerate_window,
        #)
    
    def acquire(self):
        try:
            self.value = self.get_point()
            self.timestamp = time.time()
            self.value_id += 1
        except:
            traceback.print_exc()
            
        super().acquire()
        
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
        self.start_time = start_time
        self.observation_fields = ['chronos', 'point']
        if self.continuous_monitor_name is None:
            self.observations = []
            self.observe = True
            while self.observe == True:
                chronos = time.time() - start_time
                try:
                    point = self.get_point()
                except:
                    traceback.print_exc()
                    point = np.nan
                self.observations.append([chronos, point])
                gevent.sleep(self.sleeptime)
        elif self.continuous_monitor_name == 'device':
            pass
        else:
            self.continuous_monitor()
            
    def continuous_monitor(self):
        status = subprocess.getoutput('%s status' % self.continuous_monitor_name)
        if not '%s is running' % self.continuous_monitor_name in status:
           os.system('%s start &' % self.continuous_monitor_name)
    
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
        try:
            timestamps, data = self.get_history()
            
            closest = np.argmin(np.abs(timestamps - timestamp))
            
            corresponding_point = data[:, closest] 
                        
        except:
            corresponding_point = self.get_point()
        
        return corresponding_point   
    
    def get_observations_from_history(self, start, end=np.inf):
        
        timestamps, data = self.get_history(start=start, end=end)
        offset_timestamps = [ts - start for ts in timestamps]
        return offset_timestamps, data
    
    def get_observations(self):
        if self.continuous_monitor_name not in [None, 'device']:
            self.observations = self.get_observations_from_history(start=self.start_time)
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
        return list(map(float, character_sequence.split(';')))
        
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
        try:
            start = seq1.index(seq2[:alignment_length])
        except ValueError:
            start = -1
        merged = seq1[:start] + seq2 
        return merged, int((len(merged) - len(seq1))/8)
    

class counter(monitor):
    
    def __init__(self,
                 device_name='i11-ma-c00/ca/cpt.2',
                 attribute_name='Ext_Eiger',
                 sleeptime=0.005):
        
        monitor.__init__(self)
        
        self.device = tango.DeviceProxy(device_name)
        self.attribute = tango.AttributeProxy('%s/%s' % (device_name, attribute_name))
    
    def stop(self):
        return self.device.stop()
    
    def start(self):
        if self.get_state() != 'STANDBY':
            self.stop()
            return self.device.start()
        
    def get_point(self):
        try:
            return self.attribute.read().value
        except:
            return np.nan
        
    def set_total_buffer_size(self, buffer_size=10000):
        try:
            self.device.totalNbPoint = buffer_size
        except:
            pass
        
    def get_frequency(self):
        return self.device.frequency
    
    def set_total_buffer_duration(self, duration):
        frequency = self.get_frequency()
        buffer_size = int(math.ceil(duration * frequency))
        try:
            self.stop()
            self.set_total_buffer_size(buffer_size)
        except:
            traceback.print_exc()

    def get_state(self):
        return self.device.state().name
    
    def init(self):
        self.device.init()
        
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
                 use_redis=True,
                 continuous_monitor_name=None):
        
        continuous_monitor_name = '%s_monitor' % os.path.basename(device_name).replace('.', '')
        monitor.__init__(self, name=device_name, history_size_threshold=1e6, sleeptime=sleeptime, use_redis=use_redis, continuous_monitor_name=continuous_monitor_name)
        
        if self.continuous_monitor_name is None:
            print('Anomaly in sai !')
            self.continuous_monitor_name = continuous_monitor_name
        self.device = tango.DeviceProxy(device_name)
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

                if history_data != b'0':
                    history_data, new_values = self.merge_two_overlapping_buffers(history_data, last_point_data_buffer)
                    previous_timestamp = float(self.redis.get(last_timestamp_key))
                    history_timestamp = np.frombuffer(self.redis.get(history_timestamp_key))
                else:
                    history_data, new_values = last_point_data_buffer, len(last_point_data)
                    previous_timestamp = last_timestamp - new_values*self.get_integration_time()*1.e-3
                    history_timestamp = False
                
                new_history_timestamps = np.linspace(last_timestamp, previous_timestamp, int(new_values), endpoint=False)[::-1]
                
                history_timestamp = new_history_timestamps if history_timestamp is False else np.hstack([history_timestamp, new_history_timestamps]) 
                
                #if history_timestamp is not False else new_history_timestamps
                self.redis.set(last_timestamp_key, last_timestamp)
                self.redis.set(history_data_key, history_data)
                self.redis.set(history_timestamp_key, history_timestamp.tobytes())
                
                self.history_sizes[channel] = len(history_data)/8
            
            if np.any(self.history_sizes > 1.2*self.history_size_threshold) and self.redis.get(self.clear_flag_key) == '1' or self.history_sizes.max() > 2*self.history_size_threshold: 
                for channel in range(self.number_of_channels):
                    last_timestamp_key, history_timestamp_key, history_data_key = self.keys[channel]
                    self.redis.set(history_timestamp_key, history_timestamp[-self.history_size_threshold:])
                    self.redis.set(history_data_key, history_data[-self.history_size_threshold:])
                        
            gevent.sleep(self.sleeptime)

    def get_history(self, start=-np.inf, end=np.inf):
        timestamps, intensities = [], []
        self.redis.set(self.clear_flag_key, 0)
        for channel in range(self.number_of_channels):
            last_timestamp_key, history_timestamp_key, history_data_key = self.keys[channel]
            channel_timestamps = np.frombuffer(self.redis.get(history_timestamp_key))
            channel_intensity = np.frombuffer(self.redis.get(history_data_key))
            
            mask = np.logical_and(channel_timestamps>=start, channel_timestamps<=end)
            
            channel_timestamps = channel_timestamps[mask]
            if len(channel_intensity) != len(mask):
                channel_intensity = channel_intensity[:len(mask)]
            channel_intensity = channel_intensity[mask]
            
            timestamps.append(channel_timestamps)
            intensities.append(channel_intensity)
        self.redis.set(self.clear_flag_key, 1)
        return timestamps, intensities
    
    def get_intensity_history(self, start=-np.inf, end=np.inf):
        timestamps, intensities = self.get_history(start=start, end=end)
        min_length = min(list(map(len, timestamps)))
        timestamps = np.array([item[-min_length:] for item in timestamps])
        intensities = np.array([item[-min_length:] for item in intensities])
        
        timestamps = timestamps.mean(axis=0)
        intensities = np.abs(intensities).sum(axis=0)
        return timestamps, intensities
     
    def get_point(self):
        return np.array([self.get_historized_channel_values(channel) for channel in range(self.number_of_channels)])
            
    def get_configuration(self):
        configuration = {}
        for parameter in self.configuration_fields:
            configuration[parameter] = self.device.read_attribute(parameter).value
        return configuration
    
    def get_historized_channel_values(self, channel_number):
        return self.device.read_attribute('historizedchannel%d' % channel_number).value
    
    def get_channel_current(self, channel_number):
        return self.device.read_attribute('averagechannel%d' % channel_number).value
        
    def get_channel_difference(self, channel_a, channel_b):
        a = self.get_channel_current(channel_a)
        b = self.get_channel_current(channel_b)
        channel_difference = a - b
        return channel_difference
    
    def get_total_current(self, absolute=True):
        current = 0
        for channel in range(self.number_of_channels):
            cc = self.get_channel_current(channel)
            if absolute:
                cc = abs(cc)
            current += cc
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
                 horizontal_motor_det='i11-ma-cx1/dt/dtc_ccd.1-mt_tx',
                 horizontal_motor_cam="i11-ma-cx1/dt/camx-mt_tx",
                 vertical_motor_det = 'i11-ma-cx1/dt/dtc_ccd.1-mt_tz',
                 vertical_motor_cam='i11-ma-cx1/dt/camx.1-mt_tz',
                 distance_motor='i11-ma-cx1/dt/dtc_ccd.1-mt_ts'):
                 
        sai.__init__(self,
                     device_name=device_name,
                     number_of_channels=1)
        
        self.thickness = thickness
        self.amplification = amplification
        self.attenuation_length_12650 = 267.310
        self._params = None
        
        self.named_positions_motor = tango_named_positions_motor(named_positions_motor)
        self.horizontal_motor_det = tango_motor(horizontal_motor_det)
        self.horizontal_motor_cam = tango_motor(horizontal_motor_cam)
        self.vertical_motor_det = tango_motor(vertical_motor_det)
        self.vertical_motor_cam = tango_motor(vertical_motor_cam)
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
        dat = [list(map(float, item.split())) for item in data]
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
        
    def insert(
        self, 
        horizontal_position_det=20.5,
        horizontal_position_cam=80.0,
        vertical_position_det=37.5,
        vertical_position_cam=33.0,
        distance=180.,
        min_distance=179.,
    ):
        if distance < min_distance:
            return -1
        self.named_positions_motor.set_named_position('DIODE')
        self.horizontal_motor_det.set_position(horizontal_position_det)
        self.horizontal_motor_cam.set_position(horizontal_position_cam)
        self.vertical_motor_det.set_position(vertical_position_det)
        self.vertical_motor_cam.set_position(vertical_position_cam)
        
        self.distance_motor.set_position(distance)
    
    def extract(self, vertical_position_det=37.5, horizontal_position_det=20.5, distance=350.):
        if distance < 150:
            return -1
        self.distance_motor.set_position(distance)
        self.horizontal_motor_det.set_position(horizontal_position_det)
        self.vertical_motor_det.set_position(vertical_position_det)
        self.named_positions_motor.set_named_position('Extract')
    
    def isinserted(self, insert_threshold_position=280):
        return self.named_positions_motor.get_position() < insert_threshold_position
            
    def isextracted(self):
        return not self.isinserted()
     
class xbpm(monitor):
    
    def __init__(self,
                 device_name='i11-ma-c04/dt/xbpm_diode.1-base',
                 point_attributes=['intensity', 'x', 'z', 'current1', 'current2', 'current3', 'current4']):
        
        monitor.__init__(self)
        
        self.device = tango.DeviceProxy(device_name)
        self.point_attributes = point_attributes
        sai_controller_proxy = self.device.get_property('SaiControllerProxyName')
        self.sai = sai(sai_controller_proxy['SaiControllerProxyName'][0])
        try:
            self.position = tango.DeviceProxy(device_name.replace('-base', '-pos'))
        except:
            self.position = None

    def get_point(self):
        #return self.get_historized_intensity()
        
        point = [getattr(self, 'get_%s' % a)() for a in self.point_attributes]
            
        return point
    
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
    
    def get_current1(self):
        return self.device.current1
    def get_current2(self):
        return self.device.current2
    def get_current3(self):
        return self.device.current3
    def get_current4(self):
        return self.device.current4
    
    def get_name(self):
        return self.device.dev_name()
    
    def insert(self):
        self.position.insert()

    def extract(self):
        self.position.extract()
    
    def get_position(self):
        if self.position is not None:
            return self.position.position
        
    def is_inserted(self):
        value = None
        if self.position is not None:
            try:
                value = self.position.isInserted
            except:
                pass
        return value
    
    def is_extracted(self):
        value = None
        if self.position is not  None:
            try:
                value = self.position.isExtracted
            except:
                pass
        return value
    
    def get_point_and_reference(self):
        point_and_reference = dict([(attribute, getattr(self, 'get_%s' % attribute)()) for attribute in self.point_attributes])
        point_and_reference['is_inserted'] = self.is_inserted()
        point_and_reference['name'] = self.get_name()
        point_and_reference['position'] = self.get_position()
        return point_and_reference

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
        
        self.device = tango.DeviceProxy(device_name)
        
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
        
        self.device = tango.DeviceProxy(device_name)
        
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
                 sleeptime=0.001,
                 history_size_threshold=1000,
                 use_redis=True,
                 continuous_monitor_name='focus_monitor'):
        
        camera.__init__(self, name=device_name, use_redis=use_redis, history_size_threshold=history_size_threshold, sleeptime=sleeptime, continuous_monitor_name=continuous_monitor_name)
        
        self.device = tango.DeviceProxy(device_name)
        self.device_specific = tango.DeviceProxy('%s-specific' % device_name)
        self.analyzer = tango.DeviceProxy(device_name.replace('vg', 'analyzer'))
        self.sleeptime = sleeptime
        
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
        return np.array([self.get_gaussian_fit_center_x(), 
                         self.get_gaussian_fit_center_y(), 
                         self.get_gaussianfit_amplitude(), 
                         self.get_gaussianfit_width_x(), 
                         self.get_gaussianfit_width_y(), 
                         self.get_max(), 
                         self.get_mean(), 
                         self.get_com_x(), 
                         self.get_com_y()])
        
    def can_clear_history(self):
        current_history_size = len(self.timestamps)
        return current_history_size > self.history_size_threshold * 1.2 and self.redis.get(self.clear_flag_key) == '1' or current_history_size >= 2*self.history_size_threshold
    
    def run_history(self):
        self.data = np.array([])
        self.timestamps = np.array([])
        
        while True:
            last_point_data = self.get_point()
            last_point_timestamp = time.time()
            self.redis.set(self.last_data_key, last_point_data)
            self.redis.set(self.last_timestamp_key, last_point_timestamp)
            
            self.data = np.hstack([self.data, last_point_data]) if len(self.data) else last_point_data
            self.timestamps = np.hstack([self.timestamps, last_point_timestamp])
            
            self.redis.set(self.history_data_key, self.data.tostring())
            self.redis.set(self.history_timestamp_key, self.timestamps.tostring())
            
            if self.can_clear_history():
                for item in [self.data, self.timestamps]:
                    self.data = self.data[-self.history_size_threshold*len(last_point_data):]
                    self.timestamps = self.timestamps[-self.history_size_threshold:]
                    
            gevent.sleep(self.sleeptime)
           
    def get_history(self, start=-np.inf, end=np.inf):
        self.redis.set(self.clear_flag_key, 0)
        
        try:
            timestamps = np.frombuffer(self.redis.get(self.history_timestamp_key))
            data = np.frombuffer(self.redis.get(self.history_data_key))
            point_size = self.get_point().size
            data_size = data.size/point_size
            data = data.reshape((data_size, point_size)).T
            if data_size > timestamps.size:
                data = data[:, -timestamps.size:]
            elif data_size < timestamps.size:
                timestamps = timestamps[-data_size:]

            mask = np.logical_and(timestamps>=start, timestamps<=end)
            
            timestamps = timestamps[mask]
            data = data[:, mask]
            
        except:
            print(traceback.print_exc())
            timestamps = np.array([])
            data = np.array([])
            
        self.redis.set(self.clear_flag_key, 1)
        
        return timestamps, data
    
        
    def get_point_corresponding_to_timestamp(self, timestamp):
        self.redis.set(self.clear_flag_key, 0)
        try:
            timestamps, data = self.get_history()
            
            closest = np.argmin(np.abs(timestamps - timestamp))
            
            corresponding_point = data[:, closest] 
                        
        except:
            corresponding_point = self.get_point()
        
        self.redis.set(self.clear_flag_key, 1)
    
        return corresponding_point
        

class xray_camera(basler_camera):
    
    def __init__(self,
                 insert_position=-5.76, #8.94,
                 extract_position=290.0,
                 safe_distance=180.,
                 observation_distance=175.,
                 stage_horizontal_observation_position=20.5, #25.78, #21.3,
                 stage_vertical_observation_position=37.5,
                 device_name='i11-ma-cx1/dt/camx.1-vg',
                 vertical_motor='i11-ma-cx1/dt/camx.1-mt_tz',
                 distance_motor='i11-ma-cx1/dt/dtc_ccd.1-mt_ts',
                 stage_horizontal_motor='i11-ma-cx1/dt/dtc_ccd.1-mt_tx',
                 stage_vertical_motor='i11-ma-cx1/dt/dtc_ccd.1-mt_tz',
                 focus_motor='i11-ma-cx1/dt/camx.1-mt_foc',
                 named_positions_motor='i11-ma-cx1/dt/camx1-pos',
                 use_redis=True):
                 
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
    
    def get_position(self):
        position = {
            'distance_motor': self.distance_motor.get_position(),
            'stage_vertical_motor': self.stage_vertical_motor.get_position(),
            'stage_horizontal_motor': self.stage_horizontal_motor.get_position(),
            'vertical_motor': self.vertical_motor.get_position(),
            }
        return position
    
    def extract(self):
        self.distance_motor.set_position(self.safe_distance, wait=True)
        self.vertical_motor.set_position(self.extract_position, wait=True)
        
    def set_safe_position(self):
        self.distance_motor.set_position(self.safe_distance, wait=True)
        

class oav_camera(monitor):
    
    def __init__(self, threshold=0.5, name='oav_monitor'):
        super().__init__(name=name)
        self.cam = redis_camera()
        self.threshold = threshold
        
    def get_point(self):
        img = self.cam.get_image(color=False)
        
        img[img<img.max()*self.threshold] = 0
        com = np.array(center_of_mass(img))/np.array(self.cam.get_image_dimensions())[::-1]
        
        return com
    

class jaull(monitor):
    
    def __init__(self, device_name='i11-ma-c05/vi/jaull.1', name='pressure_monitor'):
        super().__init__(name=name)
        self.device = tango.DeviceProxy(device_name)
        
    def get_point(self):
        try:
            point = self.device.pressure
        except:
            traceback.print_exc()
            point = np.nan
        return point
    
class tdl_xbpm(monitor):
    
    def __init__(self, device_name='tdl-i11-ma/dg/xbpm.1', name='position_monitor'):
        super().__init__(name=name)
        self.device = tango.DeviceProxy(device_name)
        
    def get_point(self):
        return np.array([self.device.zPos, self.device.xPos])
        
    def get_position(self, attributes=['zPos', 'xPos']):
        return dict([(attribute, self.device.read_attribute(attribute).value) for attribute in attributes])
        

class tango_monitor(monitor):

    def __init__(
        self, 
        device_name=None, 
        name='monitor', 
        attributes=[], 
        skip_attributes=[], 
        continuous_monitor_name=None,
    ):
        
        super().__init__(name=name, continuous_monitor_name=continuous_monitor_name)
        self.device = tango.DeviceProxy(device_name)
        self.attributes = attributes
        self.skip_attributes = skip_attributes
        
    def get_point(self):
        return [self.get_attribute(attribute) for attribute in self.attributes]
    
    def set_attributes(self, attributes=[]):
        self.attributes = attributes
    
    def get_attributes(self):
        return self.attributes
    
    def get_attribute(self, attribute):
        return self.device.read_attribute(attribute).value
        
    def get_position(self, attributes=None, skip_attributes=None):
        if attributes is None:
            attributes = self.device.get_attribute_list()
        if skip_attributes is None:
            skip_attributes = self.skip_attributes
        position = {}
        for attribute in attributes:
            if attribute in skip_attributes:
                continue
            try:
                if hasattr(self, f'get_{attribute}'):
                    value = getattr(self, f'get_{attribute}')()
                else:
                    value = self.get_attribute(attribute)
            except:
                value = None
            position[attribute] = value
        return position
    
    def get_state(self):
        return self.device.state().name
    
    def get_status(self):
        return self.device.status()

    def wait(self, timeout=30):
        print("waiting for monitor to get ready")
        _start = time.time()
        while self.get_state() not in ["STANDBY", "READY"] and time.time() - _start < timeout:
            gevent.sleep(self.sleeptime)
        
            
            
            
            
            
            
