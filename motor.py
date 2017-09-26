#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gevent
from gevent.monkey import patch_all
patch_all()

import PyTango
import time
from scipy.constants import h, c, angstrom, kilo, eV
from math import sin, radians


class motor(object):
      
    def get_speed(self):
        pass
    
    def set_speed(self, speed):
        pass
    
    def get_position(self):
        pass
    
    def set_position(self, position, wait=True, timeout=None):
        pass
    
    def get_offset(self):
        pass
    
    def set_offset(self, offset):
        pass
    
    def stop(self):
        pass
    
    def get_state(self):
        pass

class tango_motor(motor):
    
    def __init__(self, device_name, check_time=0.1):
        self.device_name = device_name
        self.device = PyTango.DeviceProxy(device_name)
        self.check_time = check_time
        self.observations = []
        self.observation_fields = ['chronos', 'position']
        self.monitor_sleep_time = 0.05
        self.position_attribute = 'position'
   
    def get_name(self):
        return self.device.dev_name()
        
    def get_observation_fields(self):
        return self.observation_fields
        
    def get_observations(self):
        return self.observations
        
    def get_speed(self):
        return self.device.velocity
    
    def set_speed(self, speed):
        self.device.velocity = speed
    
    def get_offset(self):
        return self.device.offset
        
    def set_offset(self, offset):
        self.device.offset = offset
        
    def get_position(self):
        return self.device.position
    
    def set_position(self, position, wait=True, wait_timeout=1, timeout=30, default_accuracy=0.01):
        start_move = time.time()
        #if 'accuracy' in self.device.get_attribute_list():
            #accuracy = self.device.accuracy
        #else:
        accuracy = default_accuracy
        if abs(self.get_position() - position) <= 3*accuracy: 
            print self.device_name, 'set_position: difference is negligible', abs(self.get_position() - position)
            print self.device_name, 'move took %s seconds' % (time.time() - start_move)
            return 0
            
        self.device.write_attribute(self.position_attribute, position)
        
        start = time.time()
        if wait == True:
            self.wait(timeout=timeout)
            if self.get_state() == 'ALARM':
                self.set_position(position)
        print self.device_name, 'move took %s seconds' % (time.time() - start_move)
        
    def wait(self, timeout=None):
        print self.device_name, 'wait'
        start = time.time()
        while self.get_state() != 'STANDBY':
            if self.get_state() == 'ALARM':
                self.device.position -= 5*self.device.accuracy
                time.sleep(5)
                self.device.position += 5*self.device.accuracy
            time.sleep(self.check_time)
            if timeout != None and abs(time.time() - start) > timeout:
                print 'timeout on wait for %s took %s' % (self.device_name, time.time() - start)
                break
        print 'wait for %s took %s' % (self.device_name, time.time() - start)
      
    def get_point(self):
        return self.get_position()
        
    def stop(self):
        self.device.stop()
        
    def get_state(self):
        return self.device.state().name
            
    def monitor(self, start_time):
        self.observe = True
        #while self.get_state() != 'STANDBY':
        while self.observe == True:
            chronos = time.time() - start_time
            point = [chronos, self.get_point()]
            self.observations.append(point)
            gevent.sleep(self.monitor_sleep_time)
            
 
class monochromator_rx_motor(tango_motor):
    
    def __init__(self, device_name='i11-ma-c03/op/mono1-mt_rx'):
                 
        tango_motor.__init__(self, device_name)
        
    def get_thetabragg(self):
        return self.device.position
    
    def get_wavelength(self, d=3.1347507142511746):
        return 2*d*sin(radians(self.get_position()))
        
    def get_energy(self):
        return h*c/(angstrom*self.get_wavelength()*kilo*eV)
        
    def get_position(self):
        return self.get_thetabragg()
        
    def get_point(self):
        return self.get_position()
        

class monochromator(tango_motor):
    
    def __init__(self,
                 device_name='i11-ma-c03/op/mono1'):
        
        tango_motor.__init__(self, device_name)
        
    def get_thetabragg(self):
        return self.device.thetabragg
    
    def get_wavelength(self):
        return self.device.Lambda
    
    def get_energy(self):
        return self.device.energy
        
    def get_position(self):
        return self.get_energy(), self.get_thetabragg(), self.get_wavelength()
        
    def get_point(self):
        return self.get_position()

class undulator(tango_motor):
    
    def __init__(self,
                 device_name='ans-c11/ei/m-u24'):
        
        tango_motor.__init__(self, device_name)
        self.position_attribute = 'gap'
        
    def get_speed(self):
        return self.device.gapVelocity
    
    def set_speed(self, speed):
        return None
    
    def get_encoder_position(self):
        return self.device.encoder2Position
       
    def get_position(self):
        return self.device.gap
        
    def get_point(self):
        return self.get_position()
        
class tango_named_positions_motor(tango_motor):
    
    def __init__(self,
                 device_name):
        
        tango_motor.__init__(self,
                             device_name)
                             
    def set_named_position(self, named_position):
        return getattr(self.device, named_position)()
        
class md2_motor(motor):
    
    def __init__(self, motor_name, md2_name='i11-ma-cx1/ex/md2'):
        self.md2 = PyTango.DeviceProxy(md2_name)
        self.motor_name = motor_name
        self.motor_full_name = '%sPosition' % motor_name
        self.check_time = 0.1
        
    def get_position(self):
        return self.md2.read_attribute(self.motor_full_name).value
    
    def set_position(self, position, wait=True, timeout=None, accuracy=1.e-3):
        if abs(self.get_position() - position) < accuracy: return
        self.md2.write_attribute(self.motor_full_name, position)
        start = time.time()
        if wait == True:
            self.wait()
            
    def wait(self, timeout=30):
        start = time.time()
        while self.get_state() != 'Ready':
            time.sleep(self.check_time)
            if timeout != None and abs(time.time() - start) > timeout:
                break
            
    def stop(self):
        self.md2.abort()
    
    def get_state(self):
        return dict([item.split('=') for item in md2.motorstates])[self.motor_name]