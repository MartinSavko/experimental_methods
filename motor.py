#!/usr/bin/env python
# -*- coding: utf-8 -*-

import PyTango
import time

class motor(object):
      
    def get_speed(self):
        pass
    
    def set_speed(self, speed):
        pass
    
    def get_position(self):
        pass
    
    def set_position(self, position, wait=True, timeout=None):
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
        
    def get_speed(self):
        return self.device.velocity
    
    def set_speed(self, speed):
        self.device.velocity = speed
        
    def get_position(self):
        return self.device.position
    
    def set_position(self, position, wait=True, timeout=30):
        start_move = time.time()
        if abs(self.device.position - position) <= 3*self.device.accuracy: 
            print self.device_name, 'set_position: difference is negligible', abs(self.device.position - position)
            print self.device_name, 'move took %s seconds' % (time.time() - start_move)
            return 0
        self.device.position = position
        start = time.time()
        if wait == True:
            self.wait(timeout=timeout)
            if self.get_state() == 'ALARM':
                self.device.position = position
        print self.device_name, 'move took %s seconds' % (time.time() - start_move)
        return position
        
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
        
    def stop(self):
        self.device.stop()
        
    def get_state(self):
        return self.device.state().name
            
def tango_named_positions_motor(tango_motor):
    
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