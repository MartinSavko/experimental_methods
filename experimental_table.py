#!/usr/bin/env python
# -*- coding: utf-8 -*-

from monitor import monitor
from tango import DeviceProxy

class experimental_table(monitor):
    
    def __init__(self, device_name='i11-ma-c05/ex/tab.2', name='position_monitor', attributes=['pitch', 'roll', 'yaw', 'zC', 'xC']):
        
        super().__init__(name=name)
        self.device = DeviceProxy(device_name)
        self.attributes = attributes
        #, 't1z', 't2z', 't3z', 't4x', 't5x']
        
    def get_point(self):
        return [self.get_attribute(attribute) for attribute in self.attributes]
    
    def set_attributes(self, attributes=['pitch', 'roll', 'yaw', 'zC', 'xC']):
        self.attributes = attributes
    
    def get_attributes(self):
        return self.attributes
    
    def get_attribute(self, attribute):
        return self.device.read_attribute(attribute).value
        
    def get_position(self, attributes=None):
        if attributes is None:
            attributes = self.attributes
        return dict([(attribute, self.get_attribute(attribute)) for attribute in attributes])
    
    def get_pitch(self):
        return self.get_attribute('pitch')
    def get_roll(self):
        return self.get_attribute('roll')
    def get_yaw(self):
        return self.get_attribute('yaw')
    def get_xC(self):
        return self.get_attribute('xC')
    def get_zC(self):
        return self.get_attribute('zC')
    
def main():
    tab2 = experimental_table()
    
if __name__ == '__main__':
    main()
           
