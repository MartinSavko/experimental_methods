# -*- coding: utf-8 -*-

import time
import gevent

try:
    import tango
except:
    import PyTango as tango
    
from md_mockup import md_mockup

class obx_mockup:
    
    #def __init__(self):
        #self.name = 'obx_mockup'
        
    def open(self):
        return 'Open'
    
    def close(self):
        return 'Close'
    
    def state(self):
        return 'State'
    
class safety_shutter(object):
    def __init__(self):
        try:
            self.shutter = tango.DeviceProxy('i11-ma-c04/ex/obx.1')
            self.security = tango.DeviceProxy('i11-ma-ce/pss/db_data-parser')
        except:
            self.shutter = obx_mockup()
        
    def open(self, checktime=0.2, timeout=10):
        start = time.time()
        if self.security.prmObt == 1 and self.state() != 'OPEN':
            self.shutter.Open()
            while self.state() != 'OPEN' and abs(time.time()-start) < timeout:
                gevent.sleep(checktime)
                self.shutter.Open()
        elif self.security.prmObt != 1:
            print('Not possible to open the safety shutter due to a security issue. Has the hutch been searched and locked?')
        
    def close(self, checktime=2., timeout=10.):
        start = time.time()
        while not self.closed() and time.time() - start < timeout:
            try:
                self.shutter.Close()
            except:
                pass
            gevent.sleep(checktime)
    
    def state(self):
        return self.shutter.State().name

    def closed(self):
        return self.state() == 'CLOSE'
    
    def isclosed(self):
        return self.closed()

    def isopen(self):
        return not self.closed()
    
