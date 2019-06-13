# -*- coding: utf-8 -*-
import PyTango
import time
import gevent
from md2_mockup import md2_mockup

class obx_mockup:
    
    #def __init__(self):
        #self.name = 'obx_mockup'
        
    def Open(self):
        return 'Open'
    
    def Close(self):
        return 'Close'
    
    def State(self):
        return 'State'
    
class safety_shutter(object):
    def __init__(self):
        try:
            self.shutter = PyTango.DeviceProxy('i11-ma-c04/ex/obx.1')
            self.security = PyTango.DeviceProxy('i11-ma-ce/pss/db_data-parser')
        except:
            self.shutter = obx_mockup()
        
    def open(self, checktime=0.2, timeout=10):
        start = time.time()
        if self.security.prmObt == 1 and self.state != 'OPEN':
            self.shutter.Open()
            while self.state() != 'OPEN' and abs(time.time()-start) < timeout:
                gevent.sleep(checktime)
                self.shutter.Open()
        elif self.security.prmObt != 1:
            print 'Not possible to open the safety shutter due to a security issue. Has the hutch been searched and locked?'
        
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