import PyTango

class protective_cover(object):
    def __init__(self):
        self.guillotine = PyTango.DeviceProxy('i11-ma-cx1/dt/guillot-ev')
        
    def insert(self):
        self.guillotine.insert()
    
    def extract(self):
        self.guillotine.extract()