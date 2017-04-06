import PyTango

class guillotine_mockup:
    def insert(self):
        return 
    def extract(self):
        return
    
class protective_cover(object):
    def __init__(self):
        try:
            self.guillotine = PyTango.DeviceProxy('i11-ma-cx1/dt/guillot-ev')
        except:
            self.guillotine = guillotine_mockup()
        
    def insert(self):
        self.guillotine.insert()
    
    def extract(self):
        self.guillotine.extract()