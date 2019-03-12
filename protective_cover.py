import PyTango

class guillotine_mockup:
    def insert(self):
        return 
    def extract(self):
        return
    def closed(self):
        return False
        
class protective_cover(object):
    def __init__(self):
        try:
            self.guillotine = PyTango.DeviceProxy('i11-ma-cx1/dt/guillot-ev')
        except:
            self.guillotine = guillotine_mockup()
        
    def closed(self):
        return self.guillotine.isInserted()
        
    def insert(self):
        if not self.closed():
            self.guillotine.insert()
    
    def extract(self):
        if self.closed():
            self.guillotine.extract()