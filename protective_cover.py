import PyTango

class guillotine_mockup:
    def insert(self):
        return 
    def extract(self):
        return
    def isclosed(self):
        return False
    def isopen(self):
        return True

class protective_cover(object):
    def __init__(self):
        try:
            self.guillotine = PyTango.DeviceProxy('i11-ma-cx1/dt/guillot-ev')
        except:
            self.guillotine = guillotine_mockup()
        
    def isclosed(self):
        return self.guillotine.isInserted()
        
    def isopen(self):
        return self.guillotine.isExtracted()
    
    def insert(self):
        if not self.isclosed():
            self.guillotine.insert()
    
    def extract(self):
        if self.isclosed():
            self.guillotine.extract()
