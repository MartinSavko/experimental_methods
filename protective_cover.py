import gevent

try:
    import tango
except:
    import PyTango as tango

COVER_OPERATION_MINIMUM_DISTANCE = 119.

class cover_mockup:
    def insert(self):
        return

    def extract(self):
        return

    def isclosed(self):
        return False

    def isopen(self):
        return True


class protective_cover(object):
    def __init__(self, wait_time=0.1):
        try:
            self.cover = tango.DeviceProxy("i11-ma-cx1/dt/guillot-ev")
            self.ts = tango.DeviceProxy("i11-ma-cx1/dt/dtc_ccd.1-mt_ts")
        except:
            self.cover = cover_mockup()
            self.ts = motor_mockup()
        
        self.wait_time = wait_time

    def isclosed(self):
        return self.cover.isInserted() and self.cover.read_attribute("isInserted").value

    def isopen(self):
        return (
            self.cover.isExtracted() and self.cover.read_attribute("isExtracted").value
        )

    def insert(self, wait=True):
        if self.ts.position >= COVER_OPERATION_MINIMUM_DISTANCE:
            self.cover.insert()
            if wait == True:
                while not self.isclosed():
                    gevent.sleep(self.wait_time)

    def extract(self, wait=True):
        if self.ts.position >= COVER_OPERATION_MINIMUM_DISTANCE:
            self.cover.extract()
            if wait == True:
                while not self.isopen():
                    gevent.sleep(self.wait_time)
