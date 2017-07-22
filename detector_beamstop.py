import PyTango

class detector_beamstop:
    def __init__(self):
        self.mt_x = PyTango.DeviceProxy('i11-ma-cx1/ex/bst.3-mt_tx')
        self.mt_z = PyTango.DeviceProxy('i11-ma-cx1/ex/bst.3-mt_tz')
    def set_z(self, position):
        self.mt_z.position = position
    def set_x(self, position):
        self.mt_z.position = position
    def get_z(self):
        return self.mt_z.position
    def get_x(self):
        return self.mt_x.position
