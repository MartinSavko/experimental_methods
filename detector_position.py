from motor import tango_motor

class detector_position:
    
    def __init__(self):
        for direction in ['ts', 'tx', 'tz']:
            setattr(self, direction, tango_motor('i11-ma-cx1/dt/dtc_ccd.1-mt_%s' % direction))