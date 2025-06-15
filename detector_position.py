from motor import tango_motor

class detector_position:
    
    # determined 2025-04-04 11h30, Gil, Bill, Rémi, Martin
    extreme = {
        "ts": 98.121,
        "tx": 12.5,
        "tz": 100.,
    }
    
    def __init__(self):
        for direction in ['ts', 'tx', 'tz']:
            setattr(self, direction, tango_motor(f'i11-ma-cx1/dt/dtc_ccd.1-mt_{direction}'))
        self.position = {}
        
    def get_position(self):
        for direction in ['ts', 'tx', 'tz']:
            self.position[direction] = getattr(getattr(self, direction), "get_position")()
        return self.position
    

    def set_position(self, position, wait=False):
        for direction in position:
            setattr(getattr(self, direction), "set_position")(position[direction], wait=wait)
        
        
