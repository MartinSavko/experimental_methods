#!/usr/bin/env python
# -*- coding: utf-8 -*-

from motor import tango_motor, detector_ts_motor

class detector_position:
    # determined 2025-04-04 11h30, Gil, Bill, Rémi, Martin
    extreme = {
        "ts": 98.121,
        "tx": 12.5,
        "tz": 100.0,
    }

    def __init__(self):
        self.ts = detector_ts_motor()
        for direction in ["tx", "tz"]:
            setattr(
                self, direction, tango_motor(f"i11-ma-cx1/dt/dtc_ccd.1-mt_{direction}")
            )
        self.position = {}

    def get_position(self):
        for direction in ["ts", "tx", "tz"]:
            self.position[direction] = getattr(
                getattr(self, direction), "get_position"
            )()
        return self.position

    def set_position(self, position, wait=False):
        print(f"set_position {position}, wait {wait}")
        for direction in position:
            getattr(getattr(self, direction), "set_position")(
                position[direction], wait=wait
            )

def main():
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("-s", "--ts", default=-1, type=float, help="move detector_s")
    parser.add_argument("-x", "--tx", default=-1, type=float, help="move detector_x")
    parser.add_argument("-z", "--tz", default=-1, type=float, help="move detector_z")

    args = parser.parse_args()
    print("args", args)
    dp = detector_position()
    position = dp.get_position()
    print(f"current position {position}")
    
    for d in ["ts", "tx", "tz"]:
        if getattr(args, d) >= 0:
            position[d] = getattr(args, d)
    
    print(f"intended position {position}")
    dp.set_position(position)
    
if __name__ == "__main__":
    main()
