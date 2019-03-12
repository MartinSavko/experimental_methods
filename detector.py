#!/usr/bin/env python
# -*- coding: utf-8 -*-

from eiger import eiger
from protective_cover import protective_cover
from detector_position import detector_position
from detector_beamstop import detector_beamstop
import gevent
import time

class detector(eiger):

    def __init__(self, host='172.19.10.26', port=80):
        eiger.__init__(self, host=host, port=port)
        self.position = detector_position()
        self.beamstop = detector_beamstop()
        self.cover = protective_cover()
        
    def insert_protective_cover(self, safe_distance=120., delta=1., timeout=3.):
        start = time.time()
        while self.position.ts.get_position() < safe_distance and time.time() - start < timeout:
            gevent.sleep(0.5)
        if time.time() - start > timeout:
            self.position.ts.set_position(safe_distance+1, wait=True)
        if self.position.ts.get_position() >= safe_distance:
            self.cover.insert()
        
    def extract_protective_cover(self, safe_distance=120., delta=1., timeout=3.):
        current_position = self.position.ts.get_position()
        if current_position < safe_distance:
            self.position.ts.set_position(safe_distance+delta, wait=True)
        if self.position.ts.get_position() >= safe_distance:
            self.cover.extract()
        self.position.ts.set_position(current_position, wait=True)
        
    def set_ts_position(self, position, wait=True):
        if abs(self.position.ts.get_position() - position) > 5.:
            self.insert_protective_cover()
        self.position.ts.set_position(position, wait=wait)
        
                
if __name__ == '__main__':
    import optparse
    parser = optparse.OptionParser() 
    parser.add_option('-i', '--ip', default="172.19.10.26", type=str, help='IP address of the server')
    parser.add_option('-p', '--port', default=80, type=int, help='port on which to which it listens to')
    
    options, args = parser.parse_args()
     
    d = detector(options.ip, options.port)