#!/usr/bin/env python
# -*- coding: utf-8 -*-

from eiger import eiger
from protective_cover import protective_cover
from detector_position import detector_position
from detector_beamstop import detector_beamstop

class detector(eiger):

    def __init__(self, host='172.19.10.26', port=80):
        eiger.__init__(self, host=host, port=port)
        self.position = detector_position()
        self.beamstop = detector_beamstop()
        self.cover = protective_cover()
        
if __name__ == '__main__':
    import optparse
    parser = optparse.OptionParser() 
    parser.add_option('-i', '--ip', default="172.19.10.26", type=str, help='IP address of the server')
    parser.add_option('-p', '--port', default=80, type=int, help='port on which to which it listens to')
    
    options, args = parser.parse_args()
     
    d = detector(options.ip, options.port)