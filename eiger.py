#!/usr/bin/env python
from detector import detector
from goniometer import goniometer
from sweep import sweep
from reference_images import reference_images
from beam_center import beam_center
from raster import raster
from protective_cover import protective_cover
from camera import camera
from resolution import resolution

if __name__ == '__main__':
    import optparse
    parser = optparse.OptionParser() 
    # testbed ip 62.12.151.50
    parser.add_option('-i', '--ip', default="172.19.10.26", type=str, help='IP address of the server')
    parser.add_option('-p', '--port', default=80, type=int, help='port on which to which it listens to')
    
    options, args = parser.parse_args()
     
    d = detector(host=options.ip, port=options.port)
    g = goniometer()