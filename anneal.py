#!/usr/bin/env python

import time
import optparse
try:
    import tango
except ImportError:
    import PyTango as tango
    
md = tango.DeviceProxy('i11-ma-cx1/ex/md3')
blade = tango.DeviceProxy('i11-ma-cx1/ex/annealing')

def main():
    parser = optparse.OptionParser()

    parser.add_option('-t', '--time', default='3', type=float, help='Block the cryostream for a given time (default %default seconds)')
    options, args = parser.parse_args()
    
    anneal(options.time)

def anneal(t):
    md.cryoisback = True
    blade.insert()
    while blade.isInserted() == False:
        pass
    time.sleep(t)
    blade.extract()
    md.cryoisback = False
    return 

if __name__ == '__main__':
    main()

