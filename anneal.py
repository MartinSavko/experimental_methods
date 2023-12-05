#!/usr/bin/env python

import time
import optparse
try:
    import tango
except ImportError:
    import PyTango as tango
    
md2 = tango.DeviceProxy('i11-ma-cx1/ex/md2')
blade = tango.DeviceProxy('i11-ma-cx1/ex/annealing')

def main():
    parser = optparse.OptionParser()

    parser.add_option('-t', '--time', default='3', type=float, help='Block the cryostream for a given time (default %default seconds)')
    options, args = parser.parse_args()
    
    anneal(options.time)

def anneal(t):
    md2.cryoisback = True
    blade.insert()
    while blade.isInserted() == False:
        pass
    time.sleep(t)
    blade.extract()
    md2.cryoisback = False
    return 

if __name__ == '__main__':
    main()

