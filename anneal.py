#!/usr/bin/python

import PyTango
import time
import optparse

md2 = PyTango.DeviceProxy('i11-ma-cx1/ex/md2')
blade = PyTango.DeviceProxy('i11-ma-cx1/ex/annealing')

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

if __name__ == '__main__':
    main()

