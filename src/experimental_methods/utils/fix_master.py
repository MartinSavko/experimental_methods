#!/usr/bin/env python
# -*- coding: utf-8 -*-

import h5py
import shutil
import numpy
import os

def is_reference(filename):
    return 'ref-' == filename[:4]
    
    
def fix_sensor_thickness(m):
    # this corrects bug in the firmware 1.5.3 of PX2's Eiger 9M
    if m['/entry/instrument/detector/sensor_thickness'].value != 0.00045:
        m['/entry/instrument/detector/sensor_thickness'].write_direct(numpy.array(0.00045))
    
    
def fix_ref_master(m):
    ntrigger = m['/entry/instrument/detector/detectorSpecific/ntrigger'].value
    nimages = m['/entry/instrument/detector/detectorSpecific/nimages'].value
    omega_increment = m['/entry/sample/goniometer/omega_increment'].value
    omega_start = m['/entry/sample/goniometer/omega'].value[0]
    omega = []
    omega_end = []
    for k in range(ntrigger):
        start = omega_start + k*90.
        end = start + nimages*omega_increment
        omega += list(numpy.arange(start, end , omega_increment))
        omega_end += list(numpy.arange(start + omega_increment, end+omega_increment , omega_increment))
        m['/entry/data/data_%06d' % (k+1,)].attrs['image_nr_low'] = int((start + omega_increment)/ omega_increment)
        m['/entry/data/data_%06d' % (k+1,)].attrs['image_nr_high'] = int(end / omega_increment)
    m['/entry/sample/goniometer/omega'].write_direct(numpy.array(omega))
    m['/entry/sample/goniometer/omega_end'].write_direct(numpy.array(omega_end))


def main():
    import optparse
    parser = optparse.OptionParser() 
    
    parser.add_option('-m', '--master', default=None, type=str, help='path to the master file to be fixed')
    
    options, args = parser.parse_args()
    
    abspath = os.path.abspath(options.master)
    dirname = os.path.dirname(abspath)
    basename = os.path.basename(abspath)
    
    archivename = '%s/.%s_0001' % (dirname, basename)
    
    if not os.path.isfile(archivename):
        shutil.copy(abspath, archivename)
        
    m = h5py.File(options.master, 'r+')
    
    fix_sensor_thickness(m)
    
    if is_reference(options.master):
        print 'is reference'
        fix_ref_master(m)
    else:
        print 'is not a reference'
    m.close()
    
    
if __name__ == '__main__':
    main()
    
    
