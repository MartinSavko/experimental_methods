#!/usr/bin/env python
import time
from raster_scan import raster_scan
import optparse
import traceback
import os

def main():
    
    parser = optparse.OptionParser()
    parser.add_option('-n', '--name_pattern' , default='grid', type=str, help='Template of files with the scan results, (default: %default)')
    parser.add_option('-d', '--directory', default='/tmp', type=str, help='Directory with the scan results, (default: %default)')
    parser.add_option('-y', '--vertical_range', default=0.1, type=float, help='Vertical scan range (default: %default)')
    parser.add_option('-x', '--horizontal_range', default=0.2, type=float, help='Horizontal scan range (default: %default)')
    parser.add_option('-r', '--number_of_rows', default=10, type=int, help='Number of rows (default: %default)')
    parser.add_option('-c', '--number_of_columns', default=1, type=int, help='Number of columns (default: %default)')
    parser.add_option('-e', '--scan_exposure_time', default=0.0045, type=float, help='Exposure time per image (default: %default')
    parser.add_option('-p', '--scan_start_angle', default=None, type=float, help='Orientation of the sample on the gonio during the grid scan. Current orientation is taken by default.')
    parser.add_option('-m', '--method', default='helical', type=str, help='use md2 rasterscan or helical (default: %default)')
    parser.add_option('-s', '--scan_range', default=0.1, type=float, help='Oscillation per line (default: %default)')
    parser.add_option('-a', '--scan_axis', default='vertical', type=str, help='Horizontal or vertical scan axis (default: %default)')
    parser.add_option('-z', '--zoom', default=None, type=int, help='Zoom at which to record the optical images. The current zoom will be used by default.')
    parser.add_option('-A', '--do_not_analyze', action="store_true", help='Do not analyze.')
    
    options, args = parser.parse_args()
    y = options.vertical_range
    x = options.horizontal_range
    r = options.number_of_rows
    c = options.number_of_columns
    e = options.scan_exposure_time
    p = options.scan_start_angle
    m = options.method
    s = options.scan_range
    a = options.scan_axis
    z = options.zoom
    name_pattern = options.name_pattern
    directory = options.directory
    
    print 'raster_scan(%s, %s, %s, %s, %s, scan_start_angle=%s, scan_range=%s, scan_axis=%s, method=%s, zoom=%s, name_pattern="%s", directory="%s")' % (y, x, r, c, e, p, s, a, m, z, name_pattern, directory)
    r = raster_scan(name_pattern, directory, y, x, number_of_rows=r, number_of_columns=c, frame_time=e, scan_start_angle=p, scan_range=s, scan_axis=a, zoom=z)
    k=0
    while k<3:
        k+=1
        try:
            r.execute()
            break
        except:
            print traceback.print_exc()
            time.sleep(1)
    
    #if options.do_not_analyze:
    #    pass
    #else:
    os.system('area_sense.py -n %s -d %s && cd %s && eog %s_*.png & ' % (name_pattern, directory, directory, name_pattern))
        #os.chdir(directory)
        #os.system('eog %s_*.png &' % name_pattern)
    
if __name__ == '__main__':
    #return 
    main()
