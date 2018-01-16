#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
helical scan
'''
import traceback
import logging
import time
import pickle
import os

import numpy as np

from omega_scan import omega_scan

class inverse_scan(omega_scan):
    
    actuator_names = ['Omega']
    
    specific_parameter_fields = set(['interleave_range',
                                    'raster'])
    
    def __init__(self, 
                 name_pattern='test_$id', 
                 directory='/tmp', 
                 scan_range=180, 
                 scan_exposure_time=18, 
                 scan_start_angle=0, 
                 angle_per_frame=0.1, 
                 image_nr_start=1,
                 interleave_range=5,
                 raster=False,
                 position=None,
                 photon_energy=None,
                 resolution=None,
                 detector_distance=None,
                 detector_vertical=None,
                 detector_horizontal=None,
                 transmission=None,
                 flux=None,
                 snapshot=False,
                 nimages_per_file=None,
                 zoom=None,
                 diagnostic=None,
                 analysis=None,
                 simulation=None):
        
        omega_scan.__init__(self,
                            name_pattern=name_pattern, 
                            directory=directory,
                            scan_range=scan_range, 
                            scan_exposure_time=scan_exposure_time,
                            angle_per_frame=angle_per_frame, 
                            image_nr_start=image_nr_start,
                            photon_energy=photon_energy,
                            resolution=resolution,
                            detector_distance=detector_distance,
                            detector_vertical=detector_vertical,
                            detector_horizontal=detector_horizontal,
                            transmission=transmission,
                            flux=flux,
                            snapshot=snapshot,
                            nimages_per_file=nimages_per_file,
                            diagnostic=diagnostic,
                            analysis=analysis,
                            simulation=simulation)
        
        self.raster = raster
        self.interleave_range = float(interleave_range)
        
        self.total_expected_exposure_time = self.scan_exposure_time * 2
        self.total_expected_wedges = int(self.scan_range/self.interleave_range) * 2
        
        self.parameter_fields = self.parameter_fields.union(inverse_scan.specific_parameter_fields)
        
    def get_wedges(self):
        direct = np.arange(self.scan_start_angle, self.scan_start_angle+self.scan_range, self.interleave_range)
        inverse = direct + 180
        wedges = np.vstack((direct, inverse)).T
        if self.raster == True:
            if np.__version__ >= '1.8.3':
                #functionality present probably already earlier
                wedges[1::2, ::] = wedges[1::2, ::-1]
            else:
                raster = []
                for k, line in enumerate(wedges):
                    if k%2 == 1:
                        line = line[::-1]
                    raster.append(line)
                wedges = np.array(raster)
        return wedges.ravel()
    
    def get_interleave_range(self):
        return self.interleave_range
    
    def get_raster(self):
        return self.raster
    
    def run(self):
        self._start = time.time()
        
        self.wedges = self.get_wedges()
        
        scan_range = self.interleave_range
        scan_exposure_time = self.interleave_range * self.scan_exposure_time / self.scan_range 
        
        self.md2_tasks_info = []
        
        for scan_start_angle in self.wedges:
            task_id = self.goniometer.omega_scan(scan_start_angle, scan_range, scan_exposure_time, wait=True)
            self.md2_tasks_info.append(self.goniometer.get_task_info(task_id))
            
    def get_nimages_per_file(self):
        return int(self.interleave_range/self.angle_per_frame)
    
    def get_nimages(self):
        return int(self.interleave_range/self.angle_per_frame)
    
    def get_ntrigger(self):
        return int(self.scan_range/self.interleave_range) * 2
    
    def get_frame_time(self):
        return 2 * self.scan_exposure_time/(self.get_ntrigger() * self.get_nimages())
        
        
def main():
    import optparse
        
    parser = optparse.OptionParser()
    parser.add_option('-n', '--name_pattern', default='inverse_test_raster_$id', type=str, help='Prefix default=%default')
    parser.add_option('-d', '--directory', default='/nfs/data/default', type=str, help='Destination directory default=%default')
    parser.add_option('-r', '--scan_range', default=180, type=float, help='Scan range [deg]')
    parser.add_option('-e', '--scan_exposure_time', default=18, type=float, help='Scan exposure time [s]')
    parser.add_option('-s', '--scan_start_angle', default=0, type=float, help='Scan start angle [deg]')
    parser.add_option('-a', '--angle_per_frame', default=0.1, type=float, help='Angle per frame [deg]')
    parser.add_option('-f', '--image_nr_start', default=1, type=int, help='Start image number [int]')
    parser.add_option('-I', '--interleave_range', default=10, type=float, help='Interleave range [deg]')
    parser.add_option('-P', '--position', default=None, type=str, help='Gonio alignment start position [dict]')
    parser.add_option('-p', '--photon_energy', default=None, type=float, help='Photon energy ')
    parser.add_option('-t', '--detector_distance', default=None, type=float, help='Detector distance')
    parser.add_option('-o', '--resolution', default=None, type=float, help='Resolution [Angstroem]')
    parser.add_option('-x', '--flux', default=None, type=float, help='Flux [ph/s]')
    parser.add_option('-m', '--transmission', default=None, type=float, help='Transmission. Number in range between 0 and 1.')
    parser.add_option('-A', '--analysis', action='store_true', help='If set will perform automatic analysis.')
    parser.add_option('-D', '--diagnostic', action='store_true', help='If set will record diagnostic information.')
    parser.add_option('-S', '--simulation', action='store_true', help='If set will simulate most of the beamline (except detector and goniometer).')
    parser.add_option('-R', '--raster', action='store_true', help='If set will record diagnostic information.')
    
    options, args = parser.parse_args()
    print 'options', options
    invs = inverse_scan(**vars(options))
    invs.execute()
    
def test():
    scan_range = 180
    scan_exposure_time = 18.
    scan_start_angle = 0
    angle_per_frame = 0.1
    interleave_range = 5
    s = inverse_scan(scan_range=scan_range, scan_exposure_time=scan_exposure_time, scan_start_angle=scan_start_angle, angle_per_frame=angle_per_frame, interleave_range=interleave_range)
    
if __name__ == '__main__':
    main()
    