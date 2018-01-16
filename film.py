#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
The purpose of this object is to record a film (a series of images) 
of a rotating sample on the goniometer as a function of goniometer axis (axes) position(s).
'''
from experiment import experiment
from camera import camera
from goniometer import goniometer
from fast_shutter import fast_shutter
import os
import pickle
import gevent

class film(experiment):
    
    specific_parameter_fields = set(['position',
                                     'zoom',
                                     'calibration',
                                     'frontlightlevel',
                                     'backlightlevel',
                                     'scan_exposure_time',
                                     'scan_start_angle',
                                     'scan_speed',
                                     'scan_range',
                                     'md2_task_info',
                                     'frontlight'])
                                     
    def __init__(self, 
                 name_pattern, 
                 directory, 
                 scan_range=360,
                 scan_exposure_time=3.6,
                 scan_start_angle=0,
                 zoom=None,
                 frontlight=False):
        
        experiment.__init__(self, 
                            name_pattern=name_pattern, 
                            directory=directory)
        
        self.scan_range = scan_range
        self.scan_exposure_time = scan_exposure_time
        self.scan_start_angle = scan_start_angle
        self.zoom = zoom
        self.frontlight = frontlight
        
        self.camera = camera()
        self.goniometer = goniometer()
        self.fastshutter = fast_shutter()
        
        self.md2_task_info = None
        
        self.images = []
        
        self.parameter_fields = self.parameter_fields.union(film.specific_parameter_fields)
        
    def get_frontlight(self):
        return self.frontlight
        
    def get_position(self):
        self.goniometer.get_position()
    
    def set_zoom(self, zoom):
        self.camera.set_zoom(zoom)
        
    def get_zoom(self):
        return self.camera.get_zoom()
        
    def get_calibration(self):
        return self.camera.get_calibration()
    
    def get_frontlightlevel(self):
        return self.camera.get_frontlightlevel()
    
    def get_backlightlevel(self):
        return self.camera.get_backlightlevel()
        
    def set_scan_range(self, scan_range):
        self.scan_range = scan_range
    
    def get_scan_range(self):
        return self.scan_range

    def set_scan_exposure_time(self, scan_exposure_time):
        self.scan_exposure_time = scan_exposure_time

    def get_scan_exposure_time(self):
        return self.scan_exposure_time
    
    def get_scan_start_angle(self):
        return self.scan_start_angle
        
    def insert_backlight(self):
        return self.goniometer.insert_backlight()
    
    def insert_frontlight(self):
        return self.goniometer.insert_frontlight()
    
    def extract_frontlight(self):
        return self.goniometer.extract_frontlight()
        
    def get_md2_task_info(self):
        return self.md2_task_info
        
    def prepare(self):
        self.check_directory(self.directory)
        
        self.goniometer.set_data_collection_phase(wait=True)
        
        self.fastshutter.disable()
        
        if self.scan_start_angle != None:
            self.goniometer.set_scan_start_angle(self.scan_start_angle)
        else:
            self.scan_start_angle = self.goniometer.get_scan_start_angle()
        self.goniometer.set_omega_position(self.scan_start_angle - 5)
        
        self.goniometer.set_scan_range(self.scan_range)
        
        self.goniometer.set_scan_exposure_time(self.scan_exposure_time)
        
        if self.zoom != None:
            self.camera.set_zoom(self.zoom)
        else:
            self.zoom = self.camera.get_zoom()
            
        self.insert_backlight()
        
        #self.insert_frontlight()
        if self.frontlight != True:
            self.extract_frontlight()
        
    def cancel(self):
        self.goniometer.abort()
        
    def run(self, task_id=None):
        last_image = None
        if task_id == None:
            task_id = self.goniometer.start_scan()
        
        while self.goniometer.is_task_running(task_id):
            new_image_id = self.camera.get_image_id()
            if new_image_id != last_image:
                last_image = new_image_id
                self.images.append([new_image_id, 
                                    self.goniometer.get_omega_position(), 
                                    self.camera.get_rgbimage()])
        
        self.md2_task_info = self.goniometer.get_task_info(task_id)
        
    def clean(self):
        self.fastshutter.enable()
        self.collect_parameters()
        self.save_parameters()
        self.save_log()
        self.save_results()
        
    def save_results(self):
        f = open('%s.pickle' % (os.path.join(self.directory, self.name_pattern),), 'w')
        pickle.dump(self.images, f)
        f.close()
    
    #def save_log(self):
        #self.log['directory'] = self.directory
        #self.log['name_pattern'] = self.name_pattern
        #self.log['timestamp'] = self.get_timestamp()
        #self.log['start_time'] = self.start_time
        #self.log['end_time'] = self.end_time
        #self.log['motor_positions'] = self.goniometer.get_aligned_position()
        #self.log['zoom'] = self.zoom
        #self.log['calibration'] = self.camera.get_calibration()
        #self.log['camera_exposure'] = self.camera.get_exposure()
        #self.log['scan_start_angle'] = self.scan_start_angle
        #self.log['scan_exposure_time'] = self.scan_exposure_time
        #self.log['scan_range'] = self.scan_range
        #self.log['backlightlevel'] = self.goniometer.get_backlightlevel()
        #self.log['frontlightlevel'] = self.goniometer.get_frontlightlevel()
        #f = open('%s.log' % (os.path.join(self.directory, self.name_pattern),), 'w')
        #pickle.dump(self.log, f)
        #f.close()
                 
def main():
    import optparse
    parser = optparse.OptionParser()
    
    parser.add_option('-r', '--scan_range', default=360, type=float, help='Scan range (default: %default)')
    parser.add_option('-e', '--scan_exposure_time', default=6, type=float, help='Exposure time (default: %default)')
    parser.add_option('-n', '--name_pattern', default='sample', type=str, help='Distinguishing name of files to acquire')
    parser.add_option('-d', '--directory', default='/tmp', type=str, help='Destination directory')
    parser.add_option('-z', '--zoom', default=None, help='Camera zoom to use, current zoom is used by default')
    parser.add_option('-f', '--frontlight', action='store_true', help='Insert frontlight.')
    
    options, args = parser.parse_args()
    
    acquisition = film(**vars(options))
    acquisition.execute()

    print 'nimages', len(acquisition.images)
    
if __name__ == '__main__':
    main()


