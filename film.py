#!/usr/bin/env python
'''
The purpose of this object is to record a film (a series of images) 
of a rotating sample on the goniometer as a function of goniometer axis (axes) position(s).
'''
from experiment import experiment
from camera import camera
from goniometer import goniometer
from fastshutter import fastshutter
import os
import pickle
import gevent

#from backlight import backlight
#from frontlight import frontlight

class film(experiment):
    def __init__(self, 
                 name_pattern, 
                 directory, 
                 scan_range=360,
                 scan_exposure_time=3.6,
                 scan_start_angle=None,
                 zoom=None):
        
        experiment.__init__(self, 
                            name_pattern=name_pattern, 
                            directory=directory)
        
        self.scan_range = scan_range
        self.scan_exposure_time = scan_exposure_time
        self.scan_start_angle = scan_start_angle
        self.zoom = zoom
        
        self.camera = camera()
        self.goniometer = goniometer()
        self.fastshutter = fastshutter()
        self.log = {}
        self.images = []
    
    def set_scan_range(self, scan_range):
        self.scan_range = scan_range
    
    def get_scan_range(self):
        return self.scan_range

    def set_scan_exposure_time(self, scan_exposure_time):
        self.scan_exposure_time = scan_exposure_time

    def get_scan_exposure_time(self):
        return self.scan_exposure_time
    
    def insert_backlight(self):
        return self.goniometer.insert_backlight()
    
    def insert_frontlight(self):
        return self.goniometer.insert_frontlight()
    
    def prepare(self):
        self.check_directory(self.directory)
        
        self.goniometer.set_data_collection_phase(wait=True)
        
        self.fastshutter.disable()
        
        if self.scan_start_angle != None:
            self.goniometer.set_scan_start_angle(scan_start_angle)
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
        
        self.insert_frontlight()
        
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
                                    self.camera.get_image(), 
                                    self.camera.get_rgbimage()])
        
    def clean(self):
        self.fastshutter.enable()

    def save(self):
        f = open('%s.pickle' % (os.path.join(self.directory, self.name_pattern),), 'w')
        pickle.dump(self.images, f)
        f.close()
        
    def save_log(self):
        self.log['directory'] = self.directory
        self.log['name_pattern'] = self.name_pattern
        self.log['timestamp'] = self.get_timestamp()
        self.log['start_time'] = self.start_time
        self.log['end_time'] = self.end_time
        self.log['motor_positions'] = self.goniometer.get_aligned_position()
        self.log['zoom'] = self.zoom
        self.log['calibration'] = self.camera.get_calibration()
        self.log['camera_exposure'] = self.camera.get_exposure()
        self.log['scan_start_angle'] = self.scan_start_angle
        self.log['scan_exposure_time'] = self.scan_exposure_time
        self.log['scan_range'] = self.scan_range
        self.log['backlightlevel'] = self.goniometer.get_backlightlevel()
        self.log['frontlightlevel'] = self.goniometer.get_frontlightlevel()
        f = open('%s.log' % (os.path.join(self.directory, self.name_pattern),), 'w')
        pickle.dump(self.log, f)
        f.close()
                 
def main():
    import optparse
    parser = optparse.OptionParser()
    
    parser.add_option('-r', '--scan_range', default=360, type=float, help='Scan range (default: %default)')
    parser.add_option('-e', '--scan_exposure_time', default=3.6, type=float, help='Exposure time (default: %default)')
    parser.add_option('-n', '--name_pattern', default='sample', type=str, help='Distinguishing name of files to acquire')
    parser.add_option('-d', '--directory', default='/tmp', type=str, help='Destination directory')
    parser.add_option('-z', '--zoom', default=None, help='Camera zoom to use, current zoom is used by default')
    
    options, args = parser.parse_args()
    
    acquisition = film(options.name_pattern, options.directory, options.scan_range, options.scan_exposure_time, options.zoom)
    acquisition.execute()
    acquisition.save()
    acquisition.save_log()
    print acquisition.log
    print 'nimages', len(acquisition.images)
if __name__ == '__main__':
    main()


