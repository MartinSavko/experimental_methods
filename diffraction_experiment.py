#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
import os
from xray_experiment import xray_experiment

class diffraction_experiment(xray_experiment):
    
    def __init__(self,
                 name_pattern, 
                 directory,
                 position=None,
                 photon_energy=None,
                 resolution=None,
                 detector_distance=None,
                 detector_vertical=None,
                 detector_horizontal=None,
                 transmission=None,
                 flux=None,
                 ntrigger=1,
                 snapshot=False,
                 zoom=None,
                 analysis=None):
        
        xray_experiment.__init__(self, 
                                name_pattern, 
                                directory,
                                position=position,
                                photon_energy=photon_energy,
                                resolution=resolution,
                                detector_distance=detector_distance,
                                detector_vertical=detector_vertical,
                                detector_horizontal=detector_horizontal,
                                transmission=transmission,
                                flux=flux,
                                ntrigger=ntrigger,
                                snapshot=snapshot,
                                zoom=zoom,
                                analysis=analysis)
        
            
    def program_detector(self):
        _start = time.time()
        self.detector.set_standard_parameters()
        self.detector.clear_monitor()
        self.detector.set_ntrigger(self.get_ntrigger())
        self.detector.set_nimages_per_file(self.get_nimages_per_file())
        self.detector.set_nimages(self.get_nimages())
        self.detector.set_name_pattern(self.get_full_name_pattern())
        self.detector.set_frame_time(self.get_frame_time())
        count_time = self.get_frame_time() - self.detector.get_detector_readout_time()
        self.detector.set_count_time(count_time)
        self.detector.set_omega(self.scan_start_angle)
        if self.angle_per_frame <= 0.01:
            self.detector.set_omega_increment(0)
        else:
            self.detector.set_omega_increment(self.angle_per_frame)
            
        self.detector.set_photon_energy(self.photon_energy)
        if self.detector.get_image_nr_start() != self.image_nr_start:
            self.detector.set_image_nr_start(self.image_nr_start)
        
        beam_center_x, beam_center_y = self.beam_center.get_beam_center()
        self.beam_center_x, self.beam_center_y = beam_center_x, beam_center_y
        self.detector.set_beam_center_x(beam_center_x)
        self.detector.set_beam_center_y(beam_center_y)
        self.detector.set_detector_distance(self.detector.position.ts.get_position()/1000.)
        self.sequence_id = self.detector.arm()[u'sequence id']
        print 'program_detector took %s' % (time.time()-_start)
    
    def prepare(self):
        _start = time.time()
        print 'set motors'
        self.goniometer.set_data_collection_phase(wait=False)
        self.set_photon_energy(self.photon_energy, wait=False)
        self.set_detector_distance(self.detector_distance, wait=False)
        self.set_detector_horizontal_position(self.detector_horizontal, wait=False)
        self.set_detector_vertical_position(self.detector_vertical, wait=False)
        self.set_transmission(self.transmission)
        
        self.check_directory(self.process_directory)
        self.program_goniometer()
        self.program_detector()
        if '$id' in self.name_pattern:
            self.name_pattern = self.name_pattern.replace('$id', str(self.sequence_id))
        self.safety_shutter.open()
        self.detector.cover.extract()
        
        # wait for all motors to finish movements
        print 'wait for motors to reach destinations'
        if self.energy_moved > 0:
            self.energy_motor.wait()
        if self.detector_ts_moved != 0:
            self.detector.position.ts.wait()
        if self.detector_ts_moved != 0:
            self.detector.position.tx.wait()
        if self.detector_tz_moved != 0:
            self.detector.position.tz.wait()
        
        # verify all parameters were set 
        #self.check_photon_energy()
        #self.check_resolution()
        #self.check_detector_distance()
        #self.check_detector_vertical()
        #self.check_detector_horizontal()
        #
        
        if self.position != None:
            self.goniometer.set_position(self.position)
            
        self.goniometer.wait()
        if self.scan_start_angle is None:
            self.scan_start_angle = self.reference_position['Omega']
        self.goniometer.set_omega_position(self.scan_start_angle)
        if self.snapshot == True:
            print 'taking image'
            self.camera.set_exposure(0.05)
            self.camera.set_zoom(self.zoom)
            self.goniometer.insert_backlight()
            self.goniometer.extract_frontlight()
            self.goniometer.set_position(self.reference_position)
            self.goniometer.wait()
            self.image = self.camera.get_image()
            self.rgbimage = self.camera.get_rgbimage()
        else:
            self.image = self.camera.get_image()
            self.rgbimage = self.camera.get_rgbimage()
        if self.goniometer.backlight_is_on():
            self.goniometer.remove_backlight()
        
        self.write_destination_namepattern(self.directory, self.name_pattern)
        self.energy_motor.turn_off()
        print 'diffraction_experiment prepare took %s' % (time.time()-_start)
    
    def actuator_monitor(self, start_time, task_id):
        self.observations = []
        self.observations_fields = ['chronos', 'omega_position']
        
        while self.goniometer.is_task_running(task_id):
            chronos = time.time() - start_time
            point = [chronos, self.goniometer.get_omega_position()]
            self.observations.append(point)
            gevent.sleep(self.monitor_sleep_time)
            
        for monitor in self.monitors:
            monitor.observe = False
            
    def save_log(self):
        '''method to save the experiment details in the log file'''
        f = open(os.path.join(self.directory, '%s.log' % self.name_pattern), 'w')
        keyvalues = self.parameters.items()
        keyvalues.sort()
        for key, value in keyvalues:
            if key not in ['image', 'rgb_image']:
                f.write('%s: %s\n' % (key, value)) 
        f.close()
        
