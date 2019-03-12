#!/usr/bin/env python
# -*- coding: utf-8 -*-
import gevent

import time
import os
from xray_experiment import xray_experiment

class diffraction_experiment(xray_experiment):
    
    specific_parameter_fields = [{'name': 'kappa', 'type': 'float', 'description': 'kappa position in degrees'},
                                 {'name': 'phi', 'type': 'float', 'description': 'phi position in degrees'},
                                 {'name': 'resolution', 'type': 'float', 'description': ''},
                                 {'name': 'detector_distance', 'type': 'float', 'description': ''},
                                 {'name': 'detector_vertical', 'type': 'float', 'description': ''},
                                 {'name': 'detector_horizontal', 'type': 'float', 'description': ''},
                                 {'name': 'nimages', 'type': 'int', 'description': ''},
                                 {'name': 'ntrigger', 'type': 'int', 'description': ''},
                                 {'name': 'nimages_per_file', 'type': 'int', 'description': ''},
                                 {'name': 'image_nr_start', 'type': 'int', 'description': ''},
                                 {'name': 'total_expected_wedges', 'type': 'float', 'description': ''},
                                 {'name': 'total_expected_exposure_time', 'type': 'float', 'description': ''},
                                 {'name': 'beam_center_x', 'type': 'float', 'description': ''},
                                 {'name': 'beam_center_y', 'type': 'float', 'description': ''},
                                 {'name': 'sequence_id', 'type': 'int', 'description': ''},
                                 {'name': 'frames_per_second', 'type': 'float', 'description': 'number of frames per second'}]
    def __init__(self,
                 name_pattern, 
                 directory,
                 frames_per_second=None,
                 position=None,
                 kappa=None,
                 phi=None,
                 chi=None,
                 photon_energy=None,
                 resolution=None,
                 detector_distance=None,
                 detector_vertical=None,
                 detector_horizontal=None,
                 transmission=None,
                 flux=None,
                 ntrigger=1,
                 snapshot=True,
                 zoom=None,
                 diagnostic=None,
                 analysis=None,
                 conclusion=None,
                 simulation=None,
                 parent=None):
        
        if hasattr(self, 'parameter_fields'):
            self.parameter_fields += diffraction_experiment.specific_parameter_fields
        else:
            self.parameter_fields = diffraction_experiment.specific_parameter_fields
        
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
                                diagnostic=diagnostic,
                                analysis=analysis,
                                conclusion=conclusion,
                                simulation=simulation,
                                parent=parent)
        
        self.description = 'Diffraction experiment, Proxima 2A, SOLEIL, %s' % time.ctime(self.timestamp)
        
        self.actuator = self.goniometer
        
        self.frames_per_second = frames_per_second
        
        if kappa == None:
            self.kappa = self.goniometer.md2.kappaposition
        else:
            self.kappa = kappa
        if phi == None:
            self.phi = self.goniometer.md2.phiposition
        else:
            self.phi = phi
        if chi == None:
            self.chi = self.goniometer.md2.chiposition
        else:
            self.chi = chi
            
        # Set resolution: detector_distance takes precedence
        # if neither specified, takes currect detector_distance 
        
        if self.detector_distance == None and self.resolution == None:
            self.detector_distance = self.detector.position.ts.get_position()
        
        if self.detector_distance != None:
            self.resolution = self.resolution_motor.get_resolution_from_distance(self.detector_distance, wavelength=self.wavelength)
        elif self.resolution != None:
            self.detector_distance = self.resolution_motor.get_distance_from_resolution(self.resolution, wavelength=self.wavelength)
            print 'self.detector_distance calculated from resolution', self.detector_distance
        else:
            print 'There seem to be a problem with logic for detector distance determination. Please check'
        
        
    def get_expected_files(self):
        self.expected_files = ['%s_master.h5' % self.name_pattern]
        nimages_per_file = self.get_nimages_per_file()
        nimages = self.get_nimages()
        ntrigger = self.get_ntrigger()
        total_number_of_images = nimages * ntrigger
        accounted_for = 0
        k = 0 
        while accounted_for < total_number_of_images:
            k += 1
            data_file = '%s_data_%06d.h5' % (self.name_pattern, k)
            self.expected_files.append(data_file)
            accounted_for += nimages_per_file
        return self.expected_files
        
    def get_nimages_per_file(self):
        return self.nimages_per_file
    
    def get_degrees_per_frame(self):
        '''get degrees per frame'''
        return self.angle_per_frame
    
    def get_fps(self):
        return self.get_frames_per_second()
    
    def get_frames_per_second(self):
        '''get frames per second'''
        return self.get_nimages()/self.scan_exposure_time
    
    def get_dps(self):
        return self.get_degrees_per_second()
    
    def get_degrees_per_second(self):
        '''get degrees per second'''
        return self.get_scan_speed()
    
    def get_scan_speed(self):
        '''get scan speed'''
        return self.scan_range/self.scan_exposure_time
    
    def get_frame_time(self):
        '''get frame time'''
        return self.scan_exposure_time/self.get_nimages()

    def get_reference_position(self):
        return self.get_position()
    
    def get_position(self):
        '''get position '''
        if self.position is None:
            return self.goniometer.get_position()
        else:
            return self.position
        
    def set_position(self, position=None):
        '''set position'''
        if position is None:
            self.position = self.goniometer.get_position()
        else:
            self.position = position
            self.goniometer.set_position(self.position)
            self.goniometer.wait()
        self.goniometer.save_position()
    
    def get_kappa(self):
        return self.kappa
    def set_kappa(self, kappa):
        self.kappa = kappa
    
    def get_phi(self):
        return self.phi
    def set_phi(self, phi):
        self.phi = phi
        
    def get_chi(self):
        return self.chi
    def set_chi(self):
        self.chi = chi
        
    def get_md2_task_info(self):
        return self.md2_task_info
    
    def set_nimages(self, nimages):
        self.nimages = nimages
    def get_nimages(self):
        return self.nimages
    
    def get_scan_range(self):
        return self.scan_range
    def set_scan_range(self, scan_range):
        self.scan_range = scan_range
        
    def get_scan_exposure_time(self):
        return self.scan_exposure_time
    def set_scan_exposure_time(self, scan_exposure_time):
        self.scan_exposure_time = scan_exposure_time
    
    def get_scan_start_angle(self):
        return self.scan_start_angle
    def set_scan_start_angle(self, scan_start_angle):
        self.scan_start_angle = scan_start_angle
        
    def get_angle_per_frame(self):
        return self.angle_per_frame
    def set_angle_per_frame(self, angle_per_frame):
        self.angle_per_frame = angle_per_frame
        
    def set_ntrigger(self, ntrigger):
        self.ntrigger = ntrigger
    def get_ntrigger(self):
        return self.ntrigger
    
    def set_resolution(self, resolution=None):
        if resolution != None:
            self.resolution = resolution
            self.resolution_motor.set_resolution(resolution)
    def get_resolution(self):
        return self.resolution_motor.get_resolution()

    def get_detector_distance(self):
        return self.detector.position.ts.get_position()
    def set_detector_distance(self, position, wait=True):
        self.detector_ts_moved = self.detector.set_ts_position(position, wait=wait)
        
    def get_detector_vertical_position(self):
        return self.detector.position.tz.get_position()
    def set_detector_vertical_position(self, position, wait=True):
        self.detector_tz_moved = self.detector.position.tz.set_position(position, wait=wait)
        
    def get_detector_horizontal_position(self):
        return self.detector.position.tx.get_position()
    def set_detector_horizontal_position(self, position, wait=True):
        self.detector_tx_moved = self.detector.position.tx.set_position(position, wait=wait)

    def get_sequence_id(self):
        return self.sequence_id
    
    def get_detector_ts_intention(self):
        return self.detector_distance
    def get_detector_tz_intention(self):
        return self.detector_vertical
    def get_detector_tx_intention(self):
        return self.detector_horizontal
    
    def get_detector_ts(self):
        return self.get_detector_distance()
        
    def get_detector_tz(self):
        return self.get_detector_vertical_position()
        
    def get_detector_tx(self):
        return self.get_detector_horizontal_position()
    
    def get_detector_vertical(self):
        return self.detector_vertical
    def set_detector_vertical(self, detector_vertical):
        self.detector_vertical = detector_vertical
        
    def get_detector_horizontal(self):
        return self.detector_horizontal
    def set_detector_horizontal(self, detector_horizontal):
        self.detector_horizontal = detector_horizontal
        
    def set_nimages_per_file(self, nimages_per_file):
        self.nimages_per_file = nimages_per_file
    def get_nimages_per_file(self):
        return self.nimages_per_file
    
    def set_image_nr_start(self):
        self.image_nr_start = image_nr_start
    def get_image_nr_start(self):
        return self.image_nr_start
    
    def set_total_expected_wedges(self, total_expected_wedges):
        self.total_expected_wedges = total_expected_wedges
    def get_total_expected_wedges(self):
        return self.total_expected_wedges
    
    def set_total_expected_exposure_time(self, total_expected_exposure_time):
        self.total_expected_exposure_time = total_expected_exposure_time
    def get_total_expected_exposure_time(self):
        return self.total_expected_exposure_time
    
    def set_beam_center_x(self, beam_center_x):
        self.beam_center_x = beam_center_x
    def get_beam_center_x(self):
        return self.beam_center_x
    
    def set_beam_center_y(self, beam_center_y):
        self.beam_center_y = beam_center_y
    def get_beam_center_y(self):
        return self.beam_center_y
        
    def program_detector(self):
        _start = time.time()
        if self.detector.get_trigger_mode() != 'exts':
            self.detector.set_trigger_mode('exts')
            
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
        if self.angle_per_frame <= 0.001:
            self.detector.set_omega_increment(0)
        else:
            self.detector.set_omega_increment(self.angle_per_frame)
        
        self.detector.set_kappa(self.kappa)
        self.detector.set_phi(self.phi)
        self.detector.set_chi(self.chi)
        
        self.detector.set_photon_energy(self.photon_energy)
        
        if self.detector.get_image_nr_start() != self.image_nr_start:
            self.detector.set_image_nr_start(self.image_nr_start)
        
        if self.simulation != True:
            beam_center_x, beam_center_y = self.beam_center.get_beam_center(wavelength=self.wavelength, ts=self.detector_distance, tx=self.detector_horizontal, tz=self.detector_vertical)
        else:
            beam_center_x, beam_center_y = 1430, 1550
        
        self.beam_center_x, self.beam_center_y = beam_center_x, beam_center_y
        self.detector.set_beam_center_x(beam_center_x)
        self.detector.set_beam_center_y(beam_center_y)
        if self.simulation == True:
            self.detector_distance = 0.25
        self.detector.set_detector_distance(self.detector_distance/1000.)
        self.sequence_id = self.detector.arm()[u'sequence id']
        print 'program_detector took %s' % (time.time()-_start)
    
    def prepare(self):
        _start = time.time()
        print 'set motors'
        
        initial_settings = []
        if self.simulation != True: 
            initial_settings.append(gevent.spawn(self.goniometer.set_data_collection_phase, wait=True))
            initial_settings.append(gevent.spawn(self.set_photon_energy, self.photon_energy, wait=True))
            initial_settings.append(gevent.spawn(self.set_detector_distance, self.detector_distance, wait=True))
            initial_settings.append(gevent.spawn(self.set_detector_horizontal_position, self.detector_horizontal, wait=True))
            initial_settings.append(gevent.spawn(self.set_detector_vertical_position, self.detector_vertical, wait=True))
            initial_settings.append(gevent.spawn(self.set_transmission, self.transmission))
        
        self.eiger_en_out.stop()
        self.eiger_en_out.start()
        
        self.check_directory(self.process_directory)
        self.program_goniometer()
        self.program_detector()
        
        print 'wait for motors to reach destinations'
        gevent.joinall(initial_settings)
        
        if self.detector.cover.closed():
            self.detector.extract_protective_cover()
            
        if '$id' in self.name_pattern:
            self.name_pattern = self.name_pattern.replace('$id', str(self.sequence_id))
        
        
                
        if self.position != None:
            self.goniometer.set_position(self.position)
            
        self.goniometer.wait()
        
        if self.scan_start_angle is None:
            self.scan_start_angle = self.reference_position['Omega']
        else:
            self.reference_position['Omega'] = self.scan_start_angle
        #self.goniometer.set_omega_position(self.scan_start_angle)
        
        if self.snapshot == True:
            print 'taking image'
            self.camera.set_exposure(0.05)
            self.goniometer.insert_backlight()
            self.camera.set_zoom(self.zoom)

            self.goniometer.extract_frontlight()
            self.goniometer.set_position(self.reference_position)
            self.goniometer.wait()
            self.image = self.get_image()
            self.rgbimage = self.get_rgbimage()

        if self.goniometer.backlight_is_on():
            self.goniometer.remove_backlight()
        
        if self.simulation != True:
            self.safety_shutter.open()
            if self.detector.cover.closed() == True:
                self.detector.cover.extract()
                gevent.sleep(2)
            
        self.write_destination_namepattern(self.directory, self.name_pattern)
        
        self.goniometer.insert_frontlight()
        
        if self.simulation != True: 
            self.energy_motor.turn_off()
        
        print 'diffraction_experiment prepare took %s' % (time.time()-_start)
        
        
