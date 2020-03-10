#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Object allows to define and carry out a collection of series of wedges of diffraction images of arbitrary slicing parameter and of arbitrary size at arbitrary reference angles.
'''
import traceback
import logging
import time
import os
import pickle
import numpy as np
import h5py
from omega_scan import omega_scan
import gevent
import pexpect
import re
import glob
import commands
import shutil
import subprocess
import xmlrpclib
import jsonpickle

class reference_images(omega_scan):
    
    actuator_names = ['Omega']
    
    specific_parameter_fields = [{'name': 'scan_start_angles', 'type': '', 'description': ''},
                                 {'name': 'dose_rate', 'type': '', 'description': ''},
                                 {'name': 'dose_limit', 'type': '', 'description': ''}, 
                                 {'name': 'vertical_scan_length', 'type': '', 'description': ''},
                                 {'name': 'vertical_step_size', 'type': '', 'description': ''},
                                 {'name': 'inverse_direction', 'type': '', 'description': ''},
                                 {'name': 'vertical_motor_speed', 'type': '', 'description': ''},
                                 {'name': 'exposure_time_per_frame', 'type': 'float', 'description': 'frame time in s'}]

    def __init__(self, 
                 name_pattern='ref-test_$id', 
                 directory='/tmp', 
                 scan_range=1., 
                 scan_exposure_time=1, 
                 scan_start_angles='[0, 90, 180, 270]', 
                 angle_per_frame=0.1, 
                 image_nr_start=1,
                 vertical_scan_length=0,
                 vertical_step_size=0.025,
                 inverse_direction=True,
                 dose_rate=0.25e6, #Grays per second
                 dose_limit=20e6, #Grays
                 i2s_at_highest_resolution=1.,
                 frames_per_second=None,
                 position=None,
                 kappa=None,
                 phi=None,
                 photon_energy=None,
                 resolution=None,
                 detector_distance=None,
                 detector_vertical=None,
                 detector_horizontal=None,
                 transmission=None,
                 flux=None,
                 snapshot=None,
                 diagnostic=None,
                 analysis=None,
                 simulation=None,
                 parent=None,
                 treatment_directory='/dev/shm',
                 xmlrpc_server='http://localhost:60006',
                 mxcube_parent_id=None,
                 mxcube_gparent_id=None): 
        
        logging.debug('reference_images __init__ len(reference_images.specific_parameter_fields) %d' % len(reference_images.specific_parameter_fields))

        if hasattr(self, 'parameter_fields'):
            logging.debug('reference_images __init__ len(self.parameter_fields) %d' % len(self.parameter_fields))
            self.parameter_fields += reference_images.specific_parameter_fields
        else:
            self.parameter_fields = reference_images.specific_parameter_fields[:]
        
        logging.debug('reference_images __init__ len(self.parameters_fields) %d' % len(self.parameter_fields))

        if isinstance(scan_start_angles, str):
            scan_start_angles = eval(scan_start_angles)
        self.scan_start_angles = scan_start_angles
        self.scan_range = float(scan_range)
        
        self.vertical_scan_length = float(vertical_scan_length)
        self.vertical_step_size = float(vertical_step_size)
        
        if self.vertical_scan_length != 0 and self.vertical_scan_length != None:
            nimages = int(self.vertical_scan_length/self.vertical_step_size)
            angle_per_frame = self.scan_range/nimages
        
        ntrigger = len(self.scan_start_angles)
        nimages_per_file = int(self.scan_range/angle_per_frame)
        
        omega_scan.__init__(self,
                            name_pattern, 
                            directory, 
                            scan_range=scan_range, 
                            scan_start_angle=self.scan_start_angles[0],
                            scan_exposure_time=scan_exposure_time, 
                            angle_per_frame=angle_per_frame, 
                            image_nr_start=image_nr_start,
                            frames_per_second=frames_per_second,
                            position=position,
                            kappa=kappa,
                            phi=phi,
                            photon_energy=photon_energy,
                            resolution=resolution,
                            detector_distance=detector_distance,
                            detector_vertical=detector_vertical,
                            detector_horizontal=detector_horizontal,
                            transmission=transmission,
                            flux=flux,
                            snapshot=snapshot,
                            ntrigger=ntrigger,
                            nimages_per_file=nimages_per_file,
                            diagnostic=diagnostic,
                            analysis=analysis,
                            simulation=simulation,
                            parent=parent,
                            mxcube_parent_id=mxcube_parent_id,
                            mxcube_gparent_id=mxcube_gparent_id)
        
        self.ntrigger = ntrigger
        
        self.total_expected_exposure_time = scan_exposure_time * ntrigger
        self.total_expected_wedges = ntrigger
        
        self.inverse_direction = inverse_direction
        self.dose_rate = dose_rate
        self.dose_limit = dose_limit
        self.i2s_at_highest_resolution = i2s_at_highest_resolution
        
        self.saved_parameters = self.load_parameters_from_file()
        
        self.treatment_directory = treatment_directory
        
        self.format_dictionary = {'directory': self.directory, 'name_pattern': self.name_pattern, 'treatment_directory': self.treatment_directory}
        self.description = 'Reference images, Proxima 2A, SOLEIL, %s' % time.ctime(self.timestamp)
        self.server = xmlrpclib.ServerProxy(xmlrpc_server)
        
    def get_nimages_per_file(self):
        if self.saved_parameters is not None:
            return self.saved_parameters['nimages_per_file']
        return int(self.scan_range/self.angle_per_frame)
    
    
    def get_exposure_time_per_frame(self):
        if self.saved_parameters is not None:
            if self.saved_parameters.has_key('exposure_time_per_frame'):
                return self.saved_parameters['exposure_time_per_frame']
        return self.scan_exposure_time/self.get_nimages()
    
    
    def get_nimages(self, epsilon=1e-3):
        if self.saved_parameters is not None:
            return self.saved_parameters['nimages']
        
        nimages = int(self.scan_range/self.angle_per_frame)
        if abs(nimages*self.angle_per_frame - self.scan_range) > epsilon:
            nimages += 1
        return nimages
        
    
    
    def get_ntrigger(self):
        if self.saved_parameters is not None:
            return self.saved_parameters['ntrigger']
        return self.ntrigger
    
    
    def get_vertical_motor_speed(self):
        if self.saved_parameters is not None:
            return self.saved_parameters['vertical_motor_speed']
        return self.vertical_scan_length/self.scan_exposure_time
    
    def get_angle_per_frame(self):
        if self.saved_parameters is not None:
            return self.saved_parameters['angle_per_frame']
        return self.angle_per_frame
    
    def save_snapshots(self):
        logging.info('save_snapshots')
        snapshots = []
        self.goniometer.insert_backlight()
        for k, scan_start_angle in enumerate(self.scan_start_angles):
            self.goniometer.set_orientation(scan_start_angle)
            imagename, image, image_id = self.camera.save_image('%s_%.2f_rgb.png' % (self.get_template(), scan_start_angle), color=True)
            snapshots.append(image)
        self.rgbimage = snapshots
        self.goniometer.extract_backlight()
        
    
    def run(self, wait=True):
        logging.info('expected files %s' % self.get_expected_files())
        if self.snapshot == True:
            self.save_snapshots()
        self._start = time.time()
        task_ids = []
        self.md2_task_info = []
        vertical_scan_length = self.get_vertical_scan_length()
        for k, scan_start_angle in enumerate(self.scan_start_angles):
            logging.info('scan_start_angle %s' % scan_start_angle)
            if vertical_scan_length == 0:
                task_id = self.goniometer.omega_scan(scan_start_angle, self.scan_range, self.scan_exposure_time, wait=wait)
            else:
                if self.inverse_direction == True:
                    vertical_scan_length = self.get_vertical_scan_length() * pow(-1, k)
                task_id = self.goniometer.vertical_helical_scan(vertical_scan_length, position, scan_start_angle, self.scan_range, self.scan_exposure_time, wait=wait)
            task_ids.append(task_id)
            self.md2_task_info.append(self.goniometer.get_task_info(task_id))
    
    def get_scan_start_angles(self):
        if os.path.isfile(self.get_parameters_filename()):
            return pickle.load(open(self.get_parameters_filename()))['scan_start_angles']
        else:
            return self.scan_start_angles
            
    def analyze(self):
        logging.info('reference_images analysis expected files %s' % self.get_expected_files())
        command = 'reference_images.py'
        sense_line = '%s -d %s -n %s -A --scan_start_angles "%s" &' % (command, self.directory, self.name_pattern, self.get_scan_start_angles())
        logging.info('analysis line %s' % sense_line)
        os.system(sense_line)
        #subprocess.call(sense_line, shell=True)

    def analyze_online(self):
        logging.info('analyze')
        self.rectify_master()
        try:
            self.generate_summed_h5()
        except:
            pass
        self.generate_cbf()
        self.run_dozor()
        self.run_dials()
        self.run_xds()
        #self.run_best()
        strategy = self.parse_best()
        logging.info('best_strategy')
        logging.info(str(strategy))
        
        return strategy

    
    def rectify_master(self, timeout=15):
        logging.info('rectify_master')
        start = time.time()
        expected_files = self.get_expected_files()
        logging.info('expected files:')
        logging.info(str(expected_files))
        while not self.expected_files_present() and time.time() - start < timeout:
            gevent.sleep(1)
        if not self.expected_files_present():
            logging.debug('expected files not present, exiting rectify_master, please check.')
            return -1
        else:
            logging.info('rectify_master: all files appeared after %.2f seconds' % (time.time()-start))
            
        for f in expected_files:
            shutil.copy2('%s/%s' % (self.directory, f), self.treatment_directory)
          
        m = h5py.File('%s/%s_master.h5' % (self.treatment_directory, self.name_pattern), 'r+')
        
        ntrigger = self.get_ntrigger()
        nimages = self.get_nimages()
        angle_per_frame = self.get_angle_per_frame()
        
        logging.info('ntrigger %s' % str(ntrigger))
        logging.info('nimages %s' % str(nimages))
        logging.info('angle_per_frame %s' % str(angle_per_frame))
        
        omega = []
        omega_end = []
        absolute_start = self.scan_start_angles[0]
        for k in range(ntrigger):
            start = self.scan_start_angles[k]
            end = start + nimages * angle_per_frame
            omega += list(np.arange(start, end, angle_per_frame))
            omega_end += list(np.arange(start+angle_per_frame, end+angle_per_frame, angle_per_frame))
            image_nr_low = int(1 + (start - absolute_start)/angle_per_frame)
            image_nr_high = image_nr_low + nimages - 1
            try:
                m['/entry/data/data_%06d' % (k+1,)].attrs['image_nr_low'] = image_nr_low
                m['/entry/data/data_%06d' % (k+1,)].attrs['image_nr_high'] = image_nr_high
            except:
                gevent.sleep(5)
                m['/entry/data/data_%06d' % (k+1,)].attrs['image_nr_low'] = image_nr_low
                m['/entry/data/data_%06d' % (k+1,)].attrs['image_nr_high'] = image_nr_high
        
        logging.info('omega %s' % str(omega))
        m['/entry/sample/goniometer/omega'].write_direct(np.array(omega))
        m['/entry/sample/goniometer/omega_end'].write_direct(np.array(omega_end))
        
        m.close()    
        
        for f in expected_files:
            shutil.copy2('%s/%s' % (self.treatment_directory, f), self.directory)
            
    def generate_summed_h5(self):
        logging.info('generate_summed_h5')
        if os.path.isfile('{directory}/{name_pattern}_sum10_master.h5'.format(**self.format_dictionary)):
            logging.info('summed images already generated')
            return
        self.format_dictionary['nimages_per_file'] = self.get_nimages_per_file()
        self.format_dictionary['treatment_directory'] = self.treatment_directory
        generate_summed_h5_line = 'cd {treatment_directory}; summer.py -n {nimages_per_file} -m {name_pattern}_master.h5'.format(**self.format_dictionary)
        os.system(generate_summed_h5_line)
        for f in ['data_000001.h5', 'master.h5']:
            a = '%s/%s_%s_%s' % (self.treatment_directory, self.name_pattern, 'sum%d' % (self.get_nimages_per_file()), f)
            logging.info('copying %s to %s ' % (a,  self.directory))
            shutil.copy2('%s/%s_%s_%s' % (self.treatment_directory, self.name_pattern, 'sum%d' % (self.get_nimages_per_file()), f), self.directory)
            os.remove('%s/%s_%s_%s' % (self.treatment_directory, self.name_pattern, 'sum%d' % (self.get_nimages_per_file()), f))
        for f in self.get_expected_files():
            os.remove(os.path.join(self.treatment_directory, f))
            
    def generate_cbf(self):
        logging.info('generate_cbf')
        generate_cbf_line = 'cd {treatment_directory}; /usr/local/bin/H5ToCBF.py -m {directory}/{name_pattern}_master.h5 -d {directory}/process'.format(**self.format_dictionary)
        if os.uname()[1] != 'process1':
            generate_cbf_line = 'ssh process1 "%s"' % generate_cbf_line
        logging.info('generate_cbf_line %s' % generate_cbf_line)
        os.system(generate_cbf_line)
        os.system('touch {directory}'.format(**self.format_dictionary))
        self.create_ordered_cbf_links()
        
    
    def get_transmission(self):
        if self.saved_parameters is not None:
            return self.saved_parameters['transmission']
        elif self.transmission is not None:
            return self.transmission
        else:
            self.transmission_motor.get_transmission()
        
    
    def create_dozor_control_card(self):
        '''
        dozor parameters
        !-----------------
        detector eiger9m
        !
        exposure 0.025
        spot_size 3
        detector_distance 200
        X-ray_wavelength 0.9801
        fraction_polarization 0.99
        pixel_min 0
        pixel_max 100000
        !
        ix_min 1
        ix_max 1067
        iy_min 1029
        iy_max 1108
        !
        orgx 1454.26
        orgy 1731.02
        oscillation_range 0.001
        image_step 1
        starting_angle 75
        !
        first_image_number 1
        number_images 1
        name_template_image mesh1_?????.cbf
        '''
        
        if self.saved_parameters is None:
            parameters = {'detector': 'eiger9m', 
                        'exposure': self.get_exposure_time_per_frame(),
                        'spot_size': 3,
                        'detector_distance': self.get_detector_distance(),
                        'X-ray_wavelength': self.get_wavelength(),
                        'fraction_polarization': 0.99,
                        'pixel_min': 0,
                        'pixel_max': 100000,
                        'ix_min': 1,
                        'ix_max': int(self.get_beam_center_x()) + 50,
                        'iy_min': int(self.get_beam_center_y()) - 50,
                        'iy_max': int(self.get_beam_center_y()) + 50,
                        'orgx': self.get_beam_center_x(),
                        'orgy': self.get_beam_center_y(),
                        'oscillation_range': self.get_angle_per_frame(),
                        'image_step': 1,
                        'starting_angle': self.scan_start_angles[0],
                        'first_image_number': 1,
                        'number_images': self.get_nimages() * self.get_ntrigger(),
                        'name_template_image': '{directory}/process/{name_pattern}_ordered_?????.cbf'.format(**self.format_dictionary)}
        else:
            parameters = {'detector': 'eiger9m', 
                        'exposure': self.saved_parameters['frame_time'],
                        'spot_size': 3,
                        'detector_distance': self.saved_parameters['detector_distance'],
                        'X-ray_wavelength': self.saved_parameters['wavelength'],
                        'fraction_polarization': 0.99,
                        'pixel_min': 0,
                        'pixel_max': 100000,
                        'ix_min': 1,
                        'ix_max': int(self.saved_parameters['beam_center_x']) + 50,
                        'iy_min': int(self.saved_parameters['beam_center_y']) - 50,
                        'iy_max': int(self.saved_parameters['beam_center_y']) + 50,
                        'orgx': int(self.saved_parameters['beam_center_x']),
                        'orgy': int(self.saved_parameters['beam_center_y']),
                        'oscillation_range': self.saved_parameters['angle_per_frame'],
                        'image_step': 1,
                        'starting_angle': self.saved_parameters['scan_start_angles'][0],
                        'first_image_number': 1,
                        'number_images': self.saved_parameters['nimages'] * self.saved_parameters['ntrigger'],
                        'name_template_image': '{directory}/process/{name_pattern}_ordered_?????.cbf'.format(**self.format_dictionary)}
            
        input_file = 'dozor parameters\n'
        input_file += '!-----------------\n'
        for parameter in ['detector', 'exposure', 'spot_size', 'detector_distance', 'X-ray_wavelength', 'fraction_polarization', 'pixel_min', 'pixel_max', 'ix_min', 'ix_max', 'iy_min', 'iy_max', 'orgx', 'orgy', 'oscillation_range', 'image_step', 'starting_angle', 'first_image_number', 'number_images', 'name_template_image']:
            input_file += '%s %s\n' % (parameter, parameters[parameter])
        input_file += 'end\n'
        
        try:
            os.makedirs(self.get_dozor_directory())
        except OSError:
            pass
        f = open(os.path.join(self.get_dozor_directory(), self.get_dozor_control_card_filename()), 'w')
        f.write(input_file)
        f.close()
                      
    
    def get_dozor_directory(self):
        return os.path.join('{directory}/process/dozor_{name_pattern}'.format(**self.format_dictionary))
    
    
    def get_dozor_control_card_filename(self):
        return '%s.dat' % self.name_pattern
    
    
    def create_ordered_cbf_links(self):
        logging.info('create_ordered_cbf_links')
        #create_ordered_cbf_links_line = 'cd {directory};'.format(**{'directory': self.directory}) + 'l=0; for k in sample_7_5_h_*.cbf; do l=$((${l}+1)); echo ${l}; echo ${k::-10}; a=$(printf "%05d" ${l}); ln -s ${k} ${k::-10}_ordered_${a}.cbf; done;'
        #print(create_ordered_cbf_links_line)
        os.chdir('{directory}/process'.format(**self.format_dictionary))
        os.system('touch {directory}/process'.format(**self.format_dictionary))
                  
        cbfs = commands.getoutput('ls %s_?????.cbf' % self.name_pattern).split('\n')
        if 'ls: cannot access' in cbfs:
            logging.info('no cbfs generated please check')
            return
        for k, cbf in enumerate(cbfs):
            try:
                os.symlink(cbf, '%s_ordered_%05d.cbf' % (self.name_pattern, k+1))
            except OSError:
                pass
    
    
    def run_dozor(self):
        logging.info('run_dozor')
        if os.path.isfile('{directory}/process/dozor_{name_pattern}/dozor_average.dat'.format(**self.format_dictionary)):
            return
        self.create_dozor_control_card()
        dozor_line = 'cd {directory}; dozor {control_card}&'.format(**{'directory': self.get_dozor_directory(), 'control_card': self.get_dozor_control_card_filename()})
        if os.uname()[1] != 'process1':
            dozor_line = 'ssh process1 "%s"&' % dozor_line
        logging.info('dozor_line %s' % dozor_line)
        os.system(dozor_line)
            
    
    def run_xds(self):
        logging.info('run_xds')
        if os.path.isfile('{directory}/process/xdsme_auto_{name_pattern}/CORRECT.LP'.format(**self.format_dictionary)):
            xds_line = ''
        else:
            xds_line = 'cd {directory}/process; ref_xdsme -p auto_{name_pattern} {name_pattern}_?????.cbf'.format(**self.format_dictionary)
        #if os.uname()[1] != 'process1':
        #    xds_line = "ssh process1 '%s'" % xds_line
        logging.info('xds_line %s' % xds_line)
        
        best_log_file = '{directory}/process/{name_pattern}_best.log'.format(**self.format_dictionary)
        if os.path.isfile(best_log_file) and os.stat(best_log_file).st_size > 200:
            return
        
        os.environ['besthome'] = '/home/experiences/proxima2a/com-proxima2a/Documents/Best'
        best_line = 'echo besthome $besthome; export besthome=/home/experiences/proxima2a/com-proxima2a/Documents/Best; /usr/local/bin/best -f eiger9m -t {exposure_time} -e none -i2s 1. -M 0.005 -S 120 -Trans {transmission} -w 0.001 -GpS {dose_rate} -dna {directory}/process/{name_pattern}_best_strategy.xml -xds {directory}/process/xdsme_auto_{name_pattern}/CORRECT.LP {directory}/process/xdsme_auto_{name_pattern}/BKGINIT.cbf {directory}/process/xdsme_auto_{name_pattern}/XDS_ASCII.HKL | tee {directory}/process/{name_pattern}_best.log '.format(**{'directory': self.directory, 'name_pattern': self.name_pattern, 'exposure_time': self.get_exposure_time_per_frame(), 'dose_rate': self.get_dose_rate(), 'dose_limit': self.get_dose_limit(), 'transmission': self.get_transmission()})
        logging.info('best_line %s' % best_line)
        if xds_line != '':
            total_line = '%s && %s' % (xds_line, best_line)
        else:
            total_line = best_line
        logging.info('total_line %s' % total_line)
        subprocess.call(total_line, shell=True)
        
    
    def run_best(self, sleeptime=1., timeout=120.):
        logging.info('run_best')
        best_log_file = '{directory}/process/{name_pattern}_best.log'.format(**self.format_dictionary)
        if os.path.isfile(best_log_file) and os.stat(best_log_file).st_size > 200:
            return
        best_line = 'best -f eiger9m -t {exposure_time} -e none -M 0.005 -S 120 -Trans {transmission} -w 0.001 -GpS {dose_rate} -DMAX {dose_limit} -dna {directory}/process/{name_pattern}_best_strategy.xml -xds {directory}/process/xdsme_auto_{name_pattern}/CORRECT.LP {directory}/process/xdsme_auto_{name_pattern}/BKGINIT.cbf {directory}/process/xdsme_auto_{name_pattern}/XDS_ASCII.HKL | tee {directory}/process/{name_pattern}_best.log '.format(**{'directory': self.directory, 'name_pattern': self.name_pattern, 'exposure_time': self.get_exposure_time_per_frame(), 'dose_rate': self.get_dose_rate(), 'dose_limit': self.get_dose_limit(), 'transmission': self.get_transmission()})
        
        correct_file = '{directory}/process/xdsme_auto_{name_pattern}/CORRECT.LP'.format(**{'directory': self.directory, 'name_pattern': self.name_pattern})
        xds_ascii_file = '{directory}/process/xdsme_auto_{name_pattern}/XDS_ASCII.HKL'.format(**{'directory': self.directory, 'name_pattern': self.name_pattern})
        bkginit_file = '{directory}/process/xdsme_auto_{name_pattern}/BKGINIT.cbf'.format(**{'directory': self.directory, 'name_pattern': self.name_pattern})
        
        start = time.time()
        while (not os.path.isfile(correct_file) or not os.path.isfile(xds_ascii_file) or not os.path.isfile(bkginit_file)) and time.time() - start < timeout:
            os.system('touch {directory}/process/xdsme_auto_{name_pattern}'.format(**{'directory': self.directory, 'name_pattern': self.name_pattern}))
            gevent.sleep(sleeptime)
        
        for f in (correct_file, xds_ascii_file, bkginit_file):
            if os.path.isfile(f):
                logging.info('file is created %s' % f)
            else:
                logging.info('file not created %s' % f)
            
        if os.uname()[1] != 'proxima2a-10':
            best_line = 'ssh proxima2a-10 "%s"' % best_line
        logging.info('best_line %s' % best_line)
        xds_dir_content = commands.getoutput('ls {directory}/process/xdsme_auto_{name_pattern}'.format(**{'directory': self.directory, 'name_pattern': self.name_pattern}))
        logging.info('xds_dir_content')
        logging.info(str(xds_dir_content))
        os.system('touch {directory}/process/xdsme_auto_{name_pattern}'.format(**{'directory': self.directory, 'name_pattern': self.name_pattern}))
        xds_dir_content2 = commands.getoutput('ls {directory}/process/xdsme_auto_{name_pattern}'.format(**{'directory': self.directory, 'name_pattern': self.name_pattern}))
        os.system('touch {directory}/process'.format(**{'directory': self.directory, 'name_pattern': self.name_pattern}))
        xds_dir_content2 = commands.getoutput('ls {directory}/process/xdsme_auto_{name_pattern}'.format(**{'directory': self.directory, 'name_pattern': self.name_pattern}))
        logging.info('xds_dir_content2')
        logging.info(str(xds_dir_content2))
        os.system('echo %s' % best_line)
        
        os.system(best_line)

    
    def parse_best(self):
        l = open('{directory}/process/{name_pattern}_best.log'.format(**{'directory': self.directory, 'name_pattern': self.name_pattern})).read()
        print('BEST strategy')
        print l
            
        '''                         Main Wedge  
                                 ================ 
        Resolution limit is set according to the given max.time               
        Resolution limit =2.48 Angstrom   Transmission =   10.0%  Distance = 275.5mm
        -----------------------------------------------------------------------------------------
                   WEDGE PARAMETERS       ||                 INFORMATION
        ----------------------------------||-----------------------------------------------------
        sub-| Phi  |Rot.  | Exposure| N.of||Over|sWedge|Exposure|Exposure| Dose  | Dose  |Comple-
         We-|start |width | /image  | ima-||-lap| width| /sWedge| total  |/sWedge| total |teness
         dge|degree|degree|     s   |  ges||    |degree|   s    |   s    | MGy   |  MGy  |  %    
        ----------------------------------||-----------------------------------------------------
         1    74.00   0.15     0.015   954|| No  143.10     14.2     14.2   3.540   3.540  100.0
        -----------------------------------------------------------------------------------------
        '''
        try:
            resolution = float(re.findall('Resolution limit =([\d\.]*) Angstrom', l)[0])
            transmission = float(re.findall('Transmission[\s=]*([\d\.]*)%', l)[0])
            distance = float(re.findall('Distance[\s=]*([\d\.]*)mm', l)[0])
            
            subwedge = ' (\d)\s*'
            start = '([\d\.]*)\s*'
            width = '([\d\.]*)\s*'
            exposure = '([\d\.]*)\s*'
            nimages = '([\d]*)\|\|'
            search = subwedge + start + width + exposure + nimages

            wedges = re.findall(search, l)
        except IndexError:
            resolution = None
            transmission = None
            distance = None
            wedges = None
            self.server.log_message('BEST analysis did not succeed')
            return
        '''
        [('1', '74.00', '0.15', '0.063', '767'),
        ('2', '189.05', '0.15', '0.161', '187')]
        '''
        strategy_text = ''
        ls = l.split('\n')
        flag = False
        for line in ls:
            if 'Main Wedge' in line:
                flag = True
            if 'Phi_start' in line:
                flag = False
                break
            if flag == True:
                strategy_text += '%s\n' % line
            
        strategy = []
        for wedge in wedges:
            wedge_parameters = {}
            wedge_parameters['resolution'] = float(resolution)
            wedge_parameters['transmission'] = float(transmission)
            wedge_parameters['distance'] = float(distance)
            wedge_parameters['order'] = int(wedge[0])
            wedge_parameters['scan_start_angle'] = float(wedge[1])
            wedge_parameters['angle_per_frame'] = float(wedge[2])
            wedge_parameters['exposure_per_frame'] = float(wedge[3])
            wedge_parameters['nimages'] = int(wedge[4])
            wedge_parameters['scan_exposure_time'] = wedge_parameters['nimages'] * wedge_parameters['exposure_per_frame']
            strategy.append(wedge_parameters)
        
        self.server.log_message('BEST recomends the following parameters:')
        for wedge_parameters in strategy:
            for key in ['scan_start_angle', 'resolution', 'transmission', 'angle_per_frame', 'exposure_per_frame', 'nimages']:
                self.server.log_message('%s: %s' % (key, wedge_parameters[key]))
        
        #self.server.log_message('\n%s\n' % strategy_text)
        
        #logging.getLogger('user_level_log').info('Best strategy %s' % strategy)
        #logging.getLogger('user_level_log').info(strategy_text)
        
        if resolution > self.get_resolution():
            logging.getLogger('user_level_log').warning('Best results indicate the current sample diffracts beyond currently set resolution, please consider approaching detector or increasing photon energy to measure diffraction to higher resolution.')
        return strategy
    
    
    def expected_files_present(self):
        expected_files = self.get_expected_files()
        for f in expected_files:
            if f not in os.listdir(self.directory):
                return False
        return True
        
    
    def run_dials(self):
        logging.info('run_dials')
        
        if os.path.isfile('{directory}/process/dials_{name_pattern}/dials.find_spots.log'.format(**self.format_dictionary)):
            return
        spot_find_line = 'source /usr/local/dials/dials_env.sh; mkdir -p {directory}/process/dials_{name_pattern}; cd {directory}/process/dials_{name_pattern}; touch {directory}; echo $(pwd); dials.find_spots shoebox=False per_image_statistics=True spotfinder.filter.ice_rings.filter=True nproc=80 ../../{name_pattern}_master.h5 &'.format(**self.format_dictionary)
        
        if os.uname()[1] != 'process1':
            spot_find_line = 'ssh process1 "%s"&' % spot_find_line
        logging.info('spot_find_line %s' % spot_find_line)
        os.system(spot_find_line)
    
    
    def parse_find_spots(self):
        def get_nspots_nimage(a):
            results = {}
            for line in a:
                try:
                    nimage, nspots, nspots_no_ice, total_intensity = map(int, re.findall('\| (\d*)\s*\| (\d*)\s*\| (\d*)\s*\| (\d*)\s*\|', line)[0])
                    #nspots, nimage = map(int, re.findall('Found (\d*) strong pixels on image (\d*)', line)[0])
                    results[nimage] = {}
                    results[nimage]['dials_spots'] = nspots_no_ice
                    results[nimage]['dials_all_spots'] = nspots
                    results[nimage]['dials_total_intensity'] = total_intensity
                except:
                    logging.info(traceback.format_exc())
            return results
        
        search_line = "grep '|' {directory}/process/dials_{name_pattern}/dials.find_spots.log".format(**{'directory': self.directory, 'name_pattern': self.name_pattern})
        a = commands.get_output(search_line).split('\n')
        results = get_nspots_nimage(a)
    
        return results
    
            
def main():
    import optparse
        
    parser = optparse.OptionParser()
    parser.add_option('-n', '--name_pattern', default='ref-test_$id', type=str, help='Prefix default=%default')
    parser.add_option('-d', '--directory', default='/nfs/data/default', type=str, help='Destination directory default=%default')
    parser.add_option('-r', '--scan_range', default=1., type=float, help='Scan range [deg]')
    parser.add_option('-e', '--scan_exposure_time', default=0.25, type=float, help='Scan exposure time [s]')
    parser.add_option('-s', '--scan_start_angles', default='[0, 90, 180, 270]', type=str, help='Scan start angles [deg]')
    parser.add_option('-a', '--angle_per_frame', default=0.1, type=float, help='Angle per frame [deg]')
    parser.add_option('-f', '--image_nr_start', default=1, type=int, help='Start image number [int]')
    parser.add_option('-v', '--vertical_scan_length', default=0, type=float, help='Vertical scan length [mm]')
    parser.add_option('-V', '--vertical_step_size', default=0.025, type=float, help='Vertical steps size [mm]')
    #parser.add_option('-I', '--inverse_dierction', action='store_true', help='If set will invese direction of subsequent vertical scans')
    parser.add_option('-R', '--dose_rate', default=0.25e6, type=float, help='Dose rate in Grays per second (default=%default)')
    parser.add_option('-L', '--dose_limit', default=15e6, type=float, help='Dose limit in Grays (default=%default)')
    parser.add_option('-i', '--position', default=None, type=str, help='Gonio alignment position [dict]')
    parser.add_option('-p', '--photon_energy', default=None, type=float, help='Photon energy ')
    parser.add_option('-t', '--detector_distance', default=None, type=float, help='Detector distance')
    parser.add_option('-o', '--resolution', default=None, type=float, help='Resolution [Angstroem]')
    parser.add_option('-x', '--flux', default=None, type=float, help='Flux [ph/s]')
    parser.add_option('-m', '--transmission', default=None, type=float, help='Transmission. Number in range between 0 and 1.')
    parser.add_option('-T', '--snapshot', action='store_true', help='If set will record snapshots.')
    parser.add_option('-A', '--analysis', action='store_true', help='If set will perform automatic analysis.')
    parser.add_option('-D', '--diagnostic', action='store_true', help='If set will record diagnostic information.')
    parser.add_option('-S', '--simulation', action='store_true', help='If set will record diagnostic information.')
    
    options, args = parser.parse_args()
    
    print 'options', options
    print 'args', args
    
    ri = reference_images(**vars(options))
    
    filename = '%s_parameters.pickle' % ri.get_template()
    
    if not os.path.isfile(filename):
        ri.execute()
    elif options.analysis == True:
        ri.analyze_online()
            
    
if __name__ == '__main__':
    main()
