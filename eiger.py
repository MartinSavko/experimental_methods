#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import time
import os
import numpy
import re
import logging
import traceback
import urllib2
from eigerclient import DEigerClient

'''
Author: Martin Savko
Contact: savko@synchrotron-soleil.fr
Date: 2018-10-16

eiger class implements high level interface to SIMPLON API of the EIGER detectors. 
It inherits from DIgerClient class developed by Dectris and provides explicit set and get methods for every writable attribute of the API and the get method for all readonly attributes. All of the API commands are directly available as class methods as well.

Examples:
e = eiger(ip="172.19.10.26", port=80)
e.initialize()
e.set_photon_energy(12650)

To collect 100 images in 'ints' trigger mode with frame time of 0.5 s, with 25 images per data file,
at 12.65keV and bslz4 compression:

frame_time = 0.5
if e.get_trigger_mode() != 'ints':
    e.set_trigger_mode('ints')
e.set_frame_time(frame_time)
e.set_count_time(frame_time - e.get_detector_readout_time())
if e.get_photon_energy() != 12650.:
    e.set_photon_energy(12650)
e.set_nimages(100)
e.set_nimages_per_file(25)
e.set_name_pattern('test_1')
e.set_compression_enabled(True)
if e.get_compression() != 'bslz4':
    e.set_compression('bslz4')
e.set_auto_summation(True)
e.set_flatfield_correction_applied(True)
e.set_countrate_correction_applied(True)
e.set_pixel_mask_applied(True)
e.set_virtual_pixel_mask_applied(True)

e.arm()
e.trigger()
e.disarm()

e.download('./') #download data to the current directory and remove them from the detector control unit (DCU) cache
e.remove_files('test_1') #remove data from the DCU cache

The class also implements helper methods to succintly represent the configuration and state of various components of the detector

e.print_detector_config()
e.print_filewriter_config()
e.print_monitor_config()
e.print_stream_config()
e.print_monitor_status()
e.print_detector_status()

'''

class eiger(DEigerClient):
    
    def __init__(self, host='172.19.10.26', port=80):
        DEigerClient.__init__(self, host=host, port=port)
        
    # detector configuration
    def set_photon_energy(self, photon_energy):
        if self.get_photon_energy() == photon_energy:
            return
        self.photon_energy = photon_energy
        if photon_energy < self.detectorConfig('photon_energy')['min']:
            print 'photon_energy: value below allowed minimal value'
            return
        return self.setDetectorConfig("photon_energy", photon_energy)
    
    def get_photon_energy(self):
        return self.detectorConfig("photon_energy")['value']
        
    def set_threshold_energy(self, threshold_energy):
        self.threshold_energy = threshold_energy
        if self.get_threshold_energy() == threshold_energy:
            return
        return self.setDetectorConfig("threshold_energy", threshold_energy)
    
    def get_threshold_energy(self):
        return self.detectorConfig("threshold_energy")['value']
        
    def set_data_collection_date(self, data_collection_date=None):
        if data_collection_date is None:
            return self.setDetectorConfig('data_collection_date', self.detectorStatus('time')['value'])
        else:
            return self.setDetectorConfig("data_collection_date", data_collection_date)
        
    def get_data_collection_date(self):
        return self.detectorConfig('data_collection_date')['value']
        
    def set_beam_center_x(self, beam_center_x):
        self.beam_center_x = beam_center_x
        return self.setDetectorConfig("beam_center_x", beam_center_x)
    
    def get_beam_center_x(self):
        return self.detectorConfig("beam_center_x")['value']
        
    def set_beam_center_y(self, beam_center_y):
        self.beam_center_y = beam_center_y
        return self.setDetectorConfig("beam_center_y", beam_center_y)
    
    def get_beam_center_y(self):
        return self.detectorConfig("beam_center_y")['value']
        
    def set_detector_distance(self, detector_distance):
        self.detector_distance = detector_distance
        return self.setDetectorConfig("detector_distance", detector_distance)
    
    def get_detector_distance(self):
        return self.detectorConfig("detector_distance")['value']
    
    def set_detector_translation(self, detector_translation):
        '''detector_translation is list of real values'''
        return self.setDetectorConfig("detector_translation", detector_translation)
        
    def get_detector_translation(self):
        return self.detectorConfig("detector_translation")['value']
        
    def set_frame_time(self, frame_time):
        if frame_time < self.detectorConfig('frame_time')['min']:
            print 'frame_time: requested value below allowed minimal value'
            return
        self.frame_time = frame_time
        return self.setDetectorConfig("frame_time", frame_time)
    
    def get_frame_time(self):
        return self.detectorConfig("frame_time")['value']
        
    def set_count_time(self, count_time):
        if count_time < self.detectorConfig('count_time')['min']:
            print 'count_time: requested value below allowed minimal value'
            return
        self.count_time = count_time
        return self.setDetectorConfig("count_time", count_time)
    
    def get_count_time(self):
        return self.detectorConfig("count_time")['value']
        
    def set_nimages(self, nimages):
        self.nimages = nimages
        return self.setDetectorConfig("nimages", nimages)
    
    def get_nimages(self):
        return self.detectorConfig("nimages")['value']
        
    def set_ntrigger(self, ntrigger):
        self.ntrigger = ntrigger
        return self.setDetectorConfig("ntrigger", ntrigger)
    
    def get_ntrigger(self):
        return self.detectorConfig("ntrigger")['value']
        
    def set_wavelength(self, wavelength):
        self.wavelength = wavelength
        if self.get_wavelength() == wavelength:
            return
        return self.setDetectorConfig("wavelength", wavelength)
    
    def get_wavelength(self):
        return self.detectorConfig("wavelength")['value']
        
    def set_summation_nimages(self, summation_nimages):
        self.summation_nimages = summation_nimages
        return self.setDetectorConfig("summation_nimages", summation_nimages)
    
    def get_summation_nimages(self, summation_nimages):
        return self.detectorConfig("summation_nimages")['value']
        
    def set_nframes_sum(self, nframes_sum):
        self.nframes_sum = nframes_sum
        return self.setDetectorConfig("nframes_sum", nframes_sum)
        
    def get_nframes_sum(self):
        return self.detectorConfig("nframes_sum")['value']    
        
    def set_element(self, element):
        self.element = element
        return self.setDetectorConfig("element", element)
    
    def get_element(self, element):
        return self.detectorConfig("element")['value']
        
    def set_trigger_mode(self, trigger_mode="exts"):
        '''one of four values possible
           ints
           inte
           exts
           exte
        '''
        self.trigger_mode = trigger_mode
        return self.setDetectorConfig("trigger_mode", trigger_mode)
    
    def get_trigger_mode(self):
        return self.detectorConfig("trigger_mode")['value']
        
    def set_omega(self, omega):
        self.omega = omega
        return self.setDetectorConfig('omega_start', omega)
    def get_omega(self):
        return self.detectorConfig('omega_start')['value']
    
    def set_omega_increment(self, omega_increment):
        self.omega_increment = omega_increment
        return self.setDetectorConfig('omega_increment', omega_increment)
    def get_omega_increment(self):
        return self.detectorConfig('omega_increment')['value']
        
    def set_omega_range_average(self, omega_increment):
        self.omega_increment = omega_increment
        return self.setDetectorConfig('omega_increment', omega_increment)
    def get_omega_range_average(self):
        return self.detectorConfig('omega_increment')['value']
        
    def set_phi(self, phi):
        self.phi = phi
        return self.setDetectorConfig('phi_start', phi)
    def get_phi(self):
        return self.detectorConfig('phi_start')['value']
        
    def set_phi_increment(self, phi_increment):
        self.phi_increment = phi_increment
        return self.setDetectorConfig('phi_increment', phi_increment)
    def get_phi_increment(self):
        return self.detectorConfig('phi_increment')['value']
    
    def set_phi_range_average(self, phi_increment):
        self.phi_increment = phi_increment
        return self.setDetectorConfig('phi_increment', phi_increment)
    def get_phi_range_average(self):
        return self.detectorConfig('phi_increment')['value']
    
    def set_chi(self, chi):
        self.chi = chi   
        return self.setDetectorConfig('chi_start', chi)
    def get_chi(self):
        return self.detectorConfig('chi_start')['value']
        
    def set_chi_increment(self, chi_increment):
        self.chi_increment = chi_increment
        return self.setDetectorConfig('chi_increment', chi_increment)
    def get_chi_increment(self):
        return self.detectorConfig('chi_increment')['value']
    
    def set_chi_range_average(self, chi_increment):
        self.chi_increment = chi_increment
        return self.setDetectorConfig('chi_increment', chi_increment)
    def get_chi_range_average(self):
        return self.detectorConfig('chi_increment')['value']
    
    def set_kappa(self, kappa):
        self.kappa = kappa
        return self.setDetectorConfig('kappa_start', kappa)
    def get_kappa(self):
        return self.detectorConfig('kappa_start')['value']
    
    def set_kappa_increment(self, kappa_increment):
        self.kappa_increment = kappa_increment
        return self.setDetectorConfig('kappa_increment', kappa_increment)
    def get_kappa_increment(self):
        return self.detectorConfig('kappa_increment')['value']
    
    def set_kappa_range_average(self, kappa_increment):
        self.kappa_increment = kappa_increment
        return self.setDetectorConfig('kappa_increment', kappa_increment)
    def get_kappa_range_average(self):
        return self.detectorConfig('kappa_increment')['value']
    
    def set_two_theta(self, two_theta):
        self.two_theta = two_theta
        return self.setDetectorConfig('two_theta_start', two_theta)
    def get_two_theta(self):
        return self.detectorConfig('two_theta_start')['value']
        
    def set_two_theta_range_average(self, two_theta_increment):
        self.two_theta_increment = two_theta_increment
        return self.setDetectorConfig('two_theta_increment', two_theta_increment)

    def get_two_theta_range_average(self):
        return self.detectorConfig('two_theta_increment')['value']
    
    def set_two_theta_increment(self, two_theta_increment):
        self.two_theta_increment = two_theta_increment
        return self.setDetectorConfig('two_theta_increment', two_theta_increment)

    def get_two_theta_increment(self):
        return self.detectorConfig('two_theta_increment')['value']
    
    def get_pixel_mask(self):
        return self.detectorConfig('pixel_mask')
        
    # Apparently writing to bit_depth_image is forbidden 2015-10-25 MS
    def set_bit_depth_image(self, bit_depth_image=16):
        return self.setDetectorConfig('bit_depth_image', bit_depth_image)

    def get_bit_depth_image(self):
        return self.detectorConfig('bit_depth_image')['value']
        
    def set_detector_readout_time(self, detector_readout_time):
        return self.setDetectorConfig("detector_readout_time", detector_readout_time)
    
    def get_detector_readout_time(self):
        return self.detectorConfig("detector_readout_time")['value']
        
    # booleans
    def set_auto_summation(self, auto_summation=True):
        return self.setDetectorConfig("auto_summation", auto_summation)
    
    def get_auto_summation(self):
        return self.detectorConfig("auto_summation")['value']
        
    def set_countrate_correction_applied(self, countrate_correction_applied=True):
        self.countrate_correction_applied = count_rate_correction_applied
        return self.setDetectorConfig('countrate_correction_applied', countrate_correction_applied)
    
    def get_countrate_correction_applied(self):
        return self.detectorConfig('countrate_correction_applied')['value']
        
    def set_pixel_mask_applied(self, pixel_mask_applied=True):
        self.pixel_mask_applied = pixel_mask_applied
        return self.setDetectorConfig('pixel_mask_applied', pixel_mask_applied)
         
    def get_pixel_mask_applied(self):
        return self.detectorConfig('pixel_mask_applied')['value']
        
    def set_test_mode(self, test_mode=False):
        self.test_mode = test_mode
        return self.setDetectorConfig('test_mode', test_mode)
    
    def get_test_mode(self):
        return self.detectorConfig('test_mode')['value']
    
    def set_flatfield_correction_applied(self, flatfield_correction_applied=True):
        self.flatfield_correction_applied = flatfield_correction_applied
        return self.setDetectorConfig('flatfield_correction_applied', flatfield_correction_applied)
    
    def get_flatfield_correction_applied(self):
        return self.detectorConfig('flatfield_correction_applied')['value']
        
    def set_virtual_pixel_correction_applied(self, virtual_pixel_correction_applied=True):
        self.virtual_pixel_correction_applied = virtual_pixel_correction_applied
        return self.setDetectorConfig('virtual_pixel_correction_applied', virtual_pixel_correction_applied)
    
    def get_virtual_pixel_correction_applied(self):
        return self.detectorConfig('virtual_pixel_correction_applied')['value']
        
    def set_efficiency_correction_applied(self, efficiency_correction_applied=True):
        self.efficiency_correction_applied = efficiency_correction_applied
        return self.setDetectorConfig('efficiency_correction_applied', efficiency_correction_applied)
    
    def get_efficiency_correction_applied(self):
        return self.detectorConfig('efficiency_correction_applied')['value']
        
    def set_compression(self, compression='lz4'):
        self.compression = compression
        return self.setDetectorConfig('compression', compression)
        
    def get_compression(self):
        return self.detectorConfig('compression')['value']
        
    def get_roi_mode(self):
        return self.detectorConfig('roi_mode')['value']
        
    def set_roi_mode(self, roi_mode='4M'):
        return self.setDetectorConfig('roi_mode', roi_mode)
        
    def get_x_pixels_in_detector(self):
        return self.detectorConfig('x_pixels_in_detector')
        
    def get_y_pixels_in_detector(self):
        return self.detectorConfig('y_pixels_in_detector')    
        
    # filewriter
    def set_name_pattern(self, name_pattern):
        self.name_pattern = name_pattern
        return self.setFileWriterConfig("name_pattern", name_pattern)
    
    def get_name_pattern(self):
        return self.fileWriterConfig("name_pattern")['value']
        
    def set_nimages_per_file(self, nimages_per_file):
        self.nimages_per_file = nimages_per_file
        return self.setFileWriterConfig("nimages_per_file", nimages_per_file)
    
    def get_nimages_per_file(self):
        return self.fileWriterConfig("nimages_per_file")['value']
    
    def set_image_nr_start(self, image_nr_start):
        self.image_nr_start = image_nr_start
        return self.setFileWriterConfig("image_nr_start", image_nr_start)
    
    def get_image_nr_start(self):
        return self.fileWriterConfig("image_nr_start")['value']
    
    def set_compression_enabled(self, compression_enabled=True):
        self.compression_enabled = compression_enabled
        return self.setFileWriterConfig("compression_enabled", compression_enabled)
    
    def get_compression_enabled(self):
        return self.fileWriterConfig("compression_enabled")['value']
    
    def get_filenames(self, name_pattern=None):
        return self.fileWriterFiles(filename=name_pattern, method='GET')
    
    def get_data_page(self, subpage=''):
        url = 'http://{0}:{1}/{2}data/{3}'.format(self._host, self._port, self._urlPrefix, subpage)
        data_page = urllib2.urlopen(url).read()
        return data_page
        
    def get_file_size_and_name(self, name_pattern=None):
        subpages = []
        filenames = self.get_filenames()
        for fname in filenames:
            dname = os.path.dirname(fname)
            if len(dname) > 1:
                destination = '%s/' % os.path.dirname(fname)
            else:
                destination = ''
            if destination not in subpages:
                subpages.append(destination)
        file_size_and_name = {}
        for subpage in subpages:
            data_page = self.get_data_page(subpage=subpage)
            search_pattern = u'<tr><td class="n"><a href="(.*)">(.*)</a></td><td class="m">.*</td><td class="s">([\d\.KMG]*)</td><td class="t">application/octet-stream</td></tr>'
        
            rfilename_filename_size = re.findall(search_pattern, data_page)
        
            for rfilename, filename, size in rfilename_filename_size:
                file_size_and_name[u'%s%s' % (subpage, filename)] = {'rfilename': u'%s' % rfilename, 'size': u'%s' % size}
        
        for fname in filenames:
            if fname not in file_size_and_name:
                print 'Possible problem: %s not found on the data_page, please check' % fname
                
        return file_size_and_name
        
    def list_files(self, name_pattern=None):
        return self.fileWriterFiles(filename=name_pattern, method='GET')
    
    def remove_files(self, name_pattern=None):
        return self.fileWriterFiles(filename=name_pattern, method='DELETE')
    
    def save_files(self, filename, destination, regex=False):
        destination_dirname = os.path.dirname(os.path.join(destination, filename))
        if not os.path.isdir(destination_dirname):
            os.makedirs(destination_dirname)
        return self.fileWriterSave(filename, destination, regex=regex)
        
    def get_filewriter_config(self):
        return self.fileWriterConfig()
    
    def get_free_space(self):
        return self.fileWriterStatus('buffer_free')['value']/1024./1024
    
    def get_buffer_free(self):
        return self.fileWriterStatus('buffer_free')['value']
        
    def get_filewriter_state(self):
        return self.fileWriterStatus('state')['value']
        
    def get_filewriter_error(self):
        return self.fileWriterStatus('error')['value']
        
    # detector status
    def get_detector_state(self):
        return self.detectorStatus('state')['value']
        
    def get_detector_error(self):
        return self.detectorStatus('error')['value']
        
    def get_detector_status(self):
        return self.detectorStatus()
    
    def get_humidity(self):
        return self.detectorStatus('board_000/th0_humidity')['value']
        
    def get_temperature(self):
        return self.detectorStatus('board_000/th0_temp')['value']
        
    # detector commands
    def arm(self):
        return self.sendDetectorCommand(u'arm')
    
    def wait(self):
        return self.sendDetectorCommand(u'wait')
        
    def trigger(self, count_time=None):
        if count_time == None:
            return self.sendDetectorCommand(u'trigger')
        else:
            return self.sendDetectorCommand(u'trigger', count_time)
        
    def disarm(self):
        return self.sendDetectorCommand(u'disarm')
        
    def cancel(self):
        return self.sendDetectorCommand(u'cancel')
        
    def abort(self):
        return self.sendDetectorCommand(u'abort')
        
    def initialize(self):
        return self.sendDetectorCommand(u'initialize')
    
    def status_update(self):
        return self.sendDetectorCommand(u'status_update')
    
    # filewriter commands
    def filewriter_clear(self):
        return self.clear_filewriter()
        
    def clear_filewriter(self):
        return self.sendFileWriterCommand(u'clear')
    
    def filewriter_initialize(self):
        return self.initialize_filewriter()
        
    def initialize_filewriter(self):
        return self.sendFileWriterCommand(u'initialize')
    
    def set_filewriter_enabled(self):
        return self.setFileWriterConfig(u'mode', u'enabled')
    
    def set_filewriter_disabled(self):
        return self.setFileWriterConfig(u'mode', u'disabled')
        
    def enable_filewriter(self):
        return self.set_filewriter_enabled()
        
    def filewriter_enable(self):
        return self.set_filewriter_enabled()
        
    def disable_filewriter(self):
        return self.set_filewriter_disabled()
    
    def filewriter_disable(self):
        return self.set_filewriter_disabled()
        
    # monitor commands
    def enable_monitor(self):
        return self.set_monitor_enabled()
    def disable_monitor(self):
        return self.set_monitor_disabled()
        
    def monitor_disable(self):
        return self.set_monitor_disabled()
        
    def monitor_enable(self):
        return self.set_monitor_enabled()
        
    def set_monitor_enabled(self):
        return self.setMonitorConfig(u'mode', u'enabled')
        
    def set_monitor_disabled(self):
        return self.setMonitorConfig(u'mode', u'disabled')
        
    def monitor_clear(self):
        return self.clear_monitor()
        
    def clear_monitor(self):
        return self.sendMonitorCommand(u'clear')
    
    def monitor_initialize(self):
        return self.initialize_monitor()
        
    def initialize_monitor(self):
        return self.sendMonitorCommand(u'initialize')
    
    def get_buffer_size(self):
        return self.monitorConfig('buffer_size')['value']
        
    def set_buffer_size(self, buffer_size):
        return self.setMonitorConfig('buffer_size', buffer_size)
        
    def get_monitor_state(self):
        return self.monitorStatus('state')['value']
        
    def get_monitor_error(self):
        return self.monitorStatus('error')['value']
        
    def get_buffer_fill_level(self):
        return self.monitorStatus('buffer_fill_level')['value']
        
    def get_monitor_dropped(self):
        return self.monitorStatus('dropped')['value']
    
    def get_monitor_image_number(self):
        return self.monitorStatus('monitor_image_number')['value']
    
    def get_next_image_number(self):
        return self.monitorStatus('next_image_number')['value']
    
    # stream commands 
    def set_stream_enabled(self):
        return self.setStreamConfig(u'mode', u'enabled')
    def set_stream_disabled(self):
        return self.setStreamConfig(u'mode', u'disabled')
    def enable_stream(self):
        return self.set_stream_enabled()
    def disable_stream(self):
        return self.set_stream_disabled()
    def stream_enable(self):
        return self.set_stream_enabled()
    def stream_disable(self):
        return self.set_stream_disabled()
        
    def get_stream_state(self):
        return self.streamStatus('state')['value']
        
    def get_stream_error(self):
        return self.streamStatus('state')['value']
        
    def get_stream_dropped(self):
        return self.streamStatus('dropped')['value']
    
    # system commmand
    def restart(self):
        return self.sendSystemCommand(u'restart')
    
    
    # useful helper methods
    def print_detector_config(self):
        for parameter in self.detectorConfig(param='keys'):
            if parameter in ['flatfield', 'pixel_mask']: # PARAMETERS:
                print '%s = %s' % (parameter.ljust(35), 'skipping ...')
            elif parameter in ['two_theta_start', 'two_theta_end', 'omega_start', 'omega_end', 'kappa_start', 'kappa_end', 'phi_start', 'phi_end', 'chi_start', 'chi_end']:
                try:
                    a = numpy.array(self.detectorConfig(parameter)['value'])
                    if len(a) < 6:
                        print '%s = %s' % (parameter.ljust(35), a) 
                    else:
                        st = str(a[:3])[:-1]
                        en = str(a[-3:])[1:]
                        print '%s = %s ..., %s (showing first and last 3 values)' % (parameter.ljust(35), st, en) 
                except:
                    print '%s = %s' % (parameter.ljust(35), "Unknown")
            else:
                try:
                    print '%s = %s' % (parameter.ljust(35), self.detectorConfig(parameter)['value']) 
                except:
                    print '%s = %s' % (parameter.ljust(35), "Unknown")
    
    def print_filewriter_config(self):
        for parameter in self.fileWriterConfig():
            try:
                print '%s = %s' % (parameter.ljust(35), self.fileWriterConfig(parameter)['value'])
            except:
                print '%s = %s' % (parameter.ljust(35), "Unknown")

    def print_monitor_config(self):
        for parameter in self.monitorConfig():
            try:
                print '%s = %s' % (parameter.ljust(35), self.monitorConfig(parameter)['value'])
            except:
                print '%s = %s' % (parameter.ljust(35), "Unknown")
                
    def print_stream_config(self):
        for parameter in self.streamConfig():
            try:
                print '%s = %s' % (parameter.ljust(35), self.streamConfig(parameter)['value'])
            except:
                print '%s = %s' % (parameter.ljust(35), "Unknown")
        
    def print_detector_status(self):
        for parameter in self.detectorStatus():
            try:
                print '%s = %s' % (parameter.ljust(35), self.detectorStatus(parameter)['value']) 
            except:
                print '%s = %s' % (parameter.ljust(35), "Unknown")
    
    def print_filewriter_status(self):
        for parameter in self.fileWriterStatus():
            try:
                print '%s = %s' % (parameter.ljust(35), self.fileWriterStatus(parameter)['value'])
            except:
                print '%s = %s' % (parameter.ljust(35), "Unknown")
                
    def print_monitor_status(self):
        for parameter in self.monitorStatus():
            try:
                print '%s = %s' % (parameter.ljust(35), self.monitorStatus(parameter)['value'])
            except:
                print '%s = %s' % (parameter.ljust(35), "Unknown")
    
    def print_stream_status(self):
        for parameter in self.streamStatus(param='keys'):
            try:
                print '%s = %s' % (parameter.ljust(35), self.streamStatus(parameter)['value'])
            except:
                print '%s = %s' % (parameter.ljust(35), "Unknown")
    # high level methods combining several calls
    # useful for setting up data collections and downloads
    def download(self, downloadpath="/tmp"):
        self.check_dir(downloadpath)
        try:
           matching = self.fileWriterFiles()
        except:
           print "could not get file list"
        if len(matching):
            try:
                [self.fileWriterSave(i, downloadpath) for i in matching]
            except:
                print "error saving - nothing deleted"
            else:
                print "Downloaded ..." 
                for i in matching:
                    print i + " to " + str(downloadpath)
                [self.fileWriterFiles(i, method = 'DELETE') for i in matching]
                print "Deteted " + str(len(matching)) + " file(s)"
    
    def collect(self):
        start_time = time.time()
        print 'going to collect {nimages} images, {count_time} sec. per frame'.format(**{'nimages': self.nimages, 'count_time': self.count_time})
        print 'name_pattern {name_pattern} '.format(**{'name_pattern': self.name_pattern})
        print 'Arm!'
        a=time.time()
        self.arm()
        print 'Arm took %s' % (time.time() - a)
        print 'Trigger!'
        if self.trigger_mode == 'ints':
            self.trigger()
        elif self.trigger_mode == 'inte':
            for k in ntrigger:
                self.trigger()
        else:
            self.wait_for_collect_to_finish()
            print 'Collect finished!'
        time.sleep(1)
        print 'Disarm!'
        self.disarm()
        print 'Collect took %s' % (time.time() - start_time)
        
    def wait_for_collect_to_finish(self):
        while self.detectorStatus('state')['value'] not in ['idle']:
            time.sleep(0.2)

    def check_dir(self, download):
        if os.path.isdir(download):
            pass
        else:
            os.makedirs(download)
    
    def check_directory(self, download):
        self.check_dir(download)
        
    def set_corrections(self, fca=False, pma=False, vpca=False, crca=False):
        self.set_flatfield_correction_applied(fca)
        self.set_countrate_correction_applied(crca)
        self.set_pixel_mask_applied(pma)
        self.set_virtual_pixel_correction_applied(vpca)
    
    def write_destination_namepattern(self, image_path, name_pattern, goimgfile='/927bis/ccd/log/.goimg/goimg.db'):
        try:
            f = open(goimgfile, 'w')
            f.write('%s %s' % (os.path.join(image_path, 'process'), name_pattern))
            f.close()
        except IOError:
            logging.info('Problem writing goimg.db %s' % (traceback.format_exc()))

    def prepare(self):
        self.clear_monitor()
        self.write_destination_namepattern(image_path=self.directory, name_pattern=self.name_pattern)

    def set_standard_parameters(self):
        for angle in ['two_theta', 'phi', 'chi', 'kappa']:
            if getattr(self, 'get_%s' % angle)() != 0:
                getattr(self, 'set_%s' % angle)(0)
            if getattr(self, 'get_%s_increment' % angle)() != 0:
                getattr(self, 'set_%s_increment' % angle)(0)
            if getattr(self, 'get_%s_increment' % angle)() != 0:
                getattr(self, 'set_%s_increment' % angle)(0)
        for option in ['compression_enabled', 'flatfield_correction_applied', 'countrate_correction_applied', 'virtual_pixel_correction_applied']:
            if not getattr(self, 'get_%s' % option)():
                setattr(self, 'set_%s' % option)(True)
        if self.get_compression() != 'bslz4':
            self.set_compression('bslz4')
        if self.get_trigger_mode() != 'exts':
            self.set_trigger_mode('exts')
        if self.get_nimages_per_file() != 100:
            self.set_nimages_per_file(100)


if __name__ == '__main__':
    import optparse
    parser = optparse.OptionParser() 
    parser.add_option('-i', '--ip', default="172.19.10.26", type=str, help='IP address of the server')
    parser.add_option('-p', '--port', default=80, type=int, help='port on which to which it listens to')
    
    options, args = parser.parse_args()
     
    e = eiger(host=options.ip, port=options.port)

