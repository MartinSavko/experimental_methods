#!/usr/bin/env python

import sys
import time
import os
import numpy
import logging
import traceback
from eigerclient import DEigerClient

'''
detector class implements higher level interface to SIMPLON API of the EIGER detectors. It inherits from DIgerClient class developed by Dectris and provides explicit set and get methods for every writable attribute of the API and the get method for all readonly attributes. All of the API commands are supported as well.

Examples:
d = detector(ip=172.19.10.26, port=80)
d.initialize()
d.set_photon_energy(12650)
print d.get_trigger_mode()

To collect 100 images in 'ints' trigger mode:
frame_time = 0.5
d.set_frame_time(frame_time)
d.set_count_time(frame_time - d.get_readout_time())
d.set_nimages(100)
d.set_nimages_per_file(25)
d.set_name_pattern('test_1')
d.set_compression('bslz4')
d.arm()
d.trigger()
d.disarm()
d.download('./') #download data to the current directory and remove them from the detector control unit (DCU) cache
d.remove_files('test_1') #remove data from the DCU cache

The class also implements helper methods to succintly represent the configuration and state of various components of the detector

d.print_detector_config()
d.print_filewriter_config()
d.print_monitor_status()
   
'''
class detector(DEigerClient):
    
    def __init__(self, host='172.19.10.26', port=80):
        DEigerClient.__init__(self, host=host, port=port)
        
    # detector configuration
    def set_photon_energy(self, photon_energy):
        if photon_energy < self.detectorConfig('photon_energy')['min']:
            print 'photon_energy: value below allowed minimal value'
            return
        self.photon_energy = photon_energy
        return self.setDetectorConfig("photon_energy", photon_energy)
    
    def get_photon_energy(self):
        return self.detectorConfig("photon_energy")['value']
        
    def set_threshold_energy(self, threshold_energy):
        self.threshold_energy = threshold_energy
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

    def get_omega_range_average(self):
        return self.detectorConfig('omega_increment')['value']
        
    def set_omega_range_average(self, omega_increment):
        self.omega_increment = omega_increment
        return self.setDetectorConfig('omega_range_average', omega_increment)
        
    def get_omega_range_average(self):
        return self.detectorConfig('omega_range_average')['value']
        
    def set_phi(self, phi):
        self.phi = phi
        return self.setDetectorConfig('phi_start', phi)
    
    def get_phi(self):
        return self.detectorConfig('phi_start')['value']
        
    def set_phi_range_average(self, phi_increment):
        self.phi_increment = phi_increment
        return self.setDetectorConfig('phi_range_average', phi_increment)
        
    def get_phi_range_average(self):
        return self.detectorConfig('phi_range_average')['value']
        
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
        
    def list_files(self, name_pattern=None):
        return self.fileWriterFiles(filename=name_pattern, method='GET')
    
    def remove_files(self, name_pattern=None):
        return self.fileWriterFiles(filename=name_pattern, method='DELETE')
    
    def save_files(self, filename, destination, regex=False):
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
    def clear_filewriter(self):
        return self.sendFileWriterCommand(u'clear')
    
    def initialize_filewriter(self):
        return self.sendFileWriterCommand(u'initialize')
    
    # monitor commands
    def clear_monitor(self):
        return self.sendMonitorCommand(u'clear')
    
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
            elif parameter in ['two_theta', 'two_theta_end', 'omega', 'omega_end', 'kappa', 'kappa_end', 'phi', 'phi_end', 'chi', 'chi_end']:
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


if __name__ == '__main__':
    import optparse
    parser = optparse.OptionParser() 
    parser.add_option('-i', '--ip', default="172.19.10.26", type=str, help='IP address of the server')
    parser.add_option('-p', '--port', default=80, type=int, help='port on which to which it listens to')
    
    options, args = parser.parse_args()
     
    d = detector(host=options.ip, port=options.port)

