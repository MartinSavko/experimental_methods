#!/usr/bin/env python
'''
Author: Martin Savko 
Contact: savko@synchrotron-soleil.fr
Date: 2016-02-09
Version: 1.0.0

eiger.py implements three classes: detector, goniometer and sweep. 

detector inherits DEigerClient. It provides explicit get and set methods for all
configuration parameters and state values of Eiger detectors as described in 
SIMPLON API v. 1.5.0.

goniometer implements higher level control of MD2 diffractometer using PyTango.

sweep allows configuration and execution of individual continuous data 
collections using the goniometer and the detector objects.

The code is used on PROXIMA 2A beamline, Synchrotron SOLEIL to collect data with
EIGER 9M detector and MD2 goniometer.



'''

import sys
import time
import os
import numpy 
import PyTango
import logging

sys.path.insert(0,"/usr/local/dectris/python")
sys.path.insert(0,"/usr/local/dectris/albula/3.1/python")

#import dectris.albula
from eigerclient import DEigerClient

class goniometer(object):
    
    def __init__(self):
        self.md2 = PyTango.DeviceProxy('i11-ma-cx1/ex/md2')
      
    def set_scan_start_angle(self, scan_start_angle):
        self.scan_start_angle = scan_start_angle
        self.md2.scanstartangle = scan_start_angle
    
    def get_scan_start_angle(self):
        return self.md2.scanstartangle
       
    def set_scan_range(self, scan_range):
        self.scan_range = scan_range
        self.md2.scanrange = scan_range
        
    def get_scan_range(self):
        return self.md2.scanrange
        
    def set_scan_exposure_time(self, scan_exposure_time):
        self.scan_exposure_time = scan_exposure_time
        self.md2.scanexposuretime = scan_exposure_time
    
    def get_scan_exposure_time(self):
        return self.md2.scanexposuretime
    
    def abort(self):
        return self.md2.abort()

    def start_scan(self):
        return self.md2.startscan()
        
    def start_helical_scan(self):
        return self.md2.startscan4d()
        
    def set_helical_start(self):
        return self.md2.setstartscan4d()
    
    def set_helical_stop(self):
        return self.md2.setstopscan4d()
        
    def getMotorState(self, motor_name):
        return self.md2.getMotorState(motor_name).name
        
    def getState(self):
        motors = ['Omega', 'AlignmentX', 'AlignmentY', 'AlignmentZ', 'CentringX', 'CentringY', 'ApertureHorizontal', 'ApertureVertical', 'CapillaryHorizontal', 'CapillaryVertical', 'ScintillatorHorizontal', 'ScintillatorVertical', 'Zoom']
        state = set([self.getMotorState(m) for m in motors])
        if len(state) == 1 and 'STANDBY' in state:
            return 'STANDBY'
        else:
            return 'MOVING'
            
    def wait(self, device=None):
        green_light = False
        while green_light is False:
            try:
                if device is None:
                    if self.getState() in ['MOVING', 'RUNNING']:
                        logging.info("MiniDiffPX2 wait" )
                    else:
                        green_light = True
                        return
                else:   
                    if device.state().name not in ['STANDBY']:
                        logging.info("Device %s wait" % device)
                    else:
                        green_light = True
                        return
            except:
                import traceback
                traceback.print_exc()
                logging.info('Problem occured in wait %s ' % device)
                logging.info(traceback.print_exc())
            time.sleep(.1)
        

    def moveToPosition(self, position={}, epsilon = 0.0002):
        print 'position %s' % position
        if position != {}:
            for motor in position:
                while abs(self.md2.read_attribute('%sPosition' % self.shortFull[motor]).value - position[motor]) > epsilon:
                    self.wait()
                    time.sleep(0.5)
                    self.md2.write_attribute('%sPosition' % self.shortFull[motor], position[motor])
                
            self.wait()
        self.wait()
        return
    
class detector(DEigerClient):
    
    # detector configuration
    def set_photon_energy(self, photon_energy):
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
        self.frame_time = frame_time
        return self.setDetectorConfig("frame_time", frame_time)
    
    def get_frame_time(self):
        return self.detectorConfig("frame_time")['value']
        
    def set_count_time(self, count_time):
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
        
    def set_trigger_mode(self, trigger_mode="ints"):
        '''one of four values possible
           ints
           inte
           exts
           exte
        '''
        self.trigger_mode = trigger_mode
        return self.setDetectorConfig("trigger_mode", trigger_mode)
    
    def get_trigger_mode(self, trigger_mode="ints"):
        return self.detectorConfig("trigger_mode")['value']
        
    def set_omega(self, omega):
        self.omega = omega
        return self.setDetectorConfig('omega', omega)
    
    def get_omega(self):
        return self.detectorConfig('omega')['value']
        
    def set_omega_range_average(self, omega_increment):
        self.omega_increment = omega_increment
        return self.setDetectorConfig('omega_range_average', omega_increment)
        
    def get_omega_range_average(self):
        return self.detectorConfig('omega_range_average')['value']
        
    def set_phi(self, phi):
        self.phi = phi
        return self.setDetectorConfig('phi', phi)
    
    def get_phi(self):
        return self.detectorConfig('phi')['value']
        
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
        
    # detector status
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
    
    def set_buffer_size(self, buffer_size):
        return self.setMonitorConfig('buffer_size', buffer_size)
        
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
        c.set_flatfield_correction_applied(fca)
        c.set_countrate_correction_applied(crca)
        c.set_pixel_mask_applied(pma)
        c.set_virtual_pixel_correction_applied(vpca)
        

class sweep(object):
    
    def __init__(self,
                 scan_range,
                 scan_exposure_time,
                 scan_start_angle,
                 angle_per_frame,
                 name_pattern,
                 image_nr_start=1,
                 beam_center_x=1530,
                 beam_center_y=1657,
                 detector_distance=0.250,
                 nimages_per_file=50,
                 trigger_mode='exts',
                 helical=False):
        
        self.goniometer = goniometer()
        self.detector = detector(host='172.19.10.26', port=80)
        
        self.distance_motor = PyTango.DeviceProxy('i11-ma-cx1/dt/dtc_ccd.1-mt_ts')
        self.wavelength_motor = PyTango.DeviceProxy('i11-ma-c03/op/mono1')
        
        self.detector.set_trigger_mode(trigger_mode)
        self.detector.set_nimages_per_file(nimages_per_file)
        self.detector.set_beam_center_x(beam_center_x)
        self.detector.set_beam_center_y(beam_center_y)
        self.detector.set_detector_distance(detector_distance)
        scan_range = float(scan_range)
        scan_exposure_time = float(scan_exposure_time)
        
        nimages, rest = divmod(scan_range, angle_per_frame)
        
        if rest > 0:
            nimages += 1
            scan_range += rest*angle_per_frame
            scan_exposure_time += rest*angle_per_frame/scan_range
            
        frame_time = scan_exposure_time/nimages
        
        self.scan_range = scan_range
        self.scan_exposure_time = scan_exposure_time
        self.scan_start_angle = scan_start_angle
        self.angle_per_frame = angle_per_frame
        
        self.nimages = int(nimages)
        self.frame_time = float(frame_time)
        self.count_time = self.frame_time - self.detector.get_detector_readout_time()
        
        self.name_pattern = name_pattern
        self.image_nr_start = image_nr_start
        self.helical = helical
        self.status = None
    
    def get_beam_center(self):
        Theta = numpy.matrix([[  1.54776707e+03,   1.65113065e+03], [  3.65108709e-01,   5.63662370e+00], [ -1.12769165e-01,   3.49706731e-03]])
        X = numpy.matrix([1., self.wavelength_motor.read_attribute('lambda').value, self.distance_motor.position])
        X = X.T
        beam_center = Theta.T * X
        beam_center_x = beam_center[0, 0]
        beam_center_y = beam_center[1, 0]
        return beam_center_x, beam_center_y

    def program_goniometer(self):
        self.goniometer.backlightison = False
        self.goniometer.set_scan_start_angle(self.scan_start_angle)
        self.goniometer.set_scan_range(self.scan_range)
        self.goniometer.set_scan_exposure_time(self.scan_exposure_time)
        
    def program_detector(self):
        self.detector.set_nimages(self.nimages)
        self.detector.set_frame_time(self.frame_time)
        self.detector.set_count_time(self.count_time)
        self.detector.set_name_pattern(self.name_pattern)
        #self.detector.set_omega(self.scan_start_angle)
        #self.detector.set_omega_range_average(self.angle_per_frame)
        #self.detector.set_phi(self.scan_start_angle)
        #self.detector.set_phi_range_average(self.angle_per_frame)
        self.detector.set_image_nr_start(self.image_nr_start)
        #beam_center_x, beam_center_y = self.get_beam_center()
        #self.detector.set_beam_center_x(beam_center_x)
        #self.detector.set_beam_center_y(beam_center_y)
        return self.detector.arm()
        
    def prepare(self):
        self.detector.clear_monitor()
        self.status = 'prepare'
        
    def collect(self):
        self.prepare()
        self.program_goniometer()
        self.series_id = self.program_detector()['sequence id']
        self.status = 'collect'
        if self.helical:
            return self.goniometer.start_helical_scan()
        return self.goniometer.start_scan()
    
    #def monitor(self):
        #self.
    
    def stop(self):
        self.goniometer.abort()
        self.detector.abort()

    def clean(self):
        self.detector.disarm()


if __name__ == '__main__':
    import optparse
    parser = optparse.OptionParser() 
    parser.add_option('-i', '--ip', default="172.19.10.25", type=str, help='IP address of the server')
    parser.add_option('-p', '--port', default=80, type=int, help='port on which to which it listens to')
    
    options, args = parser.parse_args()
    
    d = detector(host=options.ip, port=options.port)
    g = goniometer()
