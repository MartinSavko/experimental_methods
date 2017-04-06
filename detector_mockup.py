class detector_mockup:
    
    def __init__(self, host='172.19.10.26', port=80):
        self.host = host
        self.port = port
        attributes = [  "photon_energy",
                        "threshold_energy",
                        "data_collection_date",
                        "beam_center_x",
                        "beam_center_y",
                        "detector_distance    ",
                        "detector_translation",
                        "frame_time",
                        "count_time",
                        "nimages",
                        "ntrigger",
                        "wavelength",
                        "summation_nimages",
                        "nframes_sum    ",
                        "element",
                        "trigger_mode",
                        "omega_start",
                        "omega_increment",
                        "omega_range_average",
                        "phi_start",
                        "phi_increment",
                        "phi_range_average",
                        "chi_start",
                        "chi_increment",
                        "chi_range_average",
                        "kappa_start",
                        "kappa_increment",
                        "kappa_range_average",
                        "two_theta_start",
                        "two_theta_increment",
                        "two_theta_range_average",
                        "pixel_mask",
                        "bit_depth_image",
                        "detector_readout_time",
                        "auto_summation",
                        "countrate_correction_applied",
                        "pixel_mask_applied",
                        "flatfield_correction_applied",
                        "virtual_pixel_correction_applied",
                        "efficiency_correction_applied"
                        "count_rate_correction_applied",
                        "compression",
                        "roi_mode",
                        "roi_mode = roi_mode",
                        "x_pixels_in_detector",
                        "y_pixels_in_detector    ",
                        "name_pattern",
                        "nimages_per_file",
                        "image_nr_start",
                        "compression_enabled"]
        for attribute in attributes:
            setattr(self, attribute, 0)

    # detector configuration
    def set_photon_energy(self, photon_energy):
        self.photon_energy = photon_energy
    def get_photon_energy(self):
        return self.photon_energy
        
    def set_threshold_energy(self, threshold_energy):
        self.threshold_energy = threshold_energy
    def get_threshold_energy(self):
        return self.threshold_energy
        
    def set_data_collection_date(self, data_collection_date=None):
        self.data_collection_date = data_collection_date
    def get_data_collection_date(self):
        return self.data_collection_date
        
    def set_beam_center_x(self, beam_center_x):
        self.beam_center_x = beam_center_x
    def get_beam_center_x(self):
        return self.beam_center_x
        
    def set_beam_center_y(self, beam_center_y):
        self.beam_center_y = beam_center_y
    def get_beam_center_y(self):
        return self.beam_center_y
        
    def set_detector_distance(self, detector_distance):
        self.detector_distance = detector_distance
    def get_detector_distance(self):
        return self.detector_distance    
    
    def set_detector_translation(self, detector_translation):
        '''detector_translation is list of real values'''
        self.detector_translation = detector_translation
    def get_detector_translation(self):
        return self.detector_translation
        
    def set_frame_time(self, frame_time):
        self.frame_time = frame_time
    def get_frame_time(self):
        return self.frame_time
        
    def set_count_time(self, count_time):
        self.count_time = count_time
    def get_count_time(self):
        return self.count_time
        
    def set_nimages(self, nimages):
        self.nimages = nimages
    def get_nimages(self):
        return self.nimages
        
    def set_ntrigger(self, ntrigger):
        self.ntrigger = ntrigger
    def get_ntrigger(self):
        return self.ntrigger
        
    def set_wavelength(self, wavelength):
        self.wavelength = wavelength
    def get_wavelength(self):
        return self.wavelength
        
    def set_summation_nimages(self, summation_nimages):
        self.summation_nimages = summation_nimages
    def get_summation_nimages(self, summation_nimages):
        return self.summation_nimages
        
    def set_nframes_sum(self, nframes_sum):
        self.nframes_sum = nframes_sum
    def get_nframes_sum(self):
        return self.nframes_sum    
        
    def set_element(self, element):
        self.element = element
    def get_element(self, element):
        return self.element
        
    def set_trigger_mode(self, trigger_mode="exts"):
        '''one of four values possible
           ints
           inte
           exts
           exte
        '''
        self.trigger_mode = trigger_mode
    def get_trigger_mode(self):
        return self.trigger_mode
        
    def set_omega(self, omega):
        self.omega = omega
    def get_omega(self):
        return self.omega_start
    
    def set_omega_increment(self, omega_increment):
        self.omega_increment = omega_increment
    def get_omega_range_average(self):
        return self.omega_increment
        
    def set_omega_range_average(self, omega_increment):
        self.omega_increment = omega_increment
    def get_omega_range_average(self):
        return self.omega_range_average
        
    def set_phi(self, phi):
        self.phi = phi
    def get_phi(self):
        return self.phi_start
        
    def set_phi_range_average(self, phi_increment):
        self.phi_increment = phi_increment
    def get_phi_range_average(self):
        return self.phi_range_average
        
    def set_chi(self, chi):
        self.chi = chi   
    def get_chi(self):
        return self.chi_start
        
    def set_chi_range_average(self, phi_increment):
        self.chi_increment = chi_increment
    def get_chi_range_average(self):
        return self.chi_range_average

    def set_kappa(self, kappa):
        self.kappa = kappa
    def get_kappa(self):
        return self.kappa_start
        
    def set_kappa_range_average(self, kappa_increment):
        self.kappa_increment = kappa_increment
        return self.setDetectorConfig('kappa_range_average', kappa_increment)
        
    def get_kappa_range_average(self):
        return self.kappa_range_average
    
    def set_two_theta(self, two_theta):
        self.two_theta = two_theta
    def get_two_theta(self):
        return self.two_theta_start
        
    def set_two_theta_range_average(self, two_theta_increment):
        self.two_theta_increment = two_theta_increment
    def get_two_theta_range_average(self):
        return self.two_theta_range_average
    
    def get_pixel_mask(self):
        return self.pixel_mask
    
    # Apparently writing to bit_depth_image is forbidden 2015-10-25 MS
    def set_bit_depth_image(self, bit_depth_image=16):
        self.bit_depth_image = bit_depth_image
    def get_bit_depth_image(self):
        return self.bit_depth_image
        
    def set_detector_readout_time(self, detector_readout_time):
        self.detector_readout_time = detector_readout_time
    def get_detector_readout_time(self):
        return self.detector_readout_time
        
    # booleans
    def set_auto_summation(self, auto_summation=True):
        self.auto_summation = auto_summation
    
    def get_auto_summation(self):
        return self.auto_summation
        
    def set_countrate_correction_applied(self, count_rate_correction_applied=True):
        self.countrate_correction_applied = count_rate_correction_applied
    def get_countrate_correction_applied(self):
        return self.countrate_correction_applied
        
    def set_pixel_mask_applied(self, pixel_mask_applied=True):
        self.pixel_mask_applied = pixel_mask_applied
    def get_pixel_mask_applied(self):
        return self.pixel_mask_applied
        
    def set_flatfield_correction_applied(self, flatfield_correction_applied=True):
        self.flatfield_correction_applied = flatfield_correction_applied
    def get_flatfield_correction_applied(self):
        return self.flatfield_correction_applied
        
    def set_virtual_pixel_correction_applied(self, virtual_pixel_correction_applied=True):
        self.virtual_pixel_correction_applied = virtual_pixel_correction_applied
    def get_virtual_pixel_correction_applied(self):
        return self.virtual_pixel_correction_applied
        
    def set_efficiency_correction_applied(self, efficiency_correction_applied=True):
        self.efficiency_correction_applied = efficiency_correction_applied
    def get_efficiency_correction_applied(self):
        return self.efficiency_correction_applied
        
    def set_compression(self, compression='lz4'):
        self.compression = compression
    def get_compression(self):
        return self.compression
        
    def set_roi_mode(self, roi_mode='4M'):
        self.roi_mode = roi_mode
    def get_roi_mode(self):
        return self.roi_mode
    
    def get_x_pixels_in_detector(self):
        return self.x_pixels_in_detector
    def get_y_pixels_in_detector(self):
        self.y_pixels_in_detector    
        
    # filewriter
    def set_name_pattern(self, name_pattern):
        self.name_pattern = name_pattern
    def get_name_pattern(self):
        return self.name_pattern
        
    def set_nimages_per_file(self, nimages_per_file):
        self.nimages_per_file = nimages_per_file
    def get_nimages_per_file(self):
        return self.nimages_per_file
    
    def set_image_nr_start(self, image_nr_start):
        self.image_nr_start = image_nr_start
    def get_image_nr_start(self):
        return self.image_nr_start
    
    def set_compression_enabled(self, compression_enabled=True):
        self.compression_enabled = compression_enabled
    def get_compression_enabled(self):
        return self.compression_enabled
        
    def list_files(self, name_pattern=None):
        return 'GET'
    
    def remove_files(self, name_pattern=None):
        return 'DELETE'
    
    def save_files(self, filename, destination, regex=False):
        return 'SAVE'
        
    def get_filewriter_config(self):
        return 'filewriter_config'
    
    def get_free_space(self):
        return 'buffer_free'
    
    def get_buffer_free(self):
        return 'buffer_free'
        
    def get_filewriter_state(self):
        return 'filewriter_state'
        
    def get_filewriter_error(self):
        return 'filewriter_error'
        
    # detector status
    def get_detector_state(self):
        return 'detector_state'
        
    def get_detector_error(self):
        return 'detector_error'
        
    def get_detector_status(self):
        return 'detector_status'
    
    def get_humidity(self):
        return 'board_000/th0_humidity'
        
    def get_temperature(self):
        return 'board_000/th0_temp'
        
    # detector commands
    def arm(self):
        return u'arm'
        
    def trigger(self, count_time=None):
        if count_time == None:
            return u'trigger'
        else:
            return u'trigger', count_time
        
    def disarm(self):
        return u'disarm'
        
    def cancel(self):
        return u'cancel'
        
    def abort(self):
        return u'abort'
        
    def initialize(self):
        return u'initialize'
    
    def status_update(self):
        return u'status_update'
    
    # filewriter commands
    def clear_filewriter(self):
        return u'clear'
    
    def initialize_filewriter(self):
        return u'initialize'
    
    # monitor commands
    def clear_monitor(self):
        return u'clear'
    def initialize_monitor(self):
        return u'initialize'
    
    def set_buffer_size(self, buffer_size):
        self.monitor_buffer_size = buffer_size
    def get_buffer_size(self):
        return self.monitor_buffer_size
        
    def get_monitor_state(self):
        return 'monitor_state'
        
    def get_monitor_error(self):
        return 'monitor_error'
        
    def get_buffer_fill_level(self):
        return 'monitor_buffer_fill_level'
        
    def get_monitor_dropped(self):
        return 'monitor_dropped'
    
    def get_monitor_image_number(self):
        return 'monitor_image_number'
    
    def get_next_image_number(self):
        return 'next_image_number'
    
    # stream commands 
    def get_stream_state(self):
        return 'stream_state'
        
    def get_stream_error(self):
        return 'stream_error'
        
    def get_stream_dropped(self):
        return 'stream_dropped'
    
    # system commmand
    def restart(self):
        return u'restart'
    
    # useful helper methods
    def print_detector_config(self):
        return 'detector_config'
    
    def print_filewriter_config(self):
        return 'filewriter_config'
    
    def print_monitor_config(self):
        return 'monitor_config'
    
    def print_stream_config(self):
        return 'stream_config'
    
    def print_detector_status(self):
        return 'detector_status'
    
    def print_filewriter_status(self):
        return 'filewriter_status'
            
    def print_monitor_status(self):
        return 'monitor_status'
    
    def print_stream_status(self):
        return 'stream_status'
    
    def download(self, downloadpath="/tmp"):
        return
    
    def collect(self):
        return
        
    def wait_for_collect_to_finish(self):
        return
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

    def prepare(self):
        self.clear_monitor()
        self.write_destination_namepattern(image_path=self.directory, name_pattern=self.name_pattern)

    def set_standard_parameters(self):
        if self.get_two_theta != 0:
            self.set_two_theta(0)
        if self.get_two_theta_range_average() != 0:
            self.set_two_theta_range_average(0)
        if self.get_phi() != 0:
            self.set_phi(0)
        if self.get_phi_range_average() != 0:
            self.set_phi_range_average(0)
        if self.get_chi() != 0:
            self.set_chi(0)
        if self.get_chi_range_average() != 0:
            self.set_chi_range_average(0)
        if self.get_kappa() != 0:
            self.set_kappa(0)
        if self.get_kappa_range_average() != 0:
            self.set_kappa_range_average(0)
        if not self.get_compression_enabled():
            self.set_compression_enabled(True)
        if not self.get_flatfield_correction_applied():
            self.set_flatfield_correction_applied(True)
        if not self.get_countrate_correction_applied():
            self.set_countrate_correction_applied()
        if not self.get_virtual_pixel_correction_applied():
            self.set_virtual_pixel_correction_applied(True)
        if self.get_compression() != 'bslz4':
            self.set_compression('bslz4')
        if self.get_trigger_mode() != 'exts':
            self.set_trigger_mode('exts')
        if self.get_nimages_per_file() != 100:
            self.set_nimages_per_file(100)
