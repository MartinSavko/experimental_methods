from goniometer import goniometer
from detector import detector
from beam_center import beam_center

class reference_images(object):
    def __init__(self,
                 scan_range,
                 scan_exposure_time,
                 scan_start_angles, #this is an iterable
                 angle_per_frame,
                 name_pattern,
                 directory='/nfs/ruchebis/spool/2016_Run3/orphaned_collects',
                 image_nr_start=1):
                     
        self.goniometer = goniometer()
        self.detector = detector()
        self.beam_center = beam_center()
        
        scan_range = float(scan_range)
        scan_exposure_time = float(scan_exposure_time)
        
        nimages = float(scan_range)/angle_per_frame

        frame_time = scan_exposure_time/nimages
        
        self.scan_range = scan_range
        self.scan_exposure_time = scan_exposure_time
        self.scan_start_angles = scan_start_angles
        self.angle_per_frame = angle_per_frame
        
        self.nimages = int(nimages)
        self.frame_time = float(frame_time)
        self.count_time = self.frame_time - self.detector.get_detector_readout_time()
        
        self.name_pattern = name_pattern
        self.directory = directory
        self.image_nr_start = image_nr_start
        self.status = None
                    
    def program_detector(self):
        self.detector.set_ntrigger(len(self.scan_start_angles))
        if self.detector.get_compression() != 'bslz4':
            self.detector.set_compression('bslz4')
        self.detector.set_nimages_per_file(self.nimages)
        self.detector.set_nimages(self.nimages)
        self.detector.set_frame_time(self.frame_time)
        self.detector.set_count_time(self.count_time)
        self.detector.set_name_pattern(self.name_pattern)
        self.detector.set_omega(self.scan_start_angles[0])
        self.detector.set_omega_increment(self.angle_per_frame)
        self.detector.set_image_nr_start(self.image_nr_start)
        beam_center_x, beam_center_y = self.beam_center.get_beam_center()
        self.detector.set_beam_center_x(beam_center_x)
        self.detector.set_beam_center_y(beam_center_y)
        self.detector.set_detector_distance(self.beam_center.get_detector_distance() / 1000.)
        return self.detector.arm()
        
    def program_goniometer(self):
        if self.goniometer.md2.backlightison == True:
            self.goniometer.md2.backlightison = False
        self.goniometer.set_scan_range(self.scan_range)
        self.goniometer.set_scan_exposure_time(self.scan_exposure_time)
            
    def prepare(self):
        self.detector.check_dir(os.path.join(self.directory,'process'))
        self.detector.clear_monitor()
        self.detector.write_destination_namepattern(image_path=self.directory, name_pattern=self.name_pattern)
        self.status = 'prepare' 
    
    def collect(self):
        self.prepare()
        self.program_detector()
        self.program_goniometer()
        for scan_start_angle in self.scan_start_angles:
            self.goniometer.set_scan_start_angle(scan_start_angle)
            scan_id = self.goniometer.start_scan()
            while self.goniometer.md2.istaskrunning(scan_id):
                time.sleep(0.1)
        self.clean()
            
    def stop(self):
        self.goniometer.abort()
        self.detector.abort()

    def clean(self):
        self.detector.disarm() 