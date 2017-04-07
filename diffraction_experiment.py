#!/usr/bin/env python
from experiment import experiment
from detector import detector
from goniometer import goniometer
from energy import energy as energy_motor
from resolution import resolution as resolution_motor
from transmission import transmission as transmission_motor
# from flux import flux
# from filters import filters
# from camera import camera
#from beam import beam
from beam_center import beam_center
from protective_cover import protective_cover
from camera import camera

class diffraction_experiment(experiment):
    
    def __init__(self,
                 name_pattern, 
                 directory,
                 photon_energy=None,
                 resolution=None,
                 detector_distance=None,
                 transmission=None,
                 flux=None,
                 ntrigger=1,
                 snapshot=False):
        
        experiment.__init__(self, 
                            name_pattern=name_pattern, 
                            directory=directory)
        
        self.photon_energy = photon_energy
        self.resolution = resolution
        self.detector_distance = detector_distance
        self.transmission = transmission
        self.flux = flux
        self.ntrigger = ntrigger
        self.snapshot = snapshot
        
        # Necessary equipment
        self.goniometer = goniometer()
        try:
            self.beam_center = beam_center()
        except:
            from beam_center import beam_center_mockup
            self.beam_center = beam_center_mockup()
        try:
            self.detector = detedctor()
        except:
            from detector_mockup import detector_mockup
            self.detector = detector_mockup()
        try:
            self.energy_motor = energy_motor()
        except:
            from energy import energy_mockup
            self.energy_motor = energy_mockup()
        try:
            self.resolution_motor = resolution_motor()
        except:
            from resolution import resolution_mockup
            self.resolution_motor = resolution_mockup()
        try:
            self.transmission_motor = transmission_motor()
        except:
            from transmission import transmission_mockup
            self.transmission_motor = transmission_mockup()
        
        self.protective_cover = protective_cover()
        self.camera = camera()
        
    def set_photon_energy(self, photon_energy=None):
        if photon_energy is not None:
            self.photon_energy = photon_energy
            self.energy_motor.set_energy(photon_energy)

    def get_photon_energy(self):
        return self.photon_energy

    def set_resolution(self, resolution=None):
        if resolution is not None:
            self.resolution = resolution
            self.resolution_motor.set_resolution(resolution)
    def get_resolution(self):
        return self.resolution

    def set_transmission(self, transmission=None):
        if transmission is not None:
            self.transmission = transmission
            self.transmission_motor.set_transmission(transmission)
    def get_transmission(self):
        return self.transmission

    def program_detector(self):
        self.detector.set_standard_parameters()
        self.detector.clear_monitor()
        self.detector.set_ntrigger(self.get_ntrigger())
        self.detector.set_nimages_per_file(self.get_nimages_per_file())
        self.detector.set_nimages(self.get_nimages())
        self.detector.set_name_pattern(self.get_full_name_pattern())
        self.detector.set_frame_time(self.get_frame_time())
        count_time = self.get_frame_time() - self.detector.get_detector_readout_time()
        self.detector.set_count_time(count_time)
        self.detector.set_nimages(self.nimages)
        self.detector.set_omega(self.scan_start_angle)
        if self.angle_per_frame <= 0.01:
            self.detector.set_omega_increment(0)
        else:
            self.detector.set_omega_increment(self.angle_per_frame)
            
        if self.detector.get_photon_energy() != self.photon_energy:
            self.detector.set_photon_energy(self.photon_energy)
        if self.detector.get_image_nr_start() != self.image_nr_start:
            self.detector.set_image_nr_start(self.image_nr_start)
        
        beam_center_x, beam_center_y = self.beam_center.get_beam_center()
        self.detector.set_beam_center_x(beam_center_x)
        self.detector.set_beam_center_y(beam_center_y)
        self.detector.set_detector_distance(self.beam_center.get_detector_distance()/1000.)
        self.detector.arm()
        
    def program_goniometer(self):
        self.goniometer.set_scan_start_angle(self.scan_start_angle)
        self.goniometer.set_scan_range(self.scan_range)
        self.goniometer.set_scan_exposure_time(self.scan_exposure_time)
        self.goniometer.set_scan_number_of_frames(1)
        self.goniometer.set_detector_gate_pulse_enabled(True)
        self.goniometer.set_data_collection_phase()

    def prepare(self):
        self.check_directory(self.process_directory)
        self.program_goniometer()
        self.program_detector()
        
        self.set_photon_energy(self.photon_energy)
        self.set_resolution(self.resolution)
        self.set_transmission(self.transmission)
        self.protective_cover.extract()
        
        self.resolution_motor.wait()
        self.energy_motor.wait()
        self.goniometer.wait()
        
        self.goniometer.set_collect_phase()
        self.detector.clear_monitor()
        self.guillotine.extract()
        self.camera.set_zoom(self.zoom)
        self.goniometer.wait()
        self.camera.set_exposure(0.05)
        self.goniometer.wait()
        if self.scan_start_angle is None:
            self.scan_start_angle = self.reference_position['Omega']
        self.goniometer.set_omega_position(self.scan_start_angle)
        if self.snapshot == True:
            print 'taking image'
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
        
        self.write_destination_namepattern(image_path=self.directory, name_pattern=self.name_pattern)
        self.energy_motor.turn_off()
        
    def collect(self):
        return self.run()
    def measure(self):
        return self.run()
    def run():
        pass
    
    def clean(self):
        self.detector.disarm()
        self.save_parameters()
    
    def stop(self):
        self.goniometer.abort()
        self.detector.abort()
        
        