#!/usr/bin/env python
'''
single position oscillation scan
'''
import traceback
import logging
import time

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


class scan(experiment):
    ''' Will execute single continuous omega scan 
    
    The class accepts the following optional arguments:
        name_pattern=NAME_PATTERN
                                Prefix default=test_$id
        directory=DIRECTORY
                                Destination directory default=/nfs/data/default
        photon_energy=PHOTON_ENERGY
                                Photon energy
        scan_range=SCAN_RANGE
                                Scan range [deg]
        scan_exposure_time=SCAN_EXPOSURE_TIME
                                Scan exposure time [s]
        scan_start_angle=SCAN_START_ANGLE
                                Scan start angle [deg]
        angle_per_frame=ANGLE_PER_FRAME
                                Angle per frame [deg]
        image_nr_start=IMAGE_NR_START
                                Start image number [int]
        position=POSITION
                        Gonio alignment position [dict]                       
        detector_distance=DETECTOR_DISTANCE
                                Detector distance
        resolution=RESOLUTION
                                Resolution [Angstroem]
        transmission=TRANSMISSION
                                Transmission. Number in range between 0 and 1.
        flux=FLUX  Flux [ph/s]'''
    def __init__(self, 
                 name_pattern='test_$id', 
                 directory='/tmp', 
                 scan_range=180, 
                 scan_exposure_time=18, 
                 scan_start_angle=0, 
                 angle_per_frame=0.1, 
                 image_nr_start=1,
                 position=None, 
                 photon_energy=None,
                 resolution=None,
                 detector_distance=None,
                 transmission=None,
                 flux=None):
        
        experiment.__init__(self, name_pattern=name_pattern, directory=directory)
        # Scan parameters
        
        self.scan_range = float(scan_range)
        self.scan_exposure_time = float(scan_exposure_time)
        self.scan_start_angle = float(scan_start_angle)
        self.angle_per_frame = float(angle_per_frame)
        self.image_nr_start = int(image_nr_start)
        if position != None and not isinstance(position, dict):
            # This is a workaround allowing to supply position dictionary from the command line
            position = position.strip('}{')
            positions = position.split(',')
            keyvalues = [item.strip().split(':') for item in positions]
            keyvalues = [(item[0], float(item[1])) for item in keyvalues]
            self.position = dict(keyvalues)
        print 'self.position', self.position
        self.photon_energy = photon_energy
        self.resolution = resolution
        self.detector_distance = detector_distance
        self.transmission = transmission
        
        # Necessary equipment
        self.goniometer = goniometer()
        try:
            self.beam_center = beam_center()
        except:
            from beam_center import beam_center_mockup
            self.beam_center = beam_center_mockup()
        try:
            self.detector = detector()
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
            
        # private
        self._ntrigger = 1

    def get_nimages(self, epsilon=1e-3):
        nimages = int(self.scan_range/self.angle_per_frame)
        if abs(nimages*self.angle_per_frame - self.scan_range) > epsilon:
            nimages += 1
        return nimages

    def get_scan_speed(self):
        return self.scan_range/self.scan_exposure_time
    
    def get_frame_time(self):
        return self.scan_exposure_time/self.get_nimages()

    def get_position(self):
        if self.position is None:
            return self.goniometer.get_position()
        else:
            return self.position

    def set_position(self, position=None):
        if position is None:
            self.position = self.goniometer.get_position()
        else:
            self.position = position
            self.goniometer.set_position(self.position)
            self.goniometer.wait()
        self.goniometer.save_position()

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
        self.detector.set_ntrigger(self._ntrigger)
        self.detector.set_standard_parameters()
        self.detector.clear_monitor()
        self.detector.set_name_pattern(self.get_full_name_pattern())
        self.detector.set_frame_time(self.get_frame_time())
        count_time = self.get_frame_time() - self.detector.get_detector_readout_time()
        self.detector.set_count_time(count_time)
        self.detector.set_nimages(self.nimages)
        self.detector.set_omega(self.scan_start_angle)
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
        self.nimages = self.get_nimages()
        self.goniometer.set_scan_start_angle(self.scan_start_angle)
        self.goniometer.set_scan_range(self.scan_range)
        self.goniometer.set_scan_exposure_time(self.scan_exposure_time)
        self.goniometer.set_scan_number_of_frames(1)
        self.goniometer.set_detector_gate_pulse_enabled(True)
        self.goniometer.set_data_collection_phase()

    def prepare(self):
        self.check_directory()
        
        self.program_goniometer()
        
        self.set_photon_energy(self.photon_energy)
        self.set_resolution(self.resolution)
        self.set_transmission(self.transmission)
        self.protective_cover.extract()
        
        self.resolution_motor.wait()
        self.energy_motor.wait()
        self.goniometer.wait()

        self.program_detector()
        if self.goniometer.backlight_is_on():
            self.goniometer.remove_backlight()

        self.energy_motor.turn_off()

    def execute(self):
        self.prepare()
        self.run()
        self.clean()

    def collect(self):
        return self.run()
    def measure(self):
        return self.run()
    def run(self):
        self.goniometer.set_position(self.get_position())
        return self.goniometer.omega_scan(self.scan_start_angle, self.scan_range, self.scan_exposure_time, wait=True)

    def clean(self):
        self.detector.disarm()

    def stop(self):
        self.goniometer.abort()
        self.detector.abort()

def main():
    import optparse
        
    parser = optparse.OptionParser()
    parser.add_option('-n', '--name_pattern', default='test_$id', type=str, help='Prefix default=%default')
    parser.add_option('-d', '--directory', default='/nfs/data/default', type=str, help='Destination directory default=%default')
    
    parser.add_option('-r', '--scan_range', default=180, type=float, help='Scan range [deg]')
    parser.add_option('-e', '--scan_exposure_time', default=18, type=float, help='Scan exposure time [s]')
    parser.add_option('-s', '--scan_start_angle', default=0, type=float, help='Scan start angle [deg]')
    parser.add_option('-a', '--angle_per_frame', default=0.1, type=float, help='Angle per frame [deg]')
    parser.add_option('-f', '--image_nr_start', default=1, type=int, help='Start image number [int]')
    parser.add_option('-i', '--position', default=None, type=str, help='Gonio alignment position [dict]')
    parser.add_option('-p', '--photon_energy', default=None, type=float, help='Photon energy ')
    parser.add_option('-t', '--detector_distance', default=None, type=float, help='Detector distance')
    parser.add_option('-o', '--resolution', default=None, type=float, help='Resolution [Angstroem]')
    parser.add_option('-x', '--flux', default=None, type=float, help='Flux [ph/s]')
    parser.add_option('-m', '--transmission', default=None, type=float, help='Transmission. Number in range between 0 and 1.')
    
    options, args = parser.parse_args()
    print 'options', options
    s = scan(**vars(options))
    s.execute()
    
def test():
    scan_range = 180
    scan_exposure_time = 18.
    scan_start_angle = 0
    angle_per_frame = 0.1
    
    s = scan(scan_range=scan_range, scan_exposure_time=scan_exposure_time, scan_start_angle=scan_start_angle, angle_per_frame=angle_per_frame)
    
if __name__ == '__main__':
    main()