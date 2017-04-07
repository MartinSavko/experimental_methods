#!/usr/bin/env python
'''
single position oscillation scan
'''
import traceback
import logging
import time

from diffraction_experiment import diffraction_experiment
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

def check_position(position):
    if position != None and not isinstance(position, dict):
        position = position.strip('}{')
        positions = position.split(',')
        keyvalues = [item.strip().split(':') for item in positions]
        keyvalues = [(item[0], float(item[1])) for item in keyvalues]
        return dict(keyvalues)
    else:
        return position
    
class scan(diffraction_experiment):
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
                 name_pattern, 
                 directory, 
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
                 flux=None,
                 snapshot=False,
                 ntrigger=1,
                 nimages_per_file=100):
        
        diffraction_experiment.__init__(self, 
                                        name_pattern, 
                                        directory,
                                        photon_energy=None,
                                        resolution=None,
                                        detector_distance=None,
                                        transmission=None,
                                        flux=None,
                                        snapshot=False)
        # Scan parameters
        
        self.scan_range = float(scan_range)
        self.scan_exposure_time = float(scan_exposure_time)
        self.scan_start_angle = float(scan_start_angle)
        self.angle_per_frame = float(angle_per_frame)
        self.image_nr_start = int(image_nr_start)
        self.position = check_position(position)
        print 'self.position', self.position
            
        self.ntrigger = ntrigger
        self.nimages_per_file = nimages_per_file
        
    def get_nimages(self, epsilon=1e-3):
        nimages = int(self.scan_range/self.angle_per_frame)
        if abs(nimages*self.angle_per_frame - self.scan_range) > epsilon:
            nimages += 1
        return nimages
    
    def get_nimages_per_file(self):
        return self.nimages_per_file
    
    def get_ntrigger(self):
        return self.ntrigger
    
    def get_dpf(self):
        return self.angle_per_frame
    
    def get_fps(self):
        return self.get_nimages()/self.scan_exposure_time
    
    def get_dps(self):
        return self.get_scan_speed()
    
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

    def run(self):
        self.goniometer.set_position(self.get_position())
        return self.goniometer.omega_scan(self.scan_start_angle, self.scan_range, self.scan_exposure_time, wait=True)

    def analyze(self):
        process_line = 'ssh process1 "cd {directory:s}; xdsme -i "LIB=/nfs/data/plugin.so" ../{name_pattern:s}_master.h5"'.format(**{'directory': os.path.join(self.directory, 'process'), 'name_pattern': self.name_pattern})
        os.system(process_line)
        
    def save_parameters(self):
        parameters = {}
        
        parameters['timestamp'] = self.timestamp
        parameters['name_pattern'] = self.name_pattern
        parameters['directory'] = self.directory
        parameters['scan_range'] = self.scan_range
        parameters['scan_exposure_time'] = self.scan_exposure_time
        parameters['scan_start_angle'] = self.scan_start_angle
        parameters['image_nr_start'] = self.image_nr_start
        parameters['frame_time'] = self.get_frame_time()
        parameters['vertical_step_size'], parameters['horizontal_step_size'] = self.get_step_sizes()
        parameters['reference_position'] = self.reference_position
        parameters['beam_position_vertical'] = self.camera.md2.beampositionvertical
        parameters['beam_position_horizontal'] = self.camera.md2.beampositionhorizontal
        parameters['nimages'] = self.get_nimages()
        parameters['image'] = self.image
        parameters['rgb_image'] = self.rgbimage.reshape((self.image.shape[0], self.image.shape[1], 3))
        parameters['camera_calibration_horizontal'] = self.camera.get_horizontal_calibration()
        parameters['camera_calibration_vertical'] = self.camera.get_vertical_calibration()
        parameters['camera_zoom'] = self.camera.get_zoom()
        parameters['duration'] = self.end_time - self.start_time
        parameters['start_time'] = self.start_time
        parameters['end_time'] = self.end_time
        parameters['photon_energy'] = self.photon_energy
        parameters['transmission'] = self.transmission
        parameters['detector_distance'] = self.detector_distance
        parameters['resolution'] = self.resolution
        
        scipy.misc.imsave(os.path.join(self.directory, '%s_optical_bw.png' % self.name_pattern), self.image)
        scipy.misc.imsave(os.path.join(self.directory, '%s_optical_rgb.png' % self.name_pattern), self.rgbimage.reshape((self.image.shape[0], self.image.shape[1], 3)))
        
        f = open(os.path.join(self.directory, '%s_parameters.pickle' % self.name_pattern), 'w')
        pickle.dump(parameters, f)
        f.close()

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