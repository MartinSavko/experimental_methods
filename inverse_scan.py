#!/usr/bin/env python
'''
helical scan
'''
import traceback
import logging
import time

from scan import scan, check_position
from area import area

class inverse_scan(scan):
    def __init__(self, 
                 name_pattern='test_$id', 
                 directory='/tmp', 
                 scan_range=180, 
                 scan_exposure_time=18, 
                 scan_start_angle=0, 
                 angle_per_frame=0.1, 
                 image_nr_start=1,
                 interleave_range=5,
                 raster=False,
                 position=None,
                 photon_energy=None,
                 resolution=None,
                 detector_distance=None,
                 transmission=None,
                 flux=None): 
        
        scan.__init__(self,
                      name_pattern=name_pattern, 
                      directory=directory,
                      scan_range=scan_range, 
                      scan_exposure_time=scan_exposure_time, 
                      scan_start_angle=scan_start_angle,
                      angle_per_frame=angle_per_frame, 
                      image_nr_start=image_nr_start,
                      photon_energy=photon_energy,
                      resolution=resolution,
                      detector_distance=detector_distance,
                      transmission=transmission,
                      flux=flux)
        
        self.raster = raster
        self.interleave_range = float(interleave_range)
        self.n_wedges = self.scan_range/self.interleave_range
        self.area = area(rows=self.n_wedges, columns=2)
        
    def get_wedges(self):
        direct = np.arange(self.start_scan_angle, self.start_scan_angle+self.scan_range, self.interleave_range)
        inverse = direct + 180
        wedges = np.vstack((direct, inverse)).T
        return wedges.ravel()
    
    def run(self):
        self.goniometer.set_position(self.get_position())
        self.wedges = self.get_wedges()
        scan_range = self.interleave_range
        scan_exposure_time = self.interleave_range * self.scan_exposure_time / self.scan_range 
        for start_scan_angle in self.wedges:
            self.goniometer.omega_scan(scan_start_angle, scan_range, scan_exposure_time, wait=True)
     
    def get_nimages_per_file(self):
        return int(self.interleave_range/self.angle_per_frame)
    
    def get_nimages(self):
        return int(self.interleave_range/self.angle_per_frame)
    
    def get_ntrigger(self):
        return int(self.scan_range/self.interleave_range) * 2
    
    def save_parameters(self):
        parameters = {}
        
        parameters['timestamp'] = self.timestamp
        parameters['name_pattern'] = self.name_pattern
        parameters['directory'] = self.directory
        parameters['scan_range'] = self.scan_range
        parameters['scan_exposure_time'] = self.scan_exposure_time
        parameters['scan_start_angle'] = self.scan_start_angle
        parameters['interleave_range'] = self.interleave_range
        parameters['wedges'] = self.wedges
        parameters['image_nr_start'] = self.image_nr_start
        parameters['frame_time'] = self.get_frame_time()
        parameters['vertical_step_size'], parameters['horizontal_step_size'] = self.get_step_sizes()
        parameters['position_start'] = self.position_start
        parameters['position_end'] = self.position_end
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
    parser.add_option('-I', '--interleave_range', default=5, type=float, help='Interleave range [deg]')
    parser.add_option('-S', '--position_start', default=None, type=str, help='Gonio alignment start position [dict]')
    parser.add_option('-E', '--position_end', default=None, type=str, help='Gonio alignment end position [dict]')
    parser.add_option('-p', '--photon_energy', default=None, type=float, help='Photon energy ')
    parser.add_option('-t', '--detector_distance', default=None, type=float, help='Detector distance')
    parser.add_option('-o', '--resolution', default=None, type=float, help='Resolution [Angstroem]')
    parser.add_option('-x', '--flux', default=None, type=float, help='Flux [ph/s]')
    parser.add_option('-m', '--transmission', default=None, type=float, help='Transmission. Number in range between 0 and 1.')
    
    options, args = parser.parse_args()
    print 'options', options
    s = inverse_scan(**vars(options))
    s.execute()
    
def test():
    scan_range = 180
    scan_exposure_time = 18.
    scan_start_angle = 0
    angle_per_frame = 0.1
    interleave_range = 5
    s = inverse_scan(scan_range=scan_range, scan_exposure_time=scan_exposure_time, scan_start_angle=scan_start_angle, angle_per_frame=angle_per_frame, interleave_range=interleave_range)
    
if __name__ == '__main__':
    main()