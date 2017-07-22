#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import numpy as np
import scipy
import os
import pickle

from xray_experiment import xray_experiment
from fluorescence_detector import fluorescence_detector
from goniometer import goniometer
from fast_shutter import fast_shutter
from safety_shutter import safety_shutter
from transmission import transmission as transmission_motor

class fluorescence_spectrum(xray_experiment):

    def __init__(self,
                 name_pattern,
                 directory,
                 integration_time=5,
                 transmission=0.5,
                 insertion_timeout=2,
                 position=None,
                 photon_energy=None,
                 flux=None,
                 snapshot=False,
                 zoom=None,
                 analysis=None,
                 display=False):
        
        xray_experiment.__init__(self, 
                                name_pattern, 
                                directory,
                                position=position,
                                photon_energy=photon_energy,
                                transmission=transmission,
                                flux=flux,
                                snapshot=snapshot,
                                zoom=zoom,
                                analysis=analysis)
        
        self.description = 'XRF spectrum, Proxima 2A, SOLEIL, %s' % time.ctime(self.timestamp)
        self.detector = fluorescence_detector()
        self.fast_shutter = fast_shutter()
        
        self.integration_time = integration_time
        self.transmission = transmission
        self.insertion_timeout = insertion_timeout
        self.display = display
        
    def prepare(self):
        _start = time.time()
        print 'prepare'
        self.check_directory(self.directory)
        
        if self.snapshot == True:
            print 'taking image'
            self.camera.set_exposure(0.05)
            self.camera.set_zoom(self.zoom)
            self.goniometer.insert_backlight()
            self.goniometer.extract_frontlight()
            self.goniometer.set_position(self.reference_position)
            self.goniometer.wait()
            self.image = self.camera.get_image()
            self.rgbimage = self.camera.get_rgbimage()
       
        self.detector.insert()
        self.goniometer.set_data_collection_phase(wait=False)
        self.set_photon_energy(self.photon_energy, wait=True)
        self.set_transmission(self.transmission)
        
        self.detector.set_integration_time(self.integration_time)
        
        if self.safety_shutter.closed():
            self.safety_shutter.open()
       
        if self.position != None:
            self.goniometer.set_position(self.position)
        else:
            self.position = self.goniometer.get_position()
        
        self.write_destination_namepattern(self.directory, self.name_pattern)
        self.energy_motor.turn_off()
        self.goniometer.wait()
        while time.time() - _start < self.insertion_timeout:
            time.sleep(self.detector.sleeptime)
        
    def run(self):
        print 'run'
        self.fast_shutter.open()
        self.spectrum = self.detector.get_point()
        self.fast_shutter.close()
        
    def clean(self):
        print 'clean'
        self.detector.extract()
        self.end_time = time.time()
        self.save_spectrum()
        self.save_parameters()
        self.save_log()
        self.save_plot()
        
    def stop(self):
        self.fast_shutter.close()
        self.detector.stop()
    
    def abort(self):
        self.fast_shutter.md2.abort()
        self.stop()
        
    def analyze(self):
        #element analysis
        pass
    
    def get_channels(self):
        channels = np.arange(0, 2048)
        return channels
        
    def get_energies(self):
        '''return energies in eV'''
        a, b, c = self.detector.get_calibration()
        channels = self.get_channels()
        energies = channels + a + b*channels + c*channels**2
        return energies
    
    def save_spectrum(self):
        filename = os.path.join(self.directory, '%s.dat' % self.name_pattern)
        self.energies = self.get_energies()
        self.channels = self.get_channels()
        X = np.array(zip(self.channels, self.spectrum, self.energies))
        self.header ='#F %s\n#D %s\n#N %d\n#L channel  counts  energy\n' % (filename, time.ctime(self.timestamp), X.shape[1])
        
        if scipy.__version__ > '1.7.0':
            scipy.savetxt(filename, X, header=self.header)
        else:
            f = open(filename, 'a')
            f.write(self.header)
            scipy.savetxt(f, X)
            f.close()
        
    def save_plot(self):
        import pylab
        pylab.figure(figsize=(16, 9))
        pylab.title(self.description)
        pylab.xlabel('energy [eV]')
        pylab.ylabel('intensity [a.u.]')
        pylab.plot(self.energies, self.spectrum)
        pylab.xlim([0, 20480])
        pylab.savefig(os.path.join(self.directory, '%s.png' % self.name_pattern))
        if self.display:
            pylab.show()
                      
    def save_parameters(self):
        self.parameters = {}
        self.parameters['description'] = self.description
        self.parameters['nchannels'] = len(self.get_channels())
        self.parameters['timestamp'] = self.timestamp
        self.parameters['name_pattern'] = self.name_pattern
        self.parameters['directory'] = self.directory
        self.parameters['integration_time'] = self.integration_time
        self.parameters['position'] = self.position
        self.parameters['start'] = self.start_time
        self.parameters['end'] = self.end_time
        self.parameters['duration'] = self.end_time - self.start_time
        self.parameters['calibration'] = self.detector.get_calibration()
        self.parameters['energies'] = self.energies   
        self.parameters['spectrum'] = self.spectrum
        self.parameters['transmission'] = self.transmission
        self.parameters['photon_energy'] = self.photon_energy
        self.parameters['dead_time'] = self.detector.get_dead_time()
        self.parameters['real_count_time'] = self.detector.get_real_time()
        self.parameters['input_count_rate'] = self.detector.get_input_count_rate()
        self.parameters['output_count_rate'] = self.detector.get_output_count_rate()
        self.parameters['calculated_dead_time'] = self.detector.get_calculated_dead_time()
        self.parameters['events_in_run'] = self.detector.get_events_in_run()
        
        f = open(os.path.join(self.directory, '%s_parameters.pickle' % self.name_pattern), 'w')
        pickle.dump(self.parameters, f)
        f.close()
    
    def save_log(self):
        '''method to save the experiment details in the log file'''
        f = open(os.path.join(self.directory, '%s.log' % self.name_pattern), 'w')
        keyvalues = self.parameters.items()
        keyvalues.sort()
        for key, value in keyvalues:
            if key not in ['spectrum' , 'energies']:
                f.write('%s: %s\n' % (key, value)) 
        f.close()
        
def main():
    import optparse
    
    parser = optparse.OptionParser()
    
    parser.add_option('-d', '--directory', default='/tmp', type=str, help='directory (default=%default)')
    parser.add_option('-n', '--name_pattern', default='xrf', type=str, help='name_pattern (default=%default)')
    parser.add_option('-i', '--integration_time', default=5, type=float, help='integration_time (default=%default s)')
    parser.add_option('-t', '--transmission', default=0.5, type=float, help='transmission (default=%default %)')
    parser.add_option('-p', '--photon_energy', default=14000, type=float, help='transmission (default=%default eV)')
    parser.add_option('-D', '--display', action='store_true', help='Display the plot')
    
    options, args = parser.parse_args()
    
    fs = fluorescence_spectrum(options.name_pattern, 
                               options.directory, 
                               integration_time=options.integration_time, 
                               photon_energy=options.photon_energy,
                               transmission=options.transmission,
                               display=options.display)
    fs.execute()
    
if __name__ == '__main__':
    main()