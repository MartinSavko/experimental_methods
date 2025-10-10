#!/usr/bin/env python
# -*- coding: utf-8 -*-
import gevent

import traceback
import time
import pickle
import os
import pylab
import scipy

from xabs_lib import McMaster
from xray_experiment import xray_experiment
from fluorescence_detector import fluorescence_detector as detector
from motor_scan import motor_scan
from motor import tango_motor
from monitor import xbpm


class monochromator_scan(xray_experiment):
    
    def __init__(self,
                 name_pattern,
                 directory,
                 element,
                 edge,
                 scan_range=100, #eV
                 scan_speed=1, #eV/s
                 integration_time=0.25,
                 transmission=0.5,
                 insertion_timeout=2,
                 position=None,
                 photon_energy=None,
                 flux=None,
                 snapshot=False,
                 zoom=None,
                 analyze=True,
                 display=False,
                 optimize=False,
                 roi_width=250.,
                 mono_rx_motor_name='i11-ma-c03/op/mono1-mt_rx'): #eV
                 
        xray_experiment.__init__(self, 
                                name_pattern, 
                                directory,
                                position=position,
                                photon_energy=photon_energy,
                                resolution=resolution,
                                detector_distance=detector_distance,
                                detector_vertical=detector_vertical,
                                detector_horizontal=detector_horizontal,
                                transmission=transmission,
                                flux=flux,
                                snapshot=snapshot,
                                zoom=zoom,
                                analysis=analysis)
        
        self.element = element
        self.edge = edge
        self.scan_range = scan_range
        self.scan_speed = scan_speed
        self.integration_time = integration_time
        self.insertion_timeout = insertion_timeout
        self.optimize = optimize
        self.display = display
        self.roi_width = roi_width
        self.detector = detector()
        self.mono_rx_motor = tango_motor(mono_rx_motor_name)
        self.monitor_names = ['mca', 'xbpm1', 'cvd1', 'psd6']
        self.monitors = [self.detector,
                         xbpm('i11-ma-c04/dt/xbpm_diode.1'),
                         xbpm('i11-ma-c05/dt/xbpm-cvd.1'),
                         xbpm('i11-ma-c06/dt/xbpm_diode.6')]
        
        
    def measure_fluorescence(self):
        self.fast_shutter.open()
        self.detector.get_point()
        self.fast_shutter.close()
        
    def get_edge_energy(self):
        return McMaster[self.element]['edgeEnergies'][self.edge.upper()]
    
    def get_alpha_energy(self):
        return McMaster[self.element]['edgeEnergies']['%s-alpha' % self.edge.upper()[0]]
   
    def channel_from_energy(self, energy):
        a, b, c = self.detector.get_calibration()
        return (energy - a)/b

    def energy_from_channel(self, channel):
        return a + b*channel + c*channel**2
        
    def set_roi(self):
        self.alpha_energy = self.get_alpha_energy()
        roi_center = self.alpha_energy * 1.e3
        roi_start = self.channel_from_energy(roi_center - roi_width/2.)
        roi_end = self.channel_from_energy(roi_center + roi_width/2.)
        self.detector.set_roi(roi_start, roi_end)
        
    def adjust_transmission(self):
        if self.detector.get_dead_time() > 40:
            self.high_boundary = self.current_transmission
            self.new_transmission -= (self.high_boundary - self.low_boundary)/2.
        else:
            self.low_boundary = self.current_transmission
            self.new_transmission += (self.high_boundary - self.low_boundary)/2.
        self.current_transmission = self.new_transmission
        self.set_transmission(self.new_transmission)
        
    def optimize_transmission(self):
        self.current_transmission = self.transmission
        self.low_boundary = 0
        self.high_boundary = None
        
        k=0
        self.measure_fluorescence()
        while self.detector.get_dead_time() < 20 and or self.detector.get_dead_time() > 40:
            self.adjust_transmission()
            if self.get_transmission() > 50:
                break
            self.measure_fluorescence()
            k += 1
           
        print('Transmission optimized after %d steps to %.2f' % (k, self.current_transmission))
        
    def prepare(self):
        _start = time.time()
        print('prepare')
        
        self.check_directory(self.directory)
        self.write_destination_namepattern(self.directory, self.name_pattern)
        self.set_transmission(self.transmission)
            
        if self.snapshot == True:
            print('taking image')
            self.camera.set_exposure(0.05)
            self.camera.set_zoom(self.zoom)
            self.goniometer.insert_backlight()
            self.goniometer.extract_frontlight()
            self.goniometer.set_position(self.reference_position)
            self.goniometer.wait()
            self.image = self.camera.get_image()
            self.rgbimage = self.camera.get_rgbimage()
       
        if self.safety_shutter.closed():
           self.safety_shutter.open()
        self.goniometer.set_data_collection_phase(wait=True)
        self.detector.insert()
        self.detector.set_integration_time(self.integration_time)
        self.detector.set_roi()
        
        while time.time() - _start < self.insertion_timeout:
            gevent.sleep(self.detector.sleeptime)
        
        if self.optimize == True:
            edge_energy = self.get_edge_energy()
            self.optimize_at_energy = edge_energy + 0.010
            print('optimizing transmission at %.3f keV' % self.optimize_at_energy)
            self.set_photon_energy(self.optimize_at_energy, wait=True)
            self.optimize_transmission()
        else:
            self.set_transmission(self.transmission)
        
        self.start_energy = self.get_edge_energy - self.scan_range/2.
        self.end_energy = self.start_energy + self.scan_range
        
        self.energy_motor.mono.simenergy = self.start_energy
        angle_start = self.energy_motor.mono.simthetabragg
        self.energy_motor.mono.simenergy = self.end_energy
        angle_end = self.energy_motor.mono.simthetabragg
        self.scan_speed = abs(angle_end - angle_start)/60.
        print('scan_speed', self.scan_speed)
        
        print('moving to start energy %.3f' % self.start_energy)
        self.set_photon_energy(self.start_energy, wait=True)
       
        if self.position != None:
            self.goniometer.set_position(self.position)
        else:
            self.position = self.goniometer.get_position()
        
        for monitor in self.monitors:
            monitor.observe = True
            
        self.energy_motor.turn_on()
        self.mono_rx_motor.set_speed(self.scan_speed)

    def actuator_monitor(self, start_time):
        self.observations = []
        self.observation_fields = ['chronos', 'energy', 'thetabragg', 'wavelength']
        
        while self.mono_rx_motor.get_state() != 'STANDBY':
            chronos = time.time() - start_time
            point = [chronos, self.energy_motor.mono.energy, self.energy_motor.mono.thetabragg, self.energy_motor.mono.Lambda]
            self.observations.append(point)
            gevent.sleep()
            
        for monitor in self.monitors:
            monitor.observe = False
    
    def get_observations(self):
        return self.observations
        
    def get_observation_fields(self):
        return self.observation_fields

    def get_all_observations(self):
        all_observations = {}
        all_observations['actuator_monitor'] = {}
        actuator_observation_fields = self.get_observation_fields()
        all_observations['actuator_monitor']['observation_fields'] = actuator_observation_fields
        actuator_observations = self.get_observations()
        all_obsrevations['actuator_monitor']['observations'] = actuator_observations
        
        X = np.array(actuator_observations)
        chronos, thetabragg = X[:, 0], X[:, 2], 1)
        z = np.polyfit(chronos, thetabragg) 
        theta_chronos_predictor = np.poly1d(z)
        
        for monitor_name, monitor in zip(self.monitor_names, self.monitors):
            all_observations[monitor_name] = {}
            all_observations[monitor_name]['observation_fields'] = monitor.get_observation_fields()
            all_observations[monitor_name]['observations'] = monitor.get_observations()
        
        mca_observations = np.array(all_observations['mca']['observations'])
        mca_chronos = X[:, 0]
        mca_theta = theta_chronos_predictor(mca_chronos)
        mca_wavelengths = self.resolution_motor.get_wavelength_from_theta(mca_theta)
        mca_energies = self.resolution_motor.get_energy_from_wavelength(mca_wavelengths)
        
        mca_normalized_counts = X[:, 3]
        
        equidistant_energies = np.linspace(mca_energies.min(), mca_energies.max(), 200)
        
        equidistant_mca_normalized_counts = np.interp(equidistant_energies, mca_energies, mca_normalized_counts)
        
        all_observations['energies'] = equidistant_energies
        all_observations['counts'] =   equidistant_mca_normalized_counts
        self.all_observations = all_observations
        return all_obsrevations
        
    def run(self):
        self.fast_shutter.open()
        self.energy_motor.mono.energy = self.end_energy
        observers = [self.actuator_monitor()]
        for monitor in self.monitors:
            observers.append(gevent.spawn(monitor.monitor()))
            
        gevent.joinall(observers)
        
        self.fast_shutter.close()
        for monitor in self.monitors:
            monitor.observe = False
    
    def clean(self):
        print('clean')
        self.end_time = time.time()
        self.detector.extract()
        self.mono_rx_motor.set_speed(0.5)
        self.save_parameters()
        self.save_raw_results()
        self.save_raw_scan()
        self.save_log()
        self.save_plot()
    
    def parse_chooch_output(self, output):
        logging.info('parse_chooch_output')
        table = output[output.find('Table of results'):]
        tabl = table.split('\n')
        tab = numpy.array([ line.split('|') for line in tabl if line and line[0] == '|'])
        print('tab', tab)
        self.pk = float(tab[1][2])
        self.fppPeak = float(tab[1][3])
        self.fpPeak = float(tab[1][4])
        self.ip = float(tab[2][2])
        self.fppInfl = float(tab[2][3])
        self.fpInfl = float(tab[2][4])
        self.efs = self.getEfs()
        return {'pk': self.pk, 'fppPeak': self.fppPeak, 'fpPeak': self.fpPeak, 'ip': self.ip, 'fppInfl': self.fppInfl, 'fpInfl': self.fpInfl, 'efs': self.efs}
        
    def analyze(self):
        import subprocess
        self.results = {}
        chooch_parameters = {'element': self.element, 
                             'edge': self.edge,
                             'raw_file': self.raw_filename,
                             'output_ps': self.raw_filename.replace('.raw', '.ps'),
                             'output_efs': self.raw_filename.replace('.raw', '.efs')}
        
        chooch_command = 'chooch -p {output_ps} -o {output_efs} -e {element} -a {edge} {raw_file}'.format(**chooch_parameters)
        print('chooch command %s' % chooch_command)
       
        chooch_output = subprocess.getoutput(chooch_cmd)
        self.results['chooch_output'] = chooch_output
        print('chooch_output', chooch_output)
        chooch_results = self.parse_chooch_output(chooch_output)
        self.results['chooch_results'] = chooch_results
        
        f = open(os.path.join(self.directory, '%s_chooch_results.pickle' % self.name_pattern), 'wb')
        pickle.dump(self.results, f)
        f.close()
        
                               
    def stop(self):
        self.mono_rx_motor.stop()
        self.fast_shutter.close()
        self.mono_rx_motor.set_speed(0.5)
        
    def save_parameters(self):
        self.parameters = {}
        self.parameters['description'] = self.description
        self.parameters['element'] = self.element
        self.parameters['edge'] = self.edge
        self.parameters['scan_range'] = self.scan_range
        self.parameters['scan_speed'] = self.scan_speed
        self.parameters['roi_width'] = self.roi_width
        self.parameters['timestamp'] = self.timestamp
        self.parameters['name_pattern'] = self.name_pattern
        self.parameters['directory'] = self.directory
        self.parameters['integration_time'] = self.integration_time
        self.parameters['position'] = self.position
        self.parameters['start'] = self.start_time
        self.parameters['end'] = self.end_time
        self.parameters['duration'] = self.end_time - self.start_time
        self.parameters['calibration'] = self.detector.get_calibration()
        self.parameters['transmission'] = self.transmission
        self.parameters['photon_energy'] = self.photon_energy
        self.parameters['optimize'] = self.optimize
        if self.optimize == True:
            self.parameters['current_transmission'] = self.current_transmission
        if self.snapshot == True:
            self.parameters['camera_zoom'] = self.camera.get_zoom()
            self.parameters['camera_calibration_horizontal'] = self.camera.get_horizontal_calibration()
            self.parameters['camera_calibration_vertical'] = self.camera.get_vertical_calibration()
            self.parameters['beam_position_vertical'] = self.camera.md.beampositionvertical
            self.parameters['beam_position_horizontal'] = self.camera.md.beampositionhorizontal
            self.parameters['image'] = self.image
            self.parameters['rgb_image'] = self.rgbimage.reshape((self.image.shape[0], self.image.shape[1], 3))
            scipy.misc.imsave(os.path.join(self.directory, '%s_optical_bw.png' % self.name_pattern), self.image)
            scipy.misc.imsave(os.path.join(self.directory, '%s_optical_rgb.png' % self.name_pattern), self.rgbimage.reshape((self.image.shape[0], self.image.shape[1], 3)))
        
        f = open(os.path.join(self.directory, '%s_parameters.pickle' % self.name_pattern), 'wb')
        pickle.dump(self.parameters, f)
        f.close()
        
    def save_raw_results(self):
        f = open(os.path.join(self.directory, '%s_complete_results.pickle' % self.name_pattern), 'wb')
        pickle.dump(self.get_all_observations(), f)
        f.close()
               
    def save_raw_scan(self):
        self.raw_filename = os.path.join(self.directory, '%s.raw' % self.name_pattern)
        energies = self.all_observations['energies']
        counts = self.all_observations['counts']
        X = np.vstack([energies, counts]).T
        self.header = '%s\n%d\n' % (self.description, X.shape[0])
        
        if scipy.__version__ > '1.7.0':
            scipy.savetxt(self.raw_filename, X, header=self.header)
        else:
            f = open(self.raw_filename, 'a')
            f.write(self.header)
            scipy.savetxt(f, X)
            f.close()
        
    def save_plot(self):
        pylab.figure(figsize=(16, 9))
        pylab.plot(self.energies, self.counts)
        pylab.xlabel('energy [eV]')
        pylab.ylabel('normalized counts')
        pylab.title(self.description)
        pylab.savefig(os.path.join(self.directory, '%s_%s_%s_edge.png' % (self.name_pattern, self.element, self.edge))
        
    def save_log(self):
        '''method to save the experiment details in the log file'''
        f = open(os.path.join(self.directory, '%s.log' % self.name_pattern), 'w')
        keyvalues = self.parameters.items()
        keyvalues.sort()
        for key, value in keyvalues:
            if key not in ['spectrum' , 'energies', 'image', 'rgb_image']:
                f.write('%s: %s\n' % (key, value)) 
        f.close()
        
        
        
def main():
    usage = '''Program for energy scans
    
    ./Xanes.py -e <element> -s <edge> <options>
    
    '''
    
    import optparse
    
    parser = optparse.OptionParser(usage=usage)
        
    parser.add_option('-e', '--element', type=str, help='Specify the element')
    parser.add_option('-s', '--edge', type=str, help='Specify the edge')
    parser.add_option('-d', '--directory', type=str, default='/tmp/testXanes', help='Directory to store the results (default=%default)')
    parser.add_option('-n', '--name_pattern', type=str, default='escan', help='name_pattern')
    parser.add_option('-i', '--integration_time', type=float, default=0.1, help='integration time (default=%default)')
    
    options, args = parser.parse_args()
    
    print('options', options)
    print('args', args)
    
    energy_scan = energy_scan(options.name_pattern,
                              options.directory,
                              options.element,
                              options.edge,
                              options.integration_time)
              
    x.execute()
              
    
if __name__ == '__main__':
    main()
    
