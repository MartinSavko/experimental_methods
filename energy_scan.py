#!/usr/bin/env python
import gevent
from gevent.monkey import patch_all
patch_all()

import traceback
import logging
import time
import pickle
import os
import pylab
import numpy as np
import scipy

from xabs_lib import McMaster
from xray_experiment import xray_experiment
from fluorescence_detector import fluorescence_detector as detector
from motor_scan import motor_scan
from motor import tango_motor
from monitor import xbpm


class energy_scan(xray_experiment):
    
    def __init__(self,
                 name_pattern,
                 directory,
                 element,
                 edge,
                 scan_range=100, #eV
                 scan_speed=1, #eV/s
                 integration_time=0.25,
                 total_time=60.,
                 transmission=0.5,
                 insertion_timeout=2,
                 position=None,
                 photon_energy=None,
                 flux=None,
                 snapshot=False,
                 zoom=None,
                 analysis=True,
                 display=False,
                 optimize=True,
                 roi_width=250.,
                 mono_rx_motor_name='i11-ma-c03/op/mono1-mt_rx'): #eV
                 
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
        
        self.description = 'ESCAN, Proxima 2A, SOLEIL, element %s, edge %s, %s' % (element, edge, time.ctime(self.timestamp))
        self.element = element
        self.edge = edge
        self.scan_range = scan_range
        self.scan_speed = scan_speed
        self.integration_time = integration_time
        self.total_time = total_time
        self.insertion_timeout = insertion_timeout
        self.optimize = optimize
        self.display = display
        self.roi_width = roi_width
        self.detector = detector()
        
        self.mono_rx_motor = tango_motor(mono_rx_motor_name)
        
        self.monitor_names = ['mca'] + self.monitor_names
        self.monitors = [self.detector] + self.monitors
        
        self.default_mono_rx_motor_speed = 0.5
        self.monitor_sleep_time = 0.05
        
    def measure_fluorescence(self):
        self.fast_shutter.open()
        self.detector.get_point()
        self.fast_shutter.close()
        
    def get_edge_energy(self):
        return McMaster[self.element]['edgeEnergies'][self.edge.upper()] * 1e3
    
    def get_alpha_energy(self):
        return McMaster[self.element]['edgeEnergies']['%s-alpha' % self.edge.upper()[0]] * 1e3
   
    def channel_from_energy(self, energy):
        a, b, c = self.detector.get_calibration()
        return (energy - a)/b

    def energy_from_channel(self, channel):
        return a + b*channel + c*channel**2
        
    def set_roi(self):
        self.alpha_energy = self.get_alpha_energy()
        roi_center = self.alpha_energy
        roi_start = self.channel_from_energy(roi_center - self.roi_width/2.)
        roi_end = self.channel_from_energy(roi_center + self.roi_width/2.)
        self.detector.set_roi(roi_start, roi_end)
        
    def adjust_transmission(self):
        if self.detector.get_dead_time() > 40:
            self.high_boundary = self.current_transmission
            self.new_transmission = self.current_transmission - (self.high_boundary - self.low_boundary)/2.
        else:
            self.low_boundary = self.current_transmission
            if self.high_boundary == None:
                self.new_transmission = 2 * self.current_transmission
            else:
                self.new_transmission = self.current_transmission + (self.high_boundary - self.low_boundary)/2.
            
        self.current_transmission = self.new_transmission
        self.set_transmission(self.new_transmission)
        
    def optimize_transmission(self, low_dead_time=10, high_dead_time=20, max_transmission=50):
        self.current_transmission = self.transmission
        self.low_boundary = 0
        self.high_boundary = None
        
        k=0
        self.measure_fluorescence()
        while self.detector.get_dead_time() < low_dead_time or self.detector.get_dead_time() > high_dead_time:
            self.adjust_transmission()
            if self.get_transmission() > max_transmission:
                print 'Transmission optimization did not converge. Exiting at %.2f% after %d steps. Please check if the beam is actually getting on the sample.' % (max_transmission, k)
                break
            self.measure_fluorescence()
            k += 1
           
        print 'Transmission optimized after %d steps to %.2f' % (k, self.current_transmission)
        
    def prepare(self):
        _start = time.time()
        print 'prepare'
        self.mono_rx_motor.set_speed(self.default_mono_rx_motor_speed)
        self.check_directory(self.directory)
        self.write_destination_namepattern(self.directory, self.name_pattern)
        self.set_transmission(self.transmission)
            
        if self.snapshot == True:
            print 'taking image'
            self.camera.set_exposure()
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
        self.set_roi()
        
        while time.time() - _start < self.insertion_timeout:
            time.sleep(self.detector.sleeptime)
        
        if self.optimize == True:
            edge_energy = self.get_edge_energy()
            self.optimize_at_energy = edge_energy
            print 'optimizing transmission at %.3f keV' % self.optimize_at_energy
            self.set_photon_energy(self.optimize_at_energy, wait=True)
            self.optimize_transmission()
        else:
            self.set_transmission(self.transmission)
        
        self.start_energy = self.get_edge_energy() - self.scan_range/2.
        print 'self.start_energy', self.start_energy
        self.end_energy = self.start_energy + self.scan_range
        
        self.energy_motor.mono.simenergy = self.start_energy/1e3
        angle_start = self.energy_motor.mono.simthetabragg
        self.energy_motor.mono.simenergy = self.end_energy/1e3
        angle_end = self.energy_motor.mono.simthetabragg
        self.scan_speed = abs(angle_end - angle_start)/self.total_time
        print 'scan_speed', self.scan_speed
        
        print 'moving to start energy %.3f' % self.start_energy
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
        self.observations_fields = ['chronos', 'energy', 'thetabragg', 'wavelength']
        
        while self.mono_rx_motor.get_state() != 'STANDBY':
            chronos = time.time() - start_time
            point = [chronos, self.energy_motor.mono.energy, self.energy_motor.mono.thetabragg, self.energy_motor.mono.Lambda]
            self.observations.append(point)
            gevent.sleep(self.monitor_sleep_time)
            
        for monitor in self.monitors:
            monitor.observe = False
    
    def get_observations(self):
        return self.observations
        
    def get_observations_fields(self):
        return self.observations_fields

    def get_all_observations(self):
        all_observations = {}
        all_observations['actuator_monitor'] = {}
        actuator_observations_fields = self.get_observations_fields()
        all_observations['actuator_monitor']['observations_fields'] = actuator_observations_fields
        actuator_observations = self.get_observations()
        all_observations['actuator_monitor']['observations'] = actuator_observations
        
        X = np.array(actuator_observations)
        print 'actuator monitor X.shape', X.shape
        chronos, thetabragg = X[:, 0], X[:, 2]
        
        z = np.polyfit(chronos, thetabragg, 1) 
        print 'fit parameters', z
        theta_chronos_predictor = np.poly1d(z)
        pylab.figure(figsize=(16, 9))
        pylab.plot(chronos, thetabragg, label='experimental')
        pylab.plot(chronos, theta_chronos_predictor(chronos), label='from_fit')
        pylab.grid(True)
        pylab.legend()
        pylab.xlabel('chronos [s]')
        pylab.ylabel('theta bragg [deg.]')
        pylab.savefig(os.path.join(self.directory, '%s_%s_%s_theta_bragg_vs_chronos.png' % (self.name_pattern, self.element, self.edge)))
        for monitor_name, mon in zip(self.monitor_names, self.monitors):
            all_observations[monitor_name] = {}
            all_observations[monitor_name]['observations_fields'] = mon.get_observations_fields()
            all_observations[monitor_name]['observations'] = mon.get_observations()
        
        mca_observations = all_observations['mca']['observations']
       
        print 'mca_observations.len', len(mca_observations)
        
        mca_chronos = np.array([item[0] for item in mca_observations])
        mca_theta = theta_chronos_predictor(mca_chronos)
        mca_wavelengths = self.resolution_motor.get_wavelength_from_theta(mca_theta)
        mca_energies = self.resolution_motor.get_energy_from_wavelength(mca_wavelengths)
        
        mca_normalized_counts = np.array([item[3] for item in mca_observations])
        
        equidistant_energies = np.linspace(mca_energies.min(), mca_energies.max(), 200)
        
        equidistant_mca_normalized_counts = np.interp(equidistant_energies, mca_energies, mca_normalized_counts)
        
        all_observations['energies'] = equidistant_energies
        all_observations['counts'] =   equidistant_mca_normalized_counts
        self.all_observations = all_observations
        return all_observations
        
    def run(self):
        self.fast_shutter.open()
        
        self._start = time.time()
        
        observers = [gevent.spawn(self.actuator_monitor, self._start)]
        for monitor in self.monitors:
            observers.append(gevent.spawn(monitor.monitor, self._start))
        
        self.energy_motor.mono.energy = self.end_energy/1e3
            
        gevent.joinall(observers)
                
        for monitor in self.monitors:
            monitor.observe = False

        self.fast_shutter.close()
    
    def clean(self):
        print 'clean'
        self.end_time = time.time()
        self.detector.extract()
        self.mono_rx_motor.set_speed(self.default_mono_rx_motor_speed)
        self.save_parameters()
        self.save_raw_results()
        self.save_raw_scan()
        self.save_log()
        self.save_plot()
    
    def parse_chooch_output(self, output):
        logging.info('parse_chooch_output')
        table = output[output.find('Table of results'):]
        tabl = table.split('\n')
        tab = np.array([ line.split('|') for line in tabl if line and line[0] == '|'])
        print 'tab', tab
        self.pk = float(tab[1][2])
        self.fppPeak = float(tab[1][3])
        self.fpPeak = float(tab[1][4])
        self.ip = float(tab[2][2])
        self.fppInfl = float(tab[2][3])
        self.fpInfl = float(tab[2][4])
        self.efs = self.get_efs()
        return {'pk': self.pk, 'fppPeak': self.fppPeak, 'fpPeak': self.fpPeak, 'ip': self.ip, 'fppInfl': self.fppInfl, 'fpInfl': self.fpInfl, 'efs': self.efs}
    
    def get_efs(self):
        filename = os.path.join(self.directory, self.raw_filename.replace('.raw', '.efs'))
        f = open(filename)
        data = f.read().split('\n')
        efs = np.array([np.array(map(float, line.split())) for line in data if len(line.split()) == 3])
        return efs    
    
    def analyze(self):
        import commands
        self.results = {}
        chooch_parameters = {'element': self.element, 
                             'edge': self.edge,
                             'raw_file': self.raw_filename,
                             'output_ps': self.raw_filename.replace('.raw', '.ps'),
                             'output_efs': self.raw_filename.replace('.raw', '.efs')}
        
        chooch_command = 'chooch -p {output_ps} -o {output_efs} -e {element} -a {edge} {raw_file}'.format(**chooch_parameters)
        print 'chooch command %s' % chooch_command
       
        chooch_output = commands.getoutput(chooch_command)
        self.results['chooch_output'] = chooch_output
        print 'chooch_output', chooch_output
        chooch_results = self.parse_chooch_output(chooch_output)
        self.results['chooch_results'] = chooch_results
        
        f = open(os.path.join(self.directory, '%s_chooch_results.pickle' % self.name_pattern), 'w')
        pickle.dump(self.results, f)
        f.close()
        
                               
    def stop(self):
        self.mono_rx_motor.stop()
        self.fast_shutter.close()
        self.mono_rx_motor.set_speed(self.default_mono_rx_motor_speed)
        
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
            self.parameters['beam_position_vertical'] = self.camera.md2.beampositionvertical
            self.parameters['beam_position_horizontal'] = self.camera.md2.beampositionhorizontal
            self.parameters['image'] = self.image
            self.parameters['rgb_image'] = self.rgbimage.reshape((self.image.shape[0], self.image.shape[1], 3))
            scipy.misc.imsave(os.path.join(self.directory, '%s_optical_bw.png' % self.name_pattern), self.image)
            scipy.misc.imsave(os.path.join(self.directory, '%s_optical_rgb.png' % self.name_pattern), self.rgbimage.reshape((self.image.shape[0], self.image.shape[1], 3)))
        
        f = open(os.path.join(self.directory, '%s_parameters.pickle' % self.name_pattern), 'w')
        pickle.dump(self.parameters, f)
        f.close()
        
    def save_raw_results(self):
        f = open(os.path.join(self.directory, '%s_complete_results.pickle' % self.name_pattern), 'w')
        pickle.dump(self.get_all_observations(), f)
        f.close()
               
    def save_raw_scan(self):
        self.raw_filename = os.path.join(self.directory, '%s.raw' % self.name_pattern)
        self.energies = self.all_observations['energies']
        self.counts = self.all_observations['counts']
        X = np.vstack([self.energies, self.counts]).T
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
        pylab.plot(self.energies, self.counts, 'go-')
        pylab.xlabel('energy [eV]')
        pylab.ylabel('normalized counts')
        pylab.title(self.description)
        pylab.grid(True)
        pylab.savefig(os.path.join(self.directory, '%s_%s_%s_edge.png' % (self.name_pattern, self.element, self.edge)))
        
        if self.display == True:
            pylab.show()
            
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
    
    ./energy_scan.py -e <element> -s <edge> <options>
    
    '''
    
    import optparse
    
    parser = optparse.OptionParser(usage=usage)
        
    parser.add_option('-e', '--element', type=str, help='Specify the element')
    parser.add_option('-s', '--edge', type=str, help='Specify the edge')
    parser.add_option('-d', '--directory', type=str, default='/tmp/testXanes', help='Directory to store the results (default=%default)')
    parser.add_option('-n', '--name_pattern', type=str, default='escan', help='name_pattern')
    parser.add_option('-i', '--integration_time', type=float, default=0.25, help='integration time (default=%default)')
    parser.add_option('-o', '--optimize', action='store_true', help='optimize transmission')
    parser.add_option('-D', '--display', action='store_true', help='display plot')
    parser.add_option('-T', '--total_time', type=float, default=100., help='total scan time (default=%default)')
    parser.add_option('-r', '--scan_range', type=float, default=120., help='scan range (default=%default eV)')
    options, args = parser.parse_args()
    
    print 'options', options
    print 'args', args
    
    escan = energy_scan(options.name_pattern,
                        options.directory,
                        options.element,
                        options.edge,
                        integration_time=options.integration_time,
                        total_time=options.total_time,
                        optimize=options.optimize,
                        scan_range=options.scan_range,
                        display=options.display)
              
    escan.execute()
              
    
if __name__ == '__main__':
    main()
    