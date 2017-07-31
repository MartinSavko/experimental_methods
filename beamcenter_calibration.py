#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
single position oscillation scan
'''
import traceback
import logging
import time
import itertools
import os
import pickle

from experiment import experiment
from detector import detector
from goniometer import goniometer
from energy import energy as energy_motor
from transmission import transmission as transmission_motor
from omega_scan import omega_scan

class beamcenter_calibration(experiment):
    
    def __init__(self, 
                 directory,
                 name_pattern = 'pe_%.3feV_ts_%.3fmm_tx_%.3fmm_tz_%.3fmm_$id',
                 photon_energies=None,
                 tss=None,
                 txs=None,
                 tzs=None,
                 scan_range=0.1,
                 scan_exposure_time=0.025,
                 angle_per_frame=0.1,
                 analysis=None):
        
        experiment.__init__(self, 
                            name_pattern=name_pattern, 
                            directory=directory,
                            analysis=analysis)
        
        self.directory = directory
        self.name_pattern = name_pattern
        self.photon_energies = photon_energies
        self.tss = tss
        self.txs = txs
        self.tzs = tzs
        self.scan_range = scan_range
        self.scan_exposure_time = scan_exposure_time
        self.angle_per_frame = angle_per_frame
        
        self.nimages = int(self.scan_range/self.angle_per_frame)
        
        #actuators
        self.detector = detector()
        self.goniometer = goniometer()
        self.energy_motor = energy_motor()
        self.transmission_motor = transmission_motor()
        
        self.capillary_park_position = 80
        self.aperture_park_position = 80
        self.detector_beamstop_park_position = 18.5
        
    def prepare(self):
        self.detector.check_dir(self.directory)
        self.goniometer.set_data_collection_phase(wait=True)
        self.detector_beamstop_initial_position = self.detector.beamstop.get_z()
        self.detector_initial_ts = self.detector.position.ts.get_position()
        self.detector_initial_tz = self.detector.position.tz.get_position()
        self.detector_initial_tx = self.detector.position.tx.get_position()
        self.capillary_initial_position = self.goniometer.md2.capillaryverticalposition
        self.aperture_initial_position = self.goniometer.md2.apertureverticalposition
        print 'detector_beamstop_initial_position', self.detector_beamstop_initial_position
        print 'self.detector_initial_ts', self.detector_initial_ts
        print 'self.detector_initial_tx', self.detector_initial_tx
        print 'self.detector_initial_tz', self.detector_initial_tz
        print 'self.capillary_initial_position', self.capillary_initial_position
        print 'self.aperture_initial_position', self.aperture_initial_position
        
        self.goniometer.md2.capillaryverticalposition = self.capillary_park_position
        self.goniometer.wait()
        self.goniometer.md2.apertureverticalposition = self.aperture_park_position
        self.goniometer.wait()
        self.detector.beamstop.set_z(self.detector_beamstop_park_position)
        
        self.goniometer.md2.saveaperturebeamposition()
        self.goniometer.md2.savecapillarybeamposition()
        
        if self.photon_energies == None:
            self.photon_energies = [self.energy_motor.get_energy()]
        if self.tss == None:
            self.tss = [self.detector.position.ts.get_position()]
        if self.txs == None:
            self.txs = [self.detector.position.tx.get_position()]
        if self.tzs == None:
            self.tzs = [self.detector.position.tz.get_position()]    
        
        print 'photon_energies', self.photon_energies
        print 'tss', self.tss
        print 'txs', self.txs
        print 'tzs', self.tzs
        
    def get_transmission(self, photon_energy, default_transmision=0.006):
        if photon_energy > 1e3:
            photon_energy *= 1e-3
        if photon_energy > 7 and photon_energy <= 10:
            transmission = 0.005
        elif photon_energy > 14 and photon_energy<=16.5:
            transmission = 0.01
        elif photon_energy > 16.5:
            transmission = 0.02
        else:
            transmission = default_transmision
        return transmission
    
    def clean(self):
        self.save_parameters()
        self.save_log()
        self.detector.disarm()
        self.goniometer.wait()
        self.goniometer.md2.capillaryverticalposition = self.capillary_initial_position
        time.sleep(0.2)
        self.goniometer.wait()
        self.goniometer.md2.apertureverticalposition = self.aperture_initial_position
        time.sleep(0.2)
        self.goniometer.wait()
        
        self.goniometer.md2.saveaperturebeamposition()
        self.goniometer.md2.savecapillarybeamposition()
        
        self.transmission_motor.set_transmission(50)
        self.energy_motor.set_energy(12.65)
        self.detector.position.ts.set_position(350)
        self.detector.position.tx.set_position(self.detector_initial_tx)
        self.detector.position.tz.set_position(self.detector_initial_tz)
        self.detector.beamstop.set_z(self.detector_beamstop_initial_position)
    
    def efficient_order(self, sequence, current_value):
        if abs(current_value - sequence[0]) > abs(current_value - sequence[-1]):
            return sequence[::-1]
        else:
            return sequence[:]
        
    def run(self):
        self.nscans = 0
         #for pe, ts, tx ,tz in itertools.product(self.photon_energies, self.tss, self.txs, self.tzs):
        for pe in self.efficient_order(self.photon_energies, self.energy_motor.get_energy()):
            for ts in self.efficient_order(self.tss, self.detector.position.ts.get_position()):
                for tx in self.efficient_order(self.txs, self.detector.position.tx.get_position()):
                    for tz in self.efficient_order(self.tzs, self.detector.position.tz.get_position()):
                        if pe<30:
                            pe *= 1e3
                        name_pattern = self.name_pattern % (pe, ts, tx, tz)
                        print 'name_pattern', name_pattern
                        print 'photon_energy', pe
                        s = omega_scan(name_pattern, 
                                       self.directory, 
                                       scan_range=self.scan_range, 
                                       scan_exposure_time=self.scan_exposure_time,
                                       angle_per_frame=self.angle_per_frame,
                                       photon_energy=pe,
                                       detector_distance=ts,
                                       detector_vertical=tz,
                                       detector_horizontal=tx,
                                       transmission=self.get_transmission(pe),
                                       nimages_per_file=1)
                        s.execute()
                        self.nscans += 1

    def analyze(self):
        pass
        
    def save_parameters(self):
        self.parameters = {}
        
        self.parameters['timestamp'] = self.timestamp
        self.parameters['name_pattern'] = self.name_pattern
        self.parameters['directory'] = self.directory
        self.parameters['photon_energies'] = self.photon_energies
        self.parameters['tss'] = self.tss
        self.parameters['txs'] = self.txs
        self.parameters['tzs'] = self.tzs
        
        self.parameters['scan_range'] = self.scan_range
        self.parameters['scan_exposure_time'] = self.scan_exposure_time
        self.parameters['angle_per_frame'] = self.angle_per_frame
        self.parameters['nimages'] = self.nimages
        self.parameters['nscans'] = self.nscans
        self.parameters['duration'] = self.end_time - self.start_time
        self.parameters['start_time'] = self.start_time
        self.parameters['end_time'] = self.end_time

        f = open(os.path.join(self.directory, '%s_parameters.pickle' % self.__module__), 'w')
        pickle.dump(self.parameters, f)
        f.close()
    
    def save_log(self):
        '''method to save the experiment details in the log file'''
        f = open(os.path.join(self.directory, '%s.log' % self.name_pattern), 'w')
        keyvalues = self.parameters.items()
        keyvalues.sort()
        for key, value in keyvalues:
            f.write('%s: %s\n' % (key, value)) 
        f.close()
        
def main():
    import optparse
        
    parser = optparse.OptionParser()
    parser.add_option('-n', '--name_pattern', default='pe_%.3feV_ts_%.3fmm_tx_%.3fmm_tz_%.3fmm_$id', type=str, help='Prefix default=%default')
    parser.add_option('-d', '--directory', default='/nfs/ruche/proxima2a-spool/2017_Run3/%s/com-proxima2a/RAW_DATA/Commissioning/beam_center1' % time.strftime('%Y-%m-%d'), type=str, help='Destination directory default=%default')
    #parser.add_option('-d', '--directory', default='/nfs/ruche/proxima2a-spool/2017_Run3/2017-07-22/com-proxima2a/RAW_DATA/Commissioning/beam_center1', type=str, help='Destination directory default=%default')
    #parser.add_option('-r', '--scan_range', default=180, type=float, help='Scan range [deg]')
    #parser.add_option('-e', '--scan_exposure_time', default=18, type=float, help='Scan exposure time [s]')
    #parser.add_option('-s', '--scan_start_angle', default=0, type=float, help='Scan start angle [deg]')
    #parser.add_option('-a', '--angle_per_frame', default=0.1, type=float, help='Angle per frame [deg]')
    #parser.add_option('-f', '--image_nr_start', default=1, type=int, help='Start image number [int]')
    #parser.add_option('-i', '--position', default=None, type=str, help='Gonio alignment position [dict]')
    #parser.add_option('-p', '--photon_energy', default=None, type=float, help='Photon energy ')
    #parser.add_option('-t', '--detector_distance', default=None, type=float, help='Detector distance')
    #parser.add_option('-o', '--resolution', default=None, type=float, help='Resolution [Angstroem]')
    #parser.add_option('-x', '--flux', default=None, type=float, help='Flux [ph/s]')
    #parser.add_option('-m', '--transmission', default=None, type=float, help='Transmission. Number in range between 0 and 1.')
    
    options, args = parser.parse_args()
    print 'options', options
    #s = scan(**vars(options))
    #s.execute()
    import numpy as np
    distances = list(np.arange(125, 1050., 50))
    #distances = [98, 500, 1000]
    energies = [12650]  + list(np.arange(6500, 18501, 2000))
    txs = [21.30]
    tzs = [19.13]
    
    #distances = [175, 450, 875]
    #energies = [12650.]
    #txs = [19., 20., 21.30, 22., 23., 24.]
    #tzs = [10., 15., 19.13, 25., 30., 35., 40., 50.]
    
    bcc = beamcenter_calibration(options.directory, photon_energies=energies, tss=distances, txs=txs, tzs=tzs)
    bcc.execute()
    
if __name__ == '__main__':
    main()