#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
single position oscillation scan
"""
import gevent

import traceback
import logging
import time
import itertools
import os
import pickle

from experimental_methods.experiment.experiment import experiment
from experimental_methods.experiment.diffraction_experiment import diffraction_experiment
from experimental_methods.instrument.detector import detector
from experimental_methods.instrument.goniometer import goniometer
from experimental_methods.instrument.energy import energy as energy_motor
from experimental_methods.instrument.transmission import old_transmission as transmission_motor
from experimental_methods.experiment.omega_scan import omega_scan

"""
2022-02-26
exploration of appropriate transmission levels for different photon energies at 500mA
Insert  i11-ma-c02/dt/imag.1-mt_tz-pos
att.set_filter(att.carousel.positions[7])
att.imager1.isInserted = True
photon energy       transmission
    6000               0.5
    7000               0.1
    8100               0.001
   12000               0.0001
   14000               0.0001
   15000               0.0001
   17000               0.0001
   
"""


class beamcenter_calibration(diffraction_experiment):
    specific_parameter_fields = [
        {"name": "photon_energies", "type": "", "description": ""},
        {"name": "scan_range", "type": "float", "description": ""},
        {"name": "scan_exposure_time", "type": "float", "description": ""},
        {"name": "angle_per_frame", "type": "float", "description": ""},
        {"name": "nimages", "type": "int", "description": ""},
        {"name": "tss", "type": "", "description": ""},
        {"name": "txs", "type": "", "description": ""},
        {"name": "tzs", "type": "", "description": ""},
        {"name": "nscans", "type": "", "description": ""},
        {"name": "handle_detector_beamstop", "type": "bool", "description": ""},
        {"name": "direct_beam", "type": "bool", "description": ""},
    ]

    def __init__(
        self,
        directory,
        name_pattern="pe_%.3feV_ts_%.3fmm_tx_%.3fmm_tz_%.3fmm_$id",
        photon_energies=None,
        tss=None,
        txs=None,
        tzs=None,
        scan_range=0.1,
        scan_exposure_time=0.025,
        angle_per_frame=0.1,
        direct_beam=True,
        analysis=None,
        handle_detector_beamstop=False,
    ):
        if hasattr(self, "parameter_fields"):
            self.parameter_fields += beamcenter_calibration.specific_parameter_fields
        else:
            self.parameter_fields = beamcenter_calibration.specific_parameter_fields

        experiment.__init__(
            self, name_pattern=name_pattern, directory=directory, analysis=analysis
        )

        self.photon_energies = photon_energies
        self.tss = tss
        self.txs = txs
        self.tzs = tzs
        self.scan_range = scan_range
        self.scan_exposure_time = scan_exposure_time
        self.angle_per_frame = angle_per_frame
        self.direct_beam = direct_beam
        self.nimages = int(self.scan_range / self.angle_per_frame)

        # actuators
        self.detector = detector()
        self.goniometer = goniometer()
        self.energy_motor = energy_motor()
        self.transmission_motor = transmission_motor()

        self.capillary_park_position = 80
        self.aperture_park_position = 80
        self.detector_beamstop_park_position = 4.0
        self.handle_detector_beamstop = handle_detector_beamstop

    def prepare(self):
        self.detector.check_dir(self.directory)
        self.goniometer.set_data_collection_phase(wait=True)
        if self.handle_detector_beamstop:
            self.detector_beamstop_initial_position = self.detector.beamstop.get_z()
            self.detector.beamstop.disable_tracking()
        self.detector_initial_ts = self.detector.position.ts.get_position()
        self.detector_initial_tz = self.detector.position.tz.get_position()
        self.detector_initial_tx = self.detector.position.tx.get_position()
        self.capillary_initial_position = self.goniometer.md.capillaryverticalposition
        self.aperture_initial_position = self.goniometer.md.apertureverticalposition
        self.initial_position = self.goniometer.get_position()

        if self.handle_detector_beamstop:
            print(
                "detector_beamstop_initial_position",
                self.detector_beamstop_initial_position,
            )

        print("self.detector_initial_ts", self.detector_initial_ts)
        print("self.detector_initial_tx", self.detector_initial_tx)
        print("self.detector_initial_tz", self.detector_initial_tz)
        print("self.capillary_initial_position", self.capillary_initial_position)
        print("self.aperture_initial_position", self.aperture_initial_position)
        print("self.initial_position", self.initial_position)
        if self.direct_beam == True:
            self.goniometer.md.capillaryverticalposition = self.capillary_park_position
            self.goniometer.wait()
            self.goniometer.md.apertureverticalposition = self.aperture_park_position
            self.goniometer.wait()
            if self.handle_detector_beamstop:
                self.detector.beamstop.set_z(self.detector_beamstop_park_position)

            self.goniometer.md.saveaperturebeamposition()
            self.goniometer.md.savecapillarybeamposition()

        if self.photon_energies == None:
            self.photon_energies = [self.energy_motor.get_energy()]
        if self.tss == None:
            self.tss = [self.detector.position.ts.get_position()]
        if self.txs == None:
            self.txs = [self.detector.position.tx.get_position()]
        if self.tzs == None:
            self.tzs = [self.detector.position.tz.get_position()]

        print("photon_energies", self.photon_energies)
        print("tss", self.tss)
        print("txs", self.txs)
        print("tzs", self.tzs)

    def get_transmission(self, photon_energy, default_transmision=0.001):
        if photon_energy > 1e3:
            photon_energy *= 1e-3
        if photon_energy > 7 and photon_energy <= 10:
            transmission = 1.5 * default_transmision
        elif photon_energy > 14 and photon_energy <= 16.5:
            transmission = 2.0 * default_transmision
        elif photon_energy > 16.5:
            transmission = 8 * default_transmision
        else:
            transmission = default_transmision
        return transmission * 0.06

    def clean(self):
        try:
            self.collect_parameters()
        except:
            print(traceback.print_exc())

        self.save_parameters()
        self.save_log()
        self.detector.disarm()
        if self.direct_beam == True:
            self.goniometer.wait()
            self.goniometer.md.capillaryverticalposition = (
                self.capillary_initial_position
            )
            gevent.sleep(0.2)
            self.goniometer.wait()
            self.goniometer.md.apertureverticalposition = self.aperture_initial_position
            gevent.sleep(0.2)
            self.goniometer.wait()

            self.goniometer.md.saveaperturebeamposition()
            self.goniometer.md.savecapillarybeamposition()

            if self.handle_detector_beamstop:
                self.detector.beamstop.set_z(self.detector_beamstop_initial_position)

        self.transmission_motor.set_transmission(100)
        self.energy_motor.set_energy(12.65)
        self.detector.position.ts.set_position(350)
        self.detector.position.tx.set_position(self.detector_initial_tx)
        self.detector.position.tz.set_position(self.detector_initial_tz)
        self.detector.beamstop.enable_tracking()

    def efficient_order(self, sequence, current_value):
        if abs(current_value - sequence[0]) > abs(current_value - sequence[-1]):
            return sequence[::-1]
        else:
            return sequence[:]

    def run(self):
        self._start = time.time()
        self.nscans = 0
        # for pe, ts, tx ,tz in itertools.product(self.photon_energies, self.tss, self.txs, self.tzs):
        for pe in self.efficient_order(
            self.photon_energies, self.energy_motor.get_energy()
        ):
            for ts in self.efficient_order(
                self.tss, self.detector.position.ts.get_position()
            ):
                for tx in self.efficient_order(
                    self.txs, self.detector.position.tx.get_position()
                ):
                    for tz in self.efficient_order(
                        self.tzs, self.detector.position.tz.get_position()
                    ):
                        if pe < 30:
                            pe *= 1e3
                        name_pattern = self.name_pattern % (pe, ts, tx, tz)
                        print("name_pattern", name_pattern)
                        print("photon_energy", pe)
                        if self.direct_beam == True:
                            transmission = self.get_transmission(pe)
                        else:
                            transmission = None
                        if (
                            self.nscans % 10 == 0
                            and self.nscans != 0
                            and self.direct_beam != True
                        ):
                            self.initial_position["AlignmentY"] += 0.015

                        self.transmission_motor.set_transmission(transmission)

                        s = omega_scan(
                            name_pattern,
                            self.directory,
                            scan_range=self.scan_range,
                            scan_exposure_time=self.scan_exposure_time,
                            angle_per_frame=self.angle_per_frame,
                            position=self.initial_position,
                            photon_energy=pe,
                            detector_distance=ts,
                            detector_vertical=tz,
                            detector_horizontal=tx,
                            ##transmission=transmission,
                            nimages_per_file=1,
                        )

                        print("s.parameter_fields", s.parameter_fields)
                        s.execute()
                        self.nscans += 1

    def analyze(self):
        pass


def main():
    import optparse

    parser = optparse.OptionParser()
    parser.add_option(
        "-n",
        "--name_pattern",
        default="pe_%.3feV_ts_%.3fmm_tx_%.3fmm_tz_%.3fmm_$id",
        type=str,
        help="Prefix default=%default",
    )
    parser.add_option(
        "-d",
        "--directory",
        default="/nfs/data2/2022_Run1/Commissioning/beamcenter_calibration/%s/direct_beam_a"
        % time.strftime("%Y-%m-%d"),
        type=str,
        help="Destination directory default=%default",
    )
    parser.add_option(
        "-r", "--scan_range", default=0.1, type=float, help="Scan range [deg]"
    )
    parser.add_option(
        "-e",
        "--scan_exposure_time",
        default=0.1,
        type=float,
        help="Scan exposure time [s]",
    )
    # parser.add_option('-s', '--scan_start_angle', default=0, type=float, help='Scan start angle [deg]')
    parser.add_option(
        "-a", "--angle_per_frame", default=0.1, type=float, help="Angle per frame [deg]"
    )
    # parser.add_option('-f', '--image_nr_start', default=1, type=int, help='Start image number [int]')
    # parser.add_option('-i', '--position', default=None, type=str, help='Gonio alignment position [dict]')
    # parser.add_option('-p', '--photon_energy', default=None, type=float, help='Photon energy ')
    # parser.add_option('-t', '--detector_distance', default=None, type=float, help='Detector distance')
    # parser.add_option('-o', '--resolution', default=None, type=float, help='Resolution [Angstroem]')
    # parser.add_option('-x', '--flux', default=None, type=float, help='Flux [ph/s]')
    parser.add_option(
        "-D",
        "--direct_beam",
        action="store_true",
        help="Do apply transmission correction -- for direct beam measurements.",
    )
    parser.add_option(
        "-B",
        "--handle_detector_beamstop",
        action="store_true",
        help="Remove beamstop on the detector.",
    )

    options, args = parser.parse_args()
    print("options", options)
    # s = scan(**vars(options))
    # s.execute()
    import numpy as np

    distances = list(np.linspace(116, 550.0, 50))
    # distances = [125, 150, 200]
    # distances = [98, 500, 1000]
    # energies = [12.65] #[7., 8, 9, 10, 10836., 11, 12, 14, 16] #list(np.arange(6500, 18501, 1000))
    energies = list(np.linspace(6700, 17600, 13))
    energies = [12650.0] + energies + [12670.0]
    txs = [20.50]
    tzs = [48.50]

    # distances = [175, 450, 875]i
    # energies = [12650.]
    # txs = [19., 20., 21.30, 22., 23., 24.]
    # tzs = [10., 15., 19.13, 25., 30., 35., 40., 50.]

    bcc = beamcenter_calibration(
        options.directory,
        photon_energies=energies,
        tss=distances,
        txs=txs,
        tzs=tzs,
        scan_range=options.scan_range,
        scan_exposure_time=options.scan_exposure_time,
        angle_per_frame=options.angle_per_frame,
        # scan_start_angle=options.scan_start_angle,
        direct_beam=options.direct_beam,
        handle_detector_beamstop=options.handle_detector_beamstop,
    )
    bcc.execute()


if __name__ == "__main__":
    main()
