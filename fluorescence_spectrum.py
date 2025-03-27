#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import gevent

import numpy as np
import os
import pickle

from xray_experiment import xray_experiment
from fluorescence_detector import fluorescence_detector
from goniometer import goniometer
from fast_shutter import fast_shutter
from safety_shutter import safety_shutter
from transmission import transmission as transmission_motor


class fluorescence_spectrum(xray_experiment):
    specific_parameter_fields = [
        {"name": "position", "type": "", "description": ""},
        {"name": "integration_time", "type": "", "description": ""},
        {"name": "nchannels", "type": "", "description": ""},
        {"name": "calibration", "type": "", "description": ""},
        {"name": "energies", "type": "", "description": ""},
        {"name": "spectrum", "type": "", "description": ""},
        {"name": "dead_time", "type": "", "description": ""},
        {"name": "real_count_time", "type": "", "description": ""},
        {"name": "input_count_rate", "type": "", "description": ""},
        {"name": "output_count_rate", "type": "", "description": ""},
        {"name": "calculated_dead_time", "type": "", "description": ""},
        {"name": "events_in_run", "type": "", "description": ""},
        {
            "name": "detector_card",
            "type": "str",
            "description": "counting card to use (xia or xspress3)",
        },
    ]

    def __init__(
        self,
        name_pattern,
        directory,
        integration_time=5,
        transmission=1.0,
        insertion_timeout=2,
        detector_card="xia",
        position=None,
        photon_energy=None,
        flux=None,
        snapshot=False,
        zoom=None,
        diagnostic=None,
        analysis=None,
        simulation=None,
        display=False,
        parent=None,
    ):
        if hasattr(self, "parameter_fields"):
            self.parameter_fields += fluorescence_spectrum.specific_parameter_fields
        else:
            self.parameter_fields = fluorescence_spectrum.specific_parameter_fields[:]

        xray_experiment.__init__(
            self,
            name_pattern,
            directory,
            position=position,
            photon_energy=photon_energy,
            transmission=transmission,
            flux=flux,
            snapshot=snapshot,
            zoom=zoom,
            diagnostic=diagnostic,
            analysis=analysis,
            simulation=simulation,
        )

        self.description = "XRF spectrum, Proxima 2A, SOLEIL, %s" % time.ctime(
            self.timestamp
        )
        self.detector_card = detector_card
        self.detector = fluorescence_detector(device_name=self.detector_card)

        self.integration_time = integration_time
        self.transmission = transmission
        self.insertion_timeout = insertion_timeout
        self.display = display
        self.parent = parent

        self.total_expected_exposure_time = self.integration_time
        self.total_expected_wedges = 1

        self.spectrum = None
        self.energies = None

    def get_calibration(self):
        return self.detector.get_calibration()

    def get_nchannels(self):
        return len(self.get_channels())

    def get_dead_time(self):
        return self.detector.get_dead_time()

    def get_real_count_time(self):
        return self.detector.get_real_time()

    def get_input_count_rate(self):
        return self.detector.get_input_count_rate()

    def get_output_count_rate(self):
        return self.detector.get_output_count_rate()

    def get_calculated_dead_time(self):
        return self.detector.get_calculated_dead_time()

    def get_events_in_run(self):
        return self.detector.get_events_in_run()

    def prepare(self):
        _start = time.time()
        print("prepare")
        self.protective_cover.insert()
        self.check_directory(self.directory)

        if self.snapshot == True:
            print("taking image")
            self.camera.set_exposure(0.05)
            self.camera.set_zoom(self.zoom)
            self.goniometer.insert_backlight()
            self.goniometer.extract_frontlight()
            self.goniometer.set_position(self.reference_position)
            self.goniometer.wait()
            self.image = self.get_image()
            self.rgbimage = self.get_rgbimage()

        self.goniometer.set_data_collection_phase(wait=True)

        self.detector.insert()

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
        # gevent.sleep(0.1)
        self.fast_shutter.open()
        self.spectrum = self.detector.get_point()
        self.fast_shutter.close()

    def clean(self):
        print("clean")
        _start = time.time()
        self.detector.extract()
        self.end_time = time.time()
        self.save_spectrum()
        self.collect_parameters()
        self.save_parameters()
        self.save_log()
        self.save_plot()
        if self.diagnostic == True:
            self.save_diagnostics()
        print("clean finished in %.4f seconds" % (time.time() - _start))

    def stop(self):
        self.fast_shutter.close()
        self.detector.stop()

    def abort(self):
        self.fast_shutter.md.abort()
        self.stop()

    def analyze(self):
        self.save_plot()

    def get_channels(self):
        channels = np.arange(0, 2048)
        return channels

    def get_energies(self):
        """return energies in eV"""
        a, b, c = self.detector.get_calibration()
        channels = self.get_channels()
        energies = a + b * channels + c * channels**2
        return energies

    def save_spectrum(self):
        filename = os.path.join(self.directory, "%s.dat" % self.name_pattern)
        self.energies = self.get_energies()
        self.channels = self.get_channels()
        X = np.array(list(zip(self.channels, self.spectrum, self.energies)))
        self.header = "#F %s\n#D %s\n#N %d\n#L channel  counts  energy\n" % (
            filename,
            time.ctime(self.timestamp),
            X.shape[1],
        )

        try:
            np.savetxt(filename, X, header=self.header)
        except:
            f = open(filename, "a")
            f.write(self.header)
            np.savetxt(f, X)
            f.close()

    def save_plot(self):
        # return
        import pylab

        pylab.figure(figsize=(16, 9))
        pylab.title(self.description, fontsize=22)
        pylab.xlabel("energy [eV]", fontsize=18)
        pylab.ylabel("intensity [a.u.]", fontsize=18)
        pylab.plot(self.get_energies(), self.spectrum)
        pylab.xlim([0, 20480])
        pylab.savefig(
            r"%s" % os.path.join(self.directory, "%s_full.png" % self.name_pattern)
        )
        if self.display:
            pylab.show()


def main():
    import optparse

    parser = optparse.OptionParser()

    parser.add_option(
        "-d",
        "--directory",
        default="/tmp",
        type=str,
        help="directory (default=%default)",
    )
    parser.add_option(
        "-n",
        "--name_pattern",
        default="xrf",
        type=str,
        help="name_pattern (default=%default)",
    )
    parser.add_option(
        "-i",
        "--integration_time",
        default=5,
        type=float,
        help="integration_time (default=%default s)",
    )
    parser.add_option(
        "-t",
        "--transmission",
        default=1.0,
        type=float,
        help="transmission (default=%default %)",
    )
    parser.add_option(
        "-p",
        "--photon_energy",
        default=15000,
        type=float,
        help="transmission (default=%default eV)",
    )
    parser.add_option("-D", "--display", action="store_true", help="Display the plot")
    parser.add_option(
        "-c", "--detector_card", default="xia", type=str, help="counting card to use"
    )

    options, args = parser.parse_args()

    fs = fluorescence_spectrum(
        options.name_pattern,
        options.directory,
        integration_time=options.integration_time,
        photon_energy=options.photon_energy,
        transmission=options.transmission,
        display=options.display,
    )

    filename = "%s_parameters.pickle" % fs.get_template()

    if not os.path.isfile(filename):
        fs.execute()
    elif options.analysis == True:
        fs.save_spectrum()
        fs.save_plot()


if __name__ == "__main__":
    main()
