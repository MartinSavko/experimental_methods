#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Slits scan. Execute scan on a pair of slits.

"""
import gevent

import traceback
import time
import itertools
import os
import pickle
import numpy as np
import pylab
from scipy.constants import eV, h, c, angstrom, kilo, degree

from xray_experiment import xray_experiment
from monitor import Si_PIN_diode
from analysis import slit_scan_analysis


class slit_scan(xray_experiment):
    slit_types = {1: 1, 2: 1, 3: 2, 5: 2, 6: 2}

    specific_parameter_fields = [
        {"name": "slits", "type": "dict", "description": ""},
        {"name": "darkcurrent_time", "type": "float", "description": ""},
        {"name": "start_position", "type": "float", "description": ""},
        {"name": "end_position", "type": "float", "description": ""},
        {"name": "scan_speed", "type": "float", "description": ""},
        {"name": "default_speed", "type": "float", "description": ""},
        {"name": "default_gap", "type": "float", "description": ""},
        {"name": "default_pencil_gap", "type": "float", "description": ""},
        {"name": "slit_offsets", "type": "dict", "description": ""},
        {"name": "scan_gap", "type": "float", "description": ""},
    ]

    def __init__(
        self,
        name_pattern,
        directory,
        slits=1,
        start_position=2.0,
        end_position=-2.0,
        scan_speed=None,
        default_speed=None,
        default_gap=4.0,
        default_pencil_gap=0.1,
        scan_gap=None,
        darkcurrent_time=5.0,
        photon_energy=None,
        diagnostic=True,
        analysis=None,
        conclusion=None,
        simulation=None,
        display=False,
        extract=False,
    ):
        if hasattr(self, "parameter_fields"):
            self.parameter_fields += slit_scan.specific_parameter_fields
        else:
            self.parameter_fields = slit_scan.specific_parameter_fields[:]

        self.default_experiment_name = f"Slits {slits:d} scan between {start_position:.1f} and {end_position:.1f} mm"
        
        xray_experiment.__init__(
            self,
            name_pattern,
            directory,
            photon_energy=photon_energy,
            diagnostic=diagnostic,
            analysis=analysis,
            conclusion=conclusion,
            simulation=simulation,
        )

        self.slits = slits
        self.slit_type = self.slit_types[slits]
        self.alignment_slits = getattr(self, "slits%d" % slits)

        self.start_position = start_position
        self.end_position = end_position
        self.scan_speed = scan_speed
        self.default_speed = default_speed
        self.scan_gap = scan_gap

        if self.scan_speed == None:
            self.scan_speed = self.alignment_slits.scan_speed
        if self.default_speed == None:
            self.default_speed = self.alignment_slits.default_speed
        self.default_gap = default_gap
        self.default_pencil_gap = default_pencil_gap
        self.darkcurrent_time = darkcurrent_time

        self.diagnostic = diagnostic
        self.display = display
        self.extract = extract

        if self.slit_type == 1:
            self.total_expected_wedges = 4
        else:
            self.total_expected_wedges = 2

        self.wedge_time = abs(self.end_position - self.start_position) / self.scan_speed
        self.total_expected_exposure_time = self.total_expected_wedges * self.wedge_time

    
    def set_up_monitor(self):
        self.monitor_device = Si_PIN_diode()

        self.monitors_dictionary["calibrated_diode"] = self.monitor_device

        self.monitor_names += ["calibrated_diode"]
        self.monitors += [self.monitor_device]

    def get_scan_gap(self):
        if self.scan_gap is None:
            if self.slit_type == 1:
                self.scan_gap = self.default_gap
            else:
                self.scan_gap = self.default_pencil_gap
        return self.scan_gap

    def prepare(self):
        self.set_up_monitor()

        self.check_directory(self.directory)
        self.write_destination_namepattern(self.directory, self.name_pattern)

        initial_settings_a = []

        if self.slit_type == 1:
            self.alignment_slits.set_independent_mode()

        if self.simulation != True:
            initial_settings_a.append(
                gevent.spawn(self.goniometer.set_transfer_phase, wait=False)
            )
            initial_settings_a.append(
                gevent.spawn(self.set_photon_energy, self.photon_energy, wait=True)
            )

        for k in self.get_clean_slits():
            initial_settings_a.append(
                gevent.spawn(
                    getattr(getattr(self, "slits%d" % k), "set_horizontal_gap"),
                    self.default_gap,
                )
            )
            initial_settings_a.append(
                gevent.spawn(
                    getattr(getattr(self, "slits%d" % k), "set_vertical_gap"),
                    self.default_gap,
                )
            )

        if self.safety_shutter.closed():
            initial_settings_a.append(gevent.spawn(self.safety_shutter.open))

        gevent.joinall(initial_settings_a)

        initial_settings_b = []
        for k in self.get_clean_slits():
            initial_settings_b.append(
                gevent.spawn(
                    getattr(getattr(self, "slits%d" % k), "set_horizontal_position"), 0
                )
            )
            initial_settings_b.append(
                gevent.spawn(
                    getattr(getattr(self, "slits%d" % k), "set_vertical_position"), 0
                )
            )

        # initial_settings_b.append(gevent.spawn(self.monitor_device.insert))

        gevent.joinall(initial_settings_b)

        self.logger.debug("self.monitor_names %s" % self.monitor_names)
        self.logger.debug("self.monitors %s " % self.monitors)

    def execute(self):
        self.start_time = time.time()
        try:
            self.prepare()
            self.run()
        except:
            self.logger.info("Problem in preparation or execution %s" % self.__module__)
            self.logger.info(traceback.format_exc())
        finally:
            self.end_time = time.time()
            self.clean()
        if self.analysis == True:
            self.analyze()
        if self.conclusion == True:
            self.conclude()

        self.logger.info(
            "experiment execute took %.4f seconds" % (time.time() - self.start_time)
        )

    def get_alignment_actuators(self):
        return self.alignment_slits.get_alignment_actuators()

    def run(self):
        self.res = {}

        for k, actuator in enumerate(self.get_alignment_actuators()):
            self.logger.info("actuator: %d. %s" % (k, actuator.device_name))

            self.actuator = actuator
            # self.actuator_names = [self.actuator.get_name()]
            actuator.wait()
            actuator.set_position(self.start_position, timeout=None, wait=True)

            actuator.set_speed(self.scan_speed)

            if self.slit_type == 2:
                self.alignment_slits.set_pencil_scan_gap(
                    k, scan_gap=self.get_scan_gap(), wait=True
                )

            self.start_monitor()

            self.logger.debug(
                "sleep for darkcurrent_time while observation is already running"
            )
            gevent.sleep(self.darkcurrent_time)

            self.fast_shutter.open()

            move = gevent.spawn(
                actuator.set_position, self.end_position, timeout=None, wait=True
            )
            move.join()

            actuator.set_speed(self.default_speed)

            self.fast_shutter.close()

            gevent.sleep(self.darkcurrent_time)

            self.stop_monitor()

            actuator.wait()

            if self.slit_type == 2:
                self.alignment_slits.set_pencil_scan_gap(
                    k, scan_gap=self.default_gap, wait=True
                )
                actuator.set_position(0.0)
            elif self.slit_type == 1:
                actuator.set_position(self.start_position, wait=True)

            res = self.get_results()
            self.res[actuator.get_name()] = res

    def clean(self):
        self.collect_parameters()
        self.save_parameters()
        self.save_results()
        self.save_log()
        # self.save_plot()

        final_settings_a = []

        for k in self.get_clean_slits():
            final_settings_a.append(
                gevent.spawn(
                    getattr(getattr(self, "slits%d" % k), "set_horizontal_gap"),
                    self.default_gap,
                )
            )
            final_settings_a.append(
                gevent.spawn(
                    getattr(getattr(self, "slits%d" % k), "set_vertical_gap"),
                    self.default_gap,
                )
            )

        gevent.joinall(final_settings_a)

        final_settings_b = []
        for k in self.get_clean_slits():
            final_settings_b.append(
                gevent.spawn(
                    getattr(getattr(self, "slits%d" % k), "set_horizontal_position"), 0
                )
            )
            final_settings_b.append(
                gevent.spawn(
                    getattr(getattr(self, "slits%d" % k), "set_vertical_position"), 0
                )
            )
        gevent.joinall(final_settings_b)

        # if self.extract:
        # final_settings_b.append(gevent.spawn(self.monitor_device.extract))

        gevent.joinall(final_settings_b)

    def save_results(self):
        self.logger.debug("self.res")
        self.logger.debug("%s" % list(self.res.keys()))
        f = open(self.get_results_filename(), "wb")
        pickle.dump(self.res, f)
        f.close()

    def norm(self, a):
        return (a - a.mean()) / (a.max() - a.min())

    def save_plot(self):
        pylab.figure(figsize=(16, 9))
        for monitor_name, monitor in zip(self.monitor_names, self.monitors):
            r = np.array(self.results[monitor_name]["observations"])
            y = self.norm(r[:, 1])
            pylab.plot(r[:, 0], y, label=monitor_name)
        pylab.xlabel("chronos [s]")
        pylab.ylabel("intensity")
        pylab.title(self.description)
        pylab.grid(True)
        pylab.legend()
        pylab.savefig(
            os.path.join(self.directory, "%s_results.png" % (self.name_pattern,))
        )

        if self.display == True:
            pylab.show()

    def get_slit_offsets(self):
        slit_offsets = []
        for actuator in self.alignment_slits.get_alignment_actuators():
            slit_offsets.append(
                ("%s_offset" % actuator.get_name(), actuator.device.offset)
            )
        return dict(slit_offsets)

    def analyze(self):
        a = slit_scan_analysis(
            os.path.join(self.directory, "%s_parameters.pickle" % self.name_pattern)
        )
        a.analyze(display=self.display)

    def conclude(self):
        a = slit_scan_analysis(
            os.path.join(self.directory, "%s_parameters.pickle" % self.name_pattern)
        )
        a.conclude()


def main():
    import optparse

    usage = """Program will execute a slit scan
    
    ./slit_scan.py <options>
    
    """
    parser = optparse.OptionParser(usage=usage)

    parser.add_option(
        "-d",
        "--directory",
        type=str,
        default="/tmp/slit_scan",
        help="Directory to store the results (default=%default)",
    )
    parser.add_option(
        "-n", "--name_pattern", type=str, default="slit_scan", help="name_pattern"
    )
    parser.add_option("-s", "--slits", type=int, default=1, help="Slits")
    parser.add_option(
        "-b", "--start_position", type=float, default=2.0, help="Start position"
    )
    parser.add_option(
        "-e", "--end_position", type=float, default=-2.0, help="End position"
    )
    parser.add_option(
        "-p", "--photon_energy", type=float, default=None, help="Photon energy"
    )
    parser.add_option("-D", "--display", action="store_true", help="display plot")
    parser.add_option(
        "-E",
        "--extract",
        action="store_true",
        help="Extract the calibrated diode after the scan",
    )
    parser.add_option("-A", "--analysis", action="store_true", help="Analyze the scan")
    parser.add_option("-C", "--conclude", action="store_true", help="Apply the offsets")

    options, args = parser.parse_args()

    print("options", options)
    print("args", args)

    filename = (
        os.path.join(options.directory, options.name_pattern) + "_parameters.pickle"
    )

    slscan = slit_scan(
        options.name_pattern,
        options.directory,
        slits=options.slits,
        start_position=options.start_position,
        end_position=options.end_position,
        photon_energy=options.photon_energy,
        display=options.display,
        extract=options.extract,
    )

    if not os.path.isfile(filename):
        slscan.execute()
    if options.analysis == True:
        slscan.analyze()
    if options.conclude == True:
        slscan.conclude()


def analysis():
    import optparse

    parser = optparse.OptionParser()

    parser.add_option(
        "-f",
        "--filename",
        type=str,
        default="/tmp/slit_scan_parameters.pickle",
        help="File storing parameters of the slit scan",
    )

    options, args = parser.parse_args()

    print("options", options)
    print("args", args)

    ssa = slit_scan_analysis(options.filename)

    ssa.analyze()
    ssa.conclude()


if __name__ == "__main__":
    main()
    # analysis()
