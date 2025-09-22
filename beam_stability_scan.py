#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Beam stability scan.
"""
import os
import time
import gevent
from xray_experiment import xray_experiment
from monitor import xray_camera, analyzer


class beam_stability_scan(xray_experiment):
    specific_parameter_fields = [
        {
            "name": "observation_period",
            "type": "float",
            "description": "Intended observation period in s",
        },
        {
            "name": "default_slit_gap",
            "type": "float",
            "description": "Default slits gap in mm",
        },
    ]

    def __init__(
        self,
        name_pattern,
        directory,
        observation_period=300.0,
        default_slit_gap=4.0,
        transmission=None,
        diagnostic=True,
        analysis=None,
        conclusion=None,
        simulation=None,
        display=False,
        extract=False,
    ):
        if hasattr(self, "parameter_fields"):
            self.parameter_fields += beam_stability_scan.specific_parameter_fields
        else:
            self.parameter_fields = beam_stability_scan.specific_parameter_fields[:]

        xray_experiment.__init__(
            self,
            name_pattern,
            directory,
            transmission=transmission,
            diagnostic=diagnostic,
            analysis=analysis,
            conclusion=conclusion,
            simulation=simulation,
        )

        self.description = "Beam stability scan, Proxima 2A, SOLEIL, %s" % (
            time.ctime(self.timestamp),
        )
        self.observation_period = observation_period
        self.default_slit_gap = default_slit_gap
        self.extract = extract
        self.display = display
        self.monitor_device = None
        self.total_expected_exposure_time = self.observation_period
        self.total_expected_wedges = 1

    def set_up_monitor(self):
        self.monitor_device = xray_camera()

        self.auxiliary_monitor_device = analyzer(
            continuous_monitor_name="focus_monitor"
        )
        self.monitors_dictionary["analyzer"] = self.auxiliary_monitor_device
        self.monitor_names += ["analyzer"]
        self.monitors += [self.auxiliary_monitor_device]

    def prepare(self):
        self.set_up_monitor()

        self.check_directory(self.directory)
        self.write_destination_namepattern(self.directory, self.name_pattern)

        initial_settings_a = []

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
                    self.default_slit_gap,
                )
            )
            initial_settings_a.append(
                gevent.spawn(
                    getattr(getattr(self, "slits%d" % k), "set_vertical_gap"),
                    self.default_slit_gap,
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

        initial_settings_b.append(gevent.spawn(self.monitor_device.insert))

        gevent.joinall(initial_settings_b)

    def run(self):
        print("in run")
        print("self.fast_shutter", self.fast_shutter)
        self.fast_shutter.open()
        gevent.sleep(self.observation_period)
        self.fast_shutter.close()

    def clean(self):
        self.collect_parameters()
        self.save_parameters()
        self.save_results()
        self.save_log()

        final_settings_a = []

        for k in self.get_clean_slits():  # [1, 2, 3, 5, 6]:
            final_settings_a.append(
                gevent.spawn(
                    getattr(getattr(self, "slits%d" % k), "set_horizontal_gap"),
                    self.default_slit_gap,
                )
            )
            final_settings_a.append(
                gevent.spawn(
                    getattr(getattr(self, "slits%d" % k), "set_vertical_gap"),
                    self.default_slit_gap,
                )
            )

        gevent.joinall(final_settings_a)

        final_settings_b = []
        for k in self.get_clean_slits():  # [1, 2, 3, 5, 6]:
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

        if self.extract:
            final_settings_b.append(gevent.spawn(self.monitor_device.extract))

        gevent.joinall(final_settings_b)


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
    parser.add_option(
        "-s", "--observation_period", type=float, default=300, help="observation period"
    )
    parser.add_option("-D", "--display", action="store_true", help="display plot")
    parser.add_option(
        "-E",
        "--extract",
        action="store_true",
        help="Extract the calibrated diode after the scan",
    )

    options, args = parser.parse_args()

    print("options", options)
    print("args", args)

    filename = (
        os.path.join(options.directory, options.name_pattern) + "_parameters.pickle"
    )

    stability = beam_stability_scan(**vars(options))

    if not os.path.isfile(filename):
        stability.execute()


if __name__ == "__main__":
    main()
