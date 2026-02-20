#!/usr/bin/env python
# -*- coding: utf-8 -*-

from xray_experiment import xray_experiment

from motor import monochromator_rx_motor
from historian import historian

from useful_routines import (
    get_energy_from_wavelength,
    get_wavelength_from_energy,
    get_theta_from_wavelength,
    get_wavelength_from_theta,
    get_theta_from_energy,
    get_energy_from_theta,
)

Selenium_K_edge = 12658.0


class monochromator_scan(xray_experiment):
    default_speed = 0.5

    def __init__(
        self,
        name_pattern,
        directory,
        scan_start_angle,
        scan_range,
        scan_speed,
        undulator_gap=None,
    ):
        super().__init__(
            name_pattern,
            directory,
        )

        self.actuator = monochromator_rx_motor()

        self.scan_start_angle = scan_start_angle
        self.scan_range = scan_range
        self.scan_speed = scan_speed
        self.undulator_gap = undulator_gap

        self.scan_end_angle = self.scan_start_angle + self.scan_range
        self.scan_start_energy = get_energy_from_theta(self.scan_start_angle)
        self.scan_end_energy = get_energy_from_theta(self.scan_end_angle)

        self.historian = historian()

    def prepare(self):
        print("hasattr?", hasattr(self, "actuator"))
        self.actuator.set_speed(self.default_speed)
        self.actuator.set_position(self.scan_start_angle)
        if self.undulator_gap is not None:
            self.undulator.set_position(self.undulator_gap)

        self.check_directory()
        self.actuator.set_speed(self.scan_speed)
        self.fast_shutter.open()

    def run(self):
        self.actuator.set_position(self.scan_end_angle)

    def clean(self):
        self.actuator.set_speed(self.default_speed)
        self.fast_shutter.close()
        self.historian.save_history(
            self.get_template(),
            self.start_run_time,
            self.end_run_time,
            dimensions=["mono_rx", "sipin", "sai1", "sai3", "sai4", "sai5"],
        )
        super().clean()


def main():
    usage = """Program will execute a monochromator_scan
    ./monochromator_scan.py <arguments>
    """
    import os
    import pprint
    import argparse
    from useful_routines import (
        get_string_from_timestamp,
    )

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-n",
        "--name_pattern",
        type=str,
        default=f"monochromator_scan_{get_string_from_timestamp()}",
        help="name_pattern",
    )

    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        default=f'/nfs/data4/Commissioning/{get_string_from_timestamp(fmt="%Y-%m-%d")}',
        help="directory",
    )

    parser.add_argument(
        "-s",
        "--scan_start_energy",
        type=float,
        default=Selenium_K_edge - 50.0,
        help="scan start energy",
    )

    parser.add_argument(
        "-r",
        "--scan_energy_range",
        type=float,
        default=100.0,
        help="scan energy range",
    )

    parser.add_argument(
        "-e",
        "--scan_exposure_time",
        type=float,
        default=20.0,
        help="scan exposure time",
    )

    parser.add_argument(
        "-g",
        "--undulator_gap",
        type=float,
        default=None,
        help="undulator gap",
    )

    args = parser.parse_args()
    pprint.pprint(args)

    scan_start_angle = get_theta_from_energy(args.scan_start_energy)
    scan_end_angle = get_theta_from_energy(
        args.scan_start_energy + args.scan_energy_range
    )
    scan_range = scan_end_angle - scan_start_angle
    scan_speed = abs(scan_range / args.scan_exposure_time)

    print(f"scan_start_angle: {scan_start_angle:.3f}")
    print(f"scan_end_angle: {scan_end_angle:.3f}")
    print(f"scan_range: {scan_range:.3f}")
    print(f"scan_speed: {scan_speed:.6f}")

    ms = monochromator_scan(
        args.name_pattern,
        args.directory,
        scan_start_angle=scan_start_angle,
        scan_range=scan_range,
        scan_speed=scan_speed,
        undulator_gap=args.undulator_gap,
    )

    if not os.path.isfile(ms.get_parameters_filename()):
        ms.execute()


if __name__ == "__main__":
    main()
