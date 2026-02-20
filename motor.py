#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import gevent
import traceback
import numpy as np

try:
    import tango
except ImportError:
    import PyTango as tango

import time
from scipy.constants import h, c, angstrom, kilo, eV
from math import sin, radians

from useful_routines import position_valid


class motor(object):
    def __init__(
        self,
        name="unnamed",
        sleeptime=0.05,
    ):
        self.name = name
        self.sleeptime = sleeptime

    def get_speed(self):
        pass

    def set_speed(self, speed):
        pass

    def get_position(self):
        pass

    def set_position(self, position, wait=True, timeout=None):
        pass

    def get_offset(self):
        pass

    def set_offset(self, offset):
        pass

    def stop(self):
        pass

    def get_state(self):
        pass

    def get_limits(self):
        pass


class tango_motor_mockup(motor):
    def __init__(self, device_name, sleeptime=0.1):
        self.device_name = device_name


class tango_motor(motor):
    def __init__(
        self,
        device_name,
        sleeptime=0.05,
    ):
        super().__init__(
            name=device_name,
            sleeptime=sleeptime,
        )

        self.device_name = device_name
        self.device = tango.DeviceProxy(device_name)
        self.observations = []
        self.observation_fields = ["chronos", "position"]
        self.monitor_sleep_time = 0.05
        self.position_attribute = "position"

    def get_name(self):
        return self.device.dev_name()

    def get_observation_fields(self):
        return self.observation_fields

    def get_observations(self):
        return self.observations

    def get_speed(self):
        return self.device.velocity

    def set_speed(self, speed):
        self.device.velocity = speed

    def get_offset(self):
        return self.device.offset

    def set_offset(self, offset):
        self.device.offset = offset

    def get_position(self):
        return self.device.position

    def get_limits(self):
        try:
            pac = self.device.get_attribute_config("position")
            min_value = float(pac.min_value)
            max_value = float(pac.max_value)
        except:
            min_value = None
            max_value = None
        return (min_value, max_value)

    def is_close(self, position, accuracy):
        return abs(self.get_position() - position) <= accuracy

    def set_position(
        self,
        position,
        wait=True,
        wait_timeout=180,
        timeout=None,
        accuracy=0.005,
        turnoff=False,
        nattempts=7,
        debug=False,
    ):
        start_move = time.time()

        success = False

        current_position = self.get_position()
        if not position_valid(position):
            if debug:
                print(
                    f"{self.device_name}, None specified as position, staying at current position {current_position:.4f} and returing -1, took {time.time() - start_move:.4f} seconds."
                )
            return current_position

        if debug:
            print(
                f"requested position is {position} {position == np.nan} {type(position)}"
            )

        limits = self.get_limits()
        if limits[0] is not None:
            if position < limits[0]:
                position = limits[0]
        if limits[1] is not None:
            if position > limits[1]:
                position = limits[1]

        try:
            accuracy = self.device.accuracy
        except:
            if debug:
                print(
                    "could not determine accuracy from the device, using the default {accuracy}"
                )
            else:
                pass

        if self.is_close(np.abs(current_position - position), accuracy):
            success = True

        attempted = 0
        while not success and attempted < nattempts:
            if self.get_state() == "OFF":
                self.device.on()
            self.wait(timeout=wait_timeout)
            try:
                self.device.write_attribute(self.position_attribute, position)
                if wait == True:
                    self.wait(timeout=wait_timeout)
            except:
                traceback.print_exc()

            if self.is_close(position, accuracy):
                success = True
            attempted += 1
            gevent.sleep(wait_timeout / 60)

        real_position = self.get_position()

        if success:
            if debug:
                print(
                    f"{self.device_name}, move to {position:.4f} took {time.time() - start_move:.4f} seconds (current position {real_position:.4f}) (try {attempted})"
                )
        else:
            if debug:
                print(
                    f"{self.device_name}, move to {position:.4f} failed (current position {real_position:.4f}), abandoning after {time.time() - start_move:.4f} seconds (try {attempted})"
                )

        if turnoff:
            self.device.off()

        return real_position

    def wait(self, timeout=None):
        start = time.time()
        while self.get_state() != "STANDBY":
            # if self.get_state() == "ALARM":
            # self.device.position -= 5 * self.device.accuracy
            # gevent.sleep(5)
            # self.device.position += 5 * self.device.accuracy
            gevent.sleep(self.sleeptime)
            if timeout is not None and abs(time.time() - start) > timeout:
                print(
                    "timeout on wait for %s took %.4f seconds"
                    % (self.device_name, time.time() - start)
                )
                break

    def get_point(self):
        return self.get_position()

    def stop(self):
        self.device.stop()

    def get_state(self):
        return self.device.state().name

    def monitor(self, start_time, actuator_names=None):
        self.observe = True
        # while self.get_state() != 'STANDBY':
        while self.observe == True:
            chronos = time.time() - start_time
            point = [chronos, self.get_point()]
            self.observations.append(point)
            gevent.sleep(self.monitor_sleep_time)


class monochromator_pitch_motor(tango_motor):
    def __init__(self, device_name="i11-ma-c03/op/mono1-mt_rx_fine"):
        super().__init__(device_name)


class monochromator_rx_motor(tango_motor):
    def __init__(
        self,
        device_name="i11-ma-c03/op/mono1-mt_rx",
        debug_frequency=100,
        service="mono_rx",
        verbose=False,
        server=False,
    ):
        super().__init__(
            device_name,
        )

    def get_thetabragg(self):
        return self.get_position()

    def get_wavelength(self, thetabragg=None, d=3.1347507142511746):
        if thetabragg == None:
            thetabragg = self.get_thetabragg()
        return 2 * d * sin(radians(thetabragg))

    def get_energy(self, thetabragg=None):
        if thetabragg == None:
            thetabragg = self.get_thetabragg()
        wavelength = self.get_wavelength(thetabragg=thetabragg)
        return h * c / (angstrom * wavelength * kilo * eV)

    def get_command_line(self):
        return f"motor.py -m monochromator_rx_motor -s {self.service} -d {self.device_name}"


class monochromator(tango_motor):
    def __init__(self, device_name="i11-ma-c03/op/mono1"):
        super().__init__(device_name)

    def get_thetabragg(self):
        return self.device.thetabragg

    def get_wavelength(self):
        return self.device.Lambda

    def get_energy(self):
        return self.device.energy

    def get_position(self):
        return self.get_energy(), self.get_thetabragg(), self.get_wavelength()


class undulator(tango_motor):
    def __init__(self, device_name="ans-c11/ei/m-u24"):
        super().__init__(device_name)
        self.position_attribute = "gap"

    def get_speed(self):
        return self.device.gapVelocity

    def set_speed(self, speed):
        return None

    def get_encoder_position(self):
        return self.device.encoder2Position

    def get_position(self):
        return self.device.gap

    def get_point(self):
        return self.get_position()


class undulator_mockup(motor):
    def __init__(self):
        super().__init__()


class monochromator_rx_motor_mockup(motor):
    def __init__(self):
        super().__init__()


class tango_named_positions_motor(tango_motor):
    def __init__(self, device_name):
        super().__init__(device_name)

    def set_named_position(self, named_position):
        return getattr(self.device, named_position)()


class md_motor(motor):
    def __init__(self, motor_name, md_name="i11-ma-cx1/ex/md3"):
        self.md = tango.DeviceProxy(md_name)
        self.motor_name = motor_name
        self.motor_full_name = "%sPosition" % motor_name
        self.sleeptime = 0.1

    def get_position(self):
        return self.md.read_attribute(self.motor_full_name).value

    def set_position(self, position, wait=True, timeout=None, accuracy=1.0e-3):
        if abs(self.get_position() - position) < accuracy:
            return
        self.md.write_attribute(self.motor_full_name, position)
        start = time.time()
        if wait == True:
            self.wait()

    def wait(self, timeout=30):
        start = time.time()
        while self.get_state() != "Ready":
            gevent.sleep(self.sleeptime)
            if timeout != None and abs(time.time() - start) > timeout:
                break

    def stop(self):
        self.md.abort()

    def get_state(self):
        return dict([item.split("=") for item in md.motorstates])[self.motor_name]


def run_mono_rx(
    device_name="i11-ma-c03/op/mono1-mt_rx",
    service="mono_rx",
    debug_frequency=100,
    verbose=False,
    server=None,
):
    mt = monochromator_rx_motor(
        device_name=device_name,
        service=service,
        debug_frequency=debug_frequency,
        verbose=verbose,
        server=server,
    )
    mt.verbose = verbose
    mt.serve()

    sys.exit(0)


def run_tango_motor(
    device_name="i11-ma-cx1/dt/dtc_ccd.1-mt_ts",
    service=None,
    debug_frequency=100,
    verbose=False,
    server=None,
):
    mt = tango_motor(
        device_name=device_name,
        service=service,
        debug_frequency=debug_frequency,
        verbose=verbose,
        server=server,
    )
    mt.verbose = verbose
    mt.serve()

    sys.exit(0)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-m",
        "--motor_class",
        default="monochromator_rx_motor",
        type=str,
        help="motor class",
    )

    parser.add_argument(
        "-d",
        "--device_name",
        default="i11-ma-c03/op/mono1-mt_rx",
        type=str,
        help="device name",
    )

    parser.add_argument(
        "-k", "--debug_frequency", default=10, type=int, help="debug frame"
    )

    parser.add_argument(
        "-s",
        "--service",
        type=str,
        default="mono_rx",
        help="service",
    )

    parser.add_argument("-v", "--verbose", action="store_true", help="verbose")
    args = parser.parse_args()
    print(args)

    if args.motor_class == "monochromator_rx_motor":
        run_mono_rx(
            service=args.service,
            device_name=args.device_name,
            debug_frequency=args.debug_frequency,
        )
    elif args.motor_class == "tango_motor":
        run_tango_motor(
            device_name=args.device_name,
            service=args.service,
            debug_frequency=args.debug_frequency,
        )


if __name__ == "__main__":
    main()
