#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gevent
import traceback

try:
    import tango
except ImportError:
    import PyTango as tango
import time
from scipy.constants import h, c, angstrom, kilo, eV
from math import sin, radians


class motor(object):
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


class tango_motor_mockup(motor):
    def __init__(self, device_name, check_time=0.1):
        self.device_name = device_name


class tango_motor(motor):
    def __init__(self, device_name, check_time=0.1):
        self.device_name = device_name
        self.device = tango.DeviceProxy(device_name)
        self.check_time = check_time
        self.observations = []
        self.observation_fields = ["chronos", "position"]
        self.monitor_sleep_time = 0.05
        self.position_attribute = "position"
        self.name = device_name

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

    def set_position(
        self,
        position,
        wait=True,
        wait_timeout=60,
        timeout=None,
        accuracy=0.001,
        turnoff=False,
        nattempts=7,
    ):
        start_move = time.time()

        success = False
        if position == None or abs(self.get_position() - position) <= accuracy:
            position = self.get_position()
            success = True
        attempted = 0
        while not success and attempted < nattempts:
            if self.get_state() == "OFF":
                self.device.on()
            self.wait(timeout=wait_timeout / 5)
            try:
                self.device.write_attribute(self.position_attribute, position)
            except:
                traceback.print_exc()
            if wait == True:
                self.wait(timeout=wait_timeout)
            if abs(self.get_position() - position) <= accuracy:
                success = True
            attempted += 1
            gevent.sleep(wait_timeout / 60)

        if success:
            print(
                f"{self.device_name}, move to {position:.4f} took {time.time() - start_move:.4f} seconds (try {attempted})"
            )
        else:
            print(
                f"{self.device_name}, move to {position:.4f} was not successful (current position {self.get_position()}), abandoning after {time.time() - start_move:.4f} (try {attempted})"
            )
        if turnoff:
            self.device.off()

    def wait(self, timeout=None):
        start = time.time()
        while self.get_state() != "STANDBY":
            if self.get_state() == "ALARM":
                self.device.position -= 5 * self.device.accuracy
                gevent.sleep(5)
                self.device.position += 5 * self.device.accuracy
            gevent.sleep(self.check_time)
            if timeout != None and abs(time.time() - start) > timeout:
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
        tango_motor.__init__(self, device_name)

    def get_point(self):
        return self.get_position()


class monochromator_rx_motor(tango_motor):
    def __init__(self, device_name="i11-ma-c03/op/mono1-mt_rx"):
        tango_motor.__init__(self, device_name)

    def get_thetabragg(self):
        return self.device.position

    def get_wavelength(self, thetabragg=None, d=3.1347507142511746):
        if thetabragg == None:
            thetabragg = self.get_position()
        return 2 * d * sin(radians(thetabragg))

    def get_energy(self, thetabragg=None):
        if thetabragg == None:
            thetabragg = self.get_position()
        wavelength = self.get_wavelength(thetabragg=thetabragg)
        return h * c / (angstrom * wavelength * kilo * eV)

    def get_position(self):
        return self.get_thetabragg()

    def get_point(self):
        return self.get_position()


class monochromator(tango_motor):
    def __init__(self, device_name="i11-ma-c03/op/mono1"):
        tango_motor.__init__(self, device_name)

    def get_thetabragg(self):
        return self.device.thetabragg

    def get_wavelength(self):
        return self.device.Lambda

    def get_energy(self):
        return self.device.energy

    def get_position(self):
        return self.get_energy(), self.get_thetabragg(), self.get_wavelength()

    def get_point(self):
        return self.get_position()


class undulator(tango_motor):
    def __init__(self, device_name="ans-c11/ei/m-u24"):
        tango_motor.__init__(self, device_name)
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
        motor.__init__(self)


class monochromator_rx_motor_mockup(motor):
    def __init__(self):
        motor.__init__(self)


class tango_named_positions_motor(tango_motor):
    def __init__(self, device_name):
        tango_motor.__init__(self, device_name)

    def set_named_position(self, named_position):
        return getattr(self.device, named_position)()


class md_motor(motor):
    def __init__(self, motor_name, md_name="i11-ma-cx1/ex/md3"):
        self.md = tango.DeviceProxy(md_name)
        self.motor_name = motor_name
        self.motor_full_name = "%sPosition" % motor_name
        self.check_time = 0.1

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
            gevent.sleep(self.check_time)
            if timeout != None and abs(time.time() - start) > timeout:
                break

    def stop(self):
        self.md.abort()

    def get_state(self):
        return dict([item.split("=") for item in md.motorstates])[self.motor_name]
