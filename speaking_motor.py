#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import time
import traceback

from speech import speech
from motor import tango_motor as plain_tango_motor

from useful_routines import (
    get_wavelength_from_theta,
    get_energy_from_wavelength,
)

MOTOR_BROKER_PORT = 5557

class tango_motor(plain_tango_motor, speech):
    def __init__(
        self,
        device_name,
        sleeptime=0.05,
        port=MOTOR_BROKER_PORT,
        history_size_target=10000,
        debug_frequency=100,
        framerate_window=25,
        service=None,
        verbose=None,
        server=None,
    ):
        plain_tango_motor.__init__(
            self,
            device_name,
            sleeptime=sleeptime,
        )

        speech.__init__(
            self,
            port=port,
            sleeptime=sleeptime,
            history_size_target=history_size_target,
            debug_frequency=debug_frequency,
            framerate_window=framerate_window,
            service=service,
            verbose=verbose,
            server=server,
        )

    def acquire(self):
        try:
            self.value = self.get_point()
            self.timestamp = time.time()
            self.value_id += 1
        except:
            traceback.print_exc()

        super().acquire()
        
    
    def get_command_line(self, service=None, port=None):
        if port is None:
            port = self.port
        if service is None:
            service = self.service
        return f"speaking_motor.py -m tango_motor -s {service} -d {self.device_name} -p {port}"
    
    
class monochromator_rx_motor(tango_motor):
    
    def __init__(
        self,
        device_name="i11-ma-c03/op/mono1-mt_rx",
        service="mono_rx",
        sleeptime=0.05,
        port=MOTOR_BROKER_PORT,
        history_size_target=10000,
        debug_frequency=100,
        framerate_window=25,
        verbose=False,
        server=False,
    ):
        super().__init__(
            device_name,
            sleeptime=sleeptime,
            port=port,
            history_size_target=history_size_target,
            debug_frequency=debug_frequency,
            framerate_window=framerate_window,
            service=service,
            verbose=verbose,
            server=server,
        )

    def get_thetabragg(self):
        return self.get_position()
    
    def get_wavelength(self, thetabragg=None, d=3.1347507142511746):
        if thetabragg == None:
            thetabragg = self.get_thetabragg()
        wavelength = get_wavelength_from_theta(thetabragg, d=d)
        return wavelength

    def get_energy(self, thetabragg=None):
        if thetabragg == None:
            thetabragg = self.get_thetabragg()
        wavelength = self.get_wavelength(thetabragg=thetabragg)
        energy = get_energy_from_theta(thetabragg)
        return energy

    def get_command_line(self, service=None, port=None):
        if port is None:
            port = self.port
        if service is None:
            service = self.service
        return f"speaking_motor.py -m monochromator_rx_motor -s {service} -p {port}"
    

class undulator(tango_motor):
    def __init__(
        self, 
        device_name="ans-c11/ei/m-u24", 
        service="undulator",
        sleeptime=1,
        port=MOTOR_BROKER_PORT,
        history_size_target=10000,
        debug_frequency=100,
        framerate_window=25,
        verbose=False,
        server=False,
    ):
        super().__init__(
            device_name,
            sleeptime=sleeptime,
            port=port,
            history_size_target=history_size_target,
            debug_frequency=debug_frequency,
            framerate_window=framerate_window,
            service=service,
            verbose=verbose,
            server=server,
        )
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
    
    def get_command_line(self, service=None, port=None):
        if port is None:
            port = self.port
        if service is None:
            service = self.service
        return f"speaking_motor.py -m undulator -s {service} -p {port}"
    
    
def run_mono_rx(
    service="mono_rx", 
    debug_frequency=100, 
    verbose=False, 
    server=None,
    port=MOTOR_BROKER_PORT,
):
    
    mt = monochromator_rx_motor(
        service=service,
        debug_frequency=debug_frequency,
        verbose=verbose,
        server=server,
        port=port,
    )
    mt.verbose = verbose
    run_motor(mt)


def run_undulator(
    service="undulator", 
    debug_frequency=100, 
    verbose=False, 
    server=None,
    port=MOTOR_BROKER_PORT,
):
    
    mt = undulator(
        service=service,
        debug_frequency=debug_frequency,
        verbose=verbose,
        server=server,
        port=port,
    )
    mt.verbose = verbose
    run_motor(mt)
    
    
def run_tango_motor(
    device_name="i11-ma-cx1/dt/dtc_ccd.1-mt_ts", 
    service=None, 
    debug_frequency=100, 
    verbose=False, 
    server=None,
    port=MOTOR_BROKER_PORT,
):
    mt = tango_motor(
        device_name=device_name,
        service=service,
        debug_frequency=debug_frequency,
        verbose=verbose,
        server=server,
        port=port,
    )
    mt.verbose = verbose
    run_motor(mt)


def run_motor(mt):
    mt.serve()
    sys.exit(0)
    
    
def main():
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,)

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
    
    parser.add_argument(
        "-p", "--port", default=MOTOR_BROKER_PORT, type=int, help="port"
    )

    parser.add_argument("-v", "--verbose", action="store_true", help="verbose")
    args = parser.parse_args()
    print(args)
    
    if args.motor_class == "monochromator_rx_motor":
        run_mono_rx(
            service=args.service,
            port=args.port,
            debug_frequency=args.debug_frequency,
        )
    elif args.motor_class == "undulator":
        run_undulator(
            service=args.service,
            port=args.port,
            debug_frequency=args.debug_frequency,
        )
    elif args.motor_class == "tango_motor":
        run_tango_motor(
            device_name=args.device_name,
            service=args.service,
            port=args.port,
            debug_frequency=args.debug_frequency,
        )
    
if __name__ == "__main__":
    main()
