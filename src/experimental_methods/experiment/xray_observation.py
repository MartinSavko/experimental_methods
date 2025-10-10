#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
x-ray beam observation
"""

import gevent
import time
from experimental_methods.experiment.xray_experiment import xray_experiment
from experimental_methods.instrument.monitor import oav_camera, tdl_xbpm
from experimental_methods.instrument.motor import tango_motor


class xray_observation(xray_experiment):
    specific_parameters_fields = [
        {
            "name": "duration_intention",
            "type": "float",
            "description": "intended duration in seconds",
        },
        {
            "name": "fast_shutter_control",
            "type": "bool",
            "description": "whether to attempt to control fast shutter",
        },
    ]

    def __init__(
        self,
        name_pattern,
        directory,
        duration_intention=60.0,
        fast_shutter_control=False,
        photon_energy=None,
        transmission=None,
        diagnostic=True,
    ):
        if hasattr(self, "parameter_fields"):
            self.parameter_fields += xray_observation.specific_parameter_fields
        else:
            self.parameter_fields = xray_observation.specific_parameter_fields[:]

        self.default_experiment_name = "X-ray beam observation"
        
        self.duration_intention = duration_intention
        self.fast_shutter_control = fast_shutter_control

        super().__init__(
            name_pattern=name_pattern,
            directory=directory,
            photon_energy=photon_energy,
            transmission=transmission,
            diagnostic=diagnostic,
        )

        self.specific_observers = [
            {"name": "oav_camera", "object": oav_camera},
        ]

        for name, device_name in [
            ("vfm_pitch", "i11-ma-c05/op/mir.2-mt_rx"),
            ("hfm_pitch", "i11-ma-c05/op/mir.3-mt_rz"),
            ("vfm_trans", "i11-ma-c05/op/mir.2-mt_tz"),
            ("hfm_trans", "i11-ma-c05/op/mir.3-mt_tx"),
            ("tab2_tx1", "i11-ma-c05/ex/tab.2-mt_tx.1"),
            ("tab2_tx2", "i11-ma-c05/ex/tab.2-mt_tx.2"),
            ("tab2_tz1", "i11-ma-c05/ex/tab.2-mt_tz.1"),
            ("tab2_tz2", "i11-ma-c05/ex/tab.2-mt_tz.2"),
            ("tab2_tz3", "i11-ma-c05/ex/tab.2-mt_tz.3"),
            ("mono_mt_rx", "i11-ma-c03/op/mono1-mt_rx"),
            ("mono_mt_rx_fine", "i11-ma-c03/op/mono1-mt_rx_fine"),
        ]:
            so = {
                "name": name,
                "object": tango_motor,
                "kwargs": {"device_name": device_name},
            }
            self.specific_observers.append(so)

        self.initialize_observers(observers=self.specific_observers)

        self.total_expected_wedges = 1
        self.total_expected_exposure_time = duration_intention

        self.motors_to_control = []
        # ['vfm_pitch', 'hfm_pitch', 'vfm_trans', 'hfm_trans', 'tab2_tx1', 'tab2_tx2', 'tab2_tz1', 'tab2_tz2', 'tab2_tz3', 'mono_mt_rx']

    def run(self):
        gevent.sleep(self.duration_intention)

    def open_fast_shutter(self):
        if self.fast_shutter_control:
            try:
                self.fast_shutter.open()
            except:
                print("Could not open the fast shutter")

    def close_fast_shutter(self):
        if self.fast_shutter_control:
            try:
                self.fast_shutter.close()
            except:
                print("Could not close the fast shutter")

    def prepare(self):
        # super().prepare()
        self.open_fast_shutter()
        for motor in self.motors_to_control:
            self.monitors_dictionary[motor].device.off()

    def clean(self):
        self.close_fast_shutter()
        super().clean()
        for motor in self.motors_to_control:
            self.monitors_dictionary[motor].device.on()


def main():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-n",
        "--name_pattern",
        default="xray_observation",
        type=str,
        help="name pattern",
    )
    parser.add_argument("-d", "--directory", default="/tmp", type=str, help="directory")
    parser.add_argument(
        "-i",
        "--duration_intention",
        default=60.0,
        type=float,
        help="intended observation duration",
    )
    parser.add_argument(
        "-p", "--photon_energy", default=None, type=float, help="photon energy"
    )
    parser.add_argument(
        "-t", "--transmission", default=None, type=float, help="transmission"
    )
    parser.add_argument(
        "-f", "--fast_shutter_control", default=0, type=int, help="fast shutter control"
    )
    args = parser.parse_args()
    print("args", args)

    xo = xray_observation(
        name_pattern=args.name_pattern,
        directory=args.directory,
        duration_intention=args.duration_intention,
        photon_energy=args.photon_energy,
        transmission=args.transmission,
        fast_shutter_control=bool(args.fast_shutter_control),
    )

    xo.execute()


if __name__ == "__main__":
    main()
