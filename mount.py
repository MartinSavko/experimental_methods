#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sample mount
"""

import os
import time
from experiment import experiment


class mount(experiment):
    specific_parameter_fields = [
        {
            "name": "puck",
            "type": "int",
            "description": "Puck number",
        },
        {
            "name": "sample",
            "type": "int",
            "description": "Sample number",
        },
        {
            "name": "prepare_centring",
            "type": "bool",
            "description": "Prepare centring",
        },
        {
            "name": "success",
            "type": "bool",
            "description": "Success",
        },
        {
            "name": "prepare_centring",
            "type": "bool",
            "description": "prepare centring",
        },
    ]

    def __init__(
        self,
        puck,
        sample,
        wash=False,
        unload=False,
        prepare_centring=True,
        name_pattern=None,
        directory=None,
        cats_api=None,
    ):
        if hasattr(self, "parameter_fields"):
            self.parameter_fields += self.specific_parameter_fields[:]
        else:
            self.parameter_fields = self.specific_parameter_fields[:]

        self.default_experiment_name = "Sample mount"

        self.timestamp = time.time()
        self.puck = puck
        self.sample = sample
        self.wash = wash
        self.unload = unload
        self.prepare_centring = prepare_centring

        name_pattern = self.set_name_pattern(name_pattern)

        if directory is None:
            directory = os.path.join(
                os.environ["HOME"],
                "manual_optical_alignment",
            )

        experiment.__init__(
            self,
            name_pattern=name_pattern,
            directory=directory,
        )

        self.success = None
        if cats_api is None:
            from cats import cats

            self.sample_changer = cats()
        else:
            self.sample_changer = cats_api

    def get_designation(self, name_pattern=None):
        if name_pattern is None:
            if self.use_sample_changer():
                if self.unload:
                    designation = f"umount_{self.get_element()}"
                elif self.wash:
                    designation = f"wash_{self.get_element()}"
                else:
                    designation = f"mount_{self.get_element()}"
            else:
                designation = "manually_mounted"
        return designation

    def set_name_pattern(self, name_pattern=None):
        designation = self.get_designation(name_pattern)
        timestring = self.get_timestring()
        self.name_pattern = f"{designation}_{timestring}"
        return self.name_pattern

    def use_sample_changer(self):
        return not -1 in (self.puck, self.sample)

    def manually_mounted(self):
        return (
            self.anything_mounted
            and -1 in self.sample_changer.get_mounted_puck_and_sample()
        )

    def anything_mounted(self):
        return int(self.sample_changer.sample_mounted())

    def sample_mounted(self):
        mpuck, msample = self.sample_changer.get_mounted_puck_and_sample()
        return mpuck == self.puck and msample == self.sample

    def mount(self):
        if not self.sample_mounted():
            print("sample is about to be mounted ...")
            print("\n" * 5)
            self.sample_changer.mount(
                self.puck, self.sample, prepare_centring=not self.wash
            )
            if self.anything_mounted() and self.wash:
                self.sample_changer.mount(
                    self.puck, self.sample, prepare_centring=self.prepare_centring
                )
        elif self.wash and not self.manually_mounted():
            print(
                "washing the sample, (all is okay, sample changer is aware of it ...)"
            )
            print("\n" * 5)
            self.sample_changer.mount(
                self.puck, self.sample, prepare_centring=self.prepare_centring
            )

        return self.sample_mounted()

    def unmount(self):
        if self.anything_mounted() and not self.manually_mounted():
            self.sample_changer.umount()
        return not self.sample_mounted()

    def prepare(self):
        super().prepare()
        if self.use_sample_changer():  # and self.sample_changer.isoff():
            try:
                self.sample_changer.on()
            except:
                pass
        self.sample_changer.set_camera("cam14_quad")

    def run(self):
        if not self.unload:
            self.success = self.mount()
        else:
            self.success = self.unmount()
        return self.success

    def get_success(self):
        return self.success


def main():
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-p", "--puck", default=-1, type=int, help="puck")
    parser.add_argument("-s", "--sample", default=-1, type=int, help="sample")
    parser.add_argument(
        "-d",
        "--directory",
        default="/tmp/mount25",
        help="directory",
    )
    parser.add_argument(
        "-n", "--name_pattern", default=None, type=str, help="name_pattern"
    )
    parser.add_argument("-w", "--wash", action="store_true", help="wash")
    parser.add_argument(
        "-N", "--prepare_centring", action="store_false", help="prepare_centring"
    )

    args = parser.parse_args()

    print("args", args)

    m = mount(
        args.puck,
        args.sample,
        directory=args.directory,
        wash=bool(args.wash),
        prepare_centring=bool(args.prepare_centring),
    )

    m.execute()


if __name__ == "__main__":
    main()
