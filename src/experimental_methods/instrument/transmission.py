#!/usr/bin/env python
# -*- coding: utf-8 -*-


import pickle
from scipy.interpolate import interp1d, RectBivariateSpline
import numpy as np
import math
import logging
import time
import random
import sys

from experimental_methods.utils.speech import speech, defer
from experimental_methods.instrument.slits import slits1, slits2

def integrate(distribution, start, end, use_skimage=False):
    transmission = np.abs(
        distribution[start[0] : end[0] + 1, start[1] : end[1] + 1].sum()
    )
    return transmission


class transmission_mockup:
    def __init__(self):
        self.transmission = 1

    def get_transmission(self):
        return self.transmission

    def set_transmission(self, transmission):
        self.transmission = transmission


class transmission(speech):
    service = None
    server = None

    def __init__(
        self,
        slits2_reference_distribution="/usr/local/slits_reference/distribution_s2_observe_2025-09-03.npy",
        slits2_reference_ii="/usr/local/slits_reference/ii_s2_observe_2025-09-03.npy",
        reference_gap=4.0,
        reference_position=0.0,
        steps=4000,
        percent_factor=100.0,
        port=5555,
        service=None,
        verbose=False,
    ):
        logging.basicConfig(
            format="%(asctime)s|%(module)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            level=logging.INFO,
        )

        self.reference_gap = reference_gap
        self.reference_position = reference_position
        self.s2 = slits2()
        self.percent_factor = percent_factor
        self.slits2_reference_distribution = slits2_reference_distribution
        self.slits2_reference_ii = slits2_reference_ii
        self.distribution = None
        self.ii = None
        self.transmissions = None
        self.predict_gap_from_transmission = None
        self.id = random.random()
        self.verbose = verbose
        self.gaps = np.linspace(0, reference_gap, steps)
        speech.__init__(self, port=port, service=service, verbose=verbose)

    def load_reference(self):
        _start = time.time()
        self.distribution = np.load(self.slits2_reference_distribution)
        self.ii = np.load(self.slits2_reference_ii)
        self.transmissions = [
            self.get_hypothetical_transmission(gap, self.distribution)
            for gap in self.gaps
        ]
        self.predict_gap_from_transmission = interp1d(
            self.transmissions, self.gaps, fill_value=tuple([0, 4]), bounds_error=False
        )
        logging.info(f"loading reference took {time.time() - _start}")

    @defer
    def get_transmission(self):
        if self.distribution is None or self.ii is None:
            self.load_reference()

        start, end = self.get_indices_for_slit_setting()
        transmission = integrate(self.distribution, start, end)
        transmission *= self.percent_factor
        message = (
            "transmission server id %s received request to get transmission %.2f"
            % (self.id, transmission)
        )
        logging.info(message)
        self.value = transmission
        return transmission

    def get_value(self):
        value = self.get_transmission()
        print(f"transmission, in get_value {value}")
        return value

    @defer
    def get_hypothetical_transmission(self, gap, distribution):
        if self.ii is None or self.distribution is None:
            self.load_reference()

        start, end = self.get_indices_for_slit_setting(
            horizontal_gap=gap,
            vertical_gap=gap,
            vertical_center=0.0,
            horizontal_center=0.0,
        )

        hypothetical_transmission = (
            integrate(distribution, start, end) * self.percent_factor
        )
        return hypothetical_transmission

    @defer
    def set_transmission(self, transmission, factor=1, epsilon=1e-3, wait=True):
        start = time.time()
        if transmission > 100:
            logging.info(
                "transmission specified is larger then 100 percent %s setting to 100"
                % transmission
            )
            transmission = 100

        if self.predict_gap_from_transmission is None:
            self.load_reference()

        gap = self.predict_gap_from_transmission(transmission)
        _start = time.time()
        self.s2.set_horizontal_gap(gap, wait=wait)
        logging.info(f"move h took {time.time() - _start}")
        _start = time.time()
        self.s2.set_vertical_gap(gap, wait=wait)
        logging.info(f"move v took {time.time() - _start}")

        self.value = self.get_transmission()
        self.timestamp = time.time()
        self.value_id += 1
        self.sing()
        logging.info("set_transmission took %.4f" % (time.time() - start))

    def get_indices_for_slit_setting(
        self,
        horizontal_gap=None,
        vertical_gap=None,
        horizontal_center=None,
        vertical_center=None,
        npixels=4000,
        extent=(-2, 2),
    ):
        e = extent[1] - extent[0]
        pixels_per_mm = npixels / e
        if horizontal_gap is None:
            horizontal_gap = self.s2.get_horizontal_gap()
        if vertical_gap is None:
            vertical_gap = self.s2.get_vertical_gap()
        if horizontal_center is None:
            horizontal_center = self.s2.get_horizontal_position()
        if vertical_center is None:
            vertical_center = self.s2.get_vertical_position()

        horizontal_start = (-horizontal_gap / 2.0 - horizontal_center) - extent[0]
        horizontal_end = (horizontal_gap / 2.0 - horizontal_center) - extent[0]
        vertical_start = (-vertical_gap / 2.0 - vertical_center) - extent[0]
        vertical_end = (vertical_gap / 2.0 - vertical_center) - extent[0]

        horizontal_start *= pixels_per_mm
        horizontal_end *= pixels_per_mm
        vertical_start *= pixels_per_mm
        vertical_end *= pixels_per_mm

        return (int(vertical_start), int(horizontal_start)), (
            int(min(npixels - 1, vertical_end)),
            int(min(npixels - 1, horizontal_end)),
        )


def test():
    import argparse
    import gevent

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", type=int, default=0, help="master")
    args = parser.parse_args()
    t = transmission(verbose=bool(args.verbose))
    logging.info("current transmission: %s" % (t.get_transmission()))

    while t.server:
        gevent.sleep(5)


if __name__ == "__main__":
    test()
