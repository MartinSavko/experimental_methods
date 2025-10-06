#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gevent
import time

from .eiger import eiger
from .protective_cover import protective_cover
from .detector_position import detector_position
from .detector_beamstop import detector_beamstop
# from speaking_goniometer import speaking_goniometer
from .goniometer import goniometer


class detector(eiger):
    def __init__(
        self, host="172.19.10.26", port=80, pixel_size=0.075, beamstopx_offset=96.5
    ):
        eiger.__init__(self, host=host, port=port)
        self.position = detector_position()
        self.beamstop = detector_beamstop()
        self.cover = protective_cover()
        self.goniometer = goniometer()
        self.beamstopx_offset = beamstopx_offset

    def insert_protective_cover(
        self, safe_distance=120.0, delta=1.0, timeout=3.0, wait=True
    ):
        self.cover.insert(wait=wait)
        # start = time.time()
        # while (
        # self.position.ts.get_position() < safe_distance
        # and time.time() - start < timeout
        # ):
        # gevent.sleep(0.5)
        # if time.time() - start > timeout:
        # self.position.ts.set_position(safe_distance + 1, wait=True)
        # if self.position.ts.get_position() >= safe_distance:
        # self.cover.insert(wait=wait)

    def extract_protective_cover(
        self, safe_distance=120.0, delta=1.0, timeout=3.0, wait=True
    ):
        self.cover.extract(wait=wait)
        # current_position = self.position.ts.get_position()
        # if current_position < safe_distance:
        # self.position.ts.set_position(safe_distance + delta, wait=True)
        # if self.position.ts.get_position() >= safe_distance:
        # self.cover.extract(wait=wait)
        # self.position.ts.set_position(current_position, wait=True)

    def set_ts_position(self, position, tolerance=1, wait=True):
        if abs(self.position.ts.get_position() - position) > 5.0:
            self.insert_protective_cover(wait=wait)
        if self.get_beamstopx_distance(position - tolerance) > 0:
            self.position.ts.set_position(position, wait=wait)
        else:
            raise Exception(
                "Requested detector position would likely result in a collision with the goniometer. Refusing to move. Please check!"
            )

    def get_beamstopx_distance(self, detector_ts_position=None):
        if detector_ts_position is None:
            detector_ts_position = self.position.ts.get_position()
        beamstopx_position = self.goniometer.md.BeamstopXPosition

        beamstopx_distance = (
            detector_ts_position - beamstopx_position - self.beamstopx_offset
        )

        return beamstopx_distance


if __name__ == "__main__":
    import optparse

    parser = optparse.OptionParser()
    parser.add_option(
        "-i", "--ip", default="172.19.10.26", type=str, help="IP address of the server"
    )
    parser.add_option(
        "-p",
        "--port",
        default=80,
        type=int,
        help="port on which to which it listens to",
    )

    options, args = parser.parse_args()

    d = detector(options.ip, options.port)
