#!/usr/bin/env python
# -*- coding: utf-8 -*-

from eiger import eiger
from protective_cover import protective_cover, COVER_OPERATION_MINIMUM_DISTANCE
from detector_position import detector_position
from detector_beamstop import detector_beamstop


class detector(eiger):
    def __init__(
        self, host="172.19.10.26", port=80, pixel_size=0.075,
    ):
        eiger.__init__(self, host=host, port=port)
        self.position = detector_position()
        self.beamstop = detector_beamstop()
        self.cover = protective_cover()
        self.pixel_size = pixel_size


    def insert_protective_cover(
        self, delta=1.0, timeout=3.0, wait=True
    ):
        if self.position.ts.get_position() >= COVER_OPERATION_MINIMUM_DISTANCE:
            self.cover.insert(wait=wait)


    def extract_protective_cover(
        self, delta=1.0, timeout=3.0, wait=True
    ):
        if self.position.ts.get_position() >= COVER_OPERATION_MINIMUM_DISTANCE:
            self.cover.extract(wait=wait)


    def set_ts_position(self, position, accuracy=0.01, beamstop_distance_tolerance=1., wait=True):
        self.position.ts.set_position(position)
        
    def ready_for_transfer(self):
        if not self.position.ts.is_transfer_ready():
            self.position.ts.set_transfer_position()
        else:
            self.cover.insert()
 
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
