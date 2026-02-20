#!/usr/bin/env python

from beam_position_controller import speaking_bpc

if __name__ == "__main__":
    speaking_bpc(monitor="cam", actuator="horizontal_trans", channels=(1,), period=1)
