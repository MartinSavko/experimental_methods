#!/usr/bin/env python

from bpc import speaking_bpc

if __name__ == "__main__":
    speaking_bpc(monitor="cam", actuator="vertical_trans", channels=(0,), period=1)
