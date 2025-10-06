#!/usr/bin/env python

from bpc import main

if __name__ == "__main__":
    main(monitor="cam", actuator="vertical_trans", channels=(0,), period=1)
