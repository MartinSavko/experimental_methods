#!/usr/bin/env python

from experimental_methods.utils.bpc import main as bpc_main

def main():
    bpc_main(monitor="cam", actuator="vertical_trans", channels=(0,), period=1)

if __name__ == "__main__":
    main()
