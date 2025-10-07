#!/usr/bin/env python

from experimental_methods.utils.bpc import main as bpc_main

def main():
    bpc_main(monitor="cam", actuator="horizontal_trans", channels=(1,), period=1)
    
if __name__ == "__main__":
    main()
