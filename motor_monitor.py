#!/usr/bin/python

from motor import tango_motor
import gevent


def main(device_name, server=None, verbose=True, sleeptime=0.1):
    motor = tango_motor(
        device_name=device_name,
        # server=server,
        # verbose=verbose,
        sleeptime=sleeptime,
    )
    while motor.server:
        gevent.sleep(motor.sleeptime)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-d",
        "--device_name",
        default="i11-ma-cx1/dt/dtc_ccd.1-mt_ts",
        type=str,
        help="device_name",
    )
    args = parser.parse_args()

    main(args.device_name)
