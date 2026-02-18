#!/usr/bin/env python

import time
import pickle
import numpy as np

from beam_position_controller import get_bpc


def main(
    monitor="cam", actuator="vertical_trans", channels=(0,), period=1.0, ponm=False
):
    bpc = get_bpc(
        monitor=monitor, actuator=actuator, period=period, ponm=ponm, channels=channels
    )

    bpc.serve()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,)

    parser.add_argument("-m", "--monitor", default="cam", type=str, help="Monitor")
    parser.add_argument(
        "-a", "--actuator", default="vertical_trans", type=str, help="Actuator"
    )
    parser.add_argument("--ponm", default=0, type=int, help="ponm")
    parser.add_argument("-p", "--period", default=1.0, type=float, help="Period")
    parser.add_argument("-c", "--channels", default=(0,), type=tuple, help="Channels")

    args = parser.parse_args()

    main(
        monitor=args.monitor,
        actuator=args.actuator,
        period=args.period,
        ponm=bool(args.ponm),
        channels=channels,
    )
