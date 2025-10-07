#!/usr/bin/python

from experimental_methods.instrument.monitor import xray_camera


def main():
    xc = xray_camera(history_size_threshold=25000, sleeptime=0.001, use_redis=True)
    xc.run_history()


if __name__ == "__main__":
    main()
