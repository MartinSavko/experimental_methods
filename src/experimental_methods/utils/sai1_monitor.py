#!/usr/bin/python

from experimental_methods.instrument.monitor import sai


def main():
    sai1 = sai(
        device_name="i11-ma-c00/ca/sai.1",
        number_of_channels=4,
        sleeptime=5.0,
        use_redis=True,
    )
    sai1.run_history()


if __name__ == "__main__":
    main()
