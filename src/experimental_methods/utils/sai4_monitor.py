#!/usr/bin/python

from experimental_methods.instrument.monitor import sai


def main():
    sai4 = sai(
        device_name="i11-ma-c00/ca/sai.4",
        number_of_channels=4,
        sleeptime=5.0,
        use_redis=True,
    )
    sai4.run_history()


if __name__ == "__main__":
    main()
