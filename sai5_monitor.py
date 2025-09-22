#!/usr/bin/python

from monitor import sai


def main():
    sai5 = sai(
        device_name="i11-ma-c00/ca/sai.5",
        number_of_channels=4,
        sleeptime=5.0,
        use_redis=True,
    )
    sai5.run_history()


if __name__ == "__main__":
    main()
