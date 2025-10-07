#!/usr/bin/python

from experimental_methods.instrument.monitor import sai


def main():
    sai2 = sai(number_of_channels=1, use_redis=True)
    sai2.run_history()


if __name__ == "__main__":
    main()
