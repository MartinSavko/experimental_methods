#!/usr/bin/python

from monitor import xray_camera


def main():
    xc = xray_camera()
    xc.run()


if __name__ == "__main__":
    main()
