#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import sys
import glob
import logging
import gzip

sys.path.insert(0, "/usr/local/adxv_class")
from adxv_socket import adxvsocket


def get_spots(spots_file):
    try:
        spots = [
            list(map(float, item.split()))
            for item in gzip.open(spots_file, "rb")
            .read()
            .decode(encoding="ascii")
            .split("\n")[:-1]
        ]
    except:
        spots = []
    return spots


class image_monitor:
    def __init__(
        self,
        name_pattern,
        directory,
        total_expected_images=3600,
        image_format="cbf.gz",
        sleeptime=0.5,
        timeout=60,
        host="172.19.10.125",
        port=8100,
        show_spots=True,
    ):
        self.name_pattern = name_pattern
        self.directory = directory
        self.total_expected_images = total_expected_images
        self.format_directory = {"name_pattern": name_pattern, "directory": directory}
        self.sleeptime = sleeptime
        self.timeout = timeout
        self.host = host
        self.port = port
        # self.pattern = os.path.join(directory, '%s_cbf' % name_pattern, '%s_*.cbf.gz' % name_pattern)
        self.pattern = os.path.join(directory, "%s_*.cbf.gz" % name_pattern)
        # self.adxv = adxvsocket(self.host, self.port)
        self.start = time.time()
        self.show_spots = show_spots

    def run(self):
        nimages = 0
        while (time.time() - self.start < self.timeout) or (
            nimages < self.total_expected_images and nimages >= 0
        ):
            if os.path.isdir(self.directory):
                os.system("touch %s" % self.directory)

            images = glob.glob(self.pattern)
            # images = [item for item in os.listdir('./') if 'cbf.gz' in item]

            if len(images):
                images.sort()
            if len(images) > nimages:
                image_to_load = images[-1]
                self.reconnect()
                # logging.info('adxv load image %s' % (image_to_load))

                if os.path.isfile(image_to_load):
                    self.load_image(image_to_load)
                if self.show_spots:
                    spots_file = os.path.join(
                        os.path.dirname(image_to_load),
                        "spot_list",
                        os.path.basename(image_to_load).replace(".cbf.gz", ".adx.gz"),
                    )
                    # logging.info('adxv load spots %s' % (spots_file))
                    spots = get_spots(spots_file)
                    self.adxv.define_spot("blue", radius=4, box=7)
                    if spots:
                        self.adxv.load_spots(spots)
                nimages = len(images)
                self.disconnect()
                time.sleep(self.sleeptime)
            if nimages >= self.total_expected_images and nimages > 0:
                sys.exit()
            time.sleep(self.sleeptime)

    def load_image(self, image_name, raise_windows=False):
        self.adxv.load_image(image_name)
        if raise_windows:
            self.adxv.raise_window("Image")
            self.adxv.raise_window("Load")

    def reconnect(self):
        self.adxv = adxvsocket(self.host, self.port)
        try:
            self.adxv.clientsocket.connect((self.host, self.port))
            self.adxv.logger.debug(
                "Connected to Host:Port - %s:%i" % (self.host, self.port)
            )
        except Exception as e:
            self.adxv.logger.debug(e)

    def disconnect(self):
        try:
            self.adxv.clientsocket.close()
            self.adxv.logger.debug(
                "DisConnected from Host:Port - %s:%i" % (self.host, self.port)
            )
        except Exception as e:
            self.adxv.logger.debug(e)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n", "--name_pattern", default="prefix_1", type=str, help="name_pattern"
    )
    parser.add_argument(
        "-d",
        "--directory",
        default="/nfs/data2/orphaned_collects",
        type=str,
        help="directory",
    )
    parser.add_argument(
        "-t",
        "--total_expected_images",
        default=3600,
        type=int,
        help="total_expected_images",
    )
    parser.add_argument("-H", "--host", default="172.19.10.125", type=str, help="host")
    parser.add_argument("-p", "--port", default=8100, type=int, help="port")
    parser.add_argument("-T", "--timeout", default=180, type=float, help="timeout")
    parser.add_argument(
        "-s", "--dont_show_spots", action="store_false", help="don't show spots"
    )
    args = parser.parse_args()
    print("image monitor args", args)
    print("bool(args.dont_show_spots)", bool(args.dont_show_spots))

    im = image_monitor(
        args.name_pattern,
        args.directory,
        total_expected_images=args.total_expected_images,
        host=args.host,
        port=args.port,
        timeout=args.timeout,
        show_spots=bool(args.dont_show_spots),
    )
    im.run()


if __name__ == "__main__":
    main()
