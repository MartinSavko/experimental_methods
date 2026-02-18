#!/usr/bin/env python

import os
import sys

sys.path.insert(0, "/nfs/data/experimental_methods")
import eiger

d = eiger.eiger()


def main():
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,)

    parser.add_argument(
        "-d",
        "--destination",
        default="/nfs/data2/download_issues",
        type=str,
        help="Destination",
    )

    args = parser.parse_args()

    wget_line_template = "wget http://172.19.10.26/data/%s -O %s"
    for f in d.get_filenames():
        print("found rogue file %s" % f)
        destination = os.path.join(args.destination, f)
        destination_directory = os.path.dirname(destination)
        if not os.path.isdir(destination_directory):
            os.makedirs(destination_directory)
        wget_line = wget_line_template % (f, destination)
        print(wget_line)
        if not os.path.exists(destination):
            os.system(wget_line)
        size_on_disk = os.stat(destination).st_size
        size_on_server = os.stat("/mnt/eiger/%s" % f).st_size

        size_difference = abs(size_on_disk - size_on_server)
        if size_difference == 0:
            print(
                "size of the file on local filesystem and the server is the same, removing file from the server"
            )
            d.remove_files(f)
        else:
            print(
                "The size of the file on the local filesystem does not match that on the server, not removing the file please check."
            )


if __name__ == "__main__":
    main()
