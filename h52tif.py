#!/usr/bin/env python

import os
import h5py

# import hdf5plugin
import fabio


def get_header(images):
    header = {}
    for key, value in images.h5["entry/instrument/detector"].items():
        try:
            val = value[()]
        except:
            print("%s: unprintable" % key)
        else:
            print("%s: %s" % (key, val))
            header[key] = val
    return header


def main():
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-m", "--master", default=None, type=str, help="Name of the master file"
    )

    args = parser.parse_args()

    images = fabio.open(args.master)
    header = get_header(images)
    print("header")
    print(header)
    image_path = os.path.dirname(os.path.abspath(args.master))
    filename_template = os.path.basename(args.master).replace(
        "_master.h5", "_auto_%d.tif"
    )
    print("images", images)
    if images.nframes > 1:
        for idx, frame in enumerate(images):
            fname = filename_template % (idx + 1)
            print(fname)
            if os.path.isfile(fname):
                pass
            else:
                tif = fabio.tifimage.tifimage(header=header, data=frame.data)
                tif.write(fname)

    else:
        fname = filename_template.replace("_%06d", "")
        print(fname)
        if os.path.isfile(fname):
            pass
        else:
            tif = fabio.tifimage.tifimage(header=header, data=images.data)
            tif.write(fname)


if __name__ == "__main__":
    main()
