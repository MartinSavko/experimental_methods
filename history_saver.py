#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import traceback

from oav_camera import oav_camera
from speaking_goniometer import speaking_goniometer
from cameraman import cameraman

try:
    import simplejpeg
except ImportError:
    import complexjpeg as simplejpeg


def get_jpegs_from_arrays(images):
    jpegs = []
    for img in images:
        jpeg = simplejpeg.encode_jpeg(img)
        jpeg = np.frombuffer(jpeg, dtype="uint8")
        jpegs.append(jpeg)
    return jpegs


def classic_save(template, start, end, suffix, last_n):
    
    oac = oav_camera()
    filename_oav = "%s%s_%s.h5" % (
        template,
        suffix,
        "oav",
    )
    oac._save_history(filename_oav, start, end, last_n)
    
    try:
        sg = speaking_goniometer()
        filename_gonio = "%s%s_%s.h5" % (
            template,
            suffix,
            "gonio",
        )
        sg._save_history(filename_gonio, start, end, last_n)
    except:
        print("could not save goniometer history, please check")
        traceback.print_exc()
        
def modern_save(template, start, end, cameras):
    camm = cameraman()
    camm.save_history(
        template,
        start,
        end,
        local=True,
        cameras=cameras,
    )
    
def main():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--directory", type=str, help="directory")
    parser.add_argument("-n", "--name_pattern", type=str, help="filename template")
    parser.add_argument("-s", "--start", type=float, help="start")
    parser.add_argument("-e", "--end", type=float, help="end")
    parser.add_argument("-S", "--suffix", default="_history", type=str, help="suffix")
    parser.add_argument("-N", "--last_n", default=None, type=int, help="last_n")
    parser.add_argument("-m", "--mode", default="modern", type=str, help="mode")
    parser.add_argument("-c", "--cameras", default='["sample_view", "goniometer"]', type=str, help="cameras")
    args = parser.parse_args()
    print("args", args)
        
    template = os.path.join(args.directory, args.name_pattern)
    
    arguments = [
        template,
        args.start,
        args.end,
        args.suffix,
        args.last_n,
    ]
    
    if args.mode == "classic":
        print(arguments)
        classic_save(*arguments)
    else:
        del arguments[-2:]
        arguments += [eval(args.cameras)]
        print("arguments", arguments)
        modern_save(*arguments)
        
        
if __name__ == "__main__":
    main()
