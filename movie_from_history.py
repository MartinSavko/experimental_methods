#!/usr/bin/env python

import os
import sys
import numpy as np
import h5py
import pickle
import simplejpeg
import datetime
import time
import subprocess
import shutil
import pprint
# import logging
from camera import camera


def ffmpeg_running():
    return "ffmpeg" in subprocess.getoutput("ps aux | grep ffmpeg | grep -v grep")


def read_master(master, nattempts=77, sleeptime=1, mode="r"):
    read = False
    tried = 0
    _start = time.time()

    while not read and tried < nattempts:
        try:
            m = h5py.File(master, mode)
            read = True
            # print(f"master {master} read. (try {tried+1}, took {time.time() - _start:.4f} seconds).")
        except:
            while ffmpeg_running():
                time.sleep(sleeptime)
            time.sleep(sleeptime + np.random.random())
        tried += 1
    if not read:
        m = -1
    return m

def remove_images_from_master(master, remove_original=True):
    m = read_master(master, mode="r")
    new_m = h5py.File(master.replace(".h5", "_lean.h5"), "w")
    for key in m:
        if "images" not in key:
            m.copy(key, new_m)
    m.close()
    new_m.close()
    if remove_original:
        os.remove(master)
    
def get_images_hsv_ht(master):
    if master.endswith(".h5"):
        m = read_master(master, mode="r")

        if m == -1:
            sys.exit(f"Could not read {master}, please check!")
        images = m["history_images"][()]
        if len(images[0].shape) > 1:
            images = [
                np.frombuffer(simplejpeg.encode_jpeg(img), dtype="uint8")
                for img in images
            ]
        try:
            hsv = m["history_state_vectors"][()]
        except:
            hsv = []
        ht = m["history_timestamps"][()]
        m.close()
    elif master.endswith(".pickle"):
        f = open(master, "rb")
        m = pickle.load(f)
        f.close()
        images = m["jpegs"]
        ht = np.array(m["timestamps"])
        hsv = []

    return images, hsv, ht


def get_median_frame_duration(ht):
    htd = ht[1:] - ht[:-1]
    median_frame_duration = np.median(htd)
    return median_frame_duration


def generate_concat_input(
    ht, hsv, template="%06d", suffix="jpg", filename="concat.in", directory="./"
):
    template = f"{template}.{suffix}"
    text = ""
    mfd = get_median_frame_duration(ht)
    ht0 = ht - ht[0]
    ht1 = [ht0[k + 1] for k in range(len(ht0) - 1)]
    ht1.append(ht1[-1] + mfd)
    for k, (i, o) in enumerate(zip(ht0, ht1)):
        bit = "file %s\n" % os.path.abspath(get_imagename(k + 1, directory=directory))
        try:
            bit += "file_packet_metadata url=Ω=%.1f\n" % hsv[k][0]
        except:
            pass
        # bit += 'inpoint %s\n' % str(datetime.timedelta(seconds=i))[:-3]
        bit += "outpoint %s\n" % str(datetime.timedelta(seconds=o))[:-3]
        text += bit
    f = open(filename, "w")
    f.write(text)
    f.close()
    duration = ht1[-1]
    return len(ht), duration


def generate_jpegs(images, template="%06d", suffix="jpg", directory="./"):
    template = f"{template}.{suffix}"
    for k, img in enumerate(images):
        fname = get_imagename(k + 1, directory=directory)
        if type(img) is bytes:
            f = open(fname, "wb")
            f.write(img)
            f.close()
        else:
            img.tofile(fname)


def remove_jpegs(images, template="%06d", suffix="jpg", directory="./"):
    template = f"{template}.{suffix}"
    for k, img in enumerate(images):
        try:
            os.remove(get_imagename(k + 1, directory=directory))
        except:
            pass


def get_imagename(k, template="%06d.jpg", directory="./"):
    imagename = os.path.join(directory, template % (k))
    return imagename


def main(debug=False):
    # sys.exit("movie generation disabled -- boundary effects under investigation ...")
    # return
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,)

    parser.add_argument(
        "-H",
        "--history",
        default="examples/autocenter_100161_Tue_Apr_11_12:00:02_2023_element_9_09_history.h5",
        type=str,
        help="history",
    )
    parser.add_argument("-c", "--codec", default="h264", type=str, help="video codec")
    parser.add_argument(
        "-d",
        "--working_directory",
        default="/dev/shm/movies",
        type=str,
        help="working_directory",
    )
    parser.add_argument("-s", "--suffix", default="_movie.mp4", type=str, help="suffix")
    parser.add_argument(
        "-o", "--overlays", action="store_false", help="do not draw overlays"
    )
    parser.add_argument("-r", "--do_not_clean", action="store_true", help="clean jpegs")
    parser.add_argument("-n", "--nice", action="store_true", help="be nice")
    parser.add_argument("-m", "--meta", action="store_true", help="add metadata")
    parser.add_argument("-Y", "--year_and_title", action="store_true", help="year and title")
    args = parser.parse_args()

    start = time.time()

    cam = camera()

    images, hsv, ht = get_images_hsv_ht(args.history)
    try:
        zoom = cam.get_zoom(zoomposition=hsv[0][-3])
        calibration = cam.calibrations[zoom][-1]
        micron = 0.001 / calibration
    except:
        micron = 1
    if args.history.endswith(".h5"):
        hsuffix = ".h5"
    elif args.history.endswith(".pickle"):
        hsuffix = ".pickle"

    working_directory = os.path.join(
        args.working_directory,
        str(os.getuid()),
        os.path.basename(args.history).replace(hsuffix, ""),
    )
    if not os.path.isdir(working_directory):
        os.makedirs(working_directory)

    input_filename = args.history.replace(hsuffix, "_concat.txt")

    frames, duration = generate_concat_input(
        ht, hsv, filename=input_filename, directory=working_directory
    )
    output_filename = args.history.replace(hsuffix, args.suffix)
    working_output_filename = os.path.join(
        working_directory, os.path.basename(output_filename)
    )

    generate_jpegs(images, directory=working_directory)

    """ffmpeg_line = 
    nice -n 99 ffmpeg \
        -loglevel 8 \
        -x265-params log-level=0 \
        -f concat \
        -safe 0 \
        -i concat.in \
        -vf scale=320:256 \
        -vf "drawtext=text='%{metadata\:url}': \
            fontcolor=0x008000: \
            fontsize=14: \
            x=w-tw-10: \
            y=h-th-10" \
        -r 12
        -c:v libx264 \
        movie.mp4
    """

    framerate = frames / duration
    # print('framerate %.3f' % framerate)
    format_dictionary = {
        "beam_width": 10 * micron,
        "beam_height": 5 * micron,
        "microns100": 100 * micron,
        "half_beam_height": 2.5 * micron,
        "half_beam_width": 5 * micron,
        "microns50": 50 * micron,
        "metadata": "%{metadata\:url}",
    }
    format_dictionary["codec"] = args.codec
    format_dictionary["framerate"] = framerate
    format_dictionary["input_filename"] = input_filename
    format_dictionary["output_filename"] = working_output_filename
    format_dictionary["year"] = datetime.datetime.today().year
    format_dictionary["author"] = "Synchrotron SOLEIL, Proxima2A"
    format_dictionary["title"] = "Synchrotron SOLEIL, Proxima2A, %s" % str(
        datetime.datetime.fromtimestamp(ht[0])
    )
    format_dictionary[
        "year_and_title"
    ] = '-metadata author="{author:s}" -metadata year="{year:d}" -metadata title="{title:s}"'.format(
        **format_dictionary
    )
    """drawtext=\
            text='{metadata:s}':\
            fontcolor=0x008000:\
            fontsize=28:\
            x=w-tw-20:\
            y=h/2-th/2,\
        drawbox=\
            x=iw/2-{half_beam_width:.1f}:\
            y=ih/2-{half_beam_height:.1f}:\
            w={beam_width:.1f}:\
            h={beam_height:.1f}:\
            color=green@0.75:\
            t=3,\ 
        drawbox=\
            x=20:\
            y=ih-20:\
            w={microns100:.1f}:\
            h=5:\
            color=green@0.75:\
            t=fill,\ 
        drawtext=\
            textfile=/usr/local/experimental_methods/100microns.txt:\
            fontcolor=0x008000:\
            fontsize=14:\
            x=20+{microns50:.1f}-tw/2:\
            y=h-th-10"""
    text1 = "drawtext=\
                text='{metadata:s}':\
                fontcolor=0x008000:\
                fontsize=28:\
                x=w-tw-20:\
                y=h/2-th/2,"
    box1 = "drawbox=\
                x=iw/2-{half_beam_width:.1f}:\
                y=ih/2-{half_beam_height:.1f}:\
                w={beam_width:.1f}:\
                h={beam_height:.1f}:\
                color=green@0.75:\
                t=3,"
    box2 = "drawbox=\
                x=20:\
                y=ih-20:\
                w={microns100:.1f}:\
                h=5:\
                color=green@0.75:\
                t=fill,"
    text2 = "drawtext=\
                textfile=/usr/local/experimental_methods/100microns.txt:\
                fontcolor=0x008000:\
                fontsize=14:\
                x=20+{microns50:.1f}-tw/2:\
                y=h-th-10"

    format_dictionary["filters"] = (text1 + box1 + box2 + text2).format(
        **format_dictionary
    )

    ffmpeg_line = "ffmpeg -loglevel -8 -r {framerate:.3f} -f concat -safe 0 -i {input_filename:s} ".format(
        **format_dictionary
    )
    if args.nice:
        ffmpeg_line = "nice -n 99 " + ffmpeg_line
    if args.overlays:
        ffmpeg_line += '-vf "{filters:s}" '.format(**format_dictionary)
    # ffmpeg_line += '-c:v {codec:s} -x265-params "log-level=0" {year_and_title:s} -y {output_filename:s}'.format(
    # **format_dictionary
    # )

    if args.year_and_title:
        ffmpeg_line += ' {year_and_title:s} '.format(
        **format_dictionary
    )
    ffmpeg_line += '-c:v {codec:s} -x265-params "log-level=0" -y {output_filename:s}'.format(
        **format_dictionary
    )
    if debug:
        print("running:")
        print(ffmpeg_line)
    os.system(ffmpeg_line)

    if not args.do_not_clean:
        if debug:
            print("cleaning")
        remove_jpegs(images, directory=working_directory)
        try:
            os.remove(input_filename)
        except:
            pass
        #if args.h5_clean:
        if debug:
            print("h5 cleaning")
        remove_images_from_master(args.history, remove_original=not args.do_not_clean)

    # shutil.move(working_output_filename, output_filename)
    move_command = f"mv {working_output_filename} {output_filename}"

    os.system(move_command)
    end = time.time()
    # print(f"{os.path.basename(output_filename)} generated. (Encoded in {end - start:.2f} seconds).")


if __name__ == "__main__":
    main()
