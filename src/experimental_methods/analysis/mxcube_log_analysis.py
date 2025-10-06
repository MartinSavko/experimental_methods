#!/usr/bin/env python

import os
import time
import re
import numpy as np
import traceback
import pylab


def get_times(look_for, log_name, addition=""):
    time_stamp = time.time()
    line = 'grep "%s" %s %s > /tmp/%s.txt' % (look_for, log_name, addition, time_stamp)
    time.sleep(0.1)
    print(line)
    os.system(line)
    # time.sleep(3)
    n = open("/tmp/%s.txt" % time_stamp).read().split("\n")
    n = [
        re.findall("(\d\d\d\d\-\d\d-\d\d \d\d:\d\d:\d\d,\d\d\d).*", a)[0]
        for a in n
        if len(a) > 10
    ]
    return np.array([time.mktime(time.strptime(a, "%Y-%m-%d %H:%M:%S,%f")) for a in n])


def get_collection_finished(log_name):
    return get_times("root   |INFO   | xdsme_process_line ssh", log_name)


def get_collection_started(log_name):
    return get_times("PX2Collect: executing", log_name, addition="| grep -v reference")


def get_reference_analysis_started(log_name):
    return get_times("|INFO   | analysis line reference_images.py", log_name)


def get_reference_collect_started(log_name):
    return get_times("PX2Collect: executing reference_images", log_name)


def main():
    import optparse

    parser = optparse.OptionParser()
    parser.add_option("-l", "--log_name", type=str, default=None, help="log filename")
    options, args = parser.parse_args()

    try:
        s = get_reference_collect_started(options.log_name)
        print("len(s)", len(s), s)
        e = get_reference_analysis_started(options.log_name)
        print("len(e)", len(e), e)
        e = e[: len(s)]
        s = s[: len(e)]
        y = e - s
        x = np.arange(len(y))
        fit = np.polyfit(x, y, 1)
        print("fit reference_images", fit)

        pylab.figure(figsize=(16, 9))
        pylab.title("Reference images acquisition duration")
        pylab.plot(x, y)
        pylab.xlabel("# consecutive collect")
        pylab.ylabel("time [s]")
    except:
        print(traceback.print_exc())
        pass

    try:
        s = get_collection_started(options.log_name)
        print("len(s)", len(s), s)
        e = get_collection_finished(options.log_name)
        print("len(e)", len(e), e)
        e = e[: len(s)]
        y = e - s
        x = np.arange(len(y))
        fit = np.polyfit(x, y, 1)
        print("fit omega_scan", fit)
        pylab.figure(figsize=(16, 9))
        pylab.title("Omega scan acquisition duration")
        pylab.plot(x, y)
        pylab.xlabel("# consecutive collect")
        pylab.ylabel("time [s]")
    except:
        print(traceback.print_exc())
        pass

    pylab.show()


if __name__ == "__main__":
    main()
