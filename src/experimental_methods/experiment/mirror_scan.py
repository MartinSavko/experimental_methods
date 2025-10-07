#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Slits scan. Execute scan on a pair of slits.

"""

import os
import sys
import gevent
import traceback
import logging
import time
import itertools
import pickle
import numpy as np
import pylab
import glob
from scipy.constants import eV, h, c, angstrom, kilo, degree

from experimental_methods.instrument.monitor import xray_camera, analyzer
from experimental_methods.instrument.adaptive_mirror import adaptive_mirror
from camera import camera
from redis import StrictRedis
from slit_scan import slit_scan

from analysis import slit_scan_analysis


class mirror_scan_analysis(slit_scan_analysis):
    def analyze(
        self,
        observation_fields=[
            "chronos",
            "gaussian_fit_center_x",
            "gaussian_fit_center_y",
            "gaussianfit_amplitude",
            "gaussianfit_width_x",
            "gaussianfit_width_y",
            "max",
            "mean",
            "com_x",
            "com_y",
        ],
    ):
        if not os.path.isfile(self.parameters_filename):
            return
        parameters = self.get_parameters()
        results = self.get_results()
        print("self.monitor", self.monitor)
        for lame_name in list(results.keys()):
            actuator_chronos, actuator_position = self.get_observations(
                results[lame_name], "actuator_monitor"
            )
            fast_shutter_chronos, fast_shutter_state = self.get_observations(
                results[lame_name], "fast_shutter"
            )
            try:
                monitor_chronos, monitor_points = self.get_observations(
                    results[lame_name], self.monitor
                )
            except:
                monitor_points = []
            if len(monitor_points) == 0:
                return
            actuator_scan_indices = self.get_illuminated_indices(
                fast_shutter_chronos, fast_shutter_state, actuator_chronos
            )
            actuator_scan_chronos = actuator_chronos[actuator_scan_indices]
            actuator_scan_position = actuator_position[actuator_scan_indices]

            position_chronos_predictor = self.get_position_chronos_predictor(
                actuator_scan_chronos, actuator_scan_position
            )

            observation_indices = self.get_illuminated_indices(
                fast_shutter_chronos, fast_shutter_state, monitor_chronos
            )
            observation_chronos = monitor_chronos[observation_indices]
            observation_position = position_chronos_predictor(observation_chronos)

            if "analysis" not in results[lame_name]:
                results[lame_name]["analysis"] = {}
            if self.monitor not in results[lame_name]["analysis"]:
                results[lame_name]["analysis"][self.monitor] = {}

            results[lame_name]["analysis"][self.monitor][
                "actuator_position"
            ] = observation_position

            print("monitor_points.shape", monitor_points.shape)
            for k, field in enumerate(observation_fields[1:]):
                observation = monitor_points[k, :][observation_indices]

                # mask = np.logical_or(observation>256, observation<0)
                # monitor_points[mask] = np.nan
                results[lame_name]["analysis"][self.monitor][field] = observation

        self.save_results(results)


class mirror_scan(slit_scan):
    mirrors = {"vfm": "i11-ma-c05/op/mir2-vfm", "hfm": "i11-ma-c05/op/mir3-hfm"}

    specific_parameter_fields = [
        {"name": "mirror_name", "type": "str", "description": "Target mirror"},
        {"name": "channel_values", "type": "list", "description": "Mirror tensions"},
        {
            "name": "channel_values_intention",
            "type": "list",
            "description": "Mirror tensions",
        },
        {"name": "mirror_pitch", "type": "float", "description": "Mirror pitch"},
        {
            "name": "mirror_translation",
            "type": "float",
            "description": "Mirror translation",
        },
    ]

    def __init__(
        self,
        name_pattern,
        directory,
        mirror_name="vfm",
        channel_values=[],
        start_position=1.0,
        end_position=-1.0,
        scan_gap=0.05,
        scan_speed=None,
        darkcurrent_time=1.0,
        photon_energy=None,
        diagnostic=True,
        analysis=None,
        conclusion=None,
        simulation=None,
        display=False,
        extract=False,
    ):
        if hasattr(self, "parameter_fields"):
            self.parameter_fields += mirror_scan.specific_parameter_fields
        else:
            self.parameter_fields = mirror_scan.specific_parameter_fields[:]

        slit_scan.__init__(
            self,
            name_pattern,
            directory,
            slits=3,
            start_position=start_position,
            end_position=end_position,
            scan_speed=scan_speed,
            scan_gap=scan_gap,
            darkcurrent_time=darkcurrent_time,
            photon_energy=photon_energy,
            diagnostic=diagnostic,
            analysis=analysis,
            conclusion=conclusion,
            simulation=simulation,
            display=display,
            extract=extract,
        )

        self.description = (
            "Scan of %s mirror scan scan between %6.1f and %6.1f mm, Proxima 2A, SOLEIL, %s"
            % (mirror_name, start_position, end_position, time.ctime(self.timestamp))
        )
        self.channel_values_intention = channel_values
        self.mirror_name = mirror_name
        self.mirror = adaptive_mirror(self.mirror_name)
        self.redis = StrictRedis()

    def set_up_monitor(self):
        self.redis.set("beam_scan", 1)

        # self.monitor_device = xray_camera(continuous_monitor_name='focus_monitor')
        # self.monitors_dictionary['xray_camera'] = self.monitor_device
        # self.monitor_names += ['xray_camera']
        # self.monitors += [self.monitor_device]

        # self.auxiliary_monitor_device = analyzer(continuous_monitor_name='analyzer_monitor')
        # self.monitors_dictionary['analyzer'] = self.auxiliary_monitor_device
        # self.monitor_names += ['analyzer']
        # self.monitors += [self.auxiliary_monitor_device]

    def get_clean_slits(self):
        return [1, 2, 5, 6]

    def get_alignment_actuators(self):
        alignment_actuators = self.alignment_slits.get_alignment_actuators()
        print("alignment_actuators", alignment_actuators)
        if self.mirror_name == "vfm":
            actuators = alignment_actuators[1]
        else:
            actuators = alignment_actuators[0]
        print("actuators", actuators)
        return actuators

    def get_channel_values(self):
        return self.mirror.get_channel_values()

    def get_mirror_pitch(self):
        return self.mirror.get_pitch_position()

    def get_mirror_translation(self):
        return self.mirror.get_translation_position()

    def prepare(self):
        super(mirror_scan, self).prepare()

        try:
            self.mirror.set_voltages(self.channel_values_intention)
        except:
            print("did not succeed to set the tensions")
            print(traceback.print_exc())

    def handle_monitor_insertion(self):
        return

    def run(self):
        print("run", self.get_template())

        self.res = {}

        actuator = self.get_alignment_actuators()
        k = 1 if self.mirror_name == "vfm" else 0

        print("actuator", actuator)

        self.actuator = actuator
        # self.actuator_names = [self.actuator.get_name()]
        actuator.wait()
        actuator.set_position(self.start_position, timeout=None, wait=True)

        actuator.set_speed(self.scan_speed)

        if self.slit_type == 2:
            self.alignment_slits.set_pencil_scan_gap(
                k, scan_gap=self.get_scan_gap(), wait=True
            )

        self.start_monitor()

        self._observe_start = time.time()
        print("sleep for darkcurrent_time while observation is already running")
        gevent.sleep(self.darkcurrent_time)

        self.fast_shutter.open()

        move = gevent.spawn(
            actuator.set_position, self.end_position, timeout=None, wait=True
        )
        move.join()

        actuator.set_speed(self.default_speed)

        self.fast_shutter.close()

        gevent.sleep(self.darkcurrent_time)

        self._observe_stop = time.time()

        self.stop_monitor()
        self.redis.set("beam_scan", 0)

        # os.system('/nfs/data3/Martin/Research/experimental_methods/history_saver.py -d %s -n %s_basler -e %.4f -s %.4f -m xray_camera &' % (self.directory, self.name_pattern, self._observe_stop, self._observe_start))

        os.system(
            "/nfs/data3/Martin/Research/experimental_methods/history_saver.py -d %s -n %s_prosilica -e %.4f -s %.4f -m prosilica &"
            % (
                self.directory,
                self.name_pattern,
                self._observe_stop,
                self._observe_start,
            )
        )

        actuator.wait()

        # if self.slit_type == 2:
        # self.alignment_slits.set_pencil_scan_gap(k, scan_gap=self.default_gap, wait=True)
        # actuator.set_position(0.)
        # elif self.slit_type == 1:
        # actuator.set_position(self.start_position, wait=True)

        res = self.get_results()
        self.res[actuator.get_name()] = res

    def analyze(self):
        # a = mirror_scan_analysis(os.path.join(self.directory, '%s_parameters.pickle' % self.name_pattern), monitor='xray_camera')
        # a.analyze(observation_fields=['chronos', 'com_y', 'com_x'])

        a = mirror_scan_analysis(
            os.path.join(self.directory, "%s_parameters.pickle" % self.name_pattern),
            monitor="prosilica",
        )
        a.analyze(observation_fields=["chronos", "com_y", "com_x"])

        # a = mirror_scan_analysis(os.path.join(self.directory, '%s_parameters.pickle' % self.name_pattern), monitor='analyzer')
        # a.analyze() #observation_fields=['chronos', 'com_y', 'com_x'])

    def conclude(self):
        pass


def main():
    import optparse

    usage = """Program will execute a slit scan
    
    ./mirror_scan.py <options>
    
    """
    parser = optparse.OptionParser(usage=usage)

    parser.add_option(
        "-d",
        "--directory",
        type=str,
        default="/tmp/slit_scan",
        help="Directory to store the results (default=%default)",
    )
    parser.add_option(
        "-n", "--name_pattern", type=str, default="slit_scan", help="name_pattern"
    )
    parser.add_option(
        "-m", "--mirror_name", type=str, default="vfm", help="mirror_name"
    )
    parser.add_option(
        "-b", "--start_position", type=float, default=1.0, help="Start position"
    )
    parser.add_option(
        "-e", "--end_position", type=float, default=-1.0, help="End position"
    )
    parser.add_option(
        "-p", "--photon_energy", type=float, default=12650, help="Photon energy"
    )
    parser.add_option("-D", "--display", action="store_true", help="display plot")
    parser.add_option(
        "-E",
        "--extract",
        action="store_true",
        help="Extract the calibrated diode after the scan",
    )
    parser.add_option("-A", "--analysis", action="store_true", help="Analyze the scan")
    parser.add_option(
        "-C", "--conclusion", action="store_true", help="Apply the offsets"
    )

    options, args = parser.parse_args()

    print("options", options)
    print("args", args)

    filename = (
        os.path.join(options.directory, options.name_pattern) + "_parameters.pickle"
    )

    mscan = mirror_scan(**vars(options))

    if not os.path.isfile(filename):
        mscan.execute()
    if options.analysis == True:
        mscan.analyze()
    if options.conclusion == True:
        mscan.conclude()


def scan_mirror_step_by_step(mirror_name, start_stop, filename, nsteps=30):
    from slits import slits3
    from experimental_methods.instrument.fast_shutter import fast_shutter

    s3 = slits3()
    xc = xray_camera()
    cam = camera()
    fs = fast_shutter()
    fs.open()
    start, stop = start_stop
    positions = np.linspace(start, stop, nsteps)
    values = []
    for p in positions:
        if mirror_name == "hfm":
            s3.set_horizontal_position(p)
            values.append([xc.get_com_x(), cam.get_com_x()])
        else:
            s3.set_vertical_position(p)
            values.append([xc.get_com_y(), cam.get_com_y()])
    fs.close()
    values = np.array(values)
    np.save(filename, np.vstack([positions, values.T]).T)


def get_close_pairs(max_distance=2, nchannels=12):
    close_pairs = []
    for k in range(nchannels):
        for l in range(nchannels):
            if k != l:
                if (
                    (k, l) not in close_pairs
                    and (l, k) not in close_pairs
                    and abs(k - l) <= max_distance
                    and k < l
                ):
                    close_pairs.append((k, l))
    return close_pairs


def get_close_triplets(max_distance=2, nchannels=12):
    close_triplets = []
    for k in range(nchannels):
        for l in range(nchannels):
            for m in range(nchannels):
                if k != l and l != m and k != m:
                    if (
                        max([abs(k - l), abs(l - m), abs(k - m)]) <= max_distance
                        and k < l
                        and l < m
                    ):
                        if (k, l, m) not in close_triplets:
                            close_triplets.append((k, l, m))
    return close_triplets


def get_total_increments(values=[50, 0, -50], triplets=True):
    total_increments = []
    if triplets:
        increments = list(itertools.product(values, values, values))
        close = get_close_triplets()
        for triplet in close:
            for increment in increments:
                to_add = [0] * 12
                to_add[triplet[0]] = increment[0]
                to_add[triplet[1]] = increment[1]
                to_add[triplet[2]] = increment[2]
                if to_add not in total_increments:
                    total_increments.append(to_add)
    else:
        increments = list(itertools.product(values, values))
        close = get_close_pairs()
        for pair in close:
            for increment in increments:
                to_add = [0] * 12
                to_add[pair[0]] = increment[0]
                to_add[pair[1]] = increment[1]
                if to_add not in total_increments:
                    total_increments.append(to_add)

    return total_increments


def scan_mirror(
    special_directory="2020-07-16_%s_a",
    mirror_name="vfm",
    base_directory="/nfs/data3/Martin/Commissioning/mirrors",
    start_stop=[-0.8, 0.8],
):
    if mirror_name == "vfm":
        mirror = adaptive_mirror("vfm")
        # base_voltages = [50.0, 50.0, -25.0, 150.0, 150.0, 550.0, 322.0, 188.0, 40.0, -100.0, -100.0, -150.0]
        base_voltages = [
            100.0,
            100.0,
            -75.0,
            150.0,
            150.0,
            550.0,
            322.0,
            188.0,
            40.0,
            -100.0,
            -100.0,
            -200.0,
        ]
    else:
        mirror = adaptive_mirror("hfm")
        # base_voltages = [600.0, 250.0, 200.0, 50.0, 100.0, 50.0, 0.0, -150.0, -250.0, -250.0, -350.0, -400.0]
        # base_voltages = [500.0, 250.0, 200.0, 50.0, 150.0, 100.0, 50.0, -150.0, -250.0, -250.0, -350.0, -400.0]
        base_voltages = [
            500.0,
            250.0,
            200.0,
            50.0,
            200.0,
            150.0,
            100.0,
            -200.0,
            -250.0,
            -250.0,
            -350.0,
            -400.0,
        ]
    # original_vfm_voltages = [255.0, 215.0, 12.0, 170.0, 185.0, 443.0, 322.0, 188.0, 40.0, -47.0, -3.0, 88.0] # vfm.get_channel_values()
    # original_hfm_voltages = [290.0, 320.0, 265.0, 56.0, 102.0, 415.0, 37.0, -247.0, -534.0, -703.0, -1089.0, -1400.0] # hfm.get_channel_values()
    # base_voltages = [0.0, 50.0, -25.0, 150.0, 150.0, 550.0, 322.0, 188.0, 40.0, -100.0, -100.0, -100.0]

    # base_voltages = [500.0, 250.0, 200.0, 150.0, 100.0, 50.0, -50.0, -100.0, -200.0, -250.0, -350.0, -400.0] #[0.0] * 12

    directory = os.path.join(base_directory, special_directory % mirror_name)
    print("directory", directory)
    for letter in ["a"]:  # 'b', 'c']:
        # mscan = mirror_scan('base_voltages_%s' % letter , directory, mirror_name=mirror_name, channel_values=base_voltages, start_position=start_stop[0], end_position=start_stop[1])
        # mscan.execute()
        # start_stop = start_stop[::-1]
        try:
            man = mirror_scan_analysis(
                "%s/base_voltages_%s_parameters.pickle" % (directory, letter),
                monitor="prosilica",
            )
            man.analyze(observation_fields=["chronos", "com_y", "com_x"])
        except:
            print(traceback.print_exc())

    for ti in get_total_increments():
        print("total increment", ti)
        new_voltages = np.array(base_voltages[:])
        new_voltages += np.array(ti)
        name_pattern = "increment_%s" % "_".join(map(str, ti))
        # mscan = mirror_scan(name_pattern, directory, mirror_name=mirror_name, channel_values=new_voltages, start_position=start_stop[0], end_position=start_stop[1])
        # mscan.execute()
        # start_stop = start_stop[::-1]
        try:
            man = mirror_scan_analysis(
                "%s/%s_parameters.pickle" % (directory, name_pattern),
                monitor="prosilica",
            )
            man.analyze(observation_fields=["chronos", "com_y", "com_x"])
        except:
            print(traceback.print_exc())

    # for increment in [+50, -50]: #, -40, +40, +60, -60, -80, +80, +100, -100]:
    # for k in range(12):
    # print('channel %02d, increment %+d' % (k, increment))
    # new_voltages = base_voltages[:]
    # new_voltages[k] += increment
    # for letter in ['a']: #, 'b']:
    ##mscan = mirror_scan('channel_%02d_increment_%d_%s' % (k, increment, letter), directory, mirror_name=mirror_name, channel_values=new_voltages, start_position=start_stop[0], end_position=start_stop[1])
    ##mscan.execute()
    # try:
    # man = mirror_scan_analysis('%s/channel_%02d_increment_%d_%s_parameters.pickle' % (directory, k, increment, letter), monitor='prosilica')
    # man.analyze(observation_fields=['chronos', 'com_y', 'com_x'])
    # except:
    # print(traceback.print_exc())

    for letter in ["d"]:  # , 'd']:
        # mscan = mirror_scan('base_voltages_%s' % letter , directory, mirror_name=mirror_name, channel_values=base_voltages, start_position=start_stop[0], end_position=start_stop[1])
        # mscan.execute()
        try:
            man = mirror_scan_analysis(
                "%s/base_voltages_%s_parameters.pickle" % (directory, letter),
                monitor="prosilica",
            )
            man.analyze(observation_fields=["chronos", "com_y", "com_x"])
        except:
            print(traceback.print_exc())

    # if mirror_name == 'vfm':
    # mscan.slits3.set_vertical_gap(4)
    # mscan.slits3.set_vertical_position(0)
    # else:
    # mscan.slits3.set_horizontal_gap(4)
    # mscan.slits3.set_horizontal_position(0)


def plot_results(directory="/nfs/data3/Martin/Commissioning/mirrors/2020-07-16_vfm_a"):
    if "hfm" in directory:
        lame_name = "i11-ma-c05/ex/fent_h.3-mt_tx"
        point_of_interrest = "com_x"
    else:
        lame_name = "i11-ma-c05/ex/fent_v.3-mt_tz"
        point_of_interrest = "com_y"
    base = glob.glob(os.path.join(directory, "base*_results.pickle"))
    bases = []
    grid = np.linspace(-1, 1, 2000)
    for f in base:  # [1::2]:
        r = pickle.load(open(f))[lame_name]
        p = r["analysis"]["prosilica"]["actuator_position"]
        b = r["analysis"]["prosilica"][point_of_interrest]
        bi = np.interp(grid, p, b)
        bases.append(bi)
    bases = np.array(bases)
    print("bases.shape", bases.shape)
    pylab.figure()
    pylab.title("base")
    pylab.plot(grid, bases.mean(axis=0))

    results = glob.glob(os.path.join(directory, "increment*_results.pickle"))
    for f in results:
        print("result", f)
        pylab.figure()
        pylab.title(
            os.path.basename(f).replace("increment_", "").replace("_results.pickle", "")
        )
        pylab.plot(grid, np.mean(bases, axis=0), label="base")
        try:
            r = pickle.load(open(f))[lame_name]
            p = r["analysis"]["prosilica"]["actuator_position"]
            for field in [point_of_interrest]:
                b = r["analysis"]["prosilica"][field]
                label = "%s_%s" % (
                    f.replace("_results.pickle", "")
                    .replace("channel_", "")
                    .replace("increment_", "")
                    .replace(directory, ""),
                    field,
                )
                pylab.plot(p, b, label=label)
        except:
            print(traceback.print_exc())
        pylab.legend()

    # for k in range(12):
    # chan = '%02d' % k
    # print('chan', chan)
    # rf = glob.glob(os.path.join(directory, 'channel_%s_*_results.pickle' % chan))
    # pylab.figure()
    # pylab.title(chan)
    # pylab.plot(grid, np.mean(bases, axis=0), label='base')
    # for f in rf:
    # try:
    # r = pickle.load(open(f))[lame_name]
    # p = r['analysis']['prosilica']['actuator_position']
    # for field in [point_of_interrest]:
    # b = r['analysis']['prosilica'][field]
    # label = '%s_%s' % (f.replace('_results.pickle', '').replace('channel_', '').replace('_increment','').replace(directory, ''), field)
    # pylab.plot(p, b, label=label)
    # except:
    # print(traceback.print_exc())
    # pylab.legend()
    pylab.show()


def plot_results_sbs(
    directory="/nfs/data3/Martin/Commissioning/mirrors/2020-07-15_hfm_b",
):
    b = np.load(os.path.join(directory, "base_voltages_a_step_by_step.npy"))
    pb, vb = b[:, 0], b[:, 1]
    pylab.figure()
    pylab.title("base")
    pylab.plot(pb, vb)

    for k in range(0, 6):
        pylab.figure()
        for f in glob.glob(os.path.join(directory, "channel_%02d_*.npy" % k)):
            r = np.load(f)
            p = r[:, 0]
            if np.allclose(p, pb):
                v = r[:, 2]  # - vb
            else:
                v = r[:, 2]  # - vb[::-1]
            pylab.plot(
                p,
                v,
                label=os.path.basename(f)
                .replace("channel_", "")
                .replace("_increment", "")
                .replace("_a_step_by_step.npy", ""),
            )
            pylab.title("%02d" % k)
            # pylab.ylim(-10, 10)
        pylab.legend()
    pylab.show()


if __name__ == "__main__":
    # main()
    # for mirror_name in ['vfm', 'hfm']:
    # scan_mirror(mirror_name=mirror_name)
    plot_results("/nfs/data3/Martin/Commissioning/mirrors/2020-07-16_hfm_a")
    # plot_results_sbs('/nfs/data3/Martin/Commissioning/mirrors/2020-07-15_hfm_a')
