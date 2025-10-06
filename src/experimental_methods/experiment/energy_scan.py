#!/usr/bin/env python
# -*- coding: utf-8 -*-
import gevent

import traceback
import logging
import time
import pickle
import os
import pylab
import numpy as np
import scipy

from xabs_lib import McMaster
from xray_experiment import xray_experiment
from fluorescence_detector import fluorescence_detector as detector
from motor_scan import motor_scan
from motor import tango_motor
from monitor import xbpm
from motor import monochromator_rx_motor

from attenuators import attenuators
from flyscan import flyscan

try:
    import seaborn as sns

    sns.set(color_codes=True)
except:
    pass

try:
    from matplotlib import rc

    rc("text", usetex=True)
except:
    pass


class energy_scan(xray_experiment):
    specific_parameter_fields = [
        {
            "name": "element",
            "type": "str",
            "description": "chemical element two letter name",
        },
        {
            "name": "edge",
            "type": "str",
            "description": "chemical element absorption edge (K, L1, L2 or L3)",
        },
        {
            "name": "scan_range",
            "type": "float",
            "description": "total scan range of the experiment in eV, default is 60. eV",
        },
        {
            "name": "scan_speed",
            "type": "float",
            "description": "scan speed in eV/s, default is 1. eV/s",
        },
        {
            "name": "scan_step",
            "type": "float",
            "description": "scan step in eV, only relevant for shuttered acquisition",
        },
        {
            "name": "roi_width",
            "type": "float",
            "description": "width of the detector roi in eV, default is 250 eV",
        },
        {
            "name": "integration_time",
            "type": "float",
            "description": "detector count time per single point in s, default is 0.8 s",
        },
        {
            "name": "total_time",
            "type": "float",
            "description": "total duration of the measurement (shutter open time) in s",
        },
        {
            "name": "optimize",
            "type": "bool",
            "description": "whether or not to optimize transmission",
        },
        {
            "name": "optimize_at_energy",
            "type": "float",
            "description": "photon energy at which to optimize transmission in eV",
        },
        {
            "name": "edge_energy_optimize_offset",
            "type": "float",
            "description": "offset photon energy by this amount in eV compared to theoretical edge position",
        },
        {
            "name": "start_offset",
            "type": "float",
            "description": "offset start of the scan range by this amount in eV",
        },
        {
            "name": "high_dead_time",
            "type": "float",
            "description": "maximum acceptable dead time of the detector during transmission optimization",
        },
        {
            "name": "low_dead_time",
            "type": "float",
            "description": "minimum acceptable dead time of detector during transmission optimization",
        },
        {
            "name": "max_transmission",
            "type": "float",
            "description": "maximum permitted transmission in %",
        },
        {
            "name": "experiment_transmission",
            "type": "float",
            "description": "optimized transmision of the experiment in %",
        },
        {
            "name": "equidistant_spectrum",
            "type": "bool",
            "description": "whether or not to generate equidistant spectrum (using interpolation) from measured values",
        },
        {
            "name": "ignore_first_eV",
            "type": "float",
            "description": "during analysis ignore this amount of eV at the low end of the measured range",
        },
        {
            "name": "ignore_last_eV",
            "type": "float",
            "description": "during analysis ignore this amount of eV at the high end of the measured range",
        },
        {
            "name": "inverse",
            "type": "bool",
            "description": "whether or not acquire from high to low energy",
        },
        {
            "name": "shutterless",
            "type": "bool",
            "description": "whether or not acquire in shutterless mode",
        },
        {
            "name": "detector_card",
            "type": "str",
            "description": "counting card to use (xia or xspress3)",
        },
    ]

    def __init__(
        self,
        name_pattern,
        directory,
        element,
        edge,
        scan_range=100,  # eV
        scan_step=1.0,  # eV only taken into account if shutterless==False
        scan_speed=1,  # eV/s
        integration_time=0.8,  # s
        total_time=120.0,  # s
        transmission=0.05,  # %
        insertion_timeout=4,  # s
        roi_width=250.0,  # eV
        default_speed=0.5,  # deg/s
        edge_energy_optimize_offset=20.0,  # eV
        start_offset=10.0,  # eV
        high_dead_time=40.0,  # %
        low_dead_time=10.0,  # %
        max_transmission=5.0,  # %
        equidistant_spectrum=True,
        ignore_first_eV=0.0,  # option to not consider first f eV
        ignore_last_eV=0.0,  # option to not consider last f eV
        default_config_file="0U5MICROS",
        detector_card="xia",
        use_flyscan=False,
        inverse=False,
        shutterless=True,
        position=None,
        photon_energy=None,
        flux=None,
        display=False,
        optimize=True,
        snapshot=False,
        zoom=None,
        diagnostic=True,
        analysis=True,
        conclusion=None,
        simulation=None,
        parent=None,
    ):
        if hasattr(self, "parameter_fields"):
            self.parameter_fields += energy_scan.specific_parameter_fields
        else:
            self.parameter_fields = energy_scan.specific_parameter_fields[:]

        xray_experiment.__init__(
            self,
            name_pattern,
            directory,
            position=position,
            photon_energy=photon_energy,
            transmission=transmission,
            flux=flux,
            snapshot=snapshot,
            zoom=zoom,
            diagnostic=diagnostic,
            analysis=analysis,
            conclusion=conclusion,
            simulation=simulation,
        )

        self.description = "ESCAN, Proxima 2A, SOLEIL, element %s, edge %s, %s" % (
            element,
            edge,
            time.ctime(self.timestamp),
        )
        self.element = element
        self.edge = edge
        self.scan_range = scan_range
        self.scan_step = scan_step
        self.scan_speed = scan_speed
        self.integration_time = integration_time
        self.total_time = total_time
        self.insertion_timeout = insertion_timeout
        self.optimize = optimize
        self.display = display
        self.roi_width = roi_width
        self.default_speed = default_speed
        self.edge_energy_optimize_offset = edge_energy_optimize_offset
        self.start_offset = start_offset
        self.high_dead_time = high_dead_time
        self.low_dead_time = low_dead_time
        self.max_transmission = max_transmission
        self.equidistant_spectrum = equidistant_spectrum
        self.ignore_first_eV = ignore_first_eV
        self.ignore_last_eV = ignore_last_eV
        self.inverse = inverse
        self.shutterless = shutterless
        self.parent = parent
        self.use_flyscan = bool(use_flyscan)
        self.default_config_file = default_config_file

        # self.detector = detector(device_name='i11-ma-cx1/dt/dt-xspress3')
        self.detector_card = detector_card
        self.detector = detector(device_name=self.detector_card)

        if self.use_flyscan:
            self.flyscan = flyscan()
        else:
            self.detector.set_config_file(self.default_config_file)

        if self.shutterless == True:
            self.monitor_names = ["mca"] + self.monitor_names
            self.monitors = [self.detector] + self.monitors

        self.actuator = monochromator_rx_motor()
        self.attenuators = attenuators()

        self.total_expected_exposure_time = self.total_time
        self.total_expected_wedges = 1

        self.all_observations = None

        if self.parent != None:
            self.log = logging.getLogger("user_level_log")
        else:
            self.log = logging

        self.experiment_transmission = None

    def measure_fluorescence(self):
        self.log.info("measuring fluorescence")
        self.fast_shutter.open()
        time.sleep(0.2)
        self.detector.get_point()
        self.fast_shutter.close()

    def get_edge_energy(self):
        # self.log.info('get_edge_energy element %s edge %s' % (self.element, self.edge))

        edge = self.edge.upper()
        if edge[0] == "L" and len(edge) == 1:
            edge = "L3"

        edge_energy = McMaster[self.element]["edgeEnergies"][edge] * 1e3

        return edge_energy

    def get_alpha_energy(self):
        return (
            McMaster[self.element]["edgeEnergies"]["%s-alpha" % self.edge.upper()[0]]
            * 1e3
        )

    def channel_from_energy(self, energy):
        a, b, c = self.detector.get_calibration()
        return (energy - a) / b

    def energy_from_channel(self, channel):
        return a + b * channel + c * channel**2

    def get_peak_roi_start_end(self):
        self.alpha_energy = self.get_alpha_energy()
        roi_center = self.alpha_energy
        roi_start = np.floor(
            self.channel_from_energy(roi_center - self.roi_width / 2.0)
        )
        roi_end = np.ceil(self.channel_from_energy(roi_center + self.roi_width / 2.0))
        return roi_start, roi_end

    def set_peak_roi(self, channel=0):
        roi_start, roi_end = self.get_peak_roi_start_end()
        self.detector.set_roi(roi_start, roi_end, channel=channel)

    def get_compton_roi_start_end(self):
        start_energy = self.get_start_energy()
        end_energy = self.get_end_energy()
        roi_start = np.floor(
            self.channel_from_energy(start_energy - self.roi_width / 2.0)
        )
        roi_end = np.ceil(self.channel_from_energy(end_energy + self.roi_width / 2.0))
        return roi_start, roi_end

    def set_rois(self):
        p_start, p_end = self.get_peak_roi_start_end()
        c_start, c_end = self.get_compton_roi_start_end()
        self.detector.set_rois(p_start, p_end, c_start, c_end)

    def adjust_transmission(self, high_dead_time=50):
        message = "adjust_transmission"
        print(message)
        self.log.info(message)
        message = "current dead_time %.2f" % self.detector.get_dead_time()
        print(message)
        self.log.info(message)
        message = "self.get_transmission() %.3f" % self.get_transmission()
        print(message)
        self.log.info(message)
        if self.detector.get_dead_time() > high_dead_time:
            self.high_boundary = self.current_transmission
            self.new_transmission = (
                self.current_transmission
                - (self.high_boundary - self.low_boundary) / 2.0
            )
        else:
            self.low_boundary = self.current_transmission
            if self.high_boundary == None:
                self.new_transmission = 2 * self.current_transmission
            else:
                self.new_transmission = (
                    self.current_transmission
                    + (self.high_boundary - self.low_boundary) / 2.0
                )

        self.current_transmission = self.new_transmission
        message = "set new transmission %.3f" % self.new_transmission
        print(message)
        self.log.info(message)
        self.set_transmission(self.new_transmission)
        message = "self.get_transmission() %.3f" % self.get_transmission()
        print(message)
        self.log.info(message)

    def optimize_transmission(
        self,
        low_dead_time=10,
        high_dead_time=50,
        max_transmission=5,
        max_iterations=30,
        min_transmission=0.0015,
    ):
        message = f"Optimizing transmission at {self.get_photon_energy():.4f}"
        print(message)
        self.log.info(message)
        self.current_transmission = self.transmission
        message = "current_transmission %.3f" % self.current_transmission
        print(message)
        self.log.info(message)
        message = "self.get_transmission() %.3f" % self.get_transmission()
        print(message)
        self.log.info(message)
        self.low_boundary = 0
        self.high_boundary = None

        k = 0
        self.measure_fluorescence()
        self.current_dead_time = self.detector.get_dead_time()
        while (
            self.current_dead_time < low_dead_time
            or self.current_dead_time > high_dead_time
        ):
            k += 1
            self.adjust_transmission(high_dead_time)
            self.current_transmission = self.get_transmission()
            if (
                self.current_transmission >= max_transmission
                or k > max_iterations
                or self.current_transmission <= min_transmission
            ):
                if self.current_transmission >= max_transmission:
                    message = f"Transmission optimization did not converge. Exiting at {self.current_transmission:.4f}% after {k:d} steps. Please check if the beam is actually getting on the sample."
                elif self.current_transmission <= min_transmission:
                    message = f"Transmission at {self.current_transmission:.4f}%. We can not set it really much lower. Exiting. Please consider moving detector further."
                else:
                    message = f"maximum number of iterations {k:d} reached, exiting."
                print(message)
                self.log.info(message)
                break

            self.measure_fluorescence()
            self.current_dead_time = self.detector.get_dead_time()

        message = f"Transmission optimized at {self.current_transmission:.3f}%, dead_time {self.current_dead_time:.2f}, after {k:d} iterations"
        print(message)
        self.log.info(message)

    def prepare(self):
        _start = time.time()
        self.log.info("prepare")
        self.protective_cover.insert()
        self.actuator.set_speed(self.default_speed)
        self.check_directory(self.directory)
        self.write_destination_namepattern(self.directory, self.name_pattern)
        if self.transmission != None:
            self.set_transmission(self.transmission)

        if self.snapshot == True:
            self.log.info("taking image")
            self.camera.set_exposure()
            self.camera.set_zoom(self.zoom)
            self.goniometer.insert_backlight()
            self.goniometer.extract_frontlight()
            self.goniometer.set_position(self.reference_position)
            self.goniometer.wait()
            self.image = self.get_image()
            self.rgbimage = self.get_rgbimage()

        if self.safety_shutter.closed():
            self.safety_shutter.open()
        self.goniometer.set_data_collection_phase(wait=True)
        self.detector.insert()
        self.detector.set_integration_time(self.integration_time)
        self.set_rois()

        while time.time() - _start < self.insertion_timeout:
            gevent.sleep(self.detector.sleeptime)
        self.log.info("optimize? %s" % self.optimize)
        if self.optimize == True:
            self.attenuators.set_filter("07 Carbon 3mm")
            edge_energy = self.get_edge_energy()
            self.optimize_at_energy = edge_energy + self.edge_energy_optimize_offset
            self.log.info(
                "optimizing transmission at %.3f keV"
                % (self.optimize_at_energy / 1.0e3)
            )
            self.set_photon_energy(self.optimize_at_energy, wait=True)
            self.optimize_transmission(
                high_dead_time=self.high_dead_time,
                low_dead_time=self.low_dead_time,
                max_transmission=self.max_transmission,
            )
            self.experiment_transmission = self.current_transmission
        else:
            self.experiment_transmission = self.transmission
            self.set_transmission(self.transmission)

        self.start_energy = self.get_start_energy()
        message = "self.start_energy %s" % self.start_energy
        self.log.info(message)
        print(message)
        self.end_energy = self.get_end_energy()
        message = "self.end_energy %s" % self.end_energy
        self.log.info(message)
        print(message)

        self.energy_motor.mono.simenergy = self.start_energy / 1e3
        angle_start = self.energy_motor.mono.simthetabragg
        self.energy_motor.mono.simenergy = self.end_energy / 1e3
        angle_end = self.energy_motor.mono.simthetabragg

        if self.shutterless == True:
            self.scan_speed = abs(angle_end - angle_start) / self.total_time
            self.log.info("shutterless scan_speed %s" % self.scan_speed)

        if self.inverse == True:
            self.start_energy, self.end_energy = self.end_energy, self.start_energy

        self.log.info("moving to start energy %.3f" % self.start_energy)
        self.set_photon_energy(self.start_energy, wait=True)
        new_value_accepted = False
        n_tries = 7
        tried = 0
        while not new_value_accepted:
            tried += 1
            try:
                self.energy_motor.mono.energy = self.start_energy / 1.0e3
                self.log.info(
                    "self.energy_motor.mono.energy = self.start_energy/1.e3 accepted on try no %d"
                    % tried
                )
                new_value_accepted = True
            except:
                self.energy_motor.turn_on()
                gevent.sleep(0.2)

        gevent.sleep(0.5)
        self.actuator.wait()
        self.energy_motor.wait()

        if self.position != None:
            self.goniometer.set_position(self.position)
        else:
            self.position = self.goniometer.get_position()

        self.energy_motor.turn_on()
        if self.shutterless == True and not self.use_flyscan:
            self.actuator.set_speed(self.scan_speed)
        elif self.use_flyscan:
            self.flyscan.set_default_parameters()
            self.flyscan.set_integration_time(self.integration_time)
            self.flyscan.set_energy_start(self.start_energy)
            self.flyscan.set_energy_end(self.end_energy)

    def get_start_energy(self):
        return self.get_edge_energy() - self.scan_range / 2.0 + self.start_offset

    def get_end_energy(self):
        return self.get_start_energy() + self.scan_range

    def get_progress(self):
        total = abs(self.get_end_energy() / 1.0e3 - self.get_start_energy() / 1.0e3)
        try:
            passed = abs(
                self.actuator.get_energy(thetabragg=self.actuator.observations[-1][1])
                - self.get_start_energy() / 1.0e3
            )
        except:
            passed = 0
        return int(100.0 * passed / total)

    def get_point(self, start_time=None):
        try:
            if self.shutterless == True:
                x = self.actuator.get_energy(
                    thetabragg=self.actuator.observations[-1][1]
                )
                try:
                    normalized_counts = self.detector.observations[-1][3]
                    y = normalized_counts
                except:
                    y = 0
            else:
                last_observation = self.shuttered_observations[-1]
                x, y = last_observation[0], last_observation[4]
        except:
            self.log.info("get_point failed")
            self.log.info(traceback.format_exc())
            x, y = None, None

        return x, y

    def monitor(self, start_time=None):
        self.log.info("energy_scan monitor start")
        self.observations = []
        self.observation_fields = ["chronos", "progress", "energy", "mca"]

        last_point = [None, None, None, None]

        while self.observe == True:
            if start_time != None:
                chronos = time.time() - start_time
            else:
                chronos = None
            # if not self.use_flyscan:
            x, y = self.get_point(start_time)
            progress = self.get_progress()
            point = [chronos, progress, x, y]
            self.observations.append(point)
            if self.parent != None:
                if (
                    point[1] != last_point[1]
                    and point[2] != last_point[2]
                    and progress > 0
                ):
                    self.parent.emit("progressStep", (progress))
                    self.parent.emit(
                        "scanNewPoint",
                        (
                            (x < 1000 and x * 1000.0 or x),
                            y,
                        ),
                    )
                    last_point = point

            gevent.sleep(self.monitor_sleep_time)

    def run(self):
        # last_point = [None, None, None]
        self.observe = False
        if self.shutterless == True and not self.use_flyscan:
            self.fast_shutter.open()
            self.actuator.wait()
            self.energy_motor.wait()
            self.observe = True
            self.energy_motor.mono.energy = self.end_energy / 1.0e3

            while self.actuator.get_state() != "STANDBY":
                gevent.sleep(1.0)
            self.fast_shutter.close()

        elif self.use_flyscan:
            self.flyscan.run()

        elif self.shutterless == False:
            self.shuttered_observations = []
            energies = np.linspace(
                self.start_energy / 1.0e3,
                self.end_energy / 1.0e3,
                int(1 + self.scan_range / self.scan_step),
            )
            k = 0
            for energy in energies:
                k += 1
                self.energy_motor.mono.energy = energy
                self.energy_motor.wait()
                self.fast_shutter.open()
                y = self.detector.get_single_observation()
                self.fast_shutter.close()
                x = self.actuator.get_energy(thetabragg=self.actuator.get_position())

                self.shuttered_observations.append([x, y])

    def clean(self):
        self.log.info("clean")
        self.end_time = time.time()
        self.detector.extract()
        self.attenuators.set_filter("00 None", wait=False)
        self.actuator.set_speed(self.default_speed)
        self.collect_parameters()
        self.save_parameters()
        self.save_log()
        self.save_raw_scan()
        self.save_all_observations()
        self.save_raw_scan_plot()

    def get_efs_filename(self):
        return os.path.join(self.directory, "%s.efs" % self.name_pattern)

    def get_raw_filename(self):
        return os.path.join(self.directory, "%s.raw" % self.name_pattern)

    def get_ps_filename(self):
        return os.path.join(self.directory, "%s.ps" % self.name_pattern)

    def get_png_filename(self):
        return os.path.join(self.directory, "%s.png" % self.name_pattern)

    def get_chooch_results_filename(self):
        return os.path.join(
            self.directory, "%s_chooch_results.pickle" % self.name_pattern
        )

    def get_all_observations_filename(self):
        return os.path.join(
            self.directory, "%s_all_observations.pickle" % self.name_pattern
        )

    def get_efs(self):
        filename = self.get_efs_filename()
        try:
            f = open(filename)
            data = f.read().split("\n")
            efs = np.array(
                [
                    np.array(list(map(float, line.split())))
                    for line in data
                    if len(line.split()) == 3
                ]
            )
        except IOError:
            efs = np.array([])
            self.log.error("Chooch analysis failed, no efs file found.")
            self.log.info(
                "Most likely reason for chooch analysis to have failed is that the absorption edge could not be detected."
            )
            self.log.info(
                "Please verify that the element is really present by measuring the XRF spectrum."
            )
            self.log.info(
                "It is also possible that something else went wrong e.g. in the spectrum integration numerical error in the gnu scientific library integration routine, used by chooch internally, might have occured."
            )
            self.log.info(
                "If the absorption edge is present but analysis continues to resist, please call your local contact (tel. 8176) for more in-depth analysis of the problem."
            )
        return efs

    def get_raw_scan_plot_filename(self):
        return os.path.join(self.directory, "%s_raw_scan.png" % (self.name_pattern))

    def get_efs_plot_filename(self):
        return os.path.join(self.directory, "%s_efs.png" % (self.name_pattern))

    def parse_chooch_output(self, output):
        self.log.info("parse_chooch_output")
        try:
            table = output[output.find("Table of results") :]
            tabl = table.split("\n")
            tab = np.array(
                [line.split("|") for line in tabl if line and line[0] == "|"]
            )
            self.log.info("tab %s" % tab)
            self.pk = float(tab[1][2])
            self.fppPeak = float(tab[1][3])
            self.fpPeak = float(tab[1][4])
            self.ip = float(tab[2][2])
            self.fppInfl = float(tab[2][3])
            self.fpInfl = float(tab[2][4])
            self.efs = self.get_efs()
        except:
            self.pk = None
            self.fppPeak = None
            self.fpPeak = None
            self.ip = None
            self.fppInfl = None
            self.fpInfl = None
            self.efs = None
        return {
            "pk": self.pk,
            "fppPeak": self.fppPeak,
            "fpPeak": self.fpPeak,
            "ip": self.ip,
            "fppInfl": self.fppInfl,
            "fpInfl": self.fpInfl,
            "efs": self.efs,
        }

    def analyze(self):
        self.chooch_analysis()
        try:
            self.save_efs_plot()
        except:
            self.log.info(traceback.format_exc())

    def chooch_analysis(self):
        import subprocess

        chooch_results = {}
        chooch_parameters = {
            "element": self.element,
            "edge": self.edge,
            "raw_file": self.get_raw_filename(),
            "output_ps": self.get_ps_filename(),
            "output_png": self.get_png_filename(),
            "output_efs": self.get_efs_filename(),
        }

        # chooch_command = 'chooch -p {output_ps} -o {output_efs} -e {element} -a {edge} {raw_file}'.format(**chooch_parameters)
        chooch_command = "chooch -p {output_ps} -g {output_png} -o {output_efs} -e {element} -a {edge} {raw_file}".format(
            **chooch_parameters
        )
        self.log.info("chooch_line %s" % chooch_command)

        chooch_output = subprocess.getoutput(chooch_command)
        chooch_results["chooch_output"] = chooch_output
        self.log.info("chooch_output %s" % chooch_output)
        chooch_results = self.parse_chooch_output(chooch_output)

        chooch_results["chooch_results"] = chooch_results

        f = open(self.get_chooch_results_filename(), "wb")
        pickle.dump(chooch_results, f)
        f.close()

    def get_theta_chronos_predictor(self):
        all_observations = self.get_all_observations()

        X = np.array(all_observations["actuator_monitor"]["observations"])
        chronos, thetabragg = X[:, 0], X[:, 1]
        thetabragg_linear_fit = np.polyfit(chronos, thetabragg, 1)
        theta_chronos_predictor = np.poly1d(thetabragg_linear_fit)

        return theta_chronos_predictor

    def get_mca_observations(self):
        all_observations = self.get_all_observations()

        if self.shutterless == True:
            mca_observations = all_observations["mca"]["observations"]

            mca_chronos = np.array([item[0] for item in mca_observations])

            theta_chronos_predictor = self.get_theta_chronos_predictor()
            mca_theta = theta_chronos_predictor(mca_chronos)
            mca_wavelengths = self.resolution_motor.get_wavelength_from_theta(mca_theta)
            mca_energies = self.resolution_motor.get_energy_from_wavelength(
                mca_wavelengths
            )

            mca_normalized_counts = np.array([item[3] for item in mca_observations])

        else:
            mca_energies = np.array(
                [
                    observation[0]
                    for observation in all_observations["shuttered_observations"]
                ]
            )
            mca_normalized_counts = np.array(
                [
                    observation[4]
                    for observation in all_observations["shuttered_observations"]
                ]
            )

        if self.inverse == True:
            mca_energies = mca_energies[::-1]
            mca_normalized_counts = mca_normalized_counts[::-1]

        return mca_energies, mca_normalized_counts

    def get_spectrum(self, equidistant=False):
        all_observations = self.get_all_observations()

        mca_energies, mca_normalized_counts = self.get_mca_observations()

        if equidistant == True:
            energies = np.linspace(
                round(mca_energies.min()),
                round(mca_energies.max()),
                int(self.scan_range / self.scan_step + 1),
            )
            counts = np.interp(energies, mca_energies, mca_normalized_counts)
        else:
            enc = list(zip(mca_energies, mca_normalized_counts))
            enc.sort(key=lambda x: x[0])
            enc = np.array(enc)
            energies = enc[:, 0]
            counts = enc[:, 1]

        if self.ignore_first_eV != 0.0:
            X = np.array(
                [
                    (e, c)
                    for e, c in zip(energies, counts)
                    if e > energies.min() + self.ignore_first_eV
                    and e < energies.max() - self.ignore_last_eV
                ]
            )
            energies = X[:, 0]
            counts = X[:, 1]
        return energies, counts

    def get_all_observations(self):
        self.log.info("get_all_observations")
        if self.all_observations != None:
            pass
        elif os.path.isfile(self.get_all_observations_filename()):
            self.all_observations = self.load_all_observations()
        else:
            self.log.info("get_all_observations gathering")
            all_observations = {}

            all_observations["actuator_monitor"] = {}
            all_observations["actuator_monitor"][
                "observation_fields"
            ] = self.actuator.get_observation_fields()
            all_observations["actuator_monitor"][
                "observations"
            ] = self.actuator.get_observations()

            for monitor_name, mon in zip(self.monitor_names, self.monitors):
                all_observations[monitor_name] = {}
                try:
                    all_observations[monitor_name][
                        "observation_fields"
                    ] = mon.get_observation_fields()
                    all_observations[monitor_name][
                        "observations"
                    ] = mon.get_observations()
                except:
                    message = (
                        "Could not get diagnostic information from %s, please check"
                        % monitor_name
                    )
                    self.log.info(message)
                    exc = traceback.format_exc()
                    # print(exc)
                    self.log.debug(exc)

            if self.shutterless == False:
                all_observations["shuttered_observations"] = self.shuttered_observations

            self.all_observations = all_observations
        return self.all_observations

    def save_all_observations(self):
        self.log.info("save_all_observations")
        f = open(self.get_all_observations_filename(), "wb")
        pickle.dump(self.get_all_observations(), f)
        f.close()

    def load_all_observations(self):
        return pickle.load(open(self.get_all_observations_filename(), "rb"))

    def stop(self):
        self.stop_monitor()
        self.actuator.stop()
        self.fast_shutter.close()
        self.actuator.set_speed(self.default_speed)

    def save_raw_scan(self):
        raw_filename = self.get_raw_filename()
        energies, counts = self.get_spectrum(equidistant=self.equidistant_spectrum)

        X = np.vstack([energies, counts]).T

        header = "%s\n\n%d" % (self.description, X.shape[0])
        np.savetxt(raw_filename, X, header=header, comments="#")

        # if scipy.__version__ > '1.7.0':
        # scipy.savetxt(raw_filename, X, header=header)
        # else:
        # f = open(raw_filename, 'w')
        # f.write(header)
        # scipy.savetxt(f, X)
        # f.close()

    def save_raw_scan_plot(self):
        pylab.figure(figsize=(16, 9))
        energies, counts = self.get_spectrum()
        pylab.plot(energies, counts, "go-")
        pylab.xlabel("energy [eV]", fontsize=22)
        pylab.ylabel("normalized counts", fontsize=22)
        ax = pylab.gca()
        pylab.text(
            0.05,
            0.95,
            r"\# points %d" % len(energies),
            fontsize=22,
            transform=ax.transAxes,
        )
        pylab.title(self.description, fontsize=22)
        pylab.grid(True)
        pylab.savefig(self.get_raw_scan_plot_filename())

        if self.display == True:
            pylab.show()

    def save_efs_plot(self):
        efs = self.get_efs()
        pylab.figure(figsize=(16, 9))
        energies, counts = self.get_spectrum()
        self.log.debug("efs %s" % efs)
        e = efs[:, 0]
        fdp = efs[:, 1]
        fp = efs[:, 2]
        df = fdp - fp

        gdf = np.gradient(df)

        ip0 = e[np.argmin(fp)]
        ip1 = e[np.argmax(gdf)]
        ip2 = e[np.argmin(gdf)]

        pk0 = e[np.argmax(fdp)]
        pk2 = np.mean([ip1, ip2])

        pylab.plot(e, fdp, "k-", label=r"$f^{\prime\prime}$")
        pylab.plot(e, fp, "r-", label=r"$f^{\prime}$")
        pylab.plot(e, df, "g-", label=r"$\Delta f = f^{\prime\prime} - f^{\prime}$")
        pylab.plot(e, gdf, label=r"$\Delta$ gradient")

        pylab.vlines(
            ip0,
            fp.min() - 1,
            fp.min() + 1,
            colors="r",
            label=r"$f^{\prime}_{min}=%.2f$" % ip0,
        )

        pylab.vlines(
            ip1,
            df[e == ip1] - 1,
            df[e == ip1] + 1,
            colors="g",
            label=r"$\Delta f_{inf2} = %.2f$" % ip1,
        )

        pylab.vlines(
            pk2,
            df.max() - 1,
            df.max() + 1,
            colors="g",
            label=r"$\Delta f_{max} = %.2f$" % pk2,
        )

        pylab.vlines(
            pk0,
            fdp.max() - 1,
            fdp.max() + 1,
            colors="k",
            label=r"$f^{\prime\prime}_{max}=%.2f$" % pk0,
        )

        pylab.vlines(
            ip2,
            df[e == ip2] - 1,
            df[e == ip2] + 1,
            colors="g",
            label=r"$\Delta f_{inf2} = %.2f$" % ip2,
        )

        pylab.xlabel("energy [eV]", fontsize=22)
        pylab.ylabel(r"$f^{\prime}$ and $f^{\prime\prime}$ [electrons]", fontsize=22)
        ax = pylab.gca()
        pylab.text(
            0.05,
            0.95,
            r"\# points %d" % len(energies),
            fontsize=22,
            transform=ax.transAxes,
        )
        pylab.title(self.description, fontsize=22)
        pylab.grid(True)
        pylab.legend(fontsize=18)
        pylab.savefig(self.get_efs_plot_filename())

        if self.display == True:
            pylab.show()


def main():
    usage = """Program for energy scans
    
    ./energy_scan.py -e <element> -s <edge> <options>
    
    """

    import optparse

    parser = optparse.OptionParser(usage=usage)

    parser.add_option("-e", "--element", type=str, help="Specify the element")
    parser.add_option("-s", "--edge", type=str, help="Specify the edge")
    parser.add_option(
        "-d",
        "--directory",
        type=str,
        default="/tmp/testXanes",
        help="Directory to store the results (default=%default)",
    )
    parser.add_option(
        "-n", "--name_pattern", type=str, default="escan", help="name_pattern"
    )
    parser.add_option(
        "-i",
        "--integration_time",
        type=float,
        default=0.25,
        help="integration time (default=%default) in seconds",
    )
    parser.add_option(
        "-o", "--optimize", action="store_true", help="optimize transmission"
    )
    parser.add_option("-D", "--display", action="store_true", help="display plot")
    parser.add_option("-I", "--inverse", action="store_true", help="inverse scan")
    parser.add_option("-A", "--analysis", action="store_true", help="perform analysis")
    parser.add_option(
        "-E",
        "--equidistant_spectrum",
        action="store_true",
        help="save raw scan with equidistant energies",
    )
    parser.add_option(
        "-G",
        "--ignore_first_eV",
        type=float,
        default=0.0,
        help="ignore first part of the spectrum",
    )
    parser.add_option(
        "-L",
        "--ignore_last_eV",
        type=float,
        default=0.0,
        help="ignore last part of the spectrum",
    )
    parser.add_option(
        "-t", "--transmission", type=float, default=0.05, help="Default transmission"
    )
    parser.add_option(
        "-T",
        "--total_time",
        type=float,
        default=100.0,
        help="total scan time (default=%default)",
    )
    parser.add_option(
        "-r",
        "--scan_range",
        type=float,
        default=100.0,
        help="scan range (default=%default eV)",
    )
    parser.add_option("-f", "--use_flyscan", type=int, default=0, help="use flyscan")
    parser.add_option(
        "-c", "--detector_card", default="xia", type=str, help="counting card to use"
    )

    options, args = parser.parse_args()

    print("options", options)
    print("args", args)

    options.use_flyscan = bool(options.use_flyscan)
    print("modified options", options)

    escan = energy_scan(**vars(options))

    filename = "%s_parameters.pickle" % escan.get_template()

    if not os.path.isfile(filename):
        escan.execute()
        # pass
    elif options.analysis == True:
        escan.save_raw_scan()
        escan.save_raw_scan_plot()
        escan.chooch_analysis()
        try:
            escan.save_efs_plot()
        except:
            traceback.print_exc()


if __name__ == "__main__":
    main()
