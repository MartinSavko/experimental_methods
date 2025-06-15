#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import pickle
import logging
import traceback

import gevent
import numpy as np

from experiment import experiment
from goniometer import goniometer
from detector import detector as detector
from energy import energy as energy_motor, energy_mockup
from motor import (
    undulator,
    monochromator_rx_motor,
    undulator_mockup,
    monochromator_rx_motor_mockup,
)
from resolution import resolution as resolution_motor, resolution_mockup
from transmission import transmission as transmission_motor, transmission_mockup
from machine_status import machine_status, machine_status_mockup
from flux import flux as flux_monitor, flux_mockup
from beam_center import beam_center, beam_center_mockup
from frontend_shutter import frontend_shutter
from safety_shutter import safety_shutter
from fast_shutter import fast_shutter

from protective_cover import protective_cover
from monitor import (
    xbpm,
    xbpm_mockup,
    eiger_en_out,
    fast_shutter_close,
    fast_shutter_open,
    trigger_eiger_on,
    trigger_eiger_off,
    Si_PIN_diode,
    sai,
    monitor,
    jaull,
    tdl_xbpm,
)
from slits import slits1, slits2, slits3, slits5, slits6, slits_mockup
from experimental_table import experimental_table
from cryostream import cryostream
from motor import tango_motor


class xray_experiment(experiment):
    specific_parameter_fields = [
        {
            "name": "photon_energy",
            "type": "float",
            "description": "photon energy of the experiment in eV",
        },
        {
            "name": "wavelength",
            "type": "float",
            "description": "experiment photon wavelength in A",
        },
        {
            "name": "transmission_intention",
            "type": "float",
            "description": "intended photon beam transmission in %",
        },
        {
            "name": "transmission",
            "type": "float",
            "description": "measured photon beam transmission in %",
        },
        {
            "name": "flux_intention",
            "type": "float",
            "description": "intended flux of the experiment in ph/s",
        },
        {
            "name": "flux",
            "type": "float",
            "description": "intended flux of the experiment in ph/s",
        },
        {
            "name": "slit_configuration",
            "type": "dict",
            "description": "slit configuration",
        },
        {
            "name": "undulator_gap",
            "type": "float",
            "description": "experiment undulator gap in mm",
        },
        {
            "name": "undulator_gap_encoder_position",
            "type": "float",
            "description": "experiment undulator gap encoder reading in mm",
        },
        {
            "name": "monitor_sleep_time",
            "type": "float",
            "description": "default pause between monitor measurements in s",
        },
        {
            "name": "beware_of_top_up",
            "type": "bool",
            "description": "try to avoid top-up",
        },
        {
            "name": "tdl_xbpm1_position",
            "type": "dict",
            "description": "electron beam position at tdl xbpm1",
        },
        {
            "name": "tdl_xbpm2_position",
            "type": "dict",
            "description": "electron beam position at tdl xbpm2",
        },
        {"name": "tdl_x", "type": "float", "description": "diaphragm x position"},
        {"name": "tdl_z", "type": "float", "description": "diaphragm z position"},
        {"name": "shutter_x", "type": "float", "description": "diaphragm x position"},
        {"name": "shutter_z", "type": "float", "description": "diaphragm z position"},
        {
            "name": "vfm_position",
            "type": "dict",
            "description": "vertical focusing mirror position",
        },
        {
            "name": "hfm_position",
            "type": "dict",
            "description": "horizontal focusing mirror position",
        },
        {
            "name": "experimental_table_status",
            "type": "dict",
            "description": "experimental table status",
        },
        {
            "name": "cryostream_status",
            "type": "dict",
            "description": "cryostream status",
        },
        {
            "name": "mono_mt_rx_position",
            "type": "dict",
            "description": "mono rx position",
        },
        {
            "name": "mono_mt_rx_fine_position",
            "type": "dict",
            "description": "mono rx_fine position",
        },
        {
            "name": "goniometer_settings",
            "type": "dict",
            "description": "all goniometer motors' positions",
        },
        {"name": "machine_current", "type": "float", "description": "machine current"},
        {"name": "xbpm_readings", "type": "dict", "description": "xbpm readings"},
        {"name": "ntrigger", "type": "int", "description": "number of triggers"},
    ]

    def __init__(
        self,
        name_pattern,
        directory,
        position=None,
        photon_energy=None,
        transmission=None,
        flux=None,
        ntrigger=1,
        snapshot=False,
        zoom=None,
        diagnostic=None,
        analysis=None,
        conclusion=None,
        simulation=None,
        monitor_sleep_time=0.05,
        parent=None,
        mxcube_parent_id=None,
        mxcube_gparent_id=None,
        beware_of_top_up=False,
        panda=False,
    ):
        logging.debug(
            "xray_experiment __init__ len(xray_experiment.specific_parameter_fields) %d"
            % len(xray_experiment.specific_parameter_fields)
        )

        if hasattr(self, "parameter_fields"):
            self.parameter_fields += xray_experiment.specific_parameter_fields
        else:
            self.parameter_fields = xray_experiment.specific_parameter_fields[:]

        logging.debug(
            "xray_experiment __init__ len(self.parameters_fields) %d"
            % len(self.parameter_fields)
        )

        experiment.__init__(
            self,
            name_pattern=name_pattern,
            directory=directory,
            diagnostic=diagnostic,
            analysis=analysis,
            conclusion=conclusion,
            simulation=simulation,
            snapshot=snapshot,
            mxcube_parent_id=mxcube_parent_id,
            mxcube_gparent_id=mxcube_gparent_id,
        )

        self.description = "X-ray experiment, Proxima 2A, SOLEIL, %s" % time.ctime(
            self.timestamp
        )
        self.position = position
        self.photon_energy = photon_energy

        self.flux = flux
        self.ntrigger = ntrigger
        self.zoom = zoom
        self.monitor_sleep_time = monitor_sleep_time
        self.parent = parent
        self.beware_of_top_up = beware_of_top_up
        self.panda = panda

        # Necessary equipment
        self.actuators = [
            {"name": "goniometer", "object": goniometer, "must": True},
            {"name": "frontend_shutter", "object": frontend_shutter, "must": True},
            {"name": "safety_shutter", "object": safety_shutter, "must": True},
            {
                "name": "fast_shutter",
                "object": fast_shutter,
                "must": True,
                "kwargs": {"panda": panda},
            },
            {
                "name": "beam_center",
                "object": beam_center,
                "mockup": beam_center_mockup,
            },
            {"name": "detector", "object": detector, "must": True},
            {
                "name": "resolution_motor",
                "object": resolution_motor,
                "mockup": resolution_mockup,
            },
            {"name": "energy_motor", "object": energy_motor, "mockup": energy_mockup},
            {"name": "flux_monitor", "object": flux_monitor, "mockup": flux_mockup},
            {
                "name": "transmission_motor",
                "object": transmission_motor,
                "mockup": transmission_mockup,
            },
            {
                "name": "machine_status",
                "object": machine_status,
                "mockup": machine_status_mockup,
            },
            {"name": "undulator", "object": undulator, "mockup": undulator_mockup},
            {
                "name": "monochromator_rx_motor",
                "object": monochromator_rx_motor,
                "mockup": monochromator_rx_motor_mockup,
            },
            {"name": "protective_cover", "object": protective_cover, "mockup": None},
            {"name": "slits1", "object": slits1, "mockup": slits_mockup},
            {"name": "slits2", "object": slits2, "mockup": slits_mockup},
            {"name": "slits3", "object": slits3, "mockup": slits_mockup},
            {"name": "slits5", "object": slits5, "mockup": slits_mockup},
            {"name": "slits6", "object": slits6, "mockup": slits_mockup},
            {
                "name": "experimental_table",
                "object": experimental_table,
                "kwargs": {
                    "attributes": [
                        "pitch",
                        "roll",
                        "yaw",
                        "zC",
                        "xC",
                        "t1z",
                        "t2z",
                        "t3z",
                        "t4x",
                        "t5x",
                    ]
                },
                "mockup": monitor,
            },
            {
                "name": "cryostream",
                "object": cryostream,
                "kwargs": {
                    "attributes": [
                        "evapHeat",
                        "evapShift",
                        "gasHeat",
                        "suctHeat",
                        "rampRate",
                        "backPressure",
                        "evapTemp",
                        "gasFlow",
                        "sampleTemp",
                        "setPoint",
                        "suctTemp",
                        "targetTemp",
                        "tempError",
                    ],
                },
                "mockup": monitor,
            },
        ]

        for name, device_name in [
            ("vfm_pitch", "i11-ma-c05/op/mir.2-mt_rx"),
            ("hfm_pitch", "i11-ma-c05/op/mir.3-mt_rz"),
            ("vfm_trans", "i11-ma-c05/op/mir.2-mt_tz"),
            ("hfm_trans", "i11-ma-c05/op/mir.3-mt_tx"),
            ("mono_mt_rx", "i11-ma-c03/op/mono1-mt_rx"),
            ("mono_mt_rx_fine", "i11-ma-c03/op/mono1-mt_rx_fine"),
        ]:
            a = {
                "name": name,
                "object": tango_motor,
                "kwargs": {"device_name": device_name},
            }
            self.actuators.append(a)

        self.initialize_actuators()

        if self.photon_energy == None and self.simulation != True:
            self.photon_energy = self.get_current_photon_energy()

        self.wavelength = self.resolution_motor.get_wavelength_from_energy(
            self.photon_energy
        )

        self.observers = [
            {
                "name": "xbpm1",
                "object": xbpm,
                "kwargs": {"device_name": "i11-ma-c04/dt/xbpm_diode.1-base"},
                "mockup": xbpm_mockup,
            },
            {
                "name": "cvd1",
                "object": xbpm,
                "kwargs": {"device_name": "i11-ma-c05/dt/xbpm-cvd.1-base"},
                "mockup": xbpm_mockup,
            },
            {
                "name": "xbpm5",
                "object": xbpm,
                "kwargs": {"device_name": "i11-ma-c06/dt/xbpm_diode.5-base"},
                "mockup": xbpm_mockup,
            },
            {
                "name": "psd5",
                "object": xbpm,
                "kwargs": {"device_name": "i11-ma-c06/dt/xbpm_diode.psd.5-base"},
                "mockup": xbpm_mockup,
            },
            {
                "name": "psd6",
                "object": xbpm,
                "kwargs": {"device_name": "i11-ma-c06/dt/xbpm_diode.6-base"},
                "mockup": xbpm_mockup,
            },
            {
                "name": "sai1",
                "object": sai,
                "kwargs": {
                    "device_name": "i11-ma-c00/ca/sai.1",
                    "continuous_monitor_name": "sai1_monitor",
                },
                "mockup": monitor,
            },
            {
                "name": "sai2",
                "object": sai,
                "kwargs": {
                    "device_name": "i11-ma-c00/ca/sai.2",
                    "number_of_channels": 1,
                    "continuous_monitor_name": "sai2_monitor",
                },
                "mockup": monitor,
            },
            {
                "name": "fast_shutter",
                "object": fast_shutter,
                "must": True,
                "kwargs": {"panda": panda},
            },
            {
                "name": "sai3",
                "object": sai,
                "kwargs": {
                    "device_name": "i11-ma-c00/ca/sai.3",
                    "continuous_monitor_name": "sai3_monitor",
                },
                "mockup": monitor,
            },
            {
                "name": "sai4",
                "object": sai,
                "kwargs": {
                    "device_name": "i11-ma-c00/ca/sai.4",
                    "continuous_monitor_name": "sai4_monitor",
                },
                "mockup": monitor,
            },
            {
                "name": "sai5",
                "object": sai,
                "kwargs": {
                    "device_name": "i11-ma-c00/ca/sai.5",
                    "continuous_monitor_name": "sai5_monitor",
                },
                "mockup": monitor,
            },
            {
                "name": "jaull05",
                "object": jaull,
                "kwargs": {"device_name": "i11-ma-c05/vi/jaull.1", "name": "jaull05"},
                "mockup": monitor,
            },
            {
                "name": "jaull06",
                "object": jaull,
                "kwargs": {"device_name": "i11-ma-c06/vi/jaull.1", "name": "jaull06"},
                "mockup": monitor,
            },
            {"name": "eiger_en_out", "object": eiger_en_out, "mockup": monitor},
            {"name": "trigger_eiger_on", "object": trigger_eiger_on, "mockup": monitor},
            {
                "name": "trigger_eiger_off",
                "object": trigger_eiger_off,
                "mockup": monitor,
            },
            {
                "name": "fast_shutter_open",
                "object": fast_shutter_open,
                "mockup": monitor,
            },
            {
                "name": "fast_shutter_close",
                "object": fast_shutter_close,
                "mockup": monitor,
            },
            {"name": "Si_PIN_diode", "object": Si_PIN_diode, "mockup": monitor},
            {
                "name": "experimental_table",
                "object": experimental_table,
                "kwargs": {"attributes": ["pitch", "roll", "yaw", "zC", "xC"]},
                "mockup": monitor,
            },
            {
                "name": "tdl_xbpm1",
                "object": tdl_xbpm,
                "kwargs": {"device_name": "tdl-i11-ma/dg/xbpm.1"},
            },
            {
                "name": "tdl_xbpm2",
                "object": tdl_xbpm,
                "kwargs": {"device_name": "tdl-i11-ma/dg/xbpm.2"},
            },
            {"name": "self", "object": self},
        ]

        for name, device_name in [
            ("vfm_pitch", "i11-ma-c05/op/mir.2-mt_rx"),
            ("hfm_pitch", "i11-ma-c05/op/mir.3-mt_rz"),
            ("vfm_trans", "i11-ma-c05/op/mir.2-mt_tz"),
            ("hfm_trans", "i11-ma-c05/op/mir.3-mt_tx"),
            ("tab2_tx1", "i11-ma-c05/ex/tab.2-mt_tx.1"),
            ("tab2_tx2", "i11-ma-c05/ex/tab.2-mt_tx.2"),
            ("tab2_tz1", "i11-ma-c05/ex/tab.2-mt_tz.1"),
            ("tab2_tz2", "i11-ma-c05/ex/tab.2-mt_tz.2"),
            ("tab2_tz3", "i11-ma-c05/ex/tab.2-mt_tz.3"),
            ("mono_mt_rx", "i11-ma-c03/op/mono1-mt_rx"),
            ("mono_mt_rx_fine", "i11-ma-c03/op/mono1-mt_rx_fine"),
            ("tdl_x", "tdl-i11-ma/vi/mtx.1"),
            ("tdl_z", "tdl-i11-ma/vi/mtz.1"),
            ("shutter_x", "i11-ma-c06/ex/shutter-mt_tx"),
            ("shutter_z", "i11-ma-c06/ex/shutter-mt_tz"),
        ]:
            o = {
                "name": name,
                "object": tango_motor,
                "kwargs": {"device_name": device_name},
            }
            self.observers.append(o)

        self.monitor_names = []
        self.monitors = []
        self.monitors_dictionary = {}

        self.initialize_observers()

        self.transmission_intention = transmission
        if self.transmission_intention is None:
            transmission = self.transmission_motor.get_transmission()
        self.transmission = transmission

        self.image = None
        self.rgbimage = None
        self._stop_flag = False

    def initialize_actuators(self):
        for actuator in self.actuators:
            kw = {}
            if "kwargs" in actuator:
                kw = actuator["kwargs"]

            try:
                setattr(self, actuator["name"], actuator["object"](**kw))
            except:
                if "must" in actuator and actuator["must"]:
                    raise
                else:
                    setattr(self, actuator["name"], actuator["mockup"](**kw))

    def initialize_observers(self, observers=None):
        if observers is None:
            observers = self.observers

        for observer in observers:
            monitor_name = observer["name"]
            self.monitor_names.append(monitor_name)

            if monitor_name == "self":
                monitor = observer["object"]
            else:
                kw = {}
                if "kwargs" in observer:
                    kw = observer["kwargs"]

                try:
                    monitor = observer["object"](**kw)
                except:
                    if "must" in observer:
                        raise
                    else:
                        monitor = observer["mockup"](**kw)
                setattr(self, monitor_name, monitor)

            self.monitors.append(monitor)
            self.monitors_dictionary[monitor_name] = monitor

    def check_top_up(self, equilibrium=3, sleeptime=1.0):
        logging.info("going to check for when the next top up is expected to occur ...")
        try:
            trigger_current = self.machine_status.get_trigger_current()
            top_up_period = self.machine_status.get_top_up_period()

            time_to_next_top_up = self.machine_status.get_time_to_next_top_up(
                trigger_current=trigger_current
            )
            while (
                (self.scan_exposure_time <= top_up_period / 4.0)
                and (time_to_next_top_up <= self.scan_exposure_time * 1.05)
                and time_to_next_top_up > 0
            ):
                logging.info(
                    "expected time to next the top up %.1f seconds, waiting for it ..."
                    % time_to_next_top_up
                )
                gevent.sleep(max(sleeptime, time_to_next_top_up / 2.0))
                time_to_next_top_up = self.machine_status.get_time_to_next_top_up(
                    trigger_current=trigger_current
                )

            time_from_last_top_up = self.machine_status.get_time_from_last_top_up()
            while time_from_last_top_up < equilibrium:
                logging.info(
                    "waiting for things to settle after the last top up (%.1f seconds ago)"
                    % time_from_last_top_up
                )
                gevent.sleep(max(sleeptime, time_from_last_top_up / 2))
                time_from_last_top_up = self.machine_status.get_time_from_last_top_up()

            logging.info(
                "time to next top up %.1f seconds, expected scan duration is %.1f seconds, executing the scan ..."
                % (time_to_next_top_up, self.scan_exposure_time)
            )
        except:
            traceback.print_exc()

    def get_machine_current(self):
        return self.machine_status.get_current()

    def get_xbpm_readings(self):
        xbpm_readings = {}
        for name in ["xbpm1", "cvd1", "xbpm5", "psd5", "psd6"]:
            try:
                reading = getattr(getattr(self, name), "get_point_and_reference")()
                xbpm_readings[name] = reading
            except:
                traceback.print_exc()
        return xbpm_readings

    def get_undulator_gap(self):
        return self.undulator.get_position()

    def get_undulator_gap_encoder_position(self):
        return self.undulator.get_encoder_position()

    def get_tdl_xbpm1_position(self):
        return self.tdl_xbpm1.get_position()

    def get_tdl_xbpm2_position(self):
        return self.tdl_xbpm2.get_position()

    def get_tdl_x(self):
        return self.tdl_x.get_position()

    def get_tdl_z(self):
        return self.tdl_z.get_position()

    def get_shutter_x(self):
        return self.shutter_x.get_position()

    def get_shutter_z(self):
        return self.shutter_z.get_position()

    def get_vfm_position(self):
        pitch = self.vfm_pitch.get_position()
        trans = self.vfm_trans.get_position()
        return {"pitch": pitch, "translation": trans}

    def get_hfm_position(self):
        pitch = self.hfm_pitch.get_position()
        trans = self.hfm_trans.get_position()
        return {"pitch": pitch, "translation": trans}

    def get_experimental_table_status(self):
        return self.experimental_table.get_position()

    def get_cryostream_status(self):
        return self.cryostream.get_position()

    def get_mono_mt_rx_position(self):
        return self.mono_mt_rx.get_position()

    def get_mono_mt_rx_fine_position(self):
        return self.mono_mt_rx_fine.get_position()

    def get_slit_configuration(self):
        slit_configuration = {}
        for k in [1, 2, 3, 5, 6]:
            conf = {}
            for direction in ["vertical", "horizontal"]:
                for attribute in ["gap", "position"]:
                    conf["%s_%s" % (direction, attribute)] = getattr(
                        getattr(self, "slits%d" % k),
                        "get_%s_%s" % (direction, attribute),
                    )()
            slit_configuration["slits%d" % (k,)] = conf
        return slit_configuration

    def get_progression(self):
        def changes(a):
            """Helper function to remove consecutive indices.
            -- trying to get out of np.gradient what skimage.match_template would return
            """
            if len(a) == 0:
                return
            elif len(a) == 1:
                return a
            indices = [True]
            for k in range(1, len(a)):
                # if a[k]-1 == a[k-1]:
                if a[k] == a[k - 1]:
                    indices.append(False)
                else:
                    indices.append(True)
            return a[np.array(indices)]

        def get_on_segments(on, off):
            segments = []
            if len(on) == 0:
                pass
            elif off is None:
                segments = [(on[-1], -1)]
            elif len(on) == len(off):
                segments = list(zip(on, off))
            elif len(on) > len(off):
                segments = list(zip(on[:-1], off))
                segments.append((on[-1], -1))
            return segments

        def get_complete_incomplete_wedges(segments):
            nsegments = len(segments)
            if nsegments and segments[-1][-1] == -1:
                complete = nsegments - 1
                incomplete = 1
            else:
                complete = nsegments
                incomplete = 0
            return complete, incomplete

        fs_observations = self.fast_shutter.get_observations()
        if (
            len(fs_observations) >= 2
            and int(fs_observations[0][1]) == 1
            and int(fs_observations[1][1]) == 1
        ):
            fs_observations[0][1] = 0
            fs_observations[1][1] = 0
        observations = np.array(fs_observations)

        if fs_observations == [] or len(observations) < 3:
            return 0
        try:
            g = np.gradient(observations[:, 1])
            ons = changes(np.where(g == 0.5)[0])
            offs = changes(np.where(g == -0.5)[0])
            if ons is None:
                return 0
            bons = [on for k, on in enumerate(ons[:-1]) if on == ons[k + 1] - 1]
            if offs is not None:
                boffs = [
                    off for k, off in enumerate(offs[:-1]) if off == offs[k + 1] - 1
                ]
            else:
                boffs = offs
        except:
            print("observations")
            print(observations)
            print("ons")
            print(ons)
            print(traceback.print_exc())

        segments = get_on_segments(bons, boffs)
        if segments == []:
            return 0

        chronos = observations[:, 0]
        total_exposure_time = 0

        for segment in segments:
            total_exposure_time += chronos[segment[1]] - chronos[segment[0]]

        progression = 100 * total_exposure_time / self.total_expected_exposure_time
        complete, incomplete = get_complete_incomplete_wedges(segments)
        if progression > 100:
            progression = 100
        if complete == self.total_expected_wedges:
            progression = 100
        return progression

    def get_point(self, start_time):
        chronos = time.time() - start_time
        progress = self.get_progression()
        return [chronos, progress]

    def monitor(self, start_time):
        if not hasattr(self, "observations"):
            self.observations = []
        self.observation_fields = ["chronos", "progress"]
        last_point = [None, None]
        while self.observe == True:
            point = self.get_point(start_time)
            progress = point[1]
            self.observations.append(point)
            if self.parent != None:
                if (
                    point[0] != None
                    and last_point[0] != point[0]
                    and last_point[1] != progress
                    and progress > 0
                ):
                    self.parent.emit("progressStep", (progress))
                    last_point = point
            gevent.sleep(self.monitor_sleep_time)

    def start_monitor(self):
        self.observe = True
        if hasattr(self, "actuator"):
            self.actuator.observe = True
            if hasattr(self, "actuator_names"):
                self.observers_threads = [
                    gevent.spawn(
                        self.actuator.monitor, self.start_time, self.actuator_names
                    )
                ]
            else:
                self.observers_threads = [
                    gevent.spawn(self.actuator.monitor, self.start_time)
                ]
        else:
            self.observers_threads = []
        for monitor in self.monitors:
            monitor.observe = True
            self.observers_threads.append(
                gevent.spawn(monitor.monitor, self.start_time)
            )

    def stop_monitor(self):
        self.observe = False
        if hasattr(self, "actuator"):
            self.actuator.observe = False
        for monitor in self.monitors:
            monitor.observe = False
        gevent.joinall(self.observers_threads)

    def get_observations(self):
        return self.observations

    def get_observation_fields(self):
        return self.observation_fields

    def get_points(self):
        return np.array(self.observations)[:, 1]

    def get_chronos(self):
        return np.array(self.observations)[:, 0]

    def set_photon_energy(self, photon_energy=None, wait=True):
        _start = time.time()
        if photon_energy > 1000:  # if true then it was specified in eV not in keV
            photon_energy *= 1e-3
        if photon_energy != None:
            self.energy_moved = self.energy_motor.set_energy(photon_energy, wait=wait)
        else:
            self.energy_moved = 0

    def get_current_photon_energy(self):
        return self.energy_motor.get_energy()

    def set_transmission(
        self, transmission=None, wait=True, tolerance=0.5, sleeptime=0.05, timeout=7.0
    ):
        if transmission is not None:
            self.transmission = transmission
            self.transmission_motor.set_transmission(transmission)
            current_transmission = self.transmission_motor.get_transmission()
            _start = time.time()
            while (
                (current_transmission is None)
                or (abs(current_transmission - transmission) >= tolerance)
            ) and (time.time() - _start < timeout):
                gevent.sleep(sleeptime)
                current_transmission = self.transmission_motor.get_transmission()

    def get_transmission(self):
        transmission = None
        if hasattr(self, "saved_parameters") and self.saved_parameters is not None:
            transmission = self.saved_parameters["transmission"]
        if transmission is not None:
            return transmission
        elif self.transmission is not None:
            return self.transmission
        else:
            self.transmission_motor.get_transmission()

    def get_transmission_intention(self):
        if hasattr(self, "transmission_intention"):
            return self.transmission_intention
        return self.transmission

    def get_flux(self):
        if self.flux_monitor != None:
            flux = self.flux_monitor.get_flux()
        else:
            flux = None
        return flux

    def get_flux_intention(self):
        return self.flux

    def program_detector(self):
        pass

    def get_goniometer_settings(self):
        return self.instrument.goniometer.get_position()

    def program_goniometer(self):
        try:
            self.instrument.goniometer.set_scan_number_of_frames(1)
            self.instrument.goniometer.set_detector_gate_pulse_enabled(True)
        except:
            self.logger.info(traceback.format_exc())

    def prepare(self):
        super().prepare()
        beamline().check_beam()
        initial_settings = []
        initial_settings.append(
            gevent.spawn(self.set_photon_energy, self.photon_energy, wait=True)
        )
        initial_settings.append(gevent.spawn(self.set_transmission, self.transmission))
        gevent.joinall(initial_settings)

    def collect(self):
        return self.run()

    def measure(self):
        return self.run()

    def acquire(self):
        return self.run()

    def get_clean_slits(self):
        return [1, 2, 3, 5, 6]

    def get_all_observations(self):
        return self.get_results()

    def get_results(self):
        _start = time.time()
        results = {}

        if hasattr(self, "actuator"):
            results["actuator_monitor"] = {
                "observation_fields": self.actuator.get_observation_fields(),
                "observations": self.actuator.get_observations(),
            }

        for monitor_name, monitor in zip(self.monitor_names, self.monitors):
            self.logger.debug("monitor_name", monitor)
            _m_start = time.time()
            try:
                if (
                    hasattr(monitor, "continuous_monitor_name")
                    and monitor.continuous_monitor_name != None
                ):
                    results[monitor_name] = {
                        "observation_fields": monitor.get_observation_fields(),
                        "observations": monitor.get_observations_from_history(
                            start=self.start_time
                        ),
                    }
                else:
                    results[monitor_name] = {
                        "observation_fields": monitor.get_observation_fields(),
                        "observations": monitor.get_observations(),
                    }
            except:
                message = (
                    "Could not get diagnostic information from %s, please check"
                    % monitor_name
                )
                print(message)
                self.logger.info(message)
                exc = traceback.format_exc()
                # print(exc)
                self.logger.debug(exc)
            self.logger.debug(
                "results from %s took %.4f seconds"
                % (monitor_name, time.time() - _m_start)
            )
        self.logger.debug("get_results took %.4f seconds" % (time.time() - _start))
        return results

    def clean(self):
        _start = time.time()
        self.collect_parameters()
        self.logger.debug("collect_parameters finished")
        clean_jobs = []
        clean_jobs.append(gevent.spawn(self.save_parameters))
        clean_jobs.append(gevent.spawn(self.save_log))

        if self.diagnostic == True:
            clean_jobs.append(gevent.spawn(self.save_diagnostics))
        gevent.joinall(clean_jobs)
        self.logger.debug("clean took %.4f seconds" % (time.time() - _start))

    def stop(self):
        self._stop_flag = True
        self.instrument.goniometer.abort()
        self.instrument.detector.abort()


