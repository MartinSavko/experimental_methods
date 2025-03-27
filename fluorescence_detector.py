#!/usr/bin/env python
# -*- coding: utf-8 -*-
import gevent
import traceback

try:
    import tango
except ImportError:
    import PyTango as tango

import time
from monitor import monitor
from goniometer import goniometer


class fluorescence_detector(monitor):
    cards = {
        "xia": "i11-ma-cx1/dt/dtc-mca_xmap.1",
        "xspress3": "i11-ma-cx1/dt/dt-xspress3",
    }

    def __init__(
        self,
        device_name="i11-ma-cx1/dt/dtc-mca_xmap.1",
        channel="channel00",
        sleeptime=0.001,
    ):
        if device_name in self.cards:
            device_name = self.cards[device_name]
        self.device = tango.DeviceProxy(device_name)
        self.channel = channel
        self.goniometer = goniometer()
        self.sleeptime = sleeptime
        self._calibration = -16.1723871876, 9.93475667754, 0.0
        self.observe = None
        if hasattr(self.device, "presetValue"):
            self.integration_time_attribute = "presetValue"
        elif hasattr(self.device, "exposureTime"):
            self.integration_time_attribute = "exposureTime"
        else:
            self.integration_time_attribute = None

        self.peak_start = 0
        self.peak_end = 1024
        self.compton_start = 956
        self.compton_end = 994
        self.min_trusted = 50

    def load_config_file(self, config_file="0U5MICROS"):
        print("load_config_file")
        self.device.loadConfigFile(config_file)

    def set_config_file(self, config_file="0U5MICROS"):
        if config_file not in [self.get_config_file(), self.get_config_file_alias()]:
            self.load_config_file(config_file=config_file)

    def get_config_file(self):
        return self.device.currentConfigFile

    def get_config_file_alias(self):
        return self.device.currentAlias

    def set_integration_time(self, integration_time, epsilon=1e-3):
        if abs(integration_time - self.get_integration_time()) > epsilon:
            try:
                setattr(self.device, self.integration_time_attribute, integration_time)
            except:
                print(f"failed setting integration_time to {integration_time}")
                traceback.print_exc()
        else:
            print(f"integration time already at the desired value {integration_time}")
            print(f"moving on ...")

    def get_integration_time(self):
        return getattr(self.device, self.integration_time_attribute)

    def set_roi(self, start, end, channel=0):
        # self.device.SetRoisFromList(["%d;%d;%d;%d;%d;%d;%d" % (channel, start, end, 50, end, start, end+250)])
        self.device.SetRoisFromList(["%d;%d;%d" % (channel, start, end)])

    def set_rois(self, start, end, c_start, c_end, min_trusted=50, channel=0):
        print(f"setting rois {start}, {end}, {c_start}, {c_end}, {min_trusted}")
        self.peak_start = int(start)
        self.peak_end = int(end)
        self.compton_start = int(c_start)
        self.compton_end = int(c_end)
        self.min_trusted = int(min_trusted)

        # try:
        # _fs = "%d"+";%d;%d"*4
        # rois = _fs % (*map(int, [channel, start, end, min_trusted, end, start, c_end, c_start, c_end]))
        # self.device.SetRoisFromList([rois])
        # except:
        # print(f'failed setting rois, please check')
        ##traceback.print_exc()

    def get_roi_start_and_roi_end(self):
        return list(map(int, self.device.getrois()[0].split(";")[1:]))

    def insert(self, wait=True, sleeptime=5):
        return
        # self.goniometer.insert_fluorescence_detector()
        # try:
        # self.device.accumulate = False
        # except:
        # pass
        # if wait:
        # time.sleep(sleeptime)

    def extract(self):
        return
        # self.goniometer.extract_fluorescence_detector()

    def cancel(self):
        self.device.Abort()

    def get_state(self):
        return self.device.State().name

    def get_counts_in_roi(self, roi=False):
        counts_in_roi = -1
        if roi:
            try:
                counts_in_roi = float(self.device.roi00_01)
            except:
                traceback.print_exc()
        else:
            try:
                counts_in_roi = float(
                    self.get_spectrum()[self.peak_start : self.peak_end].sum()
                )
            except:
                traceback.print_exc()
        return counts_in_roi

    def get_counts_uptoend_roi(self, roi=False):
        counts_uptoend_roi = -1
        if roi:
            try:
                counts_uptoend_roi = float(self.device.roi00_02)
            except:
                traceback.print_exc()
        else:
            try:
                counts_uptoend_roi = float(
                    self.get_spectrum()[self.min_trusted : self.peak_end].sum()
                )
            except:
                traceback.print_exc()
        return counts_uptoend_roi

    def get_counts_uptopeak_end_roi(self, roi=False):
        counts_uptopeak_end_roi = -1
        if roi:
            try:
                counts_uptopeak_end_roi = float(self.device.roi00_03)
            except:
                traceback.print_exc()
        else:
            try:
                counts_uptopeak_end_roi = float(
                    self.get_spectrum()[self.peak_start : self.compton_end].sum()
                )
            except:
                traceback.print_exc()
        return counts_uptopeak_end_roi

    def get_counts_compton_roi(self, roi=False):
        counts_compton_roi = -1
        if roi:
            try:
                counts_compton_roi = float(self.device.roi00_04)
            except:
                traceback.print_exc()
        else:
            try:
                counts_compton_roi = float(
                    self.get_spectrum()[self.compton_start : self.compton_end].sum()
                )
            except:
                traceback.print_exc()
        return counts_compton_roi

    def get_real_time(self):
        return self.device.realTime00

    def get_dead_time(self):
        return self.device.deadTime00

    def get_trigger_live_time(self):
        return self.device.triggerLiveTime00

    def get_input_count_rate(self):
        return float(self.device.inputCountRate00)

    def get_output_count_rate(self):
        return float(self.device.outputCountRate00)

    def get_events_in_run(self):
        return float(self.device.eventsInRun00)

    def get_calculated_dead_time(self):
        icr = self.get_input_count_rate()
        ocr = self.get_output_count_rate()
        if icr == 0:
            return 0
        return 1e2 * (1 - (ocr / icr))

    def get_calibration(self):
        return self._calibration

    def snap(self, tries=7, sleeptime=0.05):
        success = False
        n = 0
        while not success and n <= tries:
            n += 1
            try:
                self.device.Snap()
                success = True
                # print(f'snap attempt {n} succeeded!')
            except:
                print(f"snap attempt {n} failed")
                traceback.print_exc()
            time.sleep(sleeptime)

    def get_spectrum(self):
        spectrum = self.device.read_attribute(self.channel).value
        return spectrum

    def get_point(self, wait=True):
        if wait:
            integration_time = self.get_integration_time()
            while self.get_state() != "STANDBY":
                gevent.sleep(integration_time / 10)
        self.measure(wait=wait)

        return self.get_spectrum()

    def measure(self, wait=True):
        self.snap()
        if wait:
            integration_time = self.get_integration_time()
            gevent.sleep(integration_time)
            while self.get_state() != "STANDBY":
                gevent.sleep(self.sleeptime / 10)

    def get_single_observation(self, chronos=None):
        measure_start_time = time.time()
        self.measure()
        measure_end_time = time.time()

        readout_start_time = time.time()

        spectrum = self.get_spectrum()
        # print(f"current rois: peak {self.peak_start}: {self.peak_end}")
        # print(f"current rois: up_to_peak {self.min_trusted}: {self.peak_end}")
        # print(f"current rois: compton {self.compton_start}: {self.compton_end}")

        counts_in_roi = float(
            spectrum[self.peak_start : self.peak_end].sum()
        )  # self.get_counts_in_roi()
        # counts_uptoend_roi = float(spectrum[self.min_trusted: self.peak_end].sum()) # self.get_counts_uptoend_roi()
        counts_until_roi = float(spectrum[self.min_trusted : self.peak_start].sum())
        counts_after_roi = float(spectrum[self.peak_end : self.compton_end].sum())
        # counts_compton = float(spectrum[self.compton_start: self.compton_end].sum()) #self.get_counts_compton_roi()

        normalized_counts = counts_in_roi
        trusted_counts = counts_until_roi + counts_after_roi
        if trusted_counts > 0:
            normalized_counts /= trusted_counts
        # if counts_compton > 0:
        # normalized_counts /= counts_compton

        readout_end_time = time.time()
        dead_time = self.get_dead_time()
        normalized_counts *= 1.0 + dead_time / 100.0
        input_count_rate = self.get_input_count_rate()
        output_count_rate = self.get_output_count_rate()
        real_time = self.get_real_time()
        events_in_run = self.get_events_in_run()
        readout_time = readout_end_time - readout_start_time
        measure_time = measure_end_time - measure_start_time
        return [
            chronos + measure_time / 2.0,
            spectrum,
            counts_in_roi,
            normalized_counts,
            dead_time,
            input_count_rate,
            output_count_rate,
            real_time,
            events_in_run,
            measure_time,
            readout_time,
        ]

    def monitor(self, start_time):
        self.observations = []
        self.observation_fields = [
            "chronos",
            "spectrum",
            "counts_in_roi",
            "normalized_counts",
            "dead_time",
            "input_count_rate",
            "output_count_rate",
            "real_time",
            "events_in_run",
            "measure_time",
            "readout_time",
            "duration",
        ]
        while self.observe == True:
            chronos = time.time() - start_time
            observation = self.get_single_observation(chronos)
            duration = time.time() - start_time - chronos
            self.observations.append(observation + [duration])

    def get_observations(self):
        return self.observations

    def get_observation_fields(self):
        return self.observation_fields


def main():
    fd = fluorescence_detector()
    fd.set_integration_time(0.5)
    fd.get_point()


if __name__ == "__main__":
    main()
