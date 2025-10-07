#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sweep object will carry out a single sweep of contiguous crystallographic data collection using oscillation method. It uses goniometer() and detector() classes.
"""

from experimental_methods.instrument.goniometer import goniometer
from detector import detector
from beam_center import beam_center
from camera import camera
from protective_cover import protective_cover

import os


class sweep(object):
    def __init__(
        self,
        scan_range,
        scan_exposure_time,
        scan_start_angle,
        angle_per_frame,
        name_pattern,
        directory="/nfs/ruchebis/spool/2016_Run3/orphaned_collects",
        image_nr_start=1,
        helical=False,
    ):
        self.goniometer = goniometer()
        self.detector = detector()
        self.beam_center = beam_center()
        self.protective_cover = protective_cover()

        self.detector.set_trigger_mode("exts")
        self.detector.set_nimages_per_file(100)
        self.detector.set_ntrigger(1)
        scan_range = float(scan_range)
        scan_exposure_time = float(scan_exposure_time)

        nimages, rest = divmod(scan_range, angle_per_frame)

        if rest > 0:
            nimages += 1
            scan_range += rest * angle_per_frame
            scan_exposure_time += rest * angle_per_frame / scan_range

        frame_time = scan_exposure_time / nimages

        self.scan_range = scan_range
        self.scan_exposure_time = scan_exposure_time
        self.scan_start_angle = scan_start_angle
        self.angle_per_frame = angle_per_frame

        self.nimages = int(nimages)
        self.frame_time = float(frame_time)
        self.count_time = self.frame_time - self.detector.get_detector_readout_time()

        self.name_pattern = name_pattern
        self.directory = directory
        self.image_nr_start = image_nr_start
        self.helical = helical
        self.status = None

    def program_goniometer(self):
        self.goniometer.set_scan_start_angle(self.scan_start_angle)
        self.goniometer.set_scan_range(self.scan_range)
        self.goniometer.set_scan_exposure_time(self.scan_exposure_time)
        self.goniometer.set_scan_number_of_frames(1)

    def program_detector(self):
        self.detector.set_ntrigger(1)
        if self.detector.get_compression() != "bslz4":
            self.detector.set_compression("bslz4")
        if self.detector.get_trigger_mode() != "exts":
            self.detector.set_trigger_mode("exts")
        self.detector.set_nimages(self.nimages)
        self.detector.set_nimages_per_file(100)
        self.detector.set_frame_time(self.frame_time)
        self.detector.set_count_time(self.count_time)
        self.detector.set_name_pattern(self.name_pattern)
        self.detector.set_omega(self.scan_start_angle)
        self.detector.set_omega_increment(self.angle_per_frame)
        self.detector.set_image_nr_start(self.image_nr_start)
        beam_center_x, beam_center_y = self.beam_center.get_beam_center()
        print("beam_center_x, beam_center_y", beam_center_x, beam_center_y)

        self.detector.set_beam_center_x(beam_center_x)
        self.detector.set_beam_center_y(beam_center_y)
        self.detector.set_detector_distance(
            self.beam_center.get_detector_distance() / 1000.0
        )
        self.series_id = self.detector.arm()["sequence id"]

    def prepare(self):
        self.status = "prepare"
        self.detector.check_dir(os.path.join(self.directory, "process"))
        self.detector.clear_monitor()
        self.detector.write_destination_namepattern(
            image_path=self.directory, name_pattern=self.name_pattern
        )
        print("self.protective_cover.isclosed()", self.protective_cover.isclosed())
        if self.protective_cover.isclosed() == True:
            self.protective_cover.extract()
        self.goniometer.remove_backlight()
        self.program_goniometer()
        self.program_detector()

    def collect(self, wait=False):
        self.status = "collect"
        self.series_id = self.prepare()
        if wait == False:
            if self.helical:
                task_id = self.goniometer.start_helical_scan()
            else:
                task_id = self.goniometer.start_scan()
        else:
            if self.helical:
                task_id = self.goniometer.start_helical_scan()
            else:
                task_id = self.goniometer.start_scan()
            print("task %d running" % task_id, self.goniometer.is_task_running(task_id))
            self.goniometer.wait_for_task_to_finish(task_id)
            self.clean()
        return task_id

    def stop(self):
        self.goniometer.abort()
        self.detector.abort()

    def clean(self):
        self.detector.disarm()
