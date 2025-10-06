#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The purpose of this object is to record a film (a series of images) 
of a rotating sample on the goniometer as a function of goniometer axis (axes) position(s).
"""
from experiment import experiment
from camera import camera
from goniometer import goniometer
from fast_shutter import fast_shutter
import os
import pickle
import gevent
import time


class film(experiment):
    specific_parameter_fields = [
        {
            "name": "position",
            "type": "dict",
            "description": "dictionary with motor names as keys and their positions in mm as values",
        },
        {"name": "zoom", "type": "int", "description": "zoom value"},
        {
            "name": "calibration",
            "type": "array",
            "description": "camera pixel calibration mm per pixels (vertical, horizontal)",
        },
        {
            "name": "frontlightlevel",
            "type": "float",
            "description": "front light level",
        },
        {"name": "backlightlevel", "type": "float", "description": "back light level"},
        {
            "name": "scan_exposure_time",
            "type": "float",
            "description": "scan exposure time in s",
        },
        {
            "name": "scan_start_angle",
            "type": "float",
            "description": "scan start angle in degrees",
        },
        {
            "name": "scan_speed",
            "type": "float",
            "description": "scan speed in degrees per second",
        },
        {"name": "scan_range", "type": "float", "description": "scan range in degrees"},
        {
            "name": "kappa_position",
            "type": "float",
            "description": "kappa position in degrees",
        },
        {
            "name": "phi_position",
            "type": "float",
            "description": "phi position in degrees",
        },
        {
            "name": "md_task_info",
            "type": "array",
            "description": "scan diagnostic information",
        },
        {
            "name": "frontlight",
            "type": "bool",
            "description": "use frontlight to illuminate sample",
        },
    ]

    def __init__(
        self,
        name_pattern,
        directory,
        scan_range=360,
        scan_exposure_time=3.6,
        scan_start_angle=0,
        position=None,
        kappa_position=None,
        phi_position=None,
        zoom=None,
        frontlight=False,
    ):
        if hasattr(self, "parameter_fields"):
            self.parameter_fields += film.specific_parameter_fields
        else:
            self.parameter_fields = film.specific_parameter_fields[:]

        experiment.__init__(self, name_pattern=name_pattern, directory=directory)

        self.description = "Optical scan, Proxima 2A, SOLEIL, %s" % time.ctime(
            self.timestamp
        )
        self.scan_range = scan_range
        self.scan_exposure_time = scan_exposure_time
        self.scan_start_angle = scan_start_angle
        self.position = position
        self.kappa_position = kappa_position
        self.phi_position = phi_position
        self.zoom = zoom
        self.frontlight = frontlight

        self.camera = camera()
        self.goniometer = goniometer()
        self.fastshutter = fast_shutter()

        self.md_task_info = []

        self.images = []

        if position is None:
            self.position = self.get_position()
        else:
            self.position = self.goniometer.check_position(position)

        self._abort = None

    def get_position(self):
        self.goniometer.get_aligned_position()

    def set_zoom(self, zoom):
        self.camera.set_zoom(zoom)

    def get_zoom(self):
        return self.camera.get_zoom()

    def get_calibration(self):
        return self.camera.get_calibration()

    def get_frontlightlevel(self):
        return self.camera.get_frontlightlevel()

    def get_backlightlevel(self):
        return self.camera.get_backlightlevel()

    def insert_backlight(self):
        return self.goniometer.insert_backlight()

    def insert_frontlight(self):
        return self.goniometer.insert_frontlight()

    def extract_frontlight(self):
        return self.goniometer.extract_frontlight()

    def get_omega_position(self):
        return self.goniometer.get_omega_position()

    def set_omega_position(self, omega_position, wait=True):
        return self.goniometer.set_position({"Omega": omega_position}, wait=wait)

    def get_kappa_position(self):
        return self.goniometer.get_kappa_position()

    def set_kappa_position(self, kappa_position, wait=True):
        return self.goniometer.set_position({"Kappa": kappa_position}, wait=wait)

    def get_phi_position(self):
        return self.goniometer.get_kappa_position()

    def set_phi_position(self, phi_position, wait=True):
        return self.goniometer.set_position({"Phi": phi_position}, wait=wait)

    def prepare(self):
        self.check_directory(self.directory)

        if self.scan_start_angle != None:
            self.set_omega_position(self.scan_start_angle)
        else:
            self.scan_start_angle = self.goniometer.get_omega_position()

        if self.kappa_position != None:
            self.set_kappa_position(self.kappa_position)
        else:
            self.kappa_position = self.goniometer.get_kappa_position()

        if self.phi_position != None:
            self.set_phi_position(self.phi_position)
        else:
            self.phi_position = self.goniometer.get_phi_position()

        if self.zoom != None:
            self.camera.set_zoom(self.zoom)
        else:
            self.zoom = self.camera.get_zoom()

        self.insert_backlight()

        if self.frontlight != True:
            self.extract_frontlight()

    # def run(self, task_id=None):
    # last_image = None
    # if task_id is None:
    # task_id = self.goniometer.start_scan()

    # while self.goniometer.is_task_running(task_id):
    # new_image_id = self.camera.get_image_id()
    # if new_image_id != last_image:
    # last_image = new_image_id
    # self.images.append([new_image_id,
    # self.goniometer.get_omega_position(),
    # self.camera.get_rgbimage()])

    # self.md_task_info = self.goniometer.get_task_info(task_id)

    # new_image_id = self.camera.get_image_id()
    # if new_image_id != last_image_id:
    # last_image_id = new_image_id
    # self.images.append([new_image_id,
    # self.get_omega_position(),
    # self.camera.get_rgbimage()])

    def run(self, task_id=None, step=-120, delta=0.25):
        last_image_id = None
        engaged_debt = 0.0
        engaged_range = 0.0
        k = 0

        _start = time.time()
        while (
            task_id is None
            or self.goniometer.is_task_running(task_id)
            or engaged_range < self.scan_range - 1
            and self._abort != True
        ):
            if task_id is None or self.goniometer.is_task_running(task_id) == False:
                if task_id != None:
                    self.md_task_info.append(self.goniometer.get_task_info(task_id))
                if engaged_range < self.scan_range - 1:
                    task_id = self.set_omega_position(
                        self.get_omega_position() + step + delta, wait=True
                    )
                    k += 1
                    self.logger.info("%d task_id %s" % (k, task_id))
                    engaged_range += abs(step)
                    engaged_debt += delta
                    self.logger.info("engaged_range %.2f" % engaged_range)
        task_id = self.set_omega_position(
            self.get_omega_position() + engaged_debt, wait=True
        )
        _end = time.time()

        self.md_task_info.append(self.goniometer.get_task_info(task_id))
        self.logger.info("nimages %d" % len(self.images))
        self.logger.info("acquisition took %.2f" % (_end - _start))
        return _start, _end

    def cancel(self):
        self._abort = True
        self.goniometer.abort()

    def clean(self):
        self.collect_parameters()
        self.save_parameters()
        self.save_log()
        self.save_results()

    def get_results(self):
        return self.images

    def save_results(self):
        f = open("%s.pickle" % self.get_templatge(), "w")
        pickle.dump(self.get_results(), f)
        f.close()


def main():
    import optparse

    parser = optparse.OptionParser()

    parser.add_option(
        "-r",
        "--scan_range",
        default=360,
        type=float,
        help="Scan range (default: %default)",
    )
    parser.add_option(
        "-e",
        "--scan_exposure_time",
        default=6,
        type=float,
        help="Exposure time (default: %default)",
    )
    parser.add_option(
        "-n",
        "--name_pattern",
        default="sample",
        type=str,
        help="Distinguishing name of files to acquire",
    )
    parser.add_option(
        "-d", "--directory", default="/tmp", type=str, help="Destination directory"
    )
    parser.add_option(
        "-z",
        "--zoom",
        default=None,
        help="Camera zoom to use, current zoom is used by default",
    )
    parser.add_option(
        "-f", "--frontlight", action="store_true", help="Insert frontlight."
    )

    options, args = parser.parse_args()

    acquisition = film(**vars(options))
    acquisition.execute()

    self.logger.info("nimages", len(acquisition.images))


if __name__ == "__main__":
    main()
