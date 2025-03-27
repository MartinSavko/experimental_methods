#!/usr/bin/env python

import os
import sys
import time
import logging

from goniometer import goniometer

from detector import detector

from camera import camera
from oav_camera import oav_camera

from instrument import instrument

from cats import cats

from beam_center import beam_center

from fast_shutter import fast_shutter

from safety_shutter import safety_shutter

from frontend_shutter import frontend_shutter

from monitor import xray_camera, Si_PIN_diode

from transmission import transmission

from flux import flux

from mirror_scan import adaptive_mirror

from fluorescence_detector import fluorescence_detector

from energy import energy as photon_energy

from focusing import focusing

from machine_status import machine_status

from resolution import resolution

from beam_position_controller import get_bpc

from experimental_table import experimental_table

from cryostream import cryostream


class beamline:
    def __init__(self):
        self.goniometer = goniometer()
        self.detector = detector()
        self.camera = oav_camera(service="oav_camera", mode="redis_bzoom")
        self.mako = oav_camera(service="mako", mode="redis_local")
        try:
            self.cats = cats()
        except:
            self.cats = None
        self.beam_center = beam_center()
        self.shutter = safety_shutter()
        self.frontend = frontend_shutter()
        self.fast_shutter = fast_shutter()
        self.transmission = transmission()
        self.flux = flux()
        self.vfm = adaptive_mirror("vfm")
        self.hfm = adaptive_mirror("hfm")
        self.fd = fluorescence_detector()
        self.pe = photon_energy()
        self.focus = focusing()
        self.mach = machine_status()
        self.resolution = resolution()
        self.tab = experimental_table()
        self.cryo = cryostream()
        self.vbpc = get_bpc(
            monitor="cam", actuator="vertical_trans", period=0.25, ponm=False
        )
        self.hbpc = get_bpc(
            monitor="cam", actuator="horizontal_trans", period=0.25, ponm=False
        )

        self.pin = Si_PIN_diode()
        self.xc = xray_camera()
        self.tab = experimental_table()

    def beam_available(self):
        frontend_state = self.frontend.state().lower()
        shutter_state = self.shutter.state().lower()
        c1 = frontend_state not in ["fault"]
        c2_1 = shutter_state not in ["disable"]
        c2_2 = shutter_state in ["open", "close"]
        return c1 and c2_1 and c2_2

    def check_beam(self, sleeptime=1, debug_frequency=100, too_long=900):
        checks = 0
        _start = time.time()
        while not self.beam_available():
            checks += 1
            time.sleep(sleeptime)
            if checks % debug_frequency == 0:
                logging.info(f"{checks} waiting for beam ...")
        
        wait_time = time.time() - _start
        logging.info(f"beam available, we waited {wait_time} seconds")

        if wait_time > too_long:
            current_phase = self.goniometer.get_current_phase()
            if current_phase in ["Centring", "DataCollection", "Transfer"]:
                position = self.goniometer.get_aligned_position()
            os.system("beam_align.py")
            if current_phase in ["Centring", "DataCollection", "Transfer"]:
                self.goniometer.set_goniometer_phase(current_phase, wait=True)
                self.goniometer.set_position(position)

    def abort(self):
        self.cats.abort()
        self.goniometer.abort()
        self.cats.safe()
        self.cats.dry_and_soak()

    def restart_cats(self, sleeptime=3):
        os.system("pycats stop")
        os.system("catsproxy stop")
        os.system("catsproxy start")
        time.sleep(sleeptime)
        os.system("pycats start")
        time.sleep(sleeptime)
        self.cats = cats()
        self.cats.abort()
        self.goniometer.abort()
        
symbols = ["-", "\\", "|", "/"]


def expose(t):
    start = time.time()
    it = 0
    fs.open()
    while time.time() - start < t:
        sys.stdout.write(f"exposing {symbols[ it%len(symbols) ]}\r")
        sys.stdout.flush()
        it += 1
        time.sleep(0.1)
    fs.close()
    sys.stdout.write("Done!" + it * " ")


if __name__ == "__main__":
    g = goniometer()
    d = detector()
    cam = oav_camera(service="oav_camera", mode="redis_bzoom")
    cam_mako = oav_camera(service="mako", mode="redis_local")
    c = cats()
    i = instrument()
    bc = beam_center()
    saf = safety_shutter()
    front = frontend_shutter()
    fs = fast_shutter()
    pin = Si_PIN_diode()
    xc = xray_camera()
    t = transmission()
    f = flux()
    vfm = adaptive_mirror("vfm")
    hfm = adaptive_mirror("hfm")
    fd = fluorescence_detector()
    pe = photon_energy()
    focus = focusing()
    mach = machine_status()
    res = resolution()
    en = photon_energy()
    tab = experimental_table()
    cryo = cryostream()
    vbpc = get_bpc(monitor="cam", actuator="vertical_trans", period=0.25, ponm=False)
    hbpc = get_bpc(monitor="cam", actuator="horizontal_trans", period=0.25, ponm=False)
