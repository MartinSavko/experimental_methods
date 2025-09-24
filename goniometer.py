#!/usr/bin/env python
# -*- coding: utf-8 -*-

ALIGNMENTZ_REFERENCE = 0.0  # -1.298599 -0.9786 #-1.472 # -1.067  # -0.038805 #-0.156882 #0.031317 #-0.2574  # 0.18371
ALIGNMENTX_REFERENCE = 0.0  # 0.010  # +0.0145
ALIGNMENTY_REFERENCE = 0.0 # -0.53

import os
import sys
import time
import gevent
import numpy as np
from math import sin, cos, atan2, radians, sqrt, ceil
import logging
import traceback
import datetime
from scipy.optimize import minimize

try:
    from types import InstanceType
except ImportError:
    InstanceType = object
import inspect

try:
    import lmfit
    from lmfit import fit_report
except ImportError:
    logging.warning(
        "Could not lmfit minimization library, "
        + "refractive model centring will not work."
    )

try:
    if sys.version_info.major == 3:
        import tango
    else:
        import PyTango as tango
except ImportError:
    print("goniometer could not import tango")

from useful_routines import (
    get_aligned_position_from_fit_and_reference,
    get_vertical_and_horizontal_shift_between_two_positions,
    get_shift_from_aligned_position_and_reference_position,
    get_aligned_position_from_reference_position_and_shift,
    get_focus_and_orthogonal_from_position,
    get_points_in_goniometer_frame,
    get_cx_and_cy,
    positions_close,
    copy_position,
)

from md_mockup import md_mockup
from area import area
from motor import tango_motor

# https://stackoverflow.com/questions/34832573/python-decorator-to-display-passed-and-default-kwargs
def md_task(func):
    def perform(*args, **kwargs):
        debug = False
        #if "debug" in kwargs and kwargs["debug"]:
            #debug = True
        if debug:
            print("method name", func.__name__)
            print("md_task args", args)
            print("md_task kwargs", kwargs)

        argspec = inspect.getfullargspec(func)
        if debug:
            print("argspec.args", argspec.args)
            print("argspec.defaults", argspec.defaults)
        positional_count = len(argspec.args) - len(argspec.defaults)
        passed = dict(
            (k, v) for k, v in zip(argspec.args[positional_count:], argspec.defaults)
        )

        # update with kwargs
        if debug:
            print("passed", passed)
        passed.update({k: v for k, v in kwargs.items()})
        if debug:
            print("passed", passed)

        task_id = None
        tried = 0
        args[0].wait()
        while tried < passed["number_of_attempts"]:
            tried += 1
            try:
                task_id = func(*args, **kwargs)
                if debug:
                    print(f"md_task {args} success (try no. {tried}).")
                break
            except:
                # traceback.print_exc()
                if debug:
                    print(f"md_task {args} failed (try no. {tried}).")
                if tried < passed["number_of_attempts"]:
                    n_left = passed["number_of_attempts"] - tried
                    print(f"will try again {n_left} attempts left).")
                args[0].wait()
        if "wait" in passed and passed["wait"] and task_id is not None:
            if type(task_id) is list:
                _task_id = task_id[-1]
            else:
                _task_id = task_id
            args[0].wait_for_task_to_finish(_task_id)
        return task_id

    return perform


class goniometer(object):
    motorsNames = ["AlignmentX", "AlignmentY", "AlignmentZ", "CentringX", "CentringY"]

    motorShortNames = ["PhiX", "PhiY", "PhiZ", "SamX", "SamY"]
    mxcubeShortNames = ["phix", "phiy", "phiz", "sampx", "sampy"]

    shortFull = dict(list(zip(motorShortNames, motorsNames)))
    phiy_direction = +1.0  #  (-1.0 MD2)
    phiz_direction = -1.0  #  (+1.0 MD2)

    motor_name_mapping = [
        ("AlignmentX", "phix"),
        ("AlignmentY", "phiy"),
        ("AlignmentZ", "phiz"),
        ("CentringX", "sampx"),
        ("CentringY", "sampy"),
        ("Omega", "phi"),
        ("Kappa", "kappa"),
        ("Phi", "kappa_phi"),
        ("beam_x", "beam_x"),
        ("beam_y", "beam_y"),
    ]

    # 2019
    # kappa_direction=[0.29636375,  0.29377944, -0.90992064],
    # kappa_position=[-0.30655466, -0.3570731, 0.52893628],
    # phi_direction=[0.03149443, 0.03216924, -0.99469729],
    # phi_position=[-0.01467116, -0.08069945, 0.46818622],

    # 2023-11
    # kappa_direction=[0.2857927293557859, 0.29825779103438044, -0.9106980618759559],
    # kappa_position=[0.05983227148328028, -0.17418159369049926, -0.3931170045291165],
    # phi_direction=[0.05080273994228953, -0.006002697566495966, -1.0134508568340999],
    # phi_position=[0.21993931768594516, -0.03147698225694694, -2.0073121331928205],

    # 2023-12
    # kappa_direction=[0.2699214563788776, 0.2838993348700071, -1.0018455551762098],
    # kappa_position=[0.7765041345864001, 0.5561954243441296, -2.6141365642906167],
    # phi_direction=[0.017981988387908307, 0.04904500982612193, -1.1478709640779396],
    # phi_position=[0.2813575203934254, -0.030698833411830308, -2.584128291492474],
    def __init__(
        self,
        monitor_sleep_time=0.05,
        kappa_direction = np.array([-0.9135,  0.    ,  0.4067]),
        phi_direction = np.array([0., 0., -1]),
        kappa_position = np.array([-0.594,  0.070,  0.436]),
        phi_position = np.array([-0.363, -0.106, -0.174]),
        #kappa_direction=[-0.9164301037869996, 0.00019774420237233793, 0.40248106583171156],
        #kappa_position=[-2.5335236875660665, 0.07097472129826898, 1.2854699800147564],
        #phi_direction=[1.0363616375935, 1.6533913376109066, -0.9508558830149539],
        #phi_position=[-0.09964585649213484, 0.35872830962627056, 0.06522424986716158],

        #kappa_direction=[-0.9160,  0.0000,  0.4020], # MD3Up PX2A
        ##kappa_direction=[ 0.2858,  0.2983, -0.9107], # MD2 PX2A
        ## kappa_direction=[0.287606, 0.287606, -0.913545], #MD3 down P14
        #kappa_position=[-2.5335,  0.071 ,  1.2855], # MD3Up PX2A
        ## kappa_position=[0.3658, 0.4593, -1.4996], # MD3 down P14
        ## kappa_position=[ 0.0598, -0.1742, -0.3931],  # MD2 PX2A
        #phi_direction=[ 1.0364,  1.6534, -0.9509], # MD3Up PX2A
        ##phi_direction=[0, 0, 1],
        ## phi_direction=[ 0.0508, -0.006 , -1.0135],  # MD2 PX2A
        ## phi_direction=[0, 0, -1], # MD3 down P14
        #phi_position=[-0.0996,  0.3587,  0.0652], # MD3Up PX2A
        ## phi_position=[ 0.2199, -0.0315, -2.0073],  # MD2 PX2A
        ## phi_position=[-0.0213733553, 0.03060032895, -1.669944936], # MD3 down
        align_direction=[0, 0, -1],  # MD3 up & MD3 down & MD2
        tango_name="i11-ma-cx1/ex/md3",
    ):
        try:
            # self.md = tango.DeviceProxy("i11-ma-cx1/ex/md2")
            # self.md = tango.DeviceProxy("tango://172.19.10.181:18001/embl/md/1#dbase=no")
            self.md = tango.DeviceProxy(tango_name)
        except:
            from md_mockup import md_mockup

            self.md = md_mockup()

        self.monitor_sleep_time = monitor_sleep_time
        self.observe = None
        self.kappa_axis = self.get_axis(kappa_direction, kappa_position)
        self.phi_axis = self.get_axis(phi_direction, phi_position)
        self.align_direction = align_direction
        # self.redis = redis.StrictRedis()
        self.observation_fields = ["chronos", "Omega"]
        self.observations = []
        self.centringx_direction = -1.0
        self.centringy_direction = +1.0
        self.alignmenty_direction = -1.0
        self.alignmentz_direction = +1.0

        self.md_to_mxcube = dict(
            [(key, value) for key, value in self.motor_name_mapping]
        )
        self.mxcube_to_md = dict(
            [(value, key) for key, value in self.motor_name_mapping]
        )

        self.detector_distance = tango_motor("i11-ma-cx1/dt/dtc_ccd.1-mt_ts")

    def set_collect_phase(self):
        return self.set_data_collection_phase()

    def abort(self):
        return self.md.abort()

    def get_motor_state(self, motor_name):
        if isinstance(self.md, md_mockup):
            return "STANDBY"
        else:
            return self.md.getMotorState(motor_name).name

    def get_status(self):
        try:
            return self.md.read_attribute("Status").value
        except:
            return "Unknown"

    def get_state(self):
        # This solution takes approximately 2.4 ms on average
        try:
            return self.md.read_attribute("State").value.name
        except:
            return "UNKNOWN"

        # This solution takes approximately 15.5 ms on average
        # motors = ['Omega', 'AlignmentX', 'AlignmentY', 'AlignmentZ', 'CentringX', 'CentringY', 'ApertureHorizontal', 'ApertureVertical', 'CapillaryHorizontal', 'CapillaryVertical', 'ScintillatorHorizontal', 'ScintillatorVertical', 'Zoom']
        # state = set([self.get_motor_state(m) for m in motors])
        # if len(state) == 1 and 'STANDBY' in state:
        # return 'STANDBY'
        # else:
        # return 'MOVING'

    def wait(self, device=None, timeout=7):
        green_light = False
        _start = time.time()
        while self.get_status() != "Ready" and (time.time() - _start) < timeout:
            green_light = True
            gevent.sleep(0.1)

        return green_light


    def move_to_position(self, position={}, epsilon=0.0002):
        if position != {}:
            for motor in position:
                while (
                    abs(
                        self.md.read_attribute(
                            "%sPosition" % self.shortFull[motor]
                        ).value
                        - position[motor]
                    )
                    > epsilon
                ):
                    self.wait()
                    gevent.sleep(0.5)
                    self.md.write_attribute(
                        "%sPosition" % self.shortFull[motor], position[motor]
                    )

            self.wait()
        self.wait()
        return

    def get_head_type(self):
        return self.md.headtype

    def has_kappa(self):
        return self.get_head_type() == "MiniKappa"

    def get_x_and_y(self, focus, orthogonal, omega):
        return get_cx_and_cy(focus, orthogonal, omega)
    
    @md_task
    def set_position(
        self,
        position,
        number_of_attempts=3,
        wait=True,
        allclose=True,
        ignored_motors=["Chi", "beam_x", "beam_y", "kappa", "kappa_phi"],
        debug=False,
    ):
        if allclose:
            if positions_close(position, self.get_aligned_position()):
                return 0

        if not self.has_kappa():
            ignored_motors += ["Phi", "Kappa"]

        motor_name_value_list = [
            "%s=%6.4f" % (motor, position[motor])
            for motor in position
            if position[motor] not in [None, np.nan] and (motor not in ignored_motors)
        ]

        command_string = ",".join(motor_name_value_list)
        if debug:
            print(f"position {position}")
            print(f"command_string {command_string}")
        task_id = self.md.startSimultaneousMoveMotors(command_string)
        return task_id

    def save_aperture_and_capillary_beam_positions(self):
        self.md.saveaperturebeamposition()
        self.md.savecapillarybeamposition()

    def get_omega_position(self):
        return self.md.omegaposition
        # return self.get_position()["Omega"]

    def get_kappa_position(self):
        return self.md.kappaposition
        # return self.get_position()["Kappa"]

    def set_kappa_position(self, kappa_position, simple=True):
        if simple:
            self.set_position({"Kappa": kappa_position})
            # self.md.kappaposition = kappa_position
        else:
            current_position = self.get_aligned_position()
            current_kappa = current_position["Kappa"]
            current_phi = current_position["Phi"]

            x = self.get_x()

            shift = self.get_shift(
                current_kappa, current_phi, x, kappa_position, current_phi
            )

            destination = copy_position(current_position)
            destination["AlignmentY"] = shift[0]
            destination["CentringX"] = shift[1]
            destination["CentringY"] = shift[2]
            
            # destination['AlignmentZ'] += (az_destination_offset - az_current_offset)
            destination["Kappa"] = kappa_position

            self.set_position(destination)

    def get_phi_position(self):
        return self.md.phiposition

    def set_phi_position(self, phi_position, simple=True):
        if simple:
            self.md.phiposition = phi_position
        else:
            current_position = self.get_aligned_position()
            current_kappa = current_position["Kappa"]
            current_phi = current_position["Phi"]

            x = self.get_x()

            shift = self.get_shift(
                current_kappa, current_phi, x, current_kappa, phi_position
            )

            destination = copy_position(current_position)
            destination["AlignmentY"] = shift[0]
            destination["CentringX"] = shift[1]
            destination["CentringY"] = shift[2]
            # destination['CentringX'] = shift[0]
            # destination['CentringY'] = shift[1]
            # destination['AlignmentY'] = shift[2]
            # destination['AlignmentZ'] += (az_destination_offset - az_current_offset)
            destination["Phi"] = phi_position

            self.set_position(destination)

    def set_kappa_phi_position(
        self,
        kappa_position,
        phi_position,
    ):
        current_position = self.get_aligned_position()
        current_kappa = current_position["Kappa"]
        current_phi = current_position["Phi"]

        x = self.get_x()

        shift = self.get_shift(
            current_kappa, current_phi, x, kappa_position, phi_position
        )

        destination = copy_position(current_position)
        destination["AlignmentY"] = shift[0]
        destination["CentringX"] = shift[1]
        destination["CentringY"] = shift[2]
        # destination['AlignmentZ'] += (az_destination_offset - az_current_offset)
        destination["Kappa"] = kappa_position
        destination["Phi"] = phi_position

        self.set_position(destination)

    def get_chi_position(self):
        return self.md.chiposition
        # return self.get_position()["Chi"]

    def set_chi_position(self, chi_position):
        self.md.chiposition = chi_position

    def get_x(self, position=None):
        if position is None:
            position = self.get_aligned_position()
        return [
            position[motor]
            for motor in ["AlignmentY", "CentringX", "CentringY"]
        ]

    def get_centringx_position(self):
        return self.get_position()["CentringX"]

    def get_centringy_position(self):
        return self.get_position()["CentringY"]

    def get_alignmentx_position(self):
        return self.get_position()["AlignmentX"]

    def get_alignmenty_position(self):
        return self.get_position()["AlignmentY"]

    def get_alignmentz_position(self):
        return self.get_position()["AlignmentZ"]

    def get_centringtablevertical_position(self):
        return self.get_position()["CentringTableVertical"]

    def get_centringtablefocus_position(self):
        return self.get_position()["CentringTableFocus"]

    def get_zoom_position(self):
        return self.get_position()["Zoom"]

    def get_beam_x_position(self):
        return 0

    def get_beam_y_position(self):
        return 0.0

    def get_centring_x_y_tabledisplacement(self):
        x = self.get_centringx_position()
        y = self.get_centringy_position()
        return x, y, sqrt(x**2 + y**2)

    def get_omega_alpha_and_centringtabledisplacement(self):
        omega = radians(self.get_omega_position())
        x, y, centringtabledisplacement = self.get_centring_x_y_tabledisplacement()
        alpha = atan2(y, -x)
        return omega, alpha, centringtabledisplacement

    def get_centringtablevertical_position_abinitio(self):
        (
            omega,
            alpha,
            centringtabledisplacement,
        ) = self.get_omega_alpha_and_centringtabledisplacement()
        return sin(omega + alpha) * centringtabledisplacement

    def get_centringtablefocus_position_abinitio(self):
        (
            omega,
            alpha,
            centringtabledisplacement,
        ) = self.get_omega_alpha_and_centringtabledisplacement()
        return cos(omega + alpha) * centringtabledisplacement

    def get_centringtable_vertical_position_from_hypothetical_centringx_centringy_and_omega(
        self, x, y, omega
    ):
        d = sqrt(x**2 + y**2)
        alpha = atan2(y, -x)
        omega = radians(omega)
        return sin(omega + alpha) * d

    def get_centringtable_focus_position_from_hypothetical_centringx_centringy_and_omega(
        self, x, y, omega
    ):
        d = sqrt(x**2 + y**2)
        alpha = atan2(y, -x)
        omega = radians(omega)
        return cos(omega + alpha) * d

    def get_focus_and_orthogonal_from_position(
        self,
        position=None,
        centringy_direction=-1,
    ):
        if position is None:
            position = self.get_aligned_position()
        focus, vertical = get_focus_and_orthogonal_from_position(
            position,
            centringy_direction=centringy_direction,
        )
        return focus, vertical
    

    def get_aligned_position_from_fit_and_reference(
        self,
        fit_vertical,
        fit_horizontal,
        reference,
        orientation="vertical",
    ):
        aligned_position = get_aligned_position_from_fit_and_reference(
            fit_vertical,
            fit_horizontal,
            reference,
            orientation=orientation
        )
      
        return aligned_position
    
    def get_aligned_position_from_reference_position_and_x_and_y(
        self, 
        reference_position, 
        x, 
        y, 
        AlignmentZ_reference=ALIGNMENTZ_REFERENCE,
    ):
        # MD2
        # horizontal_shift = x - reference_position["AlignmentY"]
        # vertical_shift = y - reference_position["AlignmentZ"]

        # MD3Up
        horizontal_shift = x - reference_position["AlignmentZ"]
        vertical_shift = y - reference_position["AlignmentY"]

        return self.get_aligned_position_from_reference_position_and_shift(
            reference_position,
            horizontal_shift,
            vertical_shift,
            AlignmentZ_reference=AlignmentZ_reference,
        )

    def get_position(self):
        return dict(
            [(m.split("=")[0], float(m.split("=")[1])) for m in self.md.motorpositions]
        )

    def get_aligned_position(
        self,
        motor_names=[
            "AlignmentX",
            "AlignmentY",
            "AlignmentZ",
            "CentringX",
            "CentringY",
            "Kappa",
            "Phi",
            # "Chi",
            "Omega",
        ],
    ):
        return dict(
            [
                (m.split("=")[0], float(m.split("=")[1]))
                for m in self.md.motorpositions
                if m.split("=")[0] in motor_names and m.split("=")[1] != "NaN"
            ]
        )

    def get_state_vector(
        self,
        motor_names=[
            "Omega",
            "Kappa",
            "Phi",
            "CentringX",
            "CentringY",
            "AlignmentX",
            "AlignmentY",
            "AlignmentZ",
            "ScintillatorVertical",
            "Zoom",
        ],
    ):
        motor_positions_dictionary = self.get_motor_positions_dictionary()
        return [motor_positions_dictionary[motor_name] for motor_name in motor_names]
        # return [m.split('=')[1] for m in motor_positions if m.split('=')[0] in motor_names]

    def get_motor_positions_dictionary(
        self,
        motor_names=[
            "Omega",
            "Kappa",
            "Phi",
            "Chi",
            "CentringX",
            "CentringY",
            "AlignmentX",
            "AlignmentY",
            "AlignmentZ",
            "ApertureVertical",
            "ApertureHorizontal",
            "CapillaryVertical",
            "CapillaryHorizontal",
            "ScintillatorVertical",
            "ScintillatorHorizontal",
            "BeamstopX",
            "BeamstopY",
            "BeamstopZ",
            "Zoom",
            "PlateTranslation",
            "CentringTableVertical",
            "CentringTableFocus",
            "BeamstopDistance",
        ],
        logfile="/nfs/data2/log/md_motor_positions_problem.log",
    ):
        try:
            motor_positions_dictionary = dict(
                [item.split("=") for item in self.md.motorpositions]
            )
        except:
            message = "failure to read md.motorpositions attribute"
            logging.info(message)
            os.system(
                'echo "{now:s} {message:s}" >> {logfile:s}'.format(
                    logfile=logfile, message=message, now=str(datetime.datetime.now())
                )
            )
            motor_positions_dictionary = dict(
                (motor_name, np.nan) for motor_name in motor_names
            )
        return motor_positions_dictionary

    def sample_is_loaded(
        self,
        sample_size=7,
        sleeptime=0.01,
        timeout=3,
        logfile="/nfs/data2/log/md_sample_detection_problem.log",
    ):
        _start = time.time()
        sample_is_coherent = False
        all_answers = []
        while not sample_is_coherent and (time.time() - _start < timeout):
            is_loaded_sample = []
            for k in range(sample_size):
                try:
                    is_loaded_sample.append(int(self.md.SampleIsLoaded))
                    gevent.sleep(np.random.random() * sleeptime)
                except:
                    logging.debug(traceback.format_exc())
                    print(
                        "something went wrong in checking the sample presence, You may want to check. Will try to move on ..."
                    )

            median = np.median(is_loaded_sample)
            mean = np.mean(is_loaded_sample)
            sample_is_coherent = median == mean
            if not sample_is_coherent:
                message = (
                    "gonio sample detection is not coherent, you may want to check ..."
                )
                logging.info(message)
                print(message)
                os.system(
                    'echo "{now:s} {is_loaded_sample:s}" >> {logfile:s}'.format(
                        logfile=logfile,
                        is_loaded_sample=str(is_loaded_sample),
                        now=str(datetime.datetime.now()),
                    )
                )
                all_answers += is_loaded_sample
        if not sample_is_coherent:
            median = np.median(all_answers)
        return bool(median)

    def set_beamstopposition(self, position, wait=True, sleeptime=0.5, timeout=30):
        _start = time.time()
        assert position in ["PARK", "BEAM", "OFF", "TRANSFER"]
        self.md.beamstopposition = position 
        if wait:
            while self.md.beamstopposition != position and time.time() - _start < timeout:
                time.sleep(sleeptime)
        
        
    def insert_backlight(
        self,
        sleeptime=0.1,
        timeout=7,
        gain=0.0,
        exposure=50000.0,
        beamstop_safe_distance=42.11,
        detector_safe_distance=180.0,
        beamstop_z_threshold=-30,
    ):
        _start = time.time()
        self.wait()
        while not self.backlight_is_on() and (time.time() - _start) < timeout:
            try:
                # if self.md.beamstopposition == "BEAM":
                self.set_beamstopposition("PARK", wait=True)
                if self.md.beamstopzposition > beamstop_z_threshold:
                    if self.detector_distance.get_position() < detector_safe_distance:
                        self.detector_distance.set_position(
                            detector_safe_distance, wait=True
                        )
                    if self.md.beamstopxposition < beamstop_safe_distance:
                        self.set_position(
                            {"BeamstopX": beamstop_safe_distance}, wait=True
                        )
            except:
                print("failing to insert backlight ...")
                traceback.print_exc()
                gevent.sleep(sleeptime)

            try:
                self.md.backlightison = True
                self.md.cameragain = gain
                self.md.cameraexposure = exposure
            except:
                print("failing to insert backlight ...")
                traceback.print_exc()
                gevent.sleep(sleeptime)

    def insert_frontlight(self, sleeptime=0.1, timeout=7):
        _start = time.time()
        print("inserting frontlight")
        self.wait()
        while not self.frontlight_is_on() and (time.time() - _start) < timeout:
            try:
                self.md.frontlightison = True
            except:
                gevent.sleep(sleeptime)
        print("success? %s" % self.frontlight_is_on())

    def extract_backlight(self):
        self.remove_backlight()

    def remove_backlight(self, sleeptime=0.1, timeout=7, gain=40.0, exposure=50000.0):
        _start = time.time()
        while self.backlight_is_on() and (time.time() - _start) < timeout:
            try:
                self.md.backlightison = False
                self.md.cameragain = gain
                self.md.cameraexposure = exposure
            except:
                gevent.sleep(sleeptime)

    def extract_frontlight(self, sleeptime=0.1, timeout=7):
        _start = time.time()
        print("extracting frontlight")
        while self.frontlight_is_on() and (time.time() - _start) < timeout:
            try:
                self.md.frontlightison = False
            except:
                gevent.sleep(sleeptime)
        print("success? %s" % (not self.frontlight_is_on()))

    def backlight_is_on(self):
        return self.md.backlightison

    def frontlight_is_on(self):
        return self.md.frontlightison

    def get_backlightlevel(self):
        return self.md.backlightlevel

    def set_backlightlevel(self, level=10, number_of_attempts=7, sleeptime=0.5):
        n = 0
        while self.md.backlightlevel != level and n <= number_of_attempts:
            n += 1
            try:
                self.md.backlightlevel = level
            except:
                gevent.sleep(sleeptime)

    def get_frontlightlevel(self):
        return self.md.frontlightlevel

    def set_frontlightlevel(self, level=55, number_of_attempts=7, sleeptime=0.5):
        n = 0
        while self.md.frontlightlevel != level and n <= number_of_attempts:
            n += 1
            try:
                self.md.frontlightlevel = level
                success = True
            except:
                gevent.sleep(sleeptime)

    def insert_fluorescence_detector(self):
        self.md.fluodetectorisback = False

    def extract_fluorescence_detector(self):
        self.md.fluodetectorisback = True

    def insert_cryostream(self):
        self.md.cryoisback = False

    def extract_cryostream(self):
        self.md.cryoisback = True

    def park_cryostream(self):
        self.md.cryoisout = True

    def is_task_running(self, task_id):
        reply = None
        if task_id > 0:
            try:
                reply = self.md.istaskrunning(task_id)
            except:
                reply = 2
                print("Could not connect to md device")
                traceback.print_exc()
        return reply

    def get_last_task_info(self):
        return self.md.lasttaskinfo

    def get_time_from_string(self, timestring, format="%Y-%m-%d %H:%M:%S.%f"):
        micros = float(timestring[timestring.find(".") :])
        return time.mktime(time.strptime(timestring, format)) + micros

    def get_last_task_start(self):
        lasttaskinfo = self.md.lasttaskinfo
        start = lasttaskinfo[2]
        return self.get_time_from_string(start)

    def get_last_task_end(self):
        lasttaskinfo = self.md.lasttaskinfo
        end = lasttaskinfo[3]
        if end == "null":
            return None
        return self.get_time_from_string(end)

    def get_last_task_duration(self):
        start = self.get_last_task_start()
        end = self.get_last_task_end()
        if end == None:
            return time.time() - start
        return end - start

    def get_task_start(self, task_id):
        task_info = self.get_task_info(task_id)
        task_start = task_info[2]
        return self.get_time_from_string(task_start)

    def get_task_end(self, task_id):
        task_info = self.get_task_info(task_id)
        task_end = task_info[3]
        return self.get_time_from_string(task_end)

    def get_task_duration(self, task_id):
        task_start = self.get_task_start(task_id)
        task_end = self.get_task_end(task_id)
        task_duration = task_end - task_start
        return task_duration

    def get_task_info(self, task_id):
        return self.md.gettaskinfo(task_id)

    def set_detector_gate_pulse_enabled(self, value=True):
        self.md.DetectorGatePulseEnabled = value

    # @md_task
    def enable_fast_shutter(self):
        self.wait()
        self.md.FastShutterIsEnabled = True

    # @md_task
    def disable_fast_shutter(self):
        self.wait()
        self.md.FastShutterIsEnabled = False

    # @md_task
    def set_goniometer_phase(
        self, phase, wait=False, number_of_attempts=7, sleeptime=0.5
    ):
        self.task_id = self.md.startsetphase(phase)
        ##return self.task_id
        self.wait_for_task_to_finish(self.task_id)
        # self.md.currentphase = phase
        return self.task_id

    def set_data_collection_phase(self, wait=False):
        self.save_position()
        self.set_goniometer_phase("DataCollection", wait=wait)

    def set_transfer_phase(
        self,
        transfer_position={
            # "AlignmentZ": 0.1017,
            # "AlignmentY": -1.35,
            # "AlignmentX": -0.0157,
            # "CentringX": 0.431,
            # "CentringY": 0.210,
            "ApertureVertical": 83,
            # "CapillaryVertical": 0.0,
            # "Zoom": 33524.0,
            # "Omega": 0,
        },
        phase=False,
        wait=False,
    ):  # , 'Kappa': 0, 'Phi': 0
        # transfer_position={'AlignmentX': -0.42292751002956075, 'AlignmentY': -1.5267995679700732,  'AlignmentZ': -0.049934926625844867, 'ApertureVertical': 82.99996634818412, 'CapillaryHorizontal': -0.7518915227941564, 'CapillaryVertical': -0.00012483891752579357, 'CentringX': 1.644736842105263e-05, 'CentringY': 1.644736842105263e-05, 'Kappa': 0.0015625125024403275, 'Phi': 0.004218750006591797, 'Zoom': 34448.0}
        # if phase:
        self.set_goniometer_phase("Transfer", wait=wait)
        # self.set_position(transfer_position, wait=wait)

    def set_beam_location_phase(self, wait=False):
        if self.get_current_phase() != "BeamLocation":
            self.save_position()
        # self.extract_cryostream()
        self.set_goniometer_phase("BeamLocation", wait=wait)

    def set_centring_phase(self, wait=False):
        self.set_goniometer_phase("Centring", wait=wait)

    def get_current_phase(self):
        return self.md.currentphase

    @md_task
    def save_position(self, number_of_attempts=15, sleeptime=0.2):
        self.md.savecentringpositions()

    def wait_for_task_to_finish(
        self, task_id, collect_auxiliary_images=False, sleeptime=0.1
    ):
        k = 0
        self.auxiliary_images = []
        while self.is_task_running(task_id):
            if k == 0:
                logging.debug("waiting for task %d to finish" % task_id)
            gevent.sleep(sleeptime)
            k += 1

    def set_omega_relative_position(self, step):
        self.md.omegaposition += step
        # current_position = self.get_omega_position()
        # return self.set_omega_position(current_position + step)

    def set_omega_position(self, omega_position):
        self.md.omegaposition = omega_position
        # return self.set_position({"Omega": omega_position})

    def set_attribute(self, attribute, value, nattempts=3, sleeptime=0.05):
        success = False
        tried = 0
        while not success and tried <= nattempts:
            try:
                self.md.write_attribute(attribute, value)
                success = True
            except:
                tried += 1
                time.sleep(sleeptime)
        return success

    def set_zoom(self, zoom, wait=False):
        self.set_attribute("coaxialcamerazoomvalue", zoom)
        # self.md.coaxialcamerazoomvalue = zoom
        if wait:
            self.wait()

    def get_orientation(self):
        return self.get_omega_position()

    def set_orientation(self, orientation):
        self.set_omega_position(orientation)

    def check_position(self, candidate_position):
        if isinstance(candidate_position, str):
            candidate_position = eval(candidate_position)

        if candidate_position is not None and not isinstance(candidate_position, dict):
            candidate_position = candidate_position.strip("}{")
            positions = candidate_position.split(",")
            keyvalues = [item.strip().split(":") for item in positions]
            keyvalues = [(item[0], float(item[1])) for item in keyvalues]
            candidate_position = dict(keyvalues)

        if isinstance(candidate_position, dict):
            current_position = self.get_aligned_position()
            for key in candidate_position:
                if candidate_position[key] is None and key in current_position:
                    candidate_position[key] = current_position[key]
            return candidate_position
        else:
            return self.get_aligned_position()

    def get_point(self):
        return self.get_position()

    def get_observation_fields(self):
        return self.observation_fields

    def monitor(self, start_time, motor_names=["Omega"]):
        self.observations = []
        self.observation_fields = ["chronos"] + motor_names

        while self.observe == True:
            chronos = time.time() - start_time
            position = self.get_position()
            point = [chronos] + [position[motor_name] for motor_name in motor_names]
            self.observations.append(point)
            gevent.sleep(self.monitor_sleep_time)

    def get_observations(self):
        return self.observations

    def get_observation_fields(self):
        return self.observation_fields

    def get_points(self):
        return np.array(self.observations)[:, 1]

    def get_chronos(self):
        return np.array(self.observations)[:, 0]

    def get_rotation_matrix(self, axis, angle):
        rads = np.radians(angle)
        cosa = np.cos(rads)
        sina = np.sin(rads)
        I = np.diag([1] * 3)
        rotation_matrix = I * cosa + axis["mT"] * (1 - cosa) + axis["mC"] * sina
        return rotation_matrix

    def get_axis(self, direction, position):
        axis = {}
        d = np.array(direction)
        p = np.array(position)
        axis["direction"] = d
        axis["position"] = p
        axis["mT"] = self.get_mT(direction)
        axis["mC"] = self.get_mC(direction)
        return axis

    def get_mC(self, direction):
        mC = np.array(
            [
                [0.0, -direction[2], direction[1]],
                [direction[2], 0.0, -direction[0]],
                [-direction[1], direction[0], 0.0],
            ]
        )

        return mC

    def get_mT(self, direction):
        mT = np.outer(direction, direction)

        return mT

    def get_shift(self, kappa1, phi1, x, kappa2, phi2):
        tk = self.kappa_axis["position"]
        tp = self.phi_axis["position"]

        Rk2 = self.get_rotation_matrix(self.kappa_axis, kappa2)
        Rk1 = self.get_rotation_matrix(self.kappa_axis, -kappa1)
        Rp = self.get_rotation_matrix(self.phi_axis, phi2 - phi1)

        #a = tk - np.dot((tk - x), Rk1.T)
        #b = tp - np.dot((tp - a), Rp.T)
        
        a = tk - np.dot(Rk1, (tk - x))
        b = tp - np.dot(Rp, (tp - a))
        
        #shift = tk - np.dot((tk - b), Rk2.T)
        shift = tk - np.dot(Rk2, (tk - b))
        
        return shift

    def get_align_vector(self, t1, t2, kappa, phi):
        t1 = np.array(t1)
        t2 = np.array(t2)
        x = t1 - t2
        Rk = self.get_rotation_matrix(self.kappa_axis, -kappa)
        Rp = self.get_rotation_matrix(self.phi_axis, -phi)
        x = np.dot(Rp, np.dot(Rk, x)) / np.linalg.norm(x)
        c = np.dot(self.phi_axis["direction"], x)
        if c < 0.0:
            c = -c
            x = -x
        cos2a = pow(np.dot(self.kappa_axis["direction"], self.align_direction), 2)

        d = (c - cos2a) / (1 - cos2a)

        if abs(d) > 1.0:
            new_kappa = 180.0
        else:
            new_kappa = np.degrees(np.arccos(d))

        Rk = self.get_rotation_matrix(self.kappa_axis, new_kappa)
        pp = np.dot(Rk, self.phi_axis["direction"])
        xp = np.dot(Rk, x)
        d1 = self.align_direction - c * pp
        d2 = xp - c * pp

        new_phi = np.degrees(
            np.arccos(np.dot(d1, d2) / np.linalg.norm(d1) / np.linalg.norm(d2))
        )

        newaxis = {}
        newaxis["mT"] = self.get_mT(pp)
        newaxis["mC"] = self.get_mC(pp)

        Rp = self.get_rotation_matrix(newaxis, new_phi)
        d = np.abs(np.dot(self.align_direction, np.dot(Rp, xp)))
        check = np.abs(np.dot(self.align_direction, np.dot(xp, Rp)))

        if check > d:
            new_phi = -new_phi

        shift = self.get_shift(kappa, phi, 0.5 * (t1 + t2), new_kappa, new_phi)

        align_vector = new_kappa, new_phi, shift

        return align_vector

    def get_points_in_goniometer_frame(
        self,
        points,
        calibration,
        origin,
        center=np.array([160, 256, 256]),
        directions=np.array([-1, 1, 1]),
        order=[1, 2, 0],
    ):
        mm = get_points_in_goniometer_frame(
            points,
            calibration,
            origin,
            center=center,
            directions=directions,
            order=order,
        )
        return mm

    def get_point_coordinates_from_position(self, position):
        horizontal_shift = position_initial["AlignmentZ"] - position_final["AlignmentZ"]

    def get_aligned_position_from_reference_position_and_shift(
        self,
        reference_position,
        orthogonal_shift,
        along_shift,
        omega=None,
        AlignmentZ_reference=None, #ALIGNMENTZ_REFERENCE,  # 0.0100,
        epsilon=1e-3,
        debug=False,
    ):
        if AlignmentZ_reference is None:
            AlignmentZ_reference = ALIGNMENTZ_REFERENCE
        
        aligned_position = get_aligned_position_from_reference_position_and_shift(
            reference_position,
            orthogonal_shift,
            along_shift,
            omega=omega,
            AlignmentZ_reference=AlignmentZ_reference, #ALIGNMENTZ_REFERENCE,  # 0.0100,
            epsilon=epsilon,
            debug=debug,
        )
        return aligned_position
            
    def get_shift_from_aligned_position_and_reference_position(
        self,
        aligned_position,
        reference_position=None,
        omega=None,
        AlignmentZ_reference=None,
        epsilon=1.0e-3,
    ):
        if reference_position is None:
            reference_position = self.get_aligned_position()
            
        if AlignmentZ_reference is None:
            AlignmentZ_reference = ALIGNMENTZ_REFERENCE
            
        shift = get_shift_from_aligned_position_and_reference_position(
            aligned_position,
            reference_position=reference_position,
            omega=omega,
            AlignmentZ_reference=AlignmentZ_reference,
            epsilon=epsilon,
        )
        return shift

    def get_vertical_and_horizontal_shift_between_two_positions(
        self, aligned_position, reference_position=None, epsilon=1.0e-3
    ):
        if reference_position is None:
            reference_position = self.get_aligned_position()

        shift = get_vertical_and_horizontal_shift_between_two_positions(
            aligned_position,
            reference_position,
        )
        
        return shift
    
        

    def translate_from_mxcube_to_md(self, position):
        translated_position = {}

        for key in position:
            if isinstance(key, str):
                try:
                    translated_position[self.mxcube_to_md[key]] = position[key]
                except:
                    pass
                    # self.log.exception(traceback.format_exc())
            else:
                translated_position[key.actuator_name] = position[key]
        return translated_position

    def translate_from_md_to_mxcube(self, position):
        translated_position = {}

        for key in position:
            translated_position[self.md_to_mxcube[key]] = position[key]

        return translated_position




    """
            | |                    | | | |
            | |     ___ _ __   __ _| |_| |__  ___
            | |    / _ \ '_ \ / _` | __| '_ \/ __|
            | |___|  __/ | | | (_| | |_| | | \__ \
            |______\___|_| |_|\__, |\__|_| |_|___/
                        __/ |
                        |___/
    """

    def set_scan_start_angle(self, scan_start_angle):
        self.md.scanstartangle = scan_start_angle

    def get_scan_start_angle(self):
        return self.md.scanstartangle

    def set_scan_range(self, scan_range):
        self.md.scanrange = scan_range

    def get_scan_range(self):
        return self.md.scanrange

    def set_scan_exposure_time(self, scan_exposure_time):
        self.md.scanexposuretime = scan_exposure_time

    def get_scan_exposure_time(self):
        return self.md.scanexposuretime

    def set_scan_number_of_frames(self, scan_number_of_frames):
        try:
            if self.get_scan_number_of_frames() != scan_number_of_frames:
                self.wait()
                self.md.scannumberofframes = scan_number_of_frames
        except:
            logging.info(traceback.format_exc())

    def get_scan_number_of_frames(self):
        return self.md.scannumberofframes

    def set_scan_number_of_passes(self, scan_number_of_passes):
        try:
            if self.get_scan_number_of_passes() != scan_number_of_passes:
                self.md.scannumberofpasses = scan_number_of_passes
        except:
            logging.info(traceback.format_exc())

    def get_scan_number_of_passes(self):
        return self.md.scannumberofpasses
    def start_scan(self, number_of_attempts=3, wait=False):
        return self.omega_scan(mode="nonex")

    def scan(
        self,
        position=None,
        scan_start_angle=0.0,
        scan_range=400.0,
        scan_exposure_time=20.0,
    ):
        if type(position) is dict:
            self.set_position(position)
        if type(position) is list:
            assert len(position) == 2
            start, stop = position
            task_id = self.helical_scan(
                start,
                stop,
                scan_start_angle=scan_start_angle,
                scan_range=scan_range,
                scan_exposure_time=scan_exposure_time,
            )
        else:
            task_id = self.omega_scan(
                scan_start_angle=scan_start_angle,
                scan_range=scan_range,
                scan_exposure_time=scan_exposure_time,
            )
        return task_id

    def inverse_scan(
        self,
        position=None,
        wedge_range=15.0,
        scan_start_angle=0.0,
        scan_range=180.0,
        scan_exposure_time=18.0,
        inverse_direction=True,
    ):
        d_starts = np.arange(scan_start_angle, scan_range, wedge_range)
        first_stop = scan_start_angle + wedge_range
        assert len(d_starts) > 1
        d_stops = np.linspace(scan_start_angle + wedge_range, scan_range, len(d_starts))
        wedges = d_stops - d_starts
        exposure_times = scan_exposure_time * wedges / scan_range

        i_starts = d_starts + 180.0

        print(
            f"executing {scan_range} degreees inverse scan ({2*scan_range} degrees in total), total exposure time {2*scan_exposure_time} seconds"
        )
        print(f"{2*len(wedges)} wedges of {wedges[0]} degrees each")
        print(f"start angle {scan_start_angle}")
        _start = time.time()
        task_id = []
        l = 0
        if type(position) is dict:
            self.set_position(position)
            # position = None
        for k, (d, i, w, e) in enumerate(
            zip(d_starts, i_starts, wedges, exposure_times)
        ):
            _s = time.time()
            print(
                f"wedge {k} (of  {len(wedges)}) direct and inverse, {e:.2f} seconds each"
            )
            for a in (d, i):
                l += 1
                if inverse_direction and type(position) is list and l > 1:
                    position = position[::-1]
                _task_id = self.scan(
                    position=position,
                    scan_start_angle=a,
                    scan_range=w,
                    scan_exposure_time=e,
                )
                task_id.append(_task_id)
            _e = time.time()
            print(f"took {_e-_s:.2f} seconds")
        _end = time.time()
        duration = _end - _start
        overhead = duration - 2 * scan_exposure_time
        print(
            f"inverse scan took {duration:.2f} seconds, (overhead {overhead:.2f} seconds)"
        )

        return task_id

    @md_task
    def omega_scan(
        self,
        scan_start_angle=0.0,
        scan_range=1.8,
        scan_exposure_time=0.005,
        number_of_frames=1,
        number_of_passes=1,
        number_of_attempts=7,
        wait=True,
        mode="ex",
    ):
        scan_start_angle = "%6.4f" % scan_start_angle
        scan_range = "%6.4f" % scan_range
        scan_exposure_time = "%6.4f" % scan_exposure_time
        number_of_frames = "%d" % number_of_frames
        number_of_passes = "%d" % number_of_passes
        parameters = [
            number_of_frames,
            scan_start_angle,
            scan_range,
            scan_exposure_time,
            number_of_passes,
        ]

        if mode == "ex":
            self.task_id = self.md.startscanex(parameters)
        else:
            self.set_scan_number_of_frames(number_of_frames)
            self.set_scan_start_angle(scan_start_angle)
            self.set_scan_range(scan_range)
            self.set_scan_exposure_time(scan_exposure_time)
            self.set_scan_number_of_passes(number_of_passes)
            self.task_id = self.md.startscan()

        return self.task_id

    @md_task
    def helical_scan(
        self,
        start,
        stop,
        scan_start_angle,
        scan_range,
        scan_exposure_time,
        number_of_frames=1,
        number_of_passes=1,
        number_of_attempts=7,
        wait=True,
        sleeptime=0.5,
    ):
        self.set_scan_number_of_frames(number_of_frames)
        self.set_scan_number_of_passes(number_of_passes)

        scan_start_angle = "%6.4f" % (scan_start_angle % 360.0,)
        scan_range = "%6.4f" % scan_range
        scan_exposure_time = "%6.4f" % scan_exposure_time
        start_z = "%6.4f" % start["AlignmentZ"]
        start_y = "%6.4f" % start["AlignmentY"]
        stop_z = "%6.4f" % stop["AlignmentZ"]
        stop_y = "%6.4f" % stop["AlignmentY"]
        start_cx = "%6.4f" % start["CentringX"]
        start_cy = "%6.4f" % start["CentringY"]
        stop_cx = "%6.4f" % stop["CentringX"]
        stop_cy = "%6.4f" % stop["CentringY"]
        parameters = [
            scan_start_angle,
            scan_range,
            scan_exposure_time,
            start_y,
            start_z,
            start_cx,
            start_cy,
            stop_y,
            stop_z,
            stop_cx,
            stop_cy,
        ]

        self.task_id = self.md.startScan4DEx(parameters)

        return self.task_id

    # @md_task
    def raster_scan(
        self,
        vertical_range,
        horizontal_range,
        number_of_rows=None,
        number_of_columns=None,
        position=None,
        scan_start_angle=None,
        scan_exposure_time=None,
        scan_range=0.0,
        inverse_direction=True,  # boolean: true to enable passes in the reverse direction.
        use_centring_table=True,  # boolean: true to use the centring table to do the pitch movements.
        fast_scan=True,  # boolean: true to use the fast raster scan if available (power PMAC).
        vertical_step_size=0.0025,
        horizontal_step_size=0.025,
        frame_time=0.005,
        direction=0,
        number_of_passes=1,
        dark_time_between_passes=0.0,
        number_of_frames=1,
        maximum_speed=(30, 1),
        number_of_attempts=7,
        mode="ex",
        wait=True,
    ):
        if position is None:
            position = self.get_aligned_position()
        if scan_start_angle is None:
            scan_start_angle = position["Omega"]
        else:
            position["Omega"] = scan_start_angle
        if number_of_rows is None:
            number_of_rows = ceil(vertical_range / vertical_step_size)
        if number_of_columns is None:
            number_of_columns = ceil(horizontal_range / horizontal_step_size)
        print(f"requested grid shape is ({number_of_rows}, {number_of_columns})")
        if scan_exposure_time is None:
            scan_exposure_time = number_of_rows * frame_time
            requested_scan_speed = vertical_range / scan_exposure_time
            print(f"requested scan speed is {requested_scan_speed} mm/s")
            if requested_scan_speed > maximum_speed[direction]:
                print(
                    f"reqested scan speed is larger then the maximum, reducing the speed to the maximum {maximum_speed[direction]}"
                )
                scan_exposure_time = vertical_range / maximum_speed[direction]
                print(
                    f"increasing the scan exposure time to {scan_exposure_time} to allow the movement"
                )

        if direction == 0:
            frame_scan_range = scan_range / number_of_columns
        else:
            frame_scan_range = scan_range / number_of_rows

        a = area(
            range_y=vertical_range,
            range_x=horizontal_range,
            rows=number_of_rows,
            columns=number_of_columns,
            center_y=0.0,
            center_x=0.0,
        )

        grid, shifts = a.get_grid_and_shifts()

        if inverse_direction:
            grid = a.get_raster(grid, direction=direction)

        grid = grid[:, ::-1]
        grid = grid[::-1, :]

        if mode == "ex":
            # %time task_id = g.raster_scan(0.3, 0.25, mode="ex", direction=0)
            # requested scan speed is 0.5 mm/s
            # ['0.0000', '0.3000', '0.2500', '149.9997', '-1.4383', '0.1837', '0.5774', '-0.3564', '10', '120', '0.6000', '1', '1', '1']
            # CPU times: user 48.3 ms, sys: 0 ns, total: 48.3 ms
            # Wall time: 12 s

            start_position = (
                self.get_aligned_position_from_reference_position_and_shift(
                    position,
                    horizontal_range / 2.0,
                    -vertical_range / 2.0,
                )
            )
            parameters = [
                f"{scan_range:6.4f}",
                f"{vertical_range:6.4f}",
                f"{horizontal_range:6.4f}",
                f"{scan_start_angle:6.4f}",
                f"{start_position['AlignmentY']:6.4f}",
                f"{start_position['AlignmentZ']:6.4f}",
                f"{start_position['CentringX']:6.4f}",
                f"{start_position['CentringY']:6.4f}",
                f"{number_of_columns:d}",
                f"{number_of_rows:d}",
                f"{scan_exposure_time:6.4f}",
                f"{inverse_direction:d}",
                f"{use_centring_table:d}",
                f"{fast_scan:d}",
            ]
            print(f"{parameters}")
            # ['0.0000', '0.6000', '0.4000', '150.0001', '-1.6205', '0.1837', '0.6203', '-0.4008', '16', '240', '1.2000', '1', '1', '1']
            task_id = self.md.startrasterscanex(parameters)
            if wait:
                self.wait_for_task_to_finish(task_id)

        elif mode == "nonex":
            # %time task_id = g.md.startrasterscan(["0.6", "0.39", "240", "39", "1", "1", "1"])
            # CPU times: user 1.25 ms, sys: 419 µs, total: 1.67 ms
            # Wall time: 61.4 ms
            self.md.scanstartangle = scan_start_angle
            self.md.scanexposuretime = scan_exposure_time
            self.md.scanrange = scan_range
            if not positions_close(position, self.get_aligned_position()):
                self.set_position(position)
            parameters = [
                f"{vertical_range:6.4f}",
                f"{horizontal_range:6.4f}",
                f"{number_of_columns:d}",
                f"{number_of_rows:d}",
                f"{inverse_direction:d}",
                f"{use_centring_table:d}",
                f"{fast_scan:d}",
            ]
            print(f"{parameters}")

            # ['0.6000', '0.4000', '16', '240', '1', '1', '1']
            task_id = self.md.startrasterscan(parameters)
            if wait:
                self.wait_for_task_to_finish(task_id)
        else:
            # elif mode in ["universal", "helical", "horizontal", "arbitrary", "step"]:
            if mode == "helical":
                starts, stops = a.get_jumps(grid, direction=direction)

                start_positions = [
                    self.get_aligned_position_from_reference_position_and_shift(
                        position, shifts[i][1], shifts[i][0]
                    )
                    for i in starts
                ]
                stop_positions = [
                    self.get_aligned_position_from_reference_position_and_shift(
                        position, shifts[i][1], shifts[i][0]
                    )
                    for i in stops
                ]

                helical_lines = []
                for start, stop in zip(start_positions, stop_positions):
                    helical_line = [
                        start,
                        stop,
                    ]
                    helical_line.extend(
                        [scan_start_angle, scan_range, scan_exposure_time]
                    )
                    helical_lines.append(helical_line)

                task_id = []
                for helical_line in helical_lines:
                    _task_id = self.helical_scan(*helical_line)
                    task_id.append(_task_id)

            elif mode == "step":
                grid = grid.T
                positions = [
                    self.get_aligned_position_from_reference_position_and_shift(
                        position, shifts[i][1], shifts[i][0]
                    )
                    for i in grid.ravel()
                ]
                task_id = []
                for position in positions:
                    self.set_position(position)
                    _task_id = self.omega_scan(
                        scan_start_angle=scan_start_angle,
                        scan_range=frame_scan_range,
                        scan_exposure_time=frame_time,
                        number_of_passes=number_of_passes,
                        number_of_frames=number_of_frames,
                        wait=wait,
                    )
                    task_id.append(_task_id)
                    if dark_time_between_passes:
                        time.sleep(dark_time_between_passes)

            self.set_position(position)

        return task_id, grid, shifts

    def neha(
        self,
        vertical_range,
        horizontal_range,
        number_of_rows=None,
        number_of_columns=None,
        position=None,
        scan_start_angle=0,
        frame_time=0.005,
        scan_range=180.0,
        wedge_range=45.0,
        overlap=0.0,
        vertical_step_size=0.025,
        horizontal_step_size=0.025,
    ):
        if position is None:
            position = self.get_position()

        starts = np.arange(scan_start_angle, scan_range, wedge_range + overlap)
        first_stop = scan_start_angle + wedge_range
        if len(starts) > 1:
            stops = np.linspace(
                scan_start_angle + wedge_range + overlap, scan_range, len(starts)
            )
        else:
            stops = np.array([scan_range])
        wedges = stops - starts
        wedges -= overlap
        task_id = []
        for angle, wedge in zip(starts, wedges):
            _task_id = self.raster_scan(
                vertical_range,
                horizontal_range,
                number_of_rows=number_of_rows,
                number_of_columns=number_of_columns,
                position=position,
                scan_start_angle=angle,
                scan_range=wedge,
                vertical_step_size=vertical_step_size,
                horizontal_step_size=horizontal_step_size,
            )
            task_id.append(_task_id)

        return task_id

    def vertical_helical_scan(
        self,
        vertical_scan_length,
        position,
        scan_start_angle,
        scan_range,
        scan_exposure_time,
        wait=True,
    ):
        start = {}
        stop = {}
        for motor in position:
            if motor == "AlignmentZ":
                start[motor] = position[motor] + vertical_scan_length / 2.0
                stop[motor] = position[motor] - vertical_scan_length / 2.0
            else:
                start[motor] = position[motor]
                stop[motor] = position[motor]

        return self.helical_scan(
            start, stop, scan_start_angle, scan_range, scan_exposure_time, wait=wait
        )

    def start_helical_scan(self):
        return self.md.startscan4d()

    def start_scan_4d_ex(self, parameters):
        return self.md.startScan4DEx(parameters)

    def set_helical_start(self):
        return self.md.setstartscan4d()

    def set_helical_stop(self):
        return self.md.setstopscan4d()

    @md_task
    def start_raster_scan(
        self,
        vertical_range,
        horizontal_range,
        number_of_rows,
        number_of_columns,
        direction_inversion,
    ):
        self.task_id = self.md.startRasterScan(
            [
                vertical_range,
                horizontal_range,
                number_of_rows,
                number_of_columns,
                direction_inversion,
            ]
        )
        return self.task_id
