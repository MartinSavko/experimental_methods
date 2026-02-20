#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import redis
import gevent
import logging
import pickle
import time

# sys.path.insert(0, '/home/experiences/proxima2a/com-proxima2a/CATS/PyCATS_DS/pycats')
sys.path.insert(0, "/usr/local/pycats/pycats")
from catsapi import *
import catsapi

try:
    from PyTango import DeviceProxy as dp
except ImportError:
    from tango import DeviceProxy as dp

from goniometer import (
    goniometer,
    ALIGNMENTZ_REFERENCE,
    ALIGNMENTX_REFERENCE,
    ALIGNMENTY_REFERENCE,
)
from detector import detector


class cats:
    default_dewar_content = str(["UniPuck"] * 12)

    def __init__(self, host="172.19.10.125", operator=1071, monitor=10071):
        self.connection = CS8Connection()
        self.connection.connect(host, operator, monitor)
        self._type = 1
        self._toolcal = 0
        self.goniometer = goniometer()
        self.detector = detector()
        self.catsCheck = dp("i11-ma-cx1/ex/catscheck")
        self.state_params = catsapi.state_params
        self.di_params = catsapi.di_params
        self.do_params = catsapi.do_params
        self.redis = redis.StrictRedis(host=host)
        self.last_state_time = -1
        self.last_state = None
        self.last_optical_alignment_results_key = "last_optical_alignment_results"

    def set_camera(self, camera):
        self.redis.set("mxcube_camera", camera)

    def set_autoSoak(self, autosoakenable):
        try:
            self.catsCheck.autosoakenable = autosoakenable
        except:
            logging.debug("catsCheck autoSoak value unchanged")

    def open_cryovalve(self):
        self.connection.operate("opencryovalve")

    def close_cryovalve(self):
        self.connection.operate("closecryovalve")

    def back(self):
        return self.connection.operate("back(1)")

    def back_ht(self):
        return self.connection.operate("back_ht(1)")

    def on(self):
        return self.connection.powerOn()

    def off(self):
        return self.connection.powerOff()

    def abort(self):
        return self.connection.abort()

    def pause(self):
        return self.connection.pause()

    def reset(self):
        return self.connection.reset()

    def restart(self):
        return self.connection.restart()

    def regulon(self):
        return self.connection.regulon()

    def reguloff(self):
        return self.connection.reguloff()

    def message(self):
        return self.connection.message()

    def state(self, timeout=1.0):
        if (time.time() - self.last_state_time) > timeout or self.last_state is None:
            logging.debug(
                "time.time()-self.last_state_time %.4f, self.last_state %s"
                % (time.time() - self.last_state_time, self.last_state)
            )
            self.last_state_time = time.time()
            self.last_state = self.connection.state()

            logging.debug("state %s" % self.last_state)
        return self.last_state

    def closelid(self, lid):
        if lid in [1, 2, 3]:
            getattr(self.connection, "closelid%d" % lid)()
        else:
            logging.debug("%s is not a valid lid number" % lid)

    def openlid(self, lid):
        if lid == 1:
            return self.connection.openlid1()
        elif lid == 2:
            return self.connection.home_openlid2()
        elif lid == 3:
            return self.connection.openlid3()
        else:
            logging.debug("%s is not a valid lid number" % lid)

    def closelids(self):
        for k in [1, 2, 3]:
            self.closelid(k)

    def openlids(self):
        for k in [2, 1, 3]:
            self.openlid(k)

    def clear_memory(self):
        return self.connection.clear_memory()

    def acknowledge_missing_sample(self):
        return self.connection.ack_sample_memory()

    def resetmotion(self):
        return self.connection.resetmotion()

    def prepare_for_transfer(self, attempts=3):
        if not self.sample_mounted():
            self.acknowledge_missing_sample()

        self.reset()

        logging.debug("executing prepare_for_transfer")
        tried = 0
        while self.isoff() and tried <= attempts:
            self.on()
            tried += 1
            if self.message() == "Remote Mode requested":
                logging.info(
                    "Please turn the robot key to the remote operation position"
                )
                break
        transfer_jobs = []
        transfer_jobs.append(gevent.spawn(self.detector.cover.insert))
        # transfer_jobs.append(
        # gevent.spawn(self.goniometer.set_transfer_phase, wait=True)
        # )
        try:
            if self.detector.position.ts.get_position() < 200.0:
                transfer_jobs.append(
                    gevent.spawn(self.detector.position.ts.set_position, 200, wait=True)
                )
        except:
            print("could not check detector distance, please check!")
        transfer_jobs.append(gevent.spawn(self.goniometer.set_zoom(1, wait=False)))
        gevent.joinall(transfer_jobs)

    def prepare_for_centring(self, frontlightlevel=12, dark=False):
        if self.sample_mounted() == True:
            self.set_camera("oav")
            try:
                self.goniometer.set_zoom(1, wait=False)

                if dark is False:
                    self.goniometer.insert_backlight()
                    self.goniometer.extract_frontlight()
                else:
                    self.goniometer.extract_backlight()
                    self.goniometer.insert_frontlight()
                    self.goniometer.set_frontlightlevel(frontlightlevel)
            except:
                print(f"problem preparing centring, please check")
                traceback.print_exc()
            probable_position = self.get_probable_position()
            self.goniometer.set_position(probable_position)

        # elif not self.goniometer.sample_is_loaded():
        # self.acknowledge_missing_sample()
        # self.reset()

    def wait_for_trajectory(self, trajectory, sleeptime=1):
        while self.connection._is_trajectory_running(trajectory):
            gevent.sleep(sleeptime)

    def teachlid(
        self,
        lid=1,
        tool=4,
        abrakadabra="0,0,0,0,1,1,1,0,0,0,1,5,5,10,0,0,0,0.02",
        wait=False,
    ):
        _start = time.time()
        if self.isoff():
            self.on()
        if lid in [1, 2, 3]:
            command = "teach_lid"
            getattr(self.connection, "openlid%d" % lid)()
            trajectory = "%s%d(%d,%s)" % (command, lid, tool, abrakadabra)
        elif lid == 100:
            command = "teach_hotpuck"
            trajectory = "%s(%d,%s)" % (command, tool, abrakadabra)
        self.connection.operate(trajectory)
        if wait:
            self.wait_for_trajectory(command)
            print(
                "Trajectory %s at %s percent speed completed in %.2f seconds"
                % (
                    trajectory,
                    self.get_state_dictionary()["ROBOT_SPEED_RATIO"],
                    time.time() - _start,
                )
            )

    # @record_video
    def getput(
        self,
        lid,
        sample,
        x_shift=None,
        y_shift=None,
        z_shift=None,
        wait=True,
        prepare_centring=True,
        prepare_transfer=True,
        dark=False,
        sleeptime=1,
    ):
        if prepare_transfer:
            self.prepare_for_transfer()

        x_shift, y_shift, z_shift = self.get_shifts(
            x_shift=x_shift, y_shift=y_shift, z_shift=z_shift
        )

        if self.sample_mounted() == False:
            return self.put(
                lid,
                sample,
                x_shift,
                y_shift,
                z_shift,
                wait=wait,
                prepare_centring=prepare_centring,
                prepare_transfer=False,
            )
        a = self.connection.operate(
            "getput2(%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d)"
            % (
                1,
                lid,
                sample,
                0,
                0,
                0,
                0,
                self._type,
                self._toolcal,
                0,
                x_shift,
                y_shift,
                z_shift,
            )
        )

        gevent.sleep(sleeptime)
        if "getput2" not in self.state():
            logging.debug("getput2 not in state")
        gevent.sleep(sleeptime)
        if wait == True:
            while self.connection._is_trajectory_running("getput2"):
                gevent.sleep(sleeptime)

        if prepare_centring == True:
            self.prepare_for_centring(dark=dark)

        return a

    # hotpuck
    # @record_video
    def put_ht(
        self,
        lid,
        sample,
        x_shift=None,
        y_shift=None,
        z_shift=None,
        wait=True,
        prepare_centring=True,
        dark=False,
        sleeptime=1,
    ):
        if self.sample_mounted():
            lid_mounted, sample_mounted = self.get_mounted_sample_id()
            if lid == lid_mounted and sample == sample_mounted:
                print("sample already mounted")
                return
        self.prepare_for_transfer()
        self.set_autoSoak(False)

        x_shift, y_shift, z_shift = self.get_shifts(
            x_shift=x_shift, y_shift=y_shift, z_shift=z_shift
        )

        if self.sample_mounted() == True:
            return self.getput_ht(
                lid,
                sample,
                x_shift,
                y_shift,
                z_shift,
                wait=wait,
                prepare_centring=prepare_centring,
            )
        a = self.connection.put_ht(
            1, lid, sample, self._type, self._toolcal, x_shift, y_shift, z_shift
        )

        gevent.sleep(sleeptime)
        if "put_ht" not in self.state():
            print("put_ht not in state")
        gevent.sleep(sleeptime)
        if wait == True:
            while self.connection._is_trajectory_running("put_ht"):
                gevent.sleep(sleeptime)

            if not self.sample_mounted():
                pass  # self.acknowledge_missing_sample()

        if prepare_centring == True:
            self.prepare_for_centring(dark=dark)

        return a

    # @record_video
    def get_ht(self, x_shift=None, y_shift=None, z_shift=None, wait=True, sleeptime=1):
        self.prepare_for_transfer()
        self.set_autoSoak(False)

        x_shift, y_shift, z_shift = self.get_shifts(
            x_shift=x_shift, y_shift=y_shift, z_shift=z_shift
        )

        a = self.connection.get_ht(self._type, self._toolcal, x_shift, y_shift, z_shift)

        gevent.sleep(sleeptime)
        if "get_ht" not in self.state():
            print("get_ht not in state")
        gevent.sleep(sleeptime)
        if wait == True:
            while self.connection._is_trajectory_running("get_ht"):
                gevent.sleep(sleeptime)
        return a

    # @record_video
    def getput_ht(
        self,
        lid,
        sample,
        x_shift=None,
        y_shift=None,
        z_shift=None,
        wait=True,
        prepare_centring=True,
        dark=False,
        sleeptime=1,
    ):
        self.prepare_for_transfer()
        self.set_autoSoak(False)

        x_shift, y_shift, z_shift = self.get_shifts(
            x_shift=x_shift, y_shift=y_shift, z_shift=z_shift
        )

        # self.connection.getput(1, lid, sample, self._type, self._toolcal, x_shift, y_shift, z_shift)
        if self.sample_mounted() == False:
            return self.put_ht(
                lid,
                sample,
                x_shift,
                y_shift,
                z_shift,
                wait=wait,
                prepare_centring=prepare_centring,
            )

        a = self.connection.operate(
            "getput_ht(%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d)"
            % (
                1,
                lid,
                sample,
                0,
                0,
                0,
                0,
                self._type,
                self._toolcal,
                0,
                x_shift,
                y_shift,
                z_shift,
            )
        )

        gevent.sleep(sleeptime)
        if "getput_ht" not in self.state():
            print("getput_ht not in state")
        gevent.sleep(sleeptime)
        if wait == True:
            while self.connection._is_trajectory_running("getput_ht"):
                gevent.sleep(sleeptime)
            if not self.sample_mounted():
                self.acknowledge_missing_sample()

        if prepare_centring == True:
            self.prepare_for_centring(dark=dark)

        return a

    def sample_test(
        self,
        lid,
        sample,
        command="sample_test",
        x_shift=0,
        y_shift=0,
        z_shift=0,
        wait=True,
        prepare_centring=True,
        dark=False,
        sleeptime=1,
    ):
        a = self.connection.sample_test(
            1, lid, sample, self._type, self._toolcal, x_shift, y_shift, z_shift
        )

        gevent.sleep(sleeptime)
        if command not in self.state():
            logging.debug("%s not in state" % command)
        gevent.sleep(sleeptime)
        if wait == True:
            while self.connection._is_trajectory_running(command):
                gevent.sleep(sleeptime)

        return a

    def get_shifts(self, x_shift=None, y_shift=None, z_shift=None):
        if x_shift is None:
            x_shift = int(float(self.redis.get("robot_x").decode()))
        if y_shift is None:
            y_shift = int(float(self.redis.get("robot_y").decode()))
        if z_shift is None:
            z_shift = int(float(self.redis.get("robot_z").decode()))
        return x_shift, y_shift, z_shift

    # @record_video
    def put(
        self,
        lid,
        sample,
        x_shift=None,
        y_shift=None,
        z_shift=None,
        wait=True,
        prepare_centring=True,
        prepare_transfer=True,
        dark=False,
        sleeptime=1,
    ):
        sample_mounted = self.sample_mounted()
        if sample_mounted:
            lid_mounted, sample_mounted = self.get_mounted_sample_id()
            if lid == lid_mounted and sample == sample_mounted:
                logging.debug("sample already mounted")
                return

        x_shift, y_shift, z_shift = self.get_shifts(
            x_shift=x_shift, y_shift=y_shift, z_shift=z_shift
        )

        if prepare_transfer:
            self.prepare_for_transfer()

        if sample_mounted == True:
            return self.getput(
                lid,
                sample,
                x_shift,
                y_shift,
                z_shift,
                wait=wait,
                prepare_centring=prepare_centring,
                prepare_transfer=False,
                dark=dark,
            )

        a = self.connection.put(
            1, lid, sample, self._type, self._toolcal, x_shift, y_shift, z_shift
        )

        gevent.sleep(sleeptime)
        if "put" not in self.state():
            logging.debug("put not in state")
        gevent.sleep(sleeptime)
        if wait == True:
            while self.connection._is_trajectory_running("put"):
                gevent.sleep(sleeptime)

            # if not self.sample_mounted():
            # self.acknowledge_missing_sample()

        if prepare_centring == True:
            self.prepare_for_centring(dark=dark)

        return a

    # @record_video
    def get(self, x_shift=None, y_shift=None, z_shift=None, wait=True, sleeptime=1.0):
        self.prepare_for_transfer()

        x_shift, y_shift, z_shift = self.get_shifts(
            x_shift=x_shift, y_shift=y_shift, z_shift=z_shift
        )

        a = self.connection.get(self._type, self._toolcal, x_shift, y_shift, z_shift)

        gevent.sleep(sleeptime)
        if "get" not in self.state():
            logging.debug("get not in state")
        gevent.sleep(sleeptime)
        if wait == True:
            while self.connection._is_trajectory_running("get"):
                gevent.sleep(sleeptime)
        return a

    def get_mounted_sample_id(self):
        state_dictionary = self.get_state_dictionary()
        try:
            lid = int(state_dictionary["LID_NUM_SAMPLE_MOUNTED_ON_DIFFRACTOMETER"])
        except ValueError:
            lid = -1
        try:
            sample = int(state_dictionary["NUM_SAMPLE_MOUNTED_ON_DIFFRACTOMETER"])
        except ValueError:
            sample = -1

        if not self.sample_mounted():
            lid, sample = -1, -1

        return lid, sample

    def get_mounted_puck_and_sample(
        self, pucks_per_lid=3, samples_per_puck=16, lid=None, sample=None
    ):
        if (lid is None) or (sample is None):
            lid, sample = self.get_mounted_sample_id()
        if lid == -1:
            return -1, -1
        elif lid == 100:
            lid = 4
        puck, sample = divmod(sample, samples_per_puck)
        if sample > 0:
            puck += 1
        else:
            sample = 16
        puck += (lid - 1) * pucks_per_lid
        return puck, sample

    def get_element(self, separator="_"):
        puck, sample = self.get_mounted_puck_and_sample()
        element = f"{puck}{separator}{sample}"
        if puck == -1:
            element = "unknown"
        return element

    def wash(self):
        if self.sample_mounted():
            lid, sample = self.get_mounted_sample_id()
            return self.put(lid, sample)

    def get_there(self, lid, sample, x_shift=0, y_shift=0, z_shift=0):
        self.prepare_for_transfer()
        return self.connection.operate(
            "get2(%d, %d, %d, %d, %d, %d, %d, %d)"
            % (1, lid, sample, x_shift, y_shift, z_shift, 0, 1)
        )
        # return self.connection.get(self._type, self._toolcal, x_shift, y_shift, z_shift)

    def sample_mounted(self):
        return self.goniometer.sample_is_loaded()

    # @record_video
    def dry(self):
        self.connection.dry(1)

    # @record_video
    def dry_and_soak2(self):
        self.dry()
        gevent.sleep(1)
        self.wait_for_trajectory("dry")
        self.soak()

    # @record_video
    def dry_and_soak(self):
        self.connection.dry_soak(1, 2)

    # @record_video
    def sos(self):
        self.abort()
        gevent.sleep(1)
        self.home()
        self.wait_for_trajectory("home")
        self.soak()
        gevent.sleep(1)
        self.abort()

    def dry_ht(self):
        self.set_autoSoak(False)
        return self.connection.dry_ht(1)

    def soak(self):
        return self.connection.soak(1, 2)

    def soak_toolcal(self):
        return self.connection.tremp_toolcal()

    def tremp_toolcal(self):
        return self.soak_toolcal()

    def calibrate(self):
        return self.connection.toolcal()

    def home(self):
        self.connection.home(1)

    def safe(self):
        self.goniometer.abort()
        self.connection.safe(1)

    def get_puck_presence(self):
        a = np.array([int(n) for n in self.connection.di()[15 : 15 + 9]])
        logging.debug(str(a))
        return a

    def get_state(self):
        return self.state()

    def get_state_vector(self, verification_string="state(", verbose=False):
        received = False
        while not received:
            state = self.get_state()
            if verification_string == state[: len(verification_string)]:
                received = True
            else:
                gevent.sleep(0.1)

        if verbose:
            print("state", state)
        state_vector = state.strip(verification_string + ")").split(",")
        return state_vector

    def get_state_dictionary(self):
        state_vector = self.get_state_vector()
        state_dictionary = dict(list(zip(self.state_params, state_vector)))
        return state_dictionary

    def get_di(self):
        return self.connection.di()

    def get_di_vector(self, verification_string="di("):
        received = False
        while not received:
            di = self.get_di()
            if verification_string == di[: len(verification_string)]:
                received = True
            else:
                gevent.sleep(0.1)
        logging.debug("di %s " % str(di))
        di = di.strip(verification_string + ")")
        di_vector = list(map(int, di))
        return di_vector

    def get_di_dictionary(self):
        di_vector = self.get_di_vector()
        di_dictionary = dict(list(zip(self.di_params, di_vector)))
        return di_dictionary

    def get_do(self):
        return self.connection.do()

    def get_do_vector(self, verification_string="do("):
        received = False
        while not received:
            do = self.get_do()
            if verification_string == do[: len(verification_string)]:
                received = True
            else:
                gevent.sleep(0.1)
        logging.debug("do %s" % str(do))
        do = do.strip(verification_string + ")")
        do_vector = list(map(int, do))
        return do_vector

    def get_do_dictionary(self):
        do_vector = self.get_do_vector()
        do_dictionary = dict(list(zip(self.do_params, do_vector)))
        return do_dictionary

    def ison(self):
        logging.debug("ison called")
        return self.get_state_dictionary()["POWER_1_0"] == "1"

    def isoff(self):
        return self.get_state_dictionary()["POWER_1_0"] == "0"

    def get_puck_list(self):
        return [
            "CPS-5700",
            "CPS-5701",
            "CPS-4354",
            "IRM-0003",
            "CIP-01",
            "CPS-2303",
            "CPS-2302",
            "Empty",
            "PX2-test_samples",
        ]

    def get_dewar_content(self):
        if eval(self.get_dewar_content_valid()):
            dewar_content = eval(self.redis.get("dewar_content"))
        else:
            dewar_content = self.default_dewar_content
        return dewar_content

    def print_dewar_content(self):
        dc = self.get_dewar_content()
        for k, n in enumerate(dc):
            print("%d. %s" % (k + 1, n))

    def get_dewar_content_valid(self):
        return self.redis.get("dewar_content_valid").decode("utf-8")

    def set_dewar_content_valid(self, valid=1):
        self.redis.set("dewar_content_valid", valid)

    def set_dewar_content(self, dewar_content=default_dewar_content):
        self.redis.set("dewar_content", str(dewar_content))
        self.redis.set("dewar_content_valid", 1)

    def set_puck_name(self, puck, name):
        dewar_content = self.get_dewar_content()
        dewar_content[int(puck) - 1] = str(name)
        self.set_dewar_content(dewar_content=dewar_content)

    def get_probable_position(self):
        if self.goniometer.has_kappa():
            # probable_position = {
            #'Omega': 4e-05,
            #'CentringX': 0.766989,
            #'CentringY': -0.46951,
            #'AlignmentX': 0.01,
            #'AlignmentY': -0.51634,
            #'AlignmentZ': 0.183701,
            # }
            probable_position = {
                "Omega": 180.0,
                "Kappa": 0.0,
                "Phi": 0.0,
                "CentringX": 0.106,
                "CentringY": 0.101,
                "AlignmentX": ALIGNMENTX_REFERENCE,
                "AlignmentY": ALIGNMENTY_REFERENCE,
                "AlignmentZ": ALIGNMENTZ_REFERENCE,
            }
        else:
            probable_position = {
                "AlignmentZ": 0.0,  # 0.0847,
                "AlignmentY": -0.84,
                "AlignmentX": 0.0,
                "CentringX": 0,  # 0.041,
                "CentringY": 0,  # -0.579,
                # "ApertureVertical": 83.1084,
            }

        # try:
        # last_results = pickle.loads(
        # self.redis.get(self.last_optical_alignment_results_key)
        # )
        # print("last_results present")
        # print(last_results)
        # if str(last_results["mounted_sample_id"]) == str(
        # self.get_mounted_sample_id()
        # ):
        # print(
        # "mounted_sample_id is the same as the previous one, will try to make use of it"
        # )
        # probable_position = last_results["result_position"]
        # except:
        # traceback.print_exc()
        # print("last_results not available")

        # if not self.goniometer.has_kappa():
        # probable_position["AlignmentX"] = 0.160
        # else:
        # probable_position["AlignmentX"] = ALIGNMENTX_REFERENCE

        return probable_position

    def get_lid_sample_from_puck_sample(
        self, puck, sample, samples_per_puck=16, pucks_per_lid=3
    ):
        lid, remainder = divmod(puck, pucks_per_lid)
        if remainder:
            lid += 1
            remainder -= 1
        else:
            remainder = pucks_per_lid - 1
        sample += remainder * samples_per_puck
        if lid == 4:
            lid = 100
        return lid, sample

    def get_puck_sample_from_lid_sample(
        self, lid, sample, samples_per_puck=16, pucks_per_lid=3
    ):
        puck = (lid - 1) * pucks_per_lid
        puck_in_lid, sample_in_puck = divmod(sample, samples_per_puck)
        if sample_in_puck:
            puck_in_lid += 1
        else:
            sample_in_puck = samples_per_puck
        puck += puck_in_lid
        return puck, sample_in_puck

    def mount(
        self,
        puck,
        sample,
        x_shift=None,
        y_shift=None,
        z_shift=None,
        wait=True,
        prepare_centring=True,
        dark=False,
        sleeptime=1,
    ):
        lid, sample = self.get_lid_sample_from_puck_sample(puck, sample)
        if lid <= 3:
            getput = self.getput
        elif lid == 100:
            getput = self.getput_ht

        getput(
            lid,
            sample,
            x_shift=x_shift,
            y_shift=y_shift,
            z_shift=z_shift,
            wait=wait,
            prepare_centring=prepare_centring,
            dark=dark,
            sleeptime=sleeptime,
        )

    def umount(
        self, x_shift=None, y_shift=None, z_shift=None, wait=True, sleeptime=1.0
    ):
        lid, sample = self.get_mounted_sample_id()
        if lid <= 3:
            get = self.get
        elif lid == 100:
            get = self.get_ht

        get(
            x_shift=x_shift,
            y_shift=y_shift,
            z_shift=z_shift,
            wait=wait,
            sleeptime=sleeptime,
        )

    def is_path_running(self):
        state_dictionary = self.get_state_dictionary()
        return state_dictionary["PATH_RUNNING_1_0"] == "1"


class dewar_content:
    default_dewar_content = str(["UniPuck"] * 12)

    def __init__(self, host="172.19.10.125"):
        self.host = host
        self.redis = redis.StrictRedis(host=host)

    def get_dewar_content(self):
        if eval(self.get_dewar_content_valid()):
            dewar_content = eval(self.redis.get("dewar_content"))
        else:
            dewar_content = self.default_dewar_content
        return dewar_content

    def get_dewar_content_valid(self):
        return self.redis.get("dewar_content_valid").decode("utf-8")

    def set_dewar_content_valid(self, valid=1):
        self.redis.set("dewar_content_valid", valid)

    def set_dewar_content(self, dewar_content=default_dewar_content):
        self.redis.set("dewar_content", str(dewar_content))
        self.redis.set("dewar_content_valid", 1)

    def set_puck_name(self, puck, name):
        dewar_content = self.get_dewar_content()
        dewar_content[int(puck) - 1] = str(name)
        self.set_dewar_content(dewar_content=dewar_content)


class puck:
    def __init__(self, position, name, samples):
        self.position = position
        self.name = name
        self.samples = samples

    def get_position(self):
        return self.position

    def get_name(self):
        return self.name

    def get_samples(self):
        return self.samples


class dewar:
    def __init__(
        self,
        positions=list(range(1, 10)),
        names=[""] * 9,
        samples=[list(range(1, 17))] * 9,
    ):
        self.positions = positions
        self.names = names
        self.samples = samples

    def get_content(self):
        content = []
        for position, name, samples in zip(self.positions, self.names, self.samples):
            content.append(puck(position, name, samples))
        return content


def main():
    c = cats()

    import optparse

    parser = optparse.OptionParser()
    parser.add_option(
        "-c", "--command", default=None, type=str, help="Specify command to execute"
    )
    parser.add_option("-l", "--lid", default=None, type=int, help="Specify the lid")
    parser.add_option("-p", "--puck", default=None, type=int, help="Specify the puck")
    parser.add_option(
        "-s", "--sample", default=None, type=int, help="Specify the sample"
    )

    options, args = parser.parse_args()

    if options.command in ["get", "getput", "put", "get_ht", "getput_ht", "put_ht"]:
        getattr(c, options.command)(options.lid, options.sample)
    elif options.command in ["mount"]:
        getattr(c, options.command)(options.puck, options.sample)
    elif options.command in ["umount"]:
        getattr(c, options.command)()
    elif options.command in ["openlid", "closelid"]:
        getattr(c, options.command)(options.lid)
    elif options.command is None:
        sys.exit("No command specified, exiting ...")
    else:
        getattr(c, options.command)()


if __name__ == "__main__":
    main()
