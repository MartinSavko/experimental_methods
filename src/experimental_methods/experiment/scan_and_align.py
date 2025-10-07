#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Aligning procedures for MD2 on PX2A beamline at Synchrotron SOLEIL 
Martin Savko martin.savko@synchrotron-soleil.fr
"""

import os
import itertools
import pylab
import numpy as np
import math
import PyTango
import time
import pickle
import copy
import scipy.ndimage
import struct

from experimental_methods.analysis.gauss2d import fitgaussian
from experimental_methods.instrument.safety_shutter import safety_shutter
from experimental_methods.instrument.fast_shutter import fast_shutter
from experimental_methods.instrument.camera import camera


class scan_and_align(object):
    # motorsNames = ['PhiTableXAxisPosition',
    #'PhiTableYAxisPosition',
    #'PhiTableZAxisPosition',
    #'CentringTableXAxisPosition',
    #'CentringTableYAxisPosition',
    #'ApertureHorizontalPosition',
    #'ApertureVerticalPosition',
    #'CapillaryBSHorizontalPosition',
    #'CapillaryBSVerticalPosition']
    motorsNames = [
        "AlignmentXPosition",
        "AlignmentYPosition",
        "AlignmentZPosition",
        "CentringXPosition",
        "CentringYPosition",
        "ApertureHorizontalPosition",
        "ApertureVerticalPosition",
        "CapillaryHorizontalPosition",
        "CapillaryVerticalPosition",
    ]

    motorShortNames = [
        "PhiX",
        "PhiY",
        "PhiZ",
        "SamX",
        "SamY",
        "AprX",
        "AprZ",
        "CbsX",
        "CbsZ",
    ]

    shortFull = dict(list(zip(motorShortNames, motorsNames)))

    MD2_motors = {"aperture": ["AprX", "AprZ"], "capillary": ["CbsX", "CbsZ"]}

    # reference positions 100, 50, 20, 10, 05; MS 2014-11-30
    # In [169]: c.md.apertureverticalposition, c.md.aperturehorizontalposition
    # Out[169]: (93.180902078989391, -0.13247385715793916)

    # In [165]: c.md.apertureverticalposition, c.md.aperturehorizontalposition
    # Out[165]: (94.378675780425198, -0.15038437631967905)

    # In [166]: c.md.apertureverticalposition, c.md.aperturehorizontalposition
    # Out[166]: (95.590240127975875, -0.15341139014991553)

    # In [167]: c.md.apertureverticalposition, c.md.aperturehorizontalposition
    # Out[167]: (96.767501583614859, -0.14640627680598078)

    # In [168]: c.md.apertureverticalposition, c.md.aperturehorizontalposition
    # Out[168]: (97.992180241765197, -0.12219841415817673)

    Sizes = {
        "aperture_100um": 0.1,
        "aperture_50um": 0.05,
        "aperture_20um": 0.02,
        "aperture_10um": 0.01,
        "aperture_05um": 0.005,
        "capillary": 0.100,
    }
    Steps = {
        "aperture_100um": 0.2,
        "aperture_50um": 0.25,
        "aperture_20um": 0.2,
        "aperture_10um": 0.2,
        "aperture_05um": 0.5,
        "capillary": 0.2,
    }
    Extents = {
        "aperture_100um": 3,
        "aperture_50um": 2,
        "aperture_20um": 3,
        "aperture_10um": 4,
        "aperture_05um": 10,
        "capillary": 3,
    }
    # Distances = {'aperture_100um': (0, 0), 'aperture_50um': (1.1882, -0.0126), 'aperture_20um': (1.22 , -0.005), 'aperture_10um': (1.1771,  0.0013), 'aperture_05um': (1.2237,  0.028), 'capillary': (None, None)}
    # Distances = {'aperture_100um': (0, 0), 'aperture_50um': (1.1992, -0.0186), 'aperture_20um': (1.2109 , -0.003), 'aperture_10um': (1.1762,  0.0062), 'aperture_05um': (1.229 ,  0.0239), 'capillary': (None, None)} # 2014-07-15
    Distances = {
        "aperture_100um": (0, 0),
        "aperture_50um": (1.1977, -0.0179),
        "aperture_20um": (1.2117, -0.0030),
        "aperture_10um": (1.1771, 0.0070),
        "aperture_05um": (1.2247, 0.0242),
        "capillary": (None, None),
    }

    def __init__(
        self,
        what,
        aperture_index=None,
        nbsteps=None,
        lengths=None,
        extent=None,
        shape=(1, 1),
        step=None,
        motor_device="i11-ma-cx1/ex/md3",
        observable={
            "device": "i11-ma-cx1/ex/imag.1",
            "attribute": "image",
            "economy": "mean",
        },
        snap=False,
        display=True,
        exposure=0.05,
    ):
        self.start = time.time()
        self.datetime = time.asctime()
        self.motor_device = PyTango.DeviceProxy(motor_device)
        self.observable = observable
        if "device" in self.observable:
            self.sensor_device = PyTango.DeviceProxy(self.observable["device"])
        else:
            self.sensor_device = self.observable
        self.what = what
        self.aperture_index = aperture_index
        self.motors = self.MD2_motors[self.what]
        self.nbsteps = nbsteps
        self.lengths = lengths
        self.step = step
        self.extent = extent
        self.snap = snap
        self.display = display
        self.exposure = exposure

        self.fast_shutter = fast_shutter()
        self.safety_shutter = safety_shutter()
        self.camera = camera()

        self.shape = np.array(shape)
        self.results = {}
        print("lengths", self.lengths)
        print("nbsteps", self.nbsteps)
        print("step", self.step)
        print("extent", self.extent)
        print("shape", self.shape)

    def checkSteps(self):
        if self.step is None:
            self.step = self.Steps[self.getLongName()]

    def checkLengths(self):
        self.objectSize = self.Sizes[self.getLongName()]
        self.extent = self.Extents[self.getLongName()]
        if self.lengths is None:
            self.lengths = self.objectSize * self.extent * self.shape

    def checkNbSteps(self):
        if self.step is None:
            self.step = self.Steps[self.getLongName()]
        if self.nbsteps is None:
            self.nbsteps = self.lengths / (self.step * self.objectSize)

    def wait(self, device):
        while device.state().name == "MOVING":
            time.sleep(0.1)

        while device.state().name == "RUNNING":
            time.sleep(0.1)

    def wait_motor(self, motor, sleeptime=0.2):
        while self.motor_device.getMotorState(motor).name != "STANDBY":
            time.sleep(sleeptime)

    def move_to_position(self, position={}, epsilon=0.002):
        if position != {}:
            for motor in position:
                self.wait_motor(self.shortFull[motor].replace("Position", ""))
                k = 0
                while (
                    abs(
                        self.motor_device.read_attribute(self.shortFull[motor]).value
                        - position[motor]
                    )
                    > epsilon
                ):
                    k += 1
                    self.motor_device.write_attribute(
                        self.shortFull[motor], round(position[motor], 4)
                    )
                    self.wait_motor(self.shortFull[motor].replace("Position", ""))

    def rotate(self, angle, unit="radians"):
        if unit != "radians":
            angle = math.radians(angle)

        r = np.array(
            [
                [math.cos(angle), math.sin(angle), 0.0],
                [-math.sin(angle), math.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        return r

    def shift(self, displacement):
        s = np.array(
            [[1.0, 0.0, displacement[0]], [0.0, 1.0, displacement[1]], [0.0, 0.0, 1.0]]
        )
        return s

    def scale(self, factor):
        s = np.diag([factor[0], factor[1], 1.0])
        return s

    def raster(self, grid):
        gs = grid.shape
        orderedGrid = []
        for i in range(gs[0]):
            line = grid[i, :]
            if (i + 1) % 2 == 0:
                line = line[::-1]
            orderedGrid.append(line)
        return np.array(orderedGrid)

    def calculatePositions(self):
        """Calculate positions at which we will measure. 2D for now i.e. two motors only."""
        center = np.array(self.center)
        nbsteps = np.array(np.round(self.nbsteps)).astype(np.int8)
        lengths = np.array(self.lengths)

        stepsizes = lengths / nbsteps

        print("center", center)
        print("nbsteps", nbsteps)
        print("lengths", lengths)
        print("stepsizes", stepsizes)

        # adding [1] so that we can use homogeneous coordinates
        positions = list(
            itertools.product(
                list(range(int(nbsteps[0]))), list(range(int(nbsteps[1]))), [1]
            )
        )
        # print(f'positions {positions}')
        points = [np.array(position) for position in positions]
        points = np.array(points)
        # print('0 points', points)
        # print(f'nbsteps, {nbsteps, nbsteps/2, nbsteps/2.}')
        # print(f'-nbsteps / 2. {- nbsteps / 2.}')

        # print(f'self.shift(- nbsteps / 2.) {self.shift(- nbsteps / 2.)}')
        points = np.dot(self.shift(-nbsteps / 2.0), points.T).T  # shift
        # print('1 shift points', points)
        points = np.dot(self.scale(stepsizes), points.T).T  # scale
        # print('2 scaled points', points)
        points = np.dot(self.shift(center), points.T).T  # shift
        # print('3 2nd shift points', points)
        grid = np.reshape(points, np.hstack((nbsteps, 3)))
        rasteredGrid = self.raster(grid)  # raster
        # print('rasteredGrid', rasteredGrid)
        orderedPositions = rasteredGrid.reshape((int(grid.size / 3), 3))
        # print(f'orderedPositions, {orderedPositions}')
        dictionariesOfOrderedPositions = [
            {self.motors[0]: position[0], self.motors[1]: position[1]}
            for position in orderedPositions
        ]
        self.positions = positions
        self.points = points
        self.grid = grid
        self.nbsteps = nbsteps
        self.rasteredGrid = rasteredGrid
        self.orderedPositions = orderedPositions
        self.dictionariesOfOrderedPositions = dictionariesOfOrderedPositions

    def representPosition(self, position):
        if self.what == "aperture":
            return "[x: %.4f, z: %.4f]" % (position["AprX"], position["AprZ"])
        if self.what == "capillary":
            return "[x: %.4f, z: %.4f]" % (position["CbsX"], position["CbsZ"])

    def linearizedScan(self):
        positions = self.dictionariesOfOrderedPositions
        xyz = []
        lp = len(positions)
        for k, position in enumerate(positions):
            if k % 20 == 0 or k == (lp - 1):
                print(
                    "moving to position %s (%d of %d)"
                    % (self.representPosition(position), k + 1, lp)
                )
            self.positionAndValues = copy.deepcopy(position)
            self.move_to_position(position)
            self.observe()
            xyz.append(self.positionAndValues)
        self.xyz = copy.deepcopy(xyz)
        f = open("resutls_%.1f.pickle" % time.time(), "wb")
        pickle.dump(self.xyz, f)
        f.close()

    def get_lima_image(self):
        img_data = self.sensor_device.video_last_image
        if img_data[0] == "VIDEO_IMAGE":
            header_fmt = ">IHHqiiHHHH"
            _, ver, img_mode, frame_number, width, height, _, _, _, _ = struct.unpack(
                header_fmt, img_data[1][: struct.calcsize(header_fmt)]
            )
            raw_buffer = np.fromstring(img_data[1][32:], np.uint16)
            image = raw_buffer.reshape((height, width))
        return image

    def observe(self):
        time.sleep(self.exposure)

        # if self.observable['device'] == 'lima/limaccd/1':
        # image = self.get_lima_image()
        # self.positionAndValues[(self.observable['device'], self.observable['attribute'])] = image.mean()

        if (
            self.observable != "diffraction"
            and self.observable["attribute"].find("image") != -1
        ):
            self.positionAndValues[
                (self.observable["device"], self.observable["attribute"])
            ] = self.camera.get_integral_of_bright_spots()

            # if self.observable['economy'] == 'mean':
            # self.positionAndValues[(self.observable['device'], self.observable['attribute'])] = self.sensor_device.read_attribute(self.observable['attribute']).value.mean()
            # else:
            # self.positionAndValues[(self.observable['device'], self.observable['attribute'])] = self.sensor_device.read_attribute(self.observable['attribute']).value

        else:
            self.collectObject.nbFrames = 4
            self.collectObject.template = self.collectObject.template.replace(
                "CbsX", str(position["CbsX"])
            ).replace("CbsZ", str(position["CbsZ"]))
            print("template", self.collectObject.template)
            self.collectObject.collect()
            value = self.collectObject.imagePath + self.collectObject.template
            self.positionAndValues["diffraction"] = value

    def setFP(self):
        fp = PyTango.DeviceProxy("passerelle/oh/fp")
        fent_h1 = PyTango.DeviceProxy("i11-ma-c02/ex/fent_h.1")
        fent_v1 = PyTango.DeviceProxy("i11-ma-c02/ex/fent_v.1")
        fent_h1.gap = fp.fp_hfmfield
        fent_v1.gap = fp.result2

    def transmission(self, x=None):
        """Get or set the transmission"""
        # if self.test: return 0
        Fp = PyTango.DeviceProxy("i11-ma-c00/ex/fp_parser")
        if x == None:
            return Fp.TrueTrans_FP

        Ps_h = PyTango.DeviceProxy("i11-ma-c02/ex/fent_h.1")
        Ps_v = PyTango.DeviceProxy("i11-ma-c02/ex/fent_v.1")
        Const = PyTango.DeviceProxy("i11-ma-c00/ex/fpconstparser")

        truevalue = (2.0 - math.sqrt(4 - 0.04 * x)) / 0.02

        newGapFP_H = math.sqrt(
            (truevalue / 100.0) * Const.FP_Area_FWHM / Const.Ratio_FP_Gap
        )

        newGapFP_V = newGapFP_H * Const.Ratio_FP_Gap

        Ps_h.gap = newGapFP_H
        Ps_v.gap = newGapFP_V

    def setExposure(self, exposure):
        self.camera.set_exposure(exposure)

    def setPhase(self, phase_number):
        self.motor_device.PhasePosition = phase_number
        self.wait(self.motor_device)

    def setZoom(self, zoom):
        self.motor_device.ZoomPredefinedPosition = zoom
        self.wait(self.motor_device)

    def scan(self):
        # observable is dictionary which contains three entries: the 'device' refers to the device through which we access sensors, 'attribute' referring list of attributes to record and 'economy' that is intended for multidimensional measurements to indicate whether full measurement or just some global value (like e.g. mean) should be stored.
        if self.aperture_index is not None:
            self.set_aperture(self.aperture_index)
            self.wait(self.motor_device)
        self.checkSteps()
        self.checkLengths()
        self.checkNbSteps()
        self.setExposure(
            self.exposure
        )  # 0.05 for 8 bunch mode; 0.25 for 1 bunch mode, otherwise 0.005
        # startTransmission = self.transmission() # remembering starting transmission, we will put it back after the scan
        # self.setFP()
        # while self.collect.mono_mt_rx.state().name != 'OFF':
        # self.collect.safeTurnOff(self.collect.mono_mt_rx)
        # time.sleep(0.1)
        # self.collect.setEnergy(12.65)
        # self.transmission(85)

        # self.setZoom(10)
        self.putScannedObjectInBeam()
        # center will contain current values of the scanned object
        self.center = [
            self.motor_device.read_attribute(self.shortFull[motor]).value
            for motor in self.motors
        ]

        # precalculating all the measurement positions
        self.calculatePositions()
        self.safety_shutter.open()
        self.wait(self.motor_device)
        self.fast_shutter.open()
        self.motor_device.write_attribute("frontlightlevel", 0)
        self.motor_device.write_attribute("frontlightison", False)

        self.linearizedScan()
        self.duration = time.time() - self.start

        self.fast_shutter.close()
        self.setExposure(0.050)
        # self.transmission(startTransmission)
        self.putScannedObjectInBeam()
        # self.setPhase(4) # set to collect phase

    def set_aperture(self, index=1):
        self.motor_device.write_attribute("CurrentApertureDiameterIndex", index)
        self.wait_motor("ApertureVertical")
        self.wait_motor("ApertureHorizontal")

    def setWhatToScan(self, what):
        self.what = what

    def putScannedObjectInBeam(self):
        # positionAttributeOfScannedObject = {'capillary': 'CapillaryBSPredefinedPosition',
        #'aperture': 'AperturePredefinedPosition'}
        positionAttributeOfScannedObject = {
            "capillary": "CapillaryPosition",
            "aperture": "AperturePosition",
        }
        # Put scanned object (capillary beamstop or an aperture into beam
        # self.motor_device.write_attribute(positionAttributeOfScannedObject[self.what], 1)
        self.motor_device.write_attribute(
            positionAttributeOfScannedObject[self.what], "BEAM"
        )
        self.wait(self.motor_device)

    def getLongName(self):
        # apcap = {1: 'aperture_100um', 2: 'aperture_50um', 3: 'aperture_20um', 4: 'aperture_10um', 5: 'aperture_05um', 'aperture': 'aperture', 'capillary': 'capillary'}
        apcap = {
            0: "aperture_100um",
            1: "aperture_50um",
            2: "aperture_20um",
            3: "aperture_10um",
            4: "aperture_05um",
            5: "unknown",
            "aperture": "aperture",
            "capillary": "capillary",
        }
        if self.what == "aperture":
            return apcap[
                self.motor_device.read_attribute("CurrentApertureDiameterIndex").value
            ]
        return self.what

    def save_to_publisher(self):
        publisher = PyTango.DeviceProxy("passerelle/eh/md")
        current_position = self.get_current_position()
        if self.what == "aperture":
            x, z = current_position["AprX"], current_position["AprZ"]
            ap = self.getLongName()
            if ap == "aperture_05um":
                ap = ap.replace("05um", "5um")
            publisher.write_attribute(ap + "_X", x)
            publisher.write_attribute(ap + "_Z", z)
        elif self.what == "capillary":
            x, z = current_position["CbsX"], current_position["CbsZ"]
            ap = "CPBS"
            publisher.write_attribute(ap + "_X", x)
            publisher.write_attribute(ap + "_Z", z)

    def save_scan(self):
        self.results["start"] = self.start
        self.results["datetime"] = self.datetime
        self.results["what"] = self.what
        self.results["nbsteps"] = self.nbsteps
        self.results["lengths"] = self.lengths
        self.results["shape"] = self.shape
        self.results["extent"] = self.extent
        self.results["objectSize"] = self.objectSize
        self.results["motors"] = self.motors
        self.results["center"] = self.center
        self.results["points"] = self.points
        self.results["grid"] = self.grid
        self.results["rasteredGrid"] = self.rasteredGrid
        self.results["orderedPositions"] = self.orderedPositions
        self.results[
            "dictionariesOfOrderedPositions"
        ] = self.dictionariesOfOrderedPositions
        self.results["observable"] = self.observable
        self.results["positions"] = self.positions
        self.results["xyz"] = self.xyz
        self.results["X"] = self.X
        self.results["Y"] = self.Y
        self.results["Z"] = self.Z
        self.results["duration"] = self.duration
        self.results["optimum"] = (self.xopt, self.yopt)
        longName = self.getLongName()
        filename = (
            longName + "_" + "_".join(self.results["datetime"].split()) + ".pickle"
        )
        self.save_to_publisher()
        f = open(filename, "wb")
        pickle.dump(self.results, f)
        f.close()

    def align(self, optimum="max"):
        self.X, self.Y, self.Z = self.XYZ()
        if optimum == "max":
            x, y = self.singlemax()
        elif optimum == "gauss":
            x, y = self.gauss()
        elif optimum == "com":
            x, y = self.com()
        else:
            print("unexpected branch in align")

        if self.display is True:
            print("Showing 2d representation of scan")
            img = scipy.ndimage.rotate(self.Z, 90)
            pylab.imshow(img)
            filename = self.getLongName() + "_" + "%.1f" % time.time() + ".png"
            pylab.savefig(filename)
            pylab.show()
        print("optimal values determined: %f, %f" % (x, y))
        print(
            "shift from previous values: %f, %f"
            % (x - self.center[0], y - self.center[1])
        )
        position = {self.motors[0]: x, self.motors[1]: y}
        self.move_to_position(position)
        self.save_new_values()
        self.xopt = x
        self.yopt = y
        return x, y

    def save_new_values(self):
        if self.what == "aperture":
            self.motor_device.saveApertureBeamPosition()  # ApertureSaveInPosition()
        elif self.what == "capillary":
            self.motor_device.saveCapillaryBeamPosition()  # CapillaryBSSaveInPosition()

    def get_distance_matrix(self):
        listofthem = [100, 50, 20, 10, 5]
        it = np.array((0.0, 0.0))
        li = np.array(25 * [it])
        D = li.reshape((5, 5, 2))

        def get_string(number):
            return "aperture_%sum" % str(number).zfill(2)

        for k in range(len(listofthem)):
            for l in range(len(listofthem)):
                if k != l:
                    if l > k:
                        indexes = list(range(k + 1, l + 1, 1))
                    else:
                        indexes = list(range(l + 1, k + 1, 1))
                    for i in indexes:
                        string = get_string(listofthem[i])
                        D[k, l] += np.sign(l - k) * np.array(self.Distances[string])
        return D

    def get_current_position(self):
        current_position = {}
        for motor in self.motors:
            current_position[motor] = self.motor_device.read_attribute(
                self.shortFull[motor]
            ).value
        return current_position

    def get_offset_dictionary(self, offsets):
        offset_dictionary = {}
        for k in range(len(offsets)):
            offset_dictionary[k] = {}
            for l in range(len(offsets[k])):
                offset_dictionary[k][self.motors[l]] = offsets[k][::-1][l]
        return offset_dictionary

    def get_new_positions(self, reference_position, offset_dictionary):
        new_positions = {}
        for k in range(len(offset_dictionary)):
            new_positions[k] = {}
            for motor in self.motors:
                new_positions[k][motor] = (
                    reference_position[motor] + offset_dictionary[k][motor]
                )
        return new_positions

    def predict(self):
        # return
        D = self.get_distance_matrix()
        print("D")
        print(D)
        current_index = self.motor_device.read_attribute(
            "CurrentApertureDiameterIndex"
        ).value  # self.what
        print("current_index", current_index)
        reference_position = self.get_current_position()
        offsets = D[current_index, :]
        print("offsets", offsets)
        offset_dictionary = self.get_offset_dictionary(offsets)
        print("offset_dictionary", offset_dictionary)
        new_positions = self.get_new_positions(reference_position, offset_dictionary)
        print("new_positions", new_positions)
        for k in range(0, 5):
            if k != current_index:
                self.set_aperture(k)
                self.move_to_position(new_positions[k])
                self.save_new_values()
                self.save_to_publisher()

        self.set_aperture(current_index)

    def gauss(self, treshold=0.8):
        m = self.Z.max()
        Z = (self.Z > treshold * m) * self.Z
        params = self.fitGauss(Z)
        print("Gauss fit parameters", params)
        optimum = self.getTransformedPoint([params[1], params[2]])
        ig = int(round(params[1]))
        jg = int(round(params[2]))
        print("\nindex of max point", ig, jg)
        try:
            print("X[i,j]", self.X[ig][jg])
            print("Y[i,j]", self.Y[ig][jg])
            print("Z[i,j]", self.Z[ig][jg])
        except:
            import traceback

            print(traceback.print_exc())
        print("optimum from gauss", optimum)
        return optimum

    def com(self, treshold=0.5):
        print("\nresults from center of mass calculation")
        print("self.Z", self.Z)
        m = self.Z.max()
        print("m", m)
        Z = (self.Z > treshold * m) * self.Z
        print(Z)
        com = scipy.ndimage.center_of_mass(Z)
        print("com", com)
        i, j = com
        optimum = self.getTransformedPoint(com)
        print("i, j", i, j)
        i = int(round(i))
        j = int(round(j))
        print("index of max point", i, j)
        try:
            print("X[i,j]", self.X[i][j])
            print("Y[i,j]", self.Y[i][j])
            print("Z[i,j]", self.Z[i][j])
        except:
            import traceback

            print(traceback.print_exc())
        print("com", com)
        print("optimum", optimum)
        return optimum

    def singlemax(self):
        m = self.Z.max()
        i, j = np.unravel_index(self.Z.argmax(), self.Z.shape)
        print("index of max point", i, j)
        print("X[i,j]", self.X[i][j])
        print("Y[i,j]", self.Y[i][j])
        print("Z[i,j]", self.Z[i][j])
        return self.X[i][j], self.Y[i][j]

    def getTransform(self):
        shift1 = np.matrix(self.shift(-nbsteps / 2.0))
        scale = np.matrix(self.scale(stepsizes))
        shift2 = np.matrix(self.shift(center))
        transform = shift2 * scale * shift1
        return transform

    def getTransformedPoint(self, point):
        center = np.array(self.center)
        nbsteps = np.array(self.nbsteps)
        lengths = np.array(self.lengths)
        stepsizes = lengths / nbsteps

        shift1 = self.shift(-nbsteps / 2.0)
        scale1 = self.scale(stepsizes)
        shift2 = self.shift(center)
        point = np.array([point[0], point[1], 1])

        point = np.dot(shift1, point.T).T
        point = np.dot(scale1, point.T).T
        point = np.dot(shift2, point.T).T
        return point[0], point[1]

    def XYZ(self):
        """Go through the results and return X, Y, Z matrices for 3d plots"""
        x = []
        y = []
        z = []
        value = (self.observable["device"], self.observable["attribute"])
        for item in self.xyz:
            x.append(item[self.motors[0]])
            y.append(item[self.motors[1]])
            z.append(item[value])
        print(f"self.nbsteps, {self.nbsteps}")
        X, Y, Z = [np.array(l) for l in [x, y, z]]
        X, Y, Z = [np.reshape(l, self.nbsteps.astype(np.int8)) for l in [X, Y, Z]]
        Y, Z = [self.raster(l) for l in [Y, Z]]
        return X, Y, Z

    def fitGauss(self, image):
        params = fitgaussian(image)
        return params

    def plot_surface(self, X, Y, Z):
        from mpl_toolkits.mplot3d import axes3d
        from matplotlib import cm
        from matplotlib.ticker import LinearLocator, FormatStrFormatter
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(
            X,
            Y,
            Z,
            rstride=1,
            cstride=1,
            cmap=cm.coolwarm,
            linewidth=0,
            antialiased=False,
        )
        # ax.zaxis.set_major_locator(LinearLocator(10))
        ##ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.show()

    def plot_wire_frame(self, X, Y, Z):
        from mpl_toolkits.mplot3d import axes3d
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1)

        plt.show()

    def plot_surface_wire(self, X, Y, Z, filename="resultFigure.png", stride=1):
        from mpl_toolkits.mplot3d import axes3d
        from matplotlib import cm
        import matplotlib.pyplot as pltmotors

        fig = plt.figure(filename.replace(".png", ""), figsize=plt.figaspect(0.5))
        # surface

        ax = fig.add_subplot(1, 3, 1, projection="3d", title="Grey")
        surf = ax.plot_surface(
            X,
            Y,
            Z,
            rstride=stride,
            cstride=stride,
            cmap=cm.Greys,
            linewidth=0,
            antialiased=True,
        )
        ax.view_init(elev=8.0, azim=-49.0)
        fig.colorbar(surf, shrink=0.5, aspect=15)

        ax = fig.add_subplot(1, 3, 2, projection="3d", title="Bone")
        surf = ax.plot_surface(
            X,
            Y,
            Z,
            rstride=stride,
            cstride=stride,
            cmap=cm.bone,
            linewidth=0,
            antialiased=True,
        )
        ax.view_init(elev=8.0, azim=-49.0)
        fig.colorbar(surf, shrink=0.5, aspect=15)

        # wire
        ax = fig.add_subplot(1, 3, 3, projection="3d", title="Wireframe")
        wire = ax.plot_wireframe(X, Y, Z, rstride=stride, cstride=stride)
        ax.view_init(elev=8.0, azim=-49.0)
        ## mesh
        # ax = fig.add_subplot(1, 4, 3, projection='3d', title='Wireframe')
        # wire = ax.mesh(X, Y, Z, rstride=stride, cstride=stride)

        # save and display
        plt.savefig(filename)
        plt.show()


def main():
    import optparse

    usage = "Program to perform grid scan of apertures and capillary beamstop of MD2, and find it's center with respect to the beam. The program will do the scan of 100um aperture by default."
    parser = optparse.OptionParser(usage=usage)

    parser.add_option(
        "-a",
        "--aperture",
        default=None,
        type=int,
        help="scan selected aperture (1 for 100um, 2 for 50um, 3 for 20um, 4 for 10um, 5 for 5um) (default: %default)",
    )
    parser.add_option("-c", "--capillary", action="store_true", help="scan capillary")
    parser.add_option(
        "-e",
        "--extent",
        default=None,
        help="Extent of the scan in terms of the size of the object (e.g. multiple of 100um for 100um aperture) if left unspecified reasonable predetermined defaults will be used.",
    )
    parser.add_option(
        "-s",
        "--step",
        default=None,
        help="Step size in terms of the size of the object (default: %default) ",
    )
    parser.add_option(
        "-p",
        "--shape",
        default=(1, 1),
        help="Shape of the scanned area -- horizontal X vertical (default: %default)",
    )
    parser.add_option(
        "-n",
        "--nbsteps",
        default=None,
        type=int,
        help="array of steps (horizontal X vertical) (default: %default)",
    )
    parser.add_option(
        "-l",
        "--lengths",
        default=None,
        type=float,
        help="array of lengths in mm (horizontal X vertical) (default: %default)",
    )
    parser.add_option("-S", "--snap", action="store_true", help="Save snapshot")
    parser.add_option("-D", "--display", action="store_true", help="Save snapshot")
    (options, args) = parser.parse_args()
    print(options)
    print(args)

    nbsteps = options.nbsteps
    lengths = options.lengths

    if options.capillary:
        what = "capillary"
    else:
        what = "aperture"
        # a.set_aperture(options.aperture)
    # (self, what, aperture_index=None, nbsteps=None, lengths=None, extent=None, shape=(1, 1), step=0.5, motor_device='i11-ma-cx1/ex/md3', observable={'device': 'i11-ma-cx1/ex/imag.1', 'attribute': 'image', 'economy': 'mean'})
    a = scan_and_align(
        what,
        aperture_index=options.aperture,
        nbsteps=options.nbsteps,
        lengths=options.lengths,
        extent=options.extent,
        shape=options.shape,
        step=options.step,
        snap=options.snap,
        display=options.display,
    )  # , nbsteps=nbsteps, lengths=lengths)

    print("scanning", a.getLongName())
    print("a.scan(nbsteps, lengths)", nbsteps, lengths)
    a.scan()
    # self.XYZ()
    a.align(optimum="com")
    a.save_scan()

    if a.what == "aperture":
        a.predict()
    if a.snap is True:
        os.system("getSnap.py -s -m")
    print("The scan took", a.results["duration"], "seconds")


if __name__ == "__main__":
    main()
