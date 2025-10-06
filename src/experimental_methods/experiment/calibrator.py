#!/usr/bin/python
# -*- coding: utf-8 -*-

import PyTango
import time
import gauss2d
import pylab
import numpy
import pickle
import os
import math
import scipy.ndimage
import collect as Collect

# import TangoLimaVideo
import struct


class calibrator(object):
    therm_attributes = [
        "i11-ma-c00/ex/tc.1/temperature",
        "i11-ma-c04/ex/tc.1/temperature",
        "i11-ma-c05/ex/tc.1/temperature",
        "i11-ma-c06/ex/tc.1/temperature",
        "i11-ma-cx1/ex/tc.1/temperature",
        "i11-ma-cx1/ex/tc.2/temperature",
    ]

    beamposition_attributes = [
        "Zoom1_X",
        "Zoom1_Z",
        "Zoom2_X",
        "Zoom2_Z",
        "Zoom3_X",
        "Zoom3_Z",
        "Zoom4_X",
        "Zoom4_Z",
        "Zoom5_X",
        "Zoom5_Z",
        "Zoom6_X",
        "Zoom6_Z",
        "Zoom7_X",
        "Zoom7_Z",
        "Zoom8_X",
        "Zoom8_Z",
        "Zoom9_X",
        "Zoom9_Z",
        "Zoom10_X",
        "Zoom10_Z",
    ]

    aperture_cpbs_attributes = [
        "ApertureHorizontalPosition",
        "ApertureVerticalPosition",
        "CapillaryHorizontalPosition",
        "CapillaryVerticalPosition",
    ]

    def __init__(
        self, fresh=True, save=True, temperature=True, aperture=True, cpbs=True
    ):
        self.md = PyTango.DeviceProxy("i11-ma-cx1/ex/md3")
        self.imag = PyTango.DeviceProxy("i11-ma-cx1/ex/imag.1")

        # self.lima = TangoLimaVideo.TangoLimaVideo()
        # self.lima.tangoname = 'lima/limaccd/1'
        self.lima = PyTango.DeviceProxy("lima/limaccd/1")

        self.mdbp = PyTango.DeviceProxy("i11-ma-cx1/ex/md3-beamposition")
        self.thermometres = dict(
            [(therm, PyTango.AttributeProxy(therm)) for therm in self.therm_attributes]
        )
        self.zooms = list(range(1, 11))
        self.collect = Collect.collect()
        # exposures more appropriate with lower flux: ~5% of transmission
        # self.exposures = {1: 0.021,
        # 2: 0.021,
        # 3: 0.021,
        # 4: 0.021,
        # 5: 0.021,
        # 6: 0.021,
        # 7: 0.021,
        # 8: 0.021,
        # 9: 0.021,
        # 10: 0.021}

        self.exposures = {
            1: 0.005,
            2: 0.003,
            3: 0.002,
            4: 0.002,
            5: 0.002,
            6: 0.002,
            7: 0.002,
            8: 0.002,
            9: 0.003,
            10: 0.003,
        }

        # Chip mode (without cryo)
        # self.exposures = {1: 0.01,
        # 2: 0.01,
        # 3: 0.01,
        # 4: 0.01,
        # 5: 0.02,
        # 6: 0.02,
        # 7: 0.02,
        # 8: 0.05,
        # 9: 0.05,
        # 10: 0.05}

        # 8 bunch
        # self.exposures = {1: 0.05,
        # 2: 0.01,
        # 3: 0.01,
        # 4: 0.01,
        # 5: 0.01,
        # 6: 0.01,
        # 7: 0.01,
        # 8: 0.01,
        # 9: 0.01,
        # 10: 0.01}
        # 1 bunch
        # self.exposures = {1: 0.21,
        # 2: 0.1,
        # 3: 0.1,
        # 4: 0.1,
        # 5: 0.1,
        # 6: 0.1,
        # 7: 0.2,
        # 8: 0.2,
        # 9: 0.25,
        # 10: 0.25}
        # self.exposures = {1: 1,
        # 2: 1,
        # 3: 1,
        # 4: 1,
        # 5: 1,
        # 6: 1,
        # 7: 1,
        # 8: 1,
        # 9: 1,
        # 10: 1}
        self.fresh = fresh
        self.save = save
        self.images = {}
        self.cData = {}
        self.beamposition = {}
        self.infostore = {}
        self.timestamp = time.time()

    def wait(self, device):
        while device.state().name in ["MOVING", "RUNNING"]:
            time.sleep(0.1)

    def wait_motor(self, motor):
        while self.md.getMotorState(motor).name != "STANDBY":
            time.sleep(0.1)
        time.sleep(self.exposures[10])

    def loadDatabase(
        self, filename="/927bis/ccd/Database/BeamPosition/beamposition.pck"
    ):
        try:
            f = open(filename)
            database = pickle.load(f)
            f.close()
        except IOError as diag:
            print(diag)
            database = {}
        return database

    def updateDatabase(
        self, filename="/927bis/ccd/Database/BeamPosition/beamposition.pck"
    ):
        return
        database = self.loadDatabase(filename)
        self.infostore["images"] = self.images
        self.infostore["cData"] = self.cData
        self.infostore["beamposition"] = self.beamposition
        database[self.timestamp] = self.infostore
        f = open(filename, "w")
        pickle.dump(database, f)
        f.close()

    def getTemplate(self):
        f = open(
            "/home/experiences/proxima2a/com-proxima2a/bin/beamposition_output_form.txt"
        )
        template = f.read()
        f.close()
        return template

    def getDictionaryToLog(self):
        toLog = []
        date = self.getDate()
        temperature = self.getTemperature()
        aperture_cpbs = self.getApertureCpbs()
        zooms = self.getZoomPositions()
        [toLog.extend(d) for d in [date, temperature, aperture_cpbs, zooms]]
        toLog = dict(toLog)
        self.infostore["toLog"] = toLog
        return toLog

    def writeTxt(self):
        template = self.getTemplate()
        toLog = self.getDictionaryToLog()
        log = template.format(**toLog)
        nicerDate = time.ctime(self.timestamp).replace(":", "-")
        fileName = (
            "/927bis/ccd/Database/BeamPosition/suiviligne_"
            + "_".join(nicerDate.split())
            + ".txt"
        )
        f = open(fileName, "w")
        f.write(log)
        f.close()
        # os.system('rsync -avz ' + fileName + ' srv2:/nfs/ruche-proxima2a/proxima2a-soleil/SuiviLigne/')

    def getDate(self):
        return [("date", time.ctime(self.timestamp))]

    def getZoomPositions(self):
        return [
            (att, self.mdbp.read_attribute(att).value)
            for att in self.beamposition_attributes
        ]

    def getTemperature(self):
        return [
            (item[0].replace(".", ""), item[-1].read().value)
            for item in list(self.thermometres.items())
        ]

    def getApertureCpbs(self):
        return [
            (att, self.md.read_attribute(att).value)
            for att in self.aperture_cpbs_attributes
        ]

    def get_lima_image(self):
        img_data = self.lima.video_last_image
        if img_data[0] == "VIDEO_IMAGE":
            header_fmt = ">IHHqiiHHHH"
            _, ver, img_mode, frame_number, width, height, _, _, _, _ = struct.unpack(
                header_fmt, img_data[1][: struct.calcsize(header_fmt)]
            )
            raw_buffer = numpy.fromstring(img_data[1][32:], numpy.uint16)
            image = raw_buffer.reshape((height, width))
        return image

    def get_prosilica_image(self):
        return self.imag.image

    def get_image(self):
        try:
            image = self.get_prosilica_image()
        except:
            pass
        try:
            image = self.get_lima_image()
        except:
            pass
        return image

    def snapshotAtZoom(self, zoom=1):
        green = False
        self.goToZoom(zoom)
        time.sleep(self.exposures[zoom] + 0.1)
        while green == False:
            image = self.get_image()  # imag.image  # ia.inputImage
            if image.max() < 25:
                print(
                    "Max intensity in the image is rather low -- not enough light on the scintillator?!"
                )
            else:
                green = True
            time.sleep(0.02)
        # if self.fresh == True:
        # self.goToZoom(zoom)
        # time.sleep(self.exposures[zoom])
        # image = self.imag.image  # ia.inputImage
        # else:
        # a = pylab.imread(
        #'/home/proxima2/Desktop/BeamPosition/zoom_' + str(zoom) + '.png')
        # aa = pylab.mean(a, 2)
        # b = pylab.imread(
        #'/home/proxima2/Desktop/BeamPositionVide/zoom_' + str(zoom) + '_blank.png')
        # bb = pylab.mean(b, 2)
        # image = aa - bb

        if self.save:
            self.images[zoom] = image

        return image

    def goToZoom(self, zoom=1):
        # self.md.ZoomPredefinedPosition = zoom
        self.setExposure(self.exposures[zoom])
        # self.md.ZoomLevel = zoom
        self.md.coaxialcamerazoomvalue = zoom
        self.wait_motor("Zoom")

    def calibrate(self, zoom=1, close=False):
        image = self.snapshotAtZoom(zoom)
        if close is True:
            self.md.fastshutterisopen = False  # CloseFastShutter()
        # image = image * (image > 0.25 * image.max())
        if zoom in [1, 2, 3]:  # do not try to fit in zoom 1, 2 and 3
            params = [None] * 5
            self.cData[zoom] = params
            data = self.images[zoom]
            z, x = numpy.unravel_index(data.argmax(), data.shape)
        else:
            params = gauss2d.fitgaussian(image)
            self.cData[zoom] = params
            if abs(self.cData[zoom][3]) < 100.0 and abs(self.cData[zoom][4]) < 100.0:
                x = self.cData[zoom][2]
                z = self.cData[zoom][1]
            else:
                data = self.images[zoom]
                z, x = numpy.unravel_index(data.argmax(), data.shape)
        print("zoom %s" % zoom)
        print("x, y: %s %s" % (x, z))

        self.beamposition[zoom] = {"z": z, "x": x}
        self.md.write_attribute("beampositionvertical", int(z))
        self.md.write_attribute("beampositionhorizontal", int(x))

    def calibrateAll(self):
        for zoom in self.zooms:
            self.calibrate(zoom)

    def getCalibration(self, zoom=1):
        return self.cData[zoom]

    def saveImages(self):
        for zoom in self.zooms:
            try:
                pylab.imsave(
                    "zoom"
                    + str(zoom)
                    + "_"
                    + "_".join(time.asctime().split())
                    + ".png",
                    self.images[zoom],
                )
            except KeyError as diag:
                print(diag)
                print("Can't save image at zoom %s, it's not in the dictionary." % zoom)

    def updateMD2BeamPositions(self):
        for zoom in self.zooms:
            self.mdbp.write_attribute(
                "Zoom" + str(zoom) + "_X", self.beamposition[zoom]["x"]
            )
            self.mdbp.write_attribute(
                "Zoom" + str(zoom) + "_Z", self.beamposition[zoom]["z"]
            )

    def setExposure(self, exposure):
        self.imag.exposure = exposure

    def setFP(self):
        fp = PyTango.DeviceProxy("passerelle/oh/fp")
        fent_h1 = PyTango.DeviceProxy("i11-ma-c02/ex/fent_h.1")
        fent_v1 = PyTango.DeviceProxy("i11-ma-c02/ex/fent_v.1")
        fent_h1.gap = fp.fp_hfmfield
        fent_v1.gap = fp.result2

    def setPhase(self, phase_number):
        self.md.PhasePosition = phase_number
        self.wait(self.md)

    def get_values_zoom_10(self):
        x = self.mdbp.Zoom10_X
        y = self.mdbp.Zoom10_Z
        return numpy.array((x, y))

    def prepare(self):
        self.old_values_zoom_10 = self.get_values_zoom_10()

        self.collect.test = False
        # self.collect.setEnergy(12.65)
        print("Moving scintillator into beam ...")
        # self.setPhase(2)
        # self.md.scintillatorpdverticalposition += 0.95
        print("scintillator set in the beam")
        self.startTransmission = self.transmission()
        self.setFP()
        # self.transmission(95.)
        while self.collect.mono_mt_rx.state().name != "OFF":
            self.collect.safeTurnOff(self.collect.mono_mt_rx)
            time.sleep(0.1)
        while self.collect.mono_mt_rx_fine.state().name != "OFF":
            self.collect.safeTurnOff(self.collect.mono_mt_rx_fine)
            time.sleep(0.1)
        self.collect.openSafetyShutter()
        self.md.FrontLightIsOn = False
        self.md.fastshutterisopen = True  # OpenFastShutter()

    def tidy(self):
        self.md.fastshutterisopen = False  # CloseFastShutter()
        self.new_values_zoom_10 = self.get_values_zoom_10()
        # self.collect.mono_mt_rx.On()
        # self.transmission(self.startTransmission)
        # self.collect.closeSafetyShutter()
        self.setExposure(0.050)

    def get_difference_zoom_10(self):
        calib = numpy.array((0.000338, 0.000341))
        diff = self.new_values_zoom_10 - self.old_values_zoom_10
        return diff * calib * 1000

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

    def saveSnap(self):
        os.system("getSnap.py -m -s")


def main(calib, zoom):
    calib.calibrate(zoom)

    data = calib.images[zoom]

    # noise threshold at 10
    data = (data > 25 * 1) * data

    i, j = numpy.unravel_index(data.argmax(), data.shape)

    print("Zoom " + str(zoom) + " from max:", j, i)

    params = calib.getCalibration(zoom)

    # fit = gauss2d.gaussian(*params)

    (height, x, y, width_x, width_y) = params
    print("Zoom " + str(zoom) + " from com:", scipy.ndimage.center_of_mass(data)[::-1])

    print("Zoom " + str(zoom) + " from fit:", y, x)
    print(params)
    print()


if __name__ == "__main__":
    import os
    import optparse

    usage = "usage: %prog --snap"
    parser = optparse.OptionParser(usage=usage)
    parser.add_option(
        "-s", "--snap", action="store_true", help="Take snapshot when finished."
    )
    (options, args) = parser.parse_args()
    print("options, args", options, args)

    calib = calibrator(fresh=True, save=True)
    calib.prepare()
    for zoom in calib.zooms:
        main(calib, zoom)
    calib.updateDatabase()

    if options.snap is True:
        calib.saveSnap()

    calib.tidy()
    calib.updateMD2BeamPositions()
    calib.writeTxt()
