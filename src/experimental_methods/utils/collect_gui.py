# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'collect.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:

    def _fromUtf8(s):
        return s


try:
    _encoding = QtGui.QApplication.UnicodeUTF8

    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)

except AttributeError:

    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)


class Ui_Collect(object):
    def setupUi(self, Collect):
        Collect.setObjectName(_fromUtf8("Collect"))
        Collect.resize(944, 792)
        Collect.setToolTip(_fromUtf8(""))
        self.formLayoutWidget = QtGui.QWidget(Collect)
        self.formLayoutWidget.setGeometry(QtCore.QRect(20, 90, 291, 241))
        self.formLayoutWidget.setObjectName(_fromUtf8("formLayoutWidget"))
        self.primary_input = QtGui.QFormLayout(self.formLayoutWidget)
        self.primary_input.setSizeConstraint(QtGui.QLayout.SetNoConstraint)
        self.primary_input.setFieldGrowthPolicy(QtGui.QFormLayout.AllNonFixedFieldsGrow)
        self.primary_input.setObjectName(_fromUtf8("primary_input"))
        self.rangeLabel = QtGui.QLabel(self.formLayoutWidget)
        self.rangeLabel.setObjectName(_fromUtf8("rangeLabel"))
        self.primary_input.setWidget(1, QtGui.QFormLayout.LabelRole, self.rangeLabel)
        self.rangeLineEdit = QtGui.QLineEdit(self.formLayoutWidget)
        self.rangeLineEdit.setObjectName(_fromUtf8("rangeLineEdit"))
        self.primary_input.setWidget(1, QtGui.QFormLayout.FieldRole, self.rangeLineEdit)
        self.slicingLabel = QtGui.QLabel(self.formLayoutWidget)
        self.slicingLabel.setObjectName(_fromUtf8("slicingLabel"))
        self.primary_input.setWidget(2, QtGui.QFormLayout.LabelRole, self.slicingLabel)
        self.slicingLineEdit = QtGui.QLineEdit(self.formLayoutWidget)
        self.slicingLineEdit.setObjectName(_fromUtf8("slicingLineEdit"))
        self.primary_input.setWidget(
            2, QtGui.QFormLayout.FieldRole, self.slicingLineEdit
        )
        self.startLabel = QtGui.QLabel(self.formLayoutWidget)
        self.startLabel.setObjectName(_fromUtf8("startLabel"))
        self.primary_input.setWidget(3, QtGui.QFormLayout.LabelRole, self.startLabel)
        self.startLineEdit = QtGui.QLineEdit(self.formLayoutWidget)
        self.startLineEdit.setObjectName(_fromUtf8("startLineEdit"))
        self.primary_input.setWidget(3, QtGui.QFormLayout.FieldRole, self.startLineEdit)
        self.exposureLabel = QtGui.QLabel(self.formLayoutWidget)
        self.exposureLabel.setObjectName(_fromUtf8("exposureLabel"))
        self.primary_input.setWidget(4, QtGui.QFormLayout.LabelRole, self.exposureLabel)
        self.exposureLineEdit = QtGui.QLineEdit(self.formLayoutWidget)
        self.exposureLineEdit.setObjectName(_fromUtf8("exposureLineEdit"))
        self.primary_input.setWidget(
            4, QtGui.QFormLayout.FieldRole, self.exposureLineEdit
        )
        self.energyLabel = QtGui.QLabel(self.formLayoutWidget)
        self.energyLabel.setObjectName(_fromUtf8("energyLabel"))
        self.primary_input.setWidget(5, QtGui.QFormLayout.LabelRole, self.energyLabel)
        self.energyLineEdit = QtGui.QLineEdit(self.formLayoutWidget)
        self.energyLineEdit.setObjectName(_fromUtf8("energyLineEdit"))
        self.primary_input.setWidget(
            5, QtGui.QFormLayout.FieldRole, self.energyLineEdit
        )
        self.transmissionLabel = QtGui.QLabel(self.formLayoutWidget)
        self.transmissionLabel.setObjectName(_fromUtf8("transmissionLabel"))
        self.primary_input.setWidget(
            6, QtGui.QFormLayout.LabelRole, self.transmissionLabel
        )
        self.transmissionLineEdit = QtGui.QLineEdit(self.formLayoutWidget)
        self.transmissionLineEdit.setObjectName(_fromUtf8("transmissionLineEdit"))
        self.primary_input.setWidget(
            6, QtGui.QFormLayout.FieldRole, self.transmissionLineEdit
        )
        self.resolutionLabel = QtGui.QLabel(self.formLayoutWidget)
        self.resolutionLabel.setObjectName(_fromUtf8("resolutionLabel"))
        self.primary_input.setWidget(
            7, QtGui.QFormLayout.LabelRole, self.resolutionLabel
        )
        self.resolutionLineEdit = QtGui.QLineEdit(self.formLayoutWidget)
        self.resolutionLineEdit.setObjectName(_fromUtf8("resolutionLineEdit"))
        self.primary_input.setWidget(
            7, QtGui.QFormLayout.FieldRole, self.resolutionLineEdit
        )
        self.button_collect = QtGui.QPushButton(Collect)
        self.button_collect.setGeometry(QtCore.QRect(780, 390, 99, 27))
        self.button_collect.setObjectName(_fromUtf8("button_collect"))
        self.formLayoutWidget_2 = QtGui.QWidget(Collect)
        self.formLayoutWidget_2.setGeometry(QtCore.QRect(20, 10, 581, 61))
        self.formLayoutWidget_2.setObjectName(_fromUtf8("formLayoutWidget_2"))
        self.common_input = QtGui.QFormLayout(self.formLayoutWidget_2)
        self.common_input.setObjectName(_fromUtf8("common_input"))
        self.directoryLabel = QtGui.QLabel(self.formLayoutWidget_2)
        self.directoryLabel.setObjectName(_fromUtf8("directoryLabel"))
        self.common_input.setWidget(0, QtGui.QFormLayout.LabelRole, self.directoryLabel)
        self.directoryLineEdit = QtGui.QLineEdit(self.formLayoutWidget_2)
        self.directoryLineEdit.setObjectName(_fromUtf8("directoryLineEdit"))
        self.common_input.setWidget(
            0, QtGui.QFormLayout.FieldRole, self.directoryLineEdit
        )
        self.prefixLabel = QtGui.QLabel(self.formLayoutWidget_2)
        self.prefixLabel.setObjectName(_fromUtf8("prefixLabel"))
        self.common_input.setWidget(1, QtGui.QFormLayout.LabelRole, self.prefixLabel)
        self.prefixLineEdit = QtGui.QLineEdit(self.formLayoutWidget_2)
        self.prefixLineEdit.setObjectName(_fromUtf8("prefixLineEdit"))
        self.common_input.setWidget(1, QtGui.QFormLayout.FieldRole, self.prefixLineEdit)
        self.formLayoutWidget_3 = QtGui.QWidget(Collect)
        self.formLayoutWidget_3.setGeometry(QtCore.QRect(320, 90, 281, 241))
        self.formLayoutWidget_3.setObjectName(_fromUtf8("formLayoutWidget_3"))
        self.secondary_input = QtGui.QFormLayout(self.formLayoutWidget_3)
        self.secondary_input.setObjectName(_fromUtf8("secondary_input"))
        self.nimagesLabel = QtGui.QLabel(self.formLayoutWidget_3)
        self.nimagesLabel.setObjectName(_fromUtf8("nimagesLabel"))
        self.secondary_input.setWidget(
            0, QtGui.QFormLayout.LabelRole, self.nimagesLabel
        )
        self.nimagesLineEdit = QtGui.QLineEdit(self.formLayoutWidget_3)
        self.nimagesLineEdit.setObjectName(_fromUtf8("nimagesLineEdit"))
        self.secondary_input.setWidget(
            0, QtGui.QFormLayout.FieldRole, self.nimagesLineEdit
        )
        self.framesPerSecondLabel = QtGui.QLabel(self.formLayoutWidget_3)
        self.framesPerSecondLabel.setObjectName(_fromUtf8("framesPerSecondLabel"))
        self.secondary_input.setWidget(
            1, QtGui.QFormLayout.LabelRole, self.framesPerSecondLabel
        )
        self.framesPerSecondLineEdit = QtGui.QLineEdit(self.formLayoutWidget_3)
        self.framesPerSecondLineEdit.setObjectName(_fromUtf8("framesPerSecondLineEdit"))
        self.secondary_input.setWidget(
            1, QtGui.QFormLayout.FieldRole, self.framesPerSecondLineEdit
        )
        self.scanSpeedLabel = QtGui.QLabel(self.formLayoutWidget_3)
        self.scanSpeedLabel.setObjectName(_fromUtf8("scanSpeedLabel"))
        self.secondary_input.setWidget(
            2, QtGui.QFormLayout.LabelRole, self.scanSpeedLabel
        )
        self.exposurePerFrameLabel = QtGui.QLabel(self.formLayoutWidget_3)
        self.exposurePerFrameLabel.setObjectName(_fromUtf8("exposurePerFrameLabel"))
        self.secondary_input.setWidget(
            3, QtGui.QFormLayout.LabelRole, self.exposurePerFrameLabel
        )
        self.exposurePerFrameLineEdit = QtGui.QLineEdit(self.formLayoutWidget_3)
        self.exposurePerFrameLineEdit.setObjectName(
            _fromUtf8("exposurePerFrameLineEdit")
        )
        self.secondary_input.setWidget(
            3, QtGui.QFormLayout.FieldRole, self.exposurePerFrameLineEdit
        )
        self.wavelengthLabel = QtGui.QLabel(self.formLayoutWidget_3)
        self.wavelengthLabel.setObjectName(_fromUtf8("wavelengthLabel"))
        self.secondary_input.setWidget(
            4, QtGui.QFormLayout.LabelRole, self.wavelengthLabel
        )
        self.wavelengthLineEdit = QtGui.QLineEdit(self.formLayoutWidget_3)
        self.wavelengthLineEdit.setObjectName(_fromUtf8("wavelengthLineEdit"))
        self.secondary_input.setWidget(
            4, QtGui.QFormLayout.FieldRole, self.wavelengthLineEdit
        )
        self.fluxLabel = QtGui.QLabel(self.formLayoutWidget_3)
        self.fluxLabel.setObjectName(_fromUtf8("fluxLabel"))
        self.secondary_input.setWidget(5, QtGui.QFormLayout.LabelRole, self.fluxLabel)
        self.fluxLineEdit = QtGui.QLineEdit(self.formLayoutWidget_3)
        self.fluxLineEdit.setObjectName(_fromUtf8("fluxLineEdit"))
        self.secondary_input.setWidget(
            5, QtGui.QFormLayout.FieldRole, self.fluxLineEdit
        )
        self.distanceLabel = QtGui.QLabel(self.formLayoutWidget_3)
        self.distanceLabel.setObjectName(_fromUtf8("distanceLabel"))
        self.secondary_input.setWidget(
            6, QtGui.QFormLayout.LabelRole, self.distanceLabel
        )
        self.distanceLineEdit = QtGui.QLineEdit(self.formLayoutWidget_3)
        self.distanceLineEdit.setObjectName(_fromUtf8("distanceLineEdit"))
        self.secondary_input.setWidget(
            6, QtGui.QFormLayout.FieldRole, self.distanceLineEdit
        )
        self.scanSpeedLineEdit = QtGui.QLineEdit(self.formLayoutWidget_3)
        self.scanSpeedLineEdit.setText(_fromUtf8(""))
        self.scanSpeedLineEdit.setObjectName(_fromUtf8("scanSpeedLineEdit"))
        self.secondary_input.setWidget(
            2, QtGui.QFormLayout.FieldRole, self.scanSpeedLineEdit
        )
        self.widget = QtGui.QWidget(Collect)
        self.widget.setGeometry(QtCore.QRect(630, 510, 281, 251))
        self.widget.setObjectName(_fromUtf8("widget"))
        self.verticalLayout = QtGui.QVBoxLayout(self.widget)
        self.verticalLayout.setSizeConstraint(QtGui.QLayout.SetNoConstraint)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.label_fps = QtGui.QLabel(self.widget)
        self.label_fps.setObjectName(_fromUtf8("label_fps"))
        self.verticalLayout.addWidget(self.label_fps)
        self.label_dpf = QtGui.QLabel(self.widget)
        self.label_dpf.setObjectName(_fromUtf8("label_dpf"))
        self.verticalLayout.addWidget(self.label_dpf)
        self.label_dps = QtGui.QLabel(self.widget)
        self.label_dps.setObjectName(_fromUtf8("label_dps"))
        self.verticalLayout.addWidget(self.label_dps)
        self.label_detector_distance = QtGui.QLabel(self.widget)
        self.label_detector_distance.setObjectName(_fromUtf8("label_detector_distance"))
        self.verticalLayout.addWidget(self.label_detector_distance)
        self.label_wavelength = QtGui.QLabel(self.widget)
        self.label_wavelength.setObjectName(_fromUtf8("label_wavelength"))
        self.verticalLayout.addWidget(self.label_wavelength)
        self.label_epf = QtGui.QLabel(self.widget)
        self.label_epf.setObjectName(_fromUtf8("label_epf"))
        self.verticalLayout.addWidget(self.label_epf)
        self.label_flux = QtGui.QLabel(self.widget)
        self.label_flux.setObjectName(_fromUtf8("label_flux"))
        self.verticalLayout.addWidget(self.label_flux)
        self.label_nimages = QtGui.QLabel(self.widget)
        self.label_nimages.setObjectName(_fromUtf8("label_nimages"))
        self.verticalLayout.addWidget(self.label_nimages)
        self.widget1 = QtGui.QWidget(Collect)
        self.widget1.setGeometry(QtCore.QRect(610, 10, 321, 311))
        self.widget1.setObjectName(_fromUtf8("widget1"))
        self.verticalLayout_2 = QtGui.QVBoxLayout(self.widget1)
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))
        self.label_directory = QtGui.QLabel(self.widget1)
        self.label_directory.setObjectName(_fromUtf8("label_directory"))
        self.verticalLayout_2.addWidget(self.label_directory)
        self.label_name_pattern = QtGui.QLabel(self.widget1)
        self.label_name_pattern.setObjectName(_fromUtf8("label_name_pattern"))
        self.verticalLayout_2.addWidget(self.label_name_pattern)
        self.label_scan_range = QtGui.QLabel(self.widget1)
        self.label_scan_range.setObjectName(_fromUtf8("label_scan_range"))
        self.verticalLayout_2.addWidget(self.label_scan_range)
        self.label_angle_per_frame = QtGui.QLabel(self.widget1)
        self.label_angle_per_frame.setObjectName(_fromUtf8("label_angle_per_frame"))
        self.verticalLayout_2.addWidget(self.label_angle_per_frame)
        self.label_scan_start_angle = QtGui.QLabel(self.widget1)
        self.label_scan_start_angle.setObjectName(_fromUtf8("label_scan_start_angle"))
        self.verticalLayout_2.addWidget(self.label_scan_start_angle)
        self.label_scan_exposure_time = QtGui.QLabel(self.widget1)
        self.label_scan_exposure_time.setObjectName(
            _fromUtf8("label_scan_exposure_time")
        )
        self.verticalLayout_2.addWidget(self.label_scan_exposure_time)
        self.label_photon_energy = QtGui.QLabel(self.widget1)
        self.label_photon_energy.setObjectName(_fromUtf8("label_photon_energy"))
        self.verticalLayout_2.addWidget(self.label_photon_energy)
        self.label_transmission = QtGui.QLabel(self.widget1)
        self.label_transmission.setObjectName(_fromUtf8("label_transmission"))
        self.verticalLayout_2.addWidget(self.label_transmission)
        self.label_resolution = QtGui.QLabel(self.widget1)
        self.label_resolution.setObjectName(_fromUtf8("label_resolution"))
        self.verticalLayout_2.addWidget(self.label_resolution)

        self.retranslateUi(Collect)
        QtCore.QMetaObject.connectSlotsByName(Collect)

    def retranslateUi(self, Collect):
        Collect.setWindowTitle(_translate("Collect", "Collect", None))
        self.rangeLabel.setText(_translate("Collect", "range", None))
        self.rangeLineEdit.setToolTip(_translate("Collect", "total scan range", None))
        self.rangeLineEdit.setStatusTip(
            _translate("Collect", "Set the total scan range in degrees", None)
        )
        self.rangeLineEdit.setPlaceholderText(
            _translate("Collect", "total scan range in degrees", None)
        )
        self.slicingLabel.setText(_translate("Collect", "slicing", None))
        self.slicingLineEdit.setToolTip(
            _translate("Collect", "oscillation per frame", None)
        )
        self.slicingLineEdit.setStatusTip(
            _translate("Collect", "Set the slicing -- degrees per frame", None)
        )
        self.slicingLineEdit.setPlaceholderText(
            _translate("Collect", "degrees per frame", None)
        )
        self.startLabel.setText(_translate("Collect", "start", None))
        self.startLineEdit.setToolTip(_translate("Collect", "scan start angle", None))
        self.startLineEdit.setStatusTip(
            _translate("Collect", "Set the scan start angle", None)
        )
        self.startLineEdit.setPlaceholderText(
            _translate("Collect", "scan start angle in degrees", None)
        )
        self.exposureLabel.setText(_translate("Collect", "exposure", None))
        self.exposureLineEdit.setToolTip(
            _translate("Collect", "total exposure time in s", None)
        )
        self.exposureLineEdit.setStatusTip(
            _translate("Collect", "Set the scan exposure time in seconds", None)
        )
        self.exposureLineEdit.setPlaceholderText(
            _translate("Collect", "total scan exposure in s", None)
        )
        self.energyLabel.setText(_translate("Collect", "energy", None))
        self.energyLineEdit.setToolTip(
            _translate("Collect", "photon energy in keV", None)
        )
        self.energyLineEdit.setStatusTip(
            _translate("Collect", "Set the photon energy in keV", None)
        )
        self.energyLineEdit.setPlaceholderText(
            _translate("Collect", "energy in keV", None)
        )
        self.transmissionLabel.setText(_translate("Collect", "transmission", None))
        self.transmissionLineEdit.setToolTip(
            _translate("Collect", "set transmission", None)
        )
        self.transmissionLineEdit.setStatusTip(
            _translate("Collect", "Set transmission in % of maximum flux", None)
        )
        self.transmissionLineEdit.setPlaceholderText(
            _translate("Collect", "transmission in %", None)
        )
        self.resolutionLabel.setText(_translate("Collect", "resolution", None))
        self.resolutionLineEdit.setToolTip(
            _translate("Collect", "set resolution", None)
        )
        self.resolutionLineEdit.setStatusTip(
            _translate("Collect", "Set the desired resolution in Angrstroems", None)
        )
        self.resolutionLineEdit.setPlaceholderText(
            _translate("Collect", "desired resolution in A", None)
        )
        self.button_collect.setText(_translate("Collect", "collect", None))
        self.directoryLabel.setText(_translate("Collect", "directory", None))
        self.directoryLineEdit.setPlaceholderText(
            _translate("Collect", "destination path name", None)
        )
        self.prefixLabel.setText(_translate("Collect", "prefix", None))
        self.prefixLineEdit.setPlaceholderText(
            _translate("Collect", "file prefix", None)
        )
        self.nimagesLabel.setText(_translate("Collect", "nimages", None))
        self.nimagesLineEdit.setPlaceholderText(
            _translate("Collect", "number of images", None)
        )
        self.framesPerSecondLabel.setText(
            _translate("Collect", "acquisition rate", None)
        )
        self.framesPerSecondLineEdit.setPlaceholderText(
            _translate("Collect", "frames per second", None)
        )
        self.scanSpeedLabel.setText(_translate("Collect", "scan speed", None))
        self.exposurePerFrameLabel.setText(
            _translate("Collect", "exposure per frame", None)
        )
        self.exposurePerFrameLineEdit.setPlaceholderText(
            _translate("Collect", "time per frame", None)
        )
        self.wavelengthLabel.setText(_translate("Collect", "wavelength", None))
        self.wavelengthLineEdit.setPlaceholderText(
            _translate("Collect", "wavelength in A", None)
        )
        self.fluxLabel.setText(_translate("Collect", "flux", None))
        self.fluxLineEdit.setPlaceholderText(
            _translate("Collect", "flux in ph/s", None)
        )
        self.distanceLabel.setText(_translate("Collect", "detector distance", None))
        self.distanceLineEdit.setPlaceholderText(
            _translate("Collect", "detector distance", None)
        )
        self.scanSpeedLineEdit.setPlaceholderText(
            _translate("Collect", "degrees per second", None)
        )
        self.label_fps.setText(_translate("Collect", "frames per second:", None))
        self.label_dpf.setText(_translate("Collect", "degrees per frame:", None))
        self.label_dps.setText(_translate("Collect", "degrees per second:", None))
        self.label_detector_distance.setText(
            _translate("Collect", "detector distance:", None)
        )
        self.label_wavelength.setText(_translate("Collect", "wavelength:", None))
        self.label_epf.setText(_translate("Collect", "exposure per frame:", None))
        self.label_flux.setText(_translate("Collect", "flux:", None))
        self.label_nimages.setText(_translate("Collect", "nimages:", None))
        self.label_directory.setText(_translate("Collect", "directory: ", None))
        self.label_name_pattern.setText(_translate("Collect", "name_pattern:", None))
        self.label_scan_range.setText(_translate("Collect", "scan_range:", None))
        self.label_angle_per_frame.setText(
            _translate("Collect", "angle_per_frame:", None)
        )
        self.label_scan_start_angle.setText(
            _translate("Collect", "scan_start_angle:", None)
        )
        self.label_scan_exposure_time.setText(
            _translate("Collect", "scan_exposure_time:", None)
        )
        self.label_photon_energy.setText(_translate("Collect", "photon_energy:", None))
        self.label_transmission.setText(_translate("Collect", "transmission:", None))
        self.label_resolution.setText(_translate("Collect", "resolution:", None))
