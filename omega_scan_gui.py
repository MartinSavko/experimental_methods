from PyQt4.QtGui import *
from PyQt4.QtCore import *

import sys

sys.path.insert(0, '/home/smartin/ResearchEducationDevelopment/eiger')
sys.path.insert(0, '/home/smartin/ResearchEducationDevelopment/learn_pyqt')

from omega_scan import omega_scan
from collect_gui import Ui_Collect

class collect_interface(QDialog, Ui_Collect):

    def __init__(self, parent=None, directory='/spool/2017_Run4', name_pattern='collect_$id'):
        super(collect_interface, self).__init__(parent)

        self.setupUi(self)

        self.osc = omega_scan(name_pattern, directory)
                
        self.directoryLineEdit.setText(self.osc.get_directory())
        self.prefixLineEdit.setText(self.osc.get_name_pattern())
        self.rangeLineEdit.setText(str(self.osc.get_scan_range()))
        self.slicingLineEdit.setText(str(self.osc.get_angle_per_frame()))
        self.startLineEdit.setText(str(self.osc.get_scan_start_angle()))
        self.exposureLineEdit.setText(str(self.osc.get_scan_exposure_time()))
        self.energyLineEdit.setText(str(self.osc.get_photon_energy()))
        self.transmissionLineEdit.setText(str(self.osc.get_transmission()))
        self.resolutionLineEdit.setText(str(round(self.osc.get_resolution(), 2)))
                
        self.collect_parameters = {}
        
        self.get_collect_parameters()
        
        self.update_ui()
        self.update_labels()
        
        self.button_collect.released.connect(self.collect)
        
        self.rangeLineEdit.editingFinished.connect(self.range_change_effects)
        self.slicingLineEdit.editingFinished.connect(self.update_ui)
        self.exposureLineEdit.editingFinished.connect(self.update_ui)
        self.energyLineEdit.editingFinished.connect(self.update_ui)
        self.transmissionLineEdit.editingFinished.connect(self.update_ui)
        self.resolutionLineEdit.editingFinished.connect(self.update_ui)
        
        self.nimagesLineEdit.editingFinished.connect(self.nimages_change_effects)
        self.exposurePerFrameLineEdit.editingFinished.connect(self.exposure_per_frame_effects)
        self.wavelengthLineEdit.editingFinished.connect(self.wavelength_change_effects)
        self.fluxLineEdit.editingFinished.connect(self.flux_change_effects)
        self.distanceLineEdit.editingFinished.connect(self.detector_distance_change_effects)
        self.scanSpeedLineEdit.editingFinished.connect(self.scan_speed_change_effects)
        self.framesPerSecondLineEdit.editingFinished.connect(self.frames_per_second_change_effects)
                
        self.directoryLineEdit.textEdited.connect(self.update_labels)
        self.prefixLineEdit.textEdited.connect(self.update_labels)
        self.rangeLineEdit.textEdited.connect(self.update_labels)
        self.slicingLineEdit.textEdited.connect(self.update_labels)
        self.startLineEdit.textEdited.connect(self.update_labels)
        self.exposureLineEdit.textEdited.connect(self.update_labels)
        self.energyLineEdit.textEdited.connect(self.update_labels)
        self.transmissionLineEdit.textEdited.connect(self.update_labels)
        self.resolutionLineEdit.textEdited.connect(self.update_labels)
        
    def update_labels(self):
        self.label_directory.setText('directory: %s' % self.directoryLineEdit.text())
        self.label_name_pattern.setText('name_pattern: %s' % self.prefixLineEdit.text())
        self.label_scan_range.setText('scan_range: %s degrees' % self.rangeLineEdit.text())
        self.label_scan_start_angle.setText('scan_start_angle: %s degrees' % self.startLineEdit.text())
        self.label_angle_per_frame.setText('angle_per_frame: %s degrees' % self.slicingLineEdit.text())
        self.label_scan_exposure_time.setText('scan_exposure_time: %s s' % self.exposureLineEdit.text())
        self.label_photon_energy.setText('photon_energy: %s keV' % self.energyLineEdit.text())
        self.label_transmission.setText('transmission: %s ' % self.transmissionLineEdit.text())
        self.label_resolution.setText('resolution: %s A' % self.resolutionLineEdit.text())
                
    def range_change_effects(self):
        self.osc.set_scan_range(float(self.rangeLineEdit.text()))
        self.nimagesLineEdit.setText('%d' % self.osc.get_nimages())
        self.framesPerSecondLineEdit.setText(str(self.osc.get_fps()))
        self.scanSpeedLineEdit.setText(str(self.osc.get_dps()))
        self.exposurePerFrameLineEdit.setText(str(self.osc.get_frame_time()))
        
    def nimages_change_effects(self):
        nimages = int(self.nimagesLineEdit.text())
        slicing = float(self.slicingLineEdit.text())
        scan_range = nimages*slicing
        self.osc.set_scan_range(scan_range)
        self.rangeLineEdit.setText('%6.2f' % scan_range)
        self.range_change_effects()
        
    def exposure_per_frame_effects(self):
        nimages = int(self.nimagesLineEdit.text())
        scan_range = float(self.rangeLineEdit.text())
        exposure_per_frame = float(self.exposurePerFrameLineEdit.text())
        self.exposureLineEdit.setText('%6.2f' % (exposure_per_frame * nimages))
        self.update_ui()
    
    def wavelength_change_effects(self):
        wavelength = float(self.wavelengthLineEdit.text())
        resolution = float(self.resolutionLineEdit.text())
        photon_energy = self.osc.resolution_motor.get_energy_from_wavelength(wavelength=wavelength, resolution=resolution)
        self.energyLineEdit.setText('%6.2f' % photon_energy)
        self.update_ui()
        
    def flux_change_effects(self, toplevel=1e12):
        flux = float(self.fluxLineEdit.text())
        transmission = 100 * flux/toplevel
        self.transmissionLineEdit('%6.2f' % transmission)
        self.update_ui()
        
    def detector_distance_change_effects(self):
        detector_distance = float(self.distanceLineEdit.text())
        wavelength = float(self.wavelengthLineEdit())
        resolution = self.osc.resolution_motor.get_resolution(distance=detector_distance, wavelength=wavelength)
        self.resolutionLineEdit.setText('%6.2f' % resolution)
        self.update_ui()
        
    def scan_speed_change_effects(self):
        scan_speed = float(self.scanSpeedLineEdit.text())
        scan_range = float(self.rangeLineEdit.text())
        scan_exposure_time = scan_range/scan_speed
        self.exposureLineEdit.setText('%6.2f' % scan_exposure_time)
        self.update_ui()
        
    def frames_per_second_change_effects(self):
        frames_per_second = float(self.framesPerSecondLineEdit.text())
        #frames_per_second = scan_range/angle_per_frame/scan_exposure_time
        scan_range = float(self.rangeLineEdit.text())
        scan_exposure_time = float(self.exposureLineEdit.text())
        angle_per_frame = scan_range/frames_per_second/scan_exposure_time
        self.slicingLineEdit.setText('%6.2f' % angle_per_frame)
        self.update_ui()
        
    def update_ui(self):
        
        self.get_collect_parameters()
        
        omega_range = self.collect_parameters['range']
        slicing = self.collect_parameters['slicing']
        exposure = self.collect_parameters['exposure']
        transmission = self.collect_parameters['transmission']
        resolution = self.collect_parameters['resolution']
        energy = self.collect_parameters['energy']
                
        frames_per_second = 'frames per second: %6.2f' % (omega_range/slicing/exposure, )
        degrees_per_frame = 'degrees per frame: %6.2f ' % slicing
        degrees_per_second = 'degrees per second: %6.2f' % (omega_range/exposure, )
        detector_distance = 'detector distance: %6.2f m' % self.osc.resolution_motor.get_distance_from_resolution(resolution=resolution) #str(self.osc.detector_distance) #_from_resolution(resolution, energy)
        flux = 'flux: %.2e ph/s' % (0.01 * transmission * 1.e12, )
        exposure_per_frame = 'exposure per frame: %6.2f s' % (exposure/(omega_range/slicing),)
        wavelength = 'wavelength: %6.2f A' % self.osc.resolution_motor.get_wavelength_from_energy(energy)
        nimages = 'nimages: %d' % (omega_range/slicing,)
        
        self.label_fps.setText(frames_per_second)
        self.label_dpf.setText(degrees_per_frame)
        self.label_dps.setText(degrees_per_second)
        self.label_detector_distance.setText(detector_distance)
        self.label_flux.setText(flux)
        self.label_epf.setText(exposure_per_frame)
        self.label_wavelength.setText(wavelength)
        self.label_nimages.setText(nimages)
        
        self.nimagesLineEdit.setText('%d' % self.osc.get_nimages())
        self.framesPerSecondLineEdit.setText('%6.2f' % self.osc.get_fps())
        self.scanSpeedLineEdit.setText('%6.2f' % self.osc.get_dps())
        self.exposurePerFrameLineEdit.setText('%6.2f' % self.osc.get_frame_time())
        self.wavelengthLineEdit.setText('%6.2f' % (self.osc.resolution_motor.get_wavelength_from_energy(energy)/1e3, ))
        self.fluxLineEdit.setText('%.2e' % ( transmission * 0.01 * 1.e12))
        self.distanceLineEdit.setText('%6.2f' % self.osc.resolution_motor.get_distance_from_resolution(resolution=resolution))
        
    def get_collect_parameters(self):

        print 'directory', self.directoryLineEdit.text()
        directory = self.directoryLineEdit.text()
        print 'prefix', self.prefixLineEdit.text()
        prefix = self.prefixLineEdit.text()
        print 'omega_range', self.rangeLineEdit.text()
        omega_range = float(self.rangeLineEdit.text())
        print 'slicing',self.slicingLineEdit.text() 
        slicing = float(self.slicingLineEdit.text())
        print 'start', self.startLineEdit.text()
        start = float(self.startLineEdit.text())
        print 'exposure', self.exposureLineEdit.text()
        exposure = float(self.exposureLineEdit.text())
        print 'energy', self.energyLineEdit.text()
        energy = float(self.energyLineEdit.text())
        print 'transmission', self.transmissionLineEdit.text()
        transmission = float(self.transmissionLineEdit.text())
        print 'resolution', self.resolutionLineEdit.text()
        resolution = float(self.resolutionLineEdit.text())
        
        self.collect_parameters['directory'] = directory
        self.collect_parameters['prefix'] = prefix
        self.collect_parameters['range'] = omega_range
        self.collect_parameters['slicing'] = slicing
        self.collect_parameters['start'] = start
        self.collect_parameters['exposure'] = exposure
        self.collect_parameters['energy'] = energy
        self.collect_parameters['transmission'] = transmission
        self.collect_parameters['resolution'] = resolution
        
        print 'collect_parameters'
        print self.collect_parameters
        
        self.osc.set_directory(directory)
        self.osc.set_name_pattern(prefix)
        self.osc.set_scan_range(omega_range)
        self.osc.set_angle_per_frame(slicing)
        self.osc.set_scan_start_angle(start)
        self.osc.set_scan_exposure_time(exposure)
        self.osc.photon_energy = energy
        self.osc.transmission = transmission
        self.osc.resolution = resolution
        
    def collect(self):
                
        self.osc.execute()
        

def main():
    app = QApplication(sys.argv)
    window = collect_interface()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
