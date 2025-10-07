#https://www.foundationsafety.com/python-project-structure
#from project_name.main import some_function
#from project_name.subdirectory.in_subdir import some_other_function

#__all__ = ["some_function", "some_other_function"]

from experimental_methods.utils.history_saver import get_jpegs_from_arrays
from experimental_methods.utils.speech import speech, defer
from experimental_methods.utils.mdbroker import main as mdbroker_cli
from experimental_methods.instrument.goniometer import goniometer
from experimental_methods.instrument.speaking_goniometer import speaking_goniometer, main as speaking_goniometer_cli
from experimental_methods.instrument.detector import detector
from experimental_methods.instrument.oav_camera import oav_camera, main as oav_camera_cli
from experimental_methods.instrument.axis_stream import axis_camera, main as axis_stream_cli
from experimental_methods.instrument.camera import camera
from experimental_methods.instrument.cats import cats, dewar_content
from experimental_methods.instrument.transmission import transmission, transmission_mockup, main as transmission_cli
from experimental_methods.instrument.resolution import resolution, resolution_mockup
from experimental_methods.instrument.energy import energy, energy_mockup
from experimental_methods.instrument.flux import flux, flux_mockup
from experimental_methods.instrument.machine_status import machine_status, machine_status_mockup
from experimental_methods.instrument.experimental_table import experimental_table
from experimental_methods.instrument.cryostream import cryostream
from experimental_methods.instrument.beam_center import beam_center, beam_center_mockup
from experimental_methods.instrument.frontend_shutter import frontend_shutter
from experimental_methods.instrument.safety_shutter import safety_shutter
from experimental_methods.instrument.fast_shutter import fast_shutter
from experimental_methods.instrument.motor import monochromator_rx_motor
from experimental_methods.instrument.slits import slits1, slits2

from experimental_methods.utils.image_monitor import main as image_monitor_cli
from experimental_methods.utils.anneal import anneal
from experimental_methods.utils.raddose import raddose

from experimental_methods.experiment.experiment import experiment
from experimental_methods.experiment.beam_align import beam_align, main as beam_align_cli
from experimental_methods.experiment.slit_scan import slit_scan, main as slit_scan_cli
from experimental_methods.experiment.scan_and_align import scan_and_align
from experimental_methods.experiment.optical_alignment import optical_alignment, main as optical_alignment_cli
from experimental_methods.experiment.omega_scan import omega_scan, main as omega_scan_cli
from experimental_methods.experiment.inverse_scan import inverse_scan, main as inverse_scan_cli
from experimental_methods.experiment.reference_images import reference_images, main as reference_images_cli
from experimental_methods.experiment.helical_scan import helical_scan, main as helical_scan_cli
from experimental_methods.experiment.raster_scan import raster_scan, main as raster_scan_cli
from experimental_methods.experiment.diffraction_tomography import diffraction_tomography, main as diffraction_tomography_cli
from experimental_methods.experiment.nested_helical_acquisition import nested_helical_acquisition, main as nested_helical_acquisition_cli
from experimental_methods.experiment.tomography import tomography
from experimental_methods.experiment.film import film
from experimental_methods.experiment.fluorescence_spectrum import fluorescence_spectrum, main as fluorescence_spectrum_cli
from experimental_methods.experiment.energy_scan import energy_scan, main as energy_scan_cli
from experimental_methods.experiment.mount import mount, main as mount_cli


__all__ = [
    "speech",
    "defer",
    "goniometer", 
    "detector", 
    "oav_camera", 
    "camera",
    "cats",
    "dewar_content",
    "transmission", "transmission_mockup",
    "resolution", "resolution_mockup",
    "energy", "energy_mockup",
    
    "flux", "flux_mockup",
    "machine_status", "machine_status_mockup",
    "experimental_table",
    "cryostream",
    "beam_center", "beam_center_mockup",
    "frontend_shutter",
    "safety_shutter",
    "fast_shutter",
    "monochromator_rx_motor",
    "slits1", "slits2",
    
    "anneal", 
    "raddose",
    
    "experiment",
    "beam_align",
    "slit_scan",
    "scan_and_align", 
    "optical_alignment", 
    "omega_scan",
    "helical_scan",
    "raster_scan",
    "diffraction_tomography",
    "nested_helical_acquisition",
    "tomography",
    "film",
    "fluorescence_spectrum",
    "energy_scan",
    "mount"
]    
