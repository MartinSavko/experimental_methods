#https://www.foundationsafety.com/python-project-structure
#from project_name.main import some_function
#from project_name.subdirectory.in_subdir import some_other_function

#__all__ = ["some_function", "some_other_function"]

from experimental_methods.instrument.goniometer import goniometer
from experimental_methods.instrument.detector import detector
from experimental_methods.instrument.oav_camera import oav_camera
from experimental_methods.instrument.camera import camera
from experimental_methods.instrument.cats import cats, dewar_content
from experimental_methods.instrument.flux import flux
from experimental_methods.instrument.transmission import transmission
from experimental_methods.instrument.energy import energy
from experimental_methods.instrument.resolution import resolution, resolution_mockup

from experimental_methods.experiment.beam_align import beam_align
from experimental_methods.experiment.scan_and_align import scan_and_align
from experimental_methods.experiment.optical_alignment import optical_alignment
from experimental_methods.experiment.anneal import anneal
from experimental_methods.experiment.diffraction_tomography import diffraction_tomography
from experimental_methods.experiment.omega_scan import omega_scan
from experimental_methods.experiment.helical_scan import helical_scan
from experimental_methods.experiment.raster_scan import raster_scan
from experimental_methods.experiment.fluorescence_spectrum import fluorescence_spectrum
from experimental_methods.experiment.energy_scan import energy_scan

from experimental_methods.utils.history_saver import get_jpegs_from_arrays
from experimental_methods.utils.speech import speech

__all__ = [
    "goniometer", 
    "detector", 
    "oav_camera", 
    "camera",
    "cats",
    "dewar_content",
    "flux",
    "transmission",
    "energy",
    "resolution",
    "resolution_mockup",
    "speech",
    "beam_align", 
    "scan_and_align", 
    "optical_alignment", 
    "anneal", 
    "diffraction_tomography",
    "omega_scan",
    "helical_scan",
    "raster_scan",
    "fluorescence_spectrum",
    "energy_scan",
]    
