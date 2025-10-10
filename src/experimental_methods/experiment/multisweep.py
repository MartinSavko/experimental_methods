#!/usr/bin/env python
# conding: utf-8

from experimental_methods.experiment.diffraction_experiment import diffraction_experiment
from experimental_methods.experiment.omega_scan import omega_scan

"""
option 1: overload parameters, 
    if list treat it as per sweep indication
    if scalar treat it as valid per all sweeps
    if only scalar it becomes a simple omega_scan

option 2: 
    list of strategies (it might correspond roughly to a data collection group of MXCuBE)
    strategy is defined by a dictionary
    [
        {
            "name_pattern": None,
            "dictionary": None,
            "scan_start_angle": 0,
            "scan_range": 360.,
            "scan_exposure_time": 18.,
            "resolution": 1.5,
            "transmission": 15.,
            "photon_energy": 13127.,
            "angle_per_frame": 0.1,
            # position is optionaly a tuple of two dictionaries 
            # -- in this case we treat it as start and stop position of a helical scan
            "position": (
                            {  
                                "AlignmentY": 0., 
                                "AlignmentZ": 0.,
                                "CentringX": 0.6,
                                "CentringY": 0.4,
                                "Kappa": 0.,
                                "Phi": 0.,
                            },
                            {  
                                "AlignmentY": 0.2, 
                                "AlignmentZ": 0.,
                                "CentringX": 0.55,
                                "CentringY": 0.45,
                                "Kappa": 0.,
                                "Phi": 0.,
                            },
                        ),
        },
        {
            "name_pattern": None,
            "dictionary": None,
            "scan_start_angle": 0,
            "scan_range": 360.,
            "scan_exposure_time": 18.,
            "resolution": 1.5,
            "transmission": 15.,
            "photon_energy": 13127.,
            "angle_per_frame": 0.1,
            # position is optionaly a tuple of two dictionaries 
            # -- in this case we treat it as start and stop position of a helical scan
            "position": (
                            {  
                                "AlignmentY": 0., 
                                "AlignmentZ": 0.,
                                "CentringX": 0.6,
                                "CentringY": 0.4,
                                "Kappa": 47.,
                                "Phi": 90.,
                            },
                            {  
                                "AlignmentY": 0.2, 
                                "AlignmentZ": 0.,
                                "CentringX": 0.55,
                                "CentringY": 0.45,
                                "Kappa": 47.,
                                "Phi": 90.,
                            },
                        ),
        },
    ]
    Define an omega_scan object for each of the sweeps.
    within run method we will go through execution of all of the omega scans one by one or in an interleaved manner
    
    def prepare(self):
        oss = [omega_scan(**sweep) for sweep in sweeps]
        total_nimages = sum([os.get_total_images() for os in oss]
        ntriggers = sum([os.get_trigger() for os in oss]
    
    def run(self):
        # simplest case
        for os in oss:
            os.run()
    
    def clean(self):
        for os in oss:
            os.clean()

"""

class multisweep(diffraction_experiment):
    
    specific_parameter_fields = [
        {"name": "sweeps", "type": "list", "description": "sweeps"},
        {"name": "interleave_range", "type": "float", "description": "interleave_range"},
        {"name": "raster", "type": "bool", "description": "raster"},
        {"name": "nrepeats", "type": "int", "description": "nrepeats"},
        {"name": "start_position", "type": "dict", "description": "start position"},
        {"name": "end_position", "type": "dict", "description": "end position"},
        {"name": "all_positions", "type": "list", "description": "all positions"},
    ]

    def __init__(
        self,
        name_pattern,
        directory,
        sweeps,
        interleave_range=15.,
        raster=False,
    ):
    
        self.default_experiment_name = "Multi sweep experiment"
        
        diffraction_experiment.__init__(
            self,
            name_pattern=name_pattern,
            directory=directory,
        )

        self.sweeps = sweeps
        
    def prepare(self):
        self.oss = [omega_scan(**sweep) for sweep in self.sweeps]
        total_images = sum([os.get_total_images() for os in oss]
        nimages = sum([os.get_images() for os in oss]
        ntriggers = sum([os.get_trigger() for os in oss]
    
    def run(self):
        # simplest case
        for os in self.oss:
            os.run()
    
    def clean(self):
        for os in oss:
            os.clean()
    
    
        
