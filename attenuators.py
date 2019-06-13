#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PyTango import DeviceProxy as dp
import numpy as np
from monitor import xbpm
from mucal import mucal
from energy import energy

class attenuators:

    def __init__(self):
        
        self.energy = energy()

        self.filters_device = dp('i11-ma-c05/ex/att.1')
        self.filters_device.display_name = 'filters'
        self.filters = {"00 None": {'element': None, 'thickness': 0., 'position': 0.},
                        "01 Carbon 200um": {'element': 'C', 'thickness': 2e-2, 'position': 668.},
                        "02 Carbon 250um": {'element': 'C', 'thickness': 2.5e-2, 'position': 1336.},
                        "03 Carbon 300um": {'element': 'C', 'thickness': 3e-2, 'position': 2004.},
                        "04 Carbon 500um": {'element': 'C', 'thickness': 5e-2, 'position': 2672.},
                        "05 Carbon 1mm": {'element': 'C', 'thickness': 1e-1, 'position': 3340.},
                        "06 Carbon 2mm":  {'element': 'C', 'thickness': 2e-1, 'position': 4008.},
                        "07 Carbon 3mm": {'element': 'C', 'thickness': 3e-1, 'position': 4676.},
                        "08 Spare 1": {'element': None, 'thickness': 0., 'position': 5344.},
                        "09 Spare 2": {'element': None, 'thickness': 0., 'position': 6012.},
                        "10 Ref Fe 5um": {'element': 'Fe', 'thickness': 5e-4, 'position': 6680.},
                        "11 Ref Pt 5um": {'element': 'Pt', 'thickness': 5e-4, 'position': 7348.}}

        self.imager1 = dp('i11-ma-c02/dt/imag.1-mt_tz-pos')	
        self.xbpm1 = xbpm('i11-ma-c04/dt/xbpm_diode.1-base')
        self.xbpm3 = xbpm('i11-ma-c05/dt/xbpm_diode.3-base')
        self.xbpm5 = xbpm('i11-ma-c06/dt/xbpm_diode.5-base')
        self.psd6 = xbpm('i11-ma-c06/dt/xbpm_diode.6-base')
        
        self.imager1.filters = {'element': 'C', 'thickness': 20e-4}
        self.imager1.display_name = 'imager1'
        self.xbpm1.filters = {'element': 'Ti', 'thickness': 5e-4}
        self.xbpm1.display_name = 'xbpm1'
        self.xbpm3.filters = {'XBPM': {'element': 'Ti', 'thickness': 5e-4},
                              'CVD': {'element': 'C', 'thickness': 20e-4}}
        self.xbpm3.display_name = 'xbpm3'
        self.xbpm5.filters = {'XBPM': {'element': 'Ti', 'thickness': 5e-4},
                              'PSD': {'element': 'C', 'thickness': 20e-4}}
        self.xbpm5.display_name = 'xbpm5'
        self.psd6.filters = {'element': 'C', 'thickness': 20e-4}
        self.psd6.display_name = 'psd6'

        self.positioners = {1: self.imager1,
                            2: self.xbpm1,
                            3: self.xbpm3,
                            4: self.filters_device,
                            5: self.xbpm5,
                            6: self.psd6}
                                      

    def get_transmission_from_element_and_thickness(self, element, thickness):
        phot_e = self.energy.get_energy() * 1.e-3
        mu = mucal(element, phot_e)[1][5]
        return np.exp(-mu*thickness)

    def get_filter(self):
        return self.filters_device.selectedattributename

    def set_filter(self, filter_name):
        return setattr(self.filters_device, filter_name, True)

    def get_element_and_thickness(self, positioner):
        if positioner == self.filters_device:
           try:
              element_and_thickness = self.filters[self.get_filter()]
           except:
              pass
	elif positioner == self.imager1:
           if positioner.isInserted:
               element_and_thickness = positioner.filters
        elif positioner == self.xbpm5:
           if positioner.position.selectedAttributeName == 'PSD':
               element_and_thickness = positioner.filters['PSD']
           elif positioner.position.selectedAttributeName == 'XBPM':
               element_and_thickness = positioner.filters['XBPM']
        elif positioner == self.xbpm3:
           if positioner.position.selectedAttributeName == 'CVD':
               element_and_thickness = positioner.filters['CVD']
           elif positioner.position.selectedAttributeName == 'XBPM':
               element_and_thickness = positioner.filters['XBPM']

        else:
           if positioner.position.read_attribute('isInserted').value:
               element_and_thickness = positioner.filters

        try:
           element = element_and_thickness['element']
           thickness = element_and_thickness['thickness']
        except:
           element = None
           thickness = 0.
        return element, thickness

    def get_transmission(self):
        transmission = 1.
        for p in self.positioners:
            #print 'positioner', p, self.positioners[p].display_name,
            element, thickness = self.get_element_and_thickness(self.positioners[p])
            if element != None and thickness != 0.:
               p_t = self.get_transmission_from_element_and_thickness(element, thickness)
            else:
               p_t = 1.
            transmission *= p_t
            #print 'absorbed', 1-p_t
        return transmission

