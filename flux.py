#!/usr/bin/env python
# -*- coding: utf-8 -*-

from slits import slits1, slits2
import pickle
from scipy.interpolate import interp1d
import numpy as np
from transmission import transmission
from energy import energy
from machine_status import machine_status
from goniometer import goniometer
from attenuators import attenuators

class flux_mockup:
    def __init__(self):
        self.flux = 1.6e12
    def get_flux(self):
        return self.flux

class flux:

    def __init__(self, flux_table='/usr/local/bin/mxcube_local/HardwareRepository/HardwareObjects/SOLEIL/PX2/experimental_methods/flux_table.pickle', reference_current=500.):
        
        self.table = pickle.load(open(flux_table))
        self.flux_as_f_of_energy = interp1d(self.table[:, 0], self.table[:, 1], bounds_error=False, fill_value='extrapolate')
        self.reference_current = reference_current
        self.transmission = transmission()
        self.machine_status = machine_status()
        self.energy = energy()
        self.goniometer = goniometer()
        self.attenuators = attenuators()
        
        self.aperture_transmission = {0: 0.95, 1: 0.822, 2: 0.287, 3: 0.134, 4: 0.081, 5: 1.}
        self.capillary_transmission = 1.
        
    def get_current_aperture(self):
        return self.goniometer.md2.currentaperturediameterindex
    
    def get_aperture_transmission(self):
        return self.aperture_transmission[self.get_current_aperture()]
    
    def get_capillary_transmission(self):
        return self.capillary_transmission
    
    def get_machine_current(self):
        return self.machine_status.get_current()
    
    def get_transmission(self):
        return self.transmission.get_transmission()/100.
    
    def get_flux(self):
        current_factor = self.get_machine_current()/self.reference_current
        capillary_factor = self.get_capillary_transmission()
        aperture_factor = self.get_aperture_transmission()
        transmission = self.get_transmission()
        attenuators_factor = self.attenuators.get_transmission()

        tabulated_flux = self.flux_as_f_of_energy(self.energy.get_energy())
        
        return tabulated_flux * current_factor * capillary_factor * aperture_factor * attenuators_factor * transmission 
                                    
        
    def set_flux(self, flux, wait=True):
        return
        
    
def test():
    f = flux()
    import sys
    print 'current transmission', f.get_flux()
    print 'setting transmission %s' % sys.argv[1], t.set_flux(float(sys.argv[1]))
    time.sleep(1)
    print 'current transmission', t.get_flux()
    
if __name__ == '__main__':
    test()
    
