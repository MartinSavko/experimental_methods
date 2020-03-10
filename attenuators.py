#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PyTango import DeviceProxy as dp
import numpy as np
#from monitor import xbpm
from mucal import mucal
from energy import energy
import gevent

class attenuators:

    def __init__(self):
        
        self.energy = energy()

        self.carousel = dp('i11-ma-c05/ex/att.1')
        self.imager1 = dp('i11-ma-c02/dt/imag.1-mt_tz-pos') 
        self.xbpm1 = dp('i11-ma-c04/dt/xbpm_diode.1-pos')
        self.xbpm3 = dp('i11-ma-c05/dt/xbpm_diode.3-pos')
        self.xbpm5 = dp('i11-ma-c06/dt/xbpm_diode.5-pos')
        self.psd6 = dp('i11-ma-c06/dt/xbpm_diode.6-pos')
        
        self.carousel.display_name = 'filters'
        self.carousel.filters = {"00 None": {'element': None, 'thickness': 0., 'position': 0.},
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
                                 "11 Ref Pt 5um": {'element': 'Pt', 'thickness': 5e-4, 'position': 7348.},
                                 "UNKNOWN": {'element': None, 'thickness': 0., 'position': None}}
        self.carousel.positions = {0: "00 None",
                                   1: "01 Carbon 200um",
                                   2: "02 Carbon 250um",
                                   3: "03 Carbon 300um",
                                   4: "04 Carbon 500um",
                                   5: "05 Carbon 1mm",
                                   6: "06 Carbon 2mm",
                                   7: "07 Carbon 3mm",
                                   8: "10 Ref Fe 5um",
                                   9: "11 Ref Pt 5um"}

        self.imager1.display_name = 'imager1'
        self.imager1.filters = {'isInserted': {'element': 'C', 'thickness': 500e-4},
                                'isExtracted': {'element': None, 'thickness': 0.}}
        self.imager1.positions = {0: 'isExtracted',
                                  1: 'isInserted'}
        
        self.xbpm1.display_name = 'xbpm1'        
        self.xbpm1.filters = {'isInserted': {'element': 'Ti', 'thickness': 5e-4},
                              'isExtracted': {'element': None, 'thickness': 0.}}
        self.xbpm1.positions = {0: 'isExtracted',
                                1: 'isInserted'}

        self.xbpm3.display_name = 'xbpm3'
        self.xbpm3.filters = {'XBPM': {'element': 'Ti', 'thickness': 5e-4},
                              'CVD': {'element': 'C', 'thickness': 20e-4},
                              'isExtracted': {'element': None, 'thickness': 0.}}
        self.xbpm3.positions = {0: 'isExtracted',
                                1: 'CVD',
                                2: 'XBPM'}
        
        self.xbpm5.display_name = 'xbpm5'
        self.xbpm5.filters = {'XBPM': {'element': 'Ti', 'thickness': 5e-4},
                              'PSD': {'element': 'C', 'thickness': 20e-4},
                              'isExtracted': {'element': None, 'thickness': 0.}}
        self.xbpm5.positions = {0: 'isExtracted',
                                1: 'PSD',
                                2: 'XBPM'}
        
        self.psd6.display_name = 'psd6'
        self.psd6.filters = {'isInserted': {'element': 'C', 'thickness': 20e-4},
                             'isExtracted': {'element': None, 'thickness': 0.}}
        self.psd6.positions = {0: 'isExtracted',
                               1: 'isInserted'}
        
        self.positioners = {1: self.psd6,
                            2: self.xbpm5,
                            3: self.xbpm1,
                            4: self.xbpm3,
                            5: self.imager1,
                            6: self.carousel}
                                      

    def get_transmission_from_element_and_thickness(self, element, thickness, photon_energy=None):
        # photon_energy should be given in eV
        if photon_energy == None:
           phot_e = self.energy.get_energy() * 1.e-3
        else:
           phot_e = photon_energy * 1.e-3
        mu = mucal(element, phot_e)[1][5]
        return np.exp(-mu*thickness)

    def get_filter(self):
        return self.carousel.selectedattributename

    def set_filter(self, filter_name, wait=True):
        setattr(self.carousel, filter_name, True)
        if wait:
            while self.get_filter() == filter_name:
                gevent.sleep(0.1)

    def get_element_and_thickness(self, positioner):
        element_and_thickness = positioner.filters[positioner.selectedAttributeName]
        element, thickness = element_and_thickness['element'], element_and_thickness['thickness']
        return element, thickness
        
        #if positioner == self.carousel:
           #try:
              #element_and_thickness = self.carousel.filters[self.get_filter()]
           #except:
              #pass
        #elif positioner == self.imager1:
            #if positioner.isInserted:
                #element_and_thickness = positioner.filters
        #elif positioner == self.xbpm5:
            #element_and_thickness = positioner.filters[positioner.position.selectedAttributeName]
            ##if positioner.position.selectedAttributeName == 'PSD':
                ##element_and_thickness = positioner.filters['PSD']
            ##elif positioner.position.selectedAttributeName == 'XBPM':
                ##element_and_thickness = positioner.filters['XBPM']
        #elif positioner == self.xbpm3:
            #if positioner.position.selectedAttributeName == 'CVD':
                #element_and_thickness = positioner.filters['CVD']
            #elif positioner.position.selectedAttributeName == 'XBPM':
                #element_and_thickness = positioner.filters['XBPM']
        #else:
           #if positioner.position.read_attribute('isInserted').value:
               #element_and_thickness = positioner.filters

        #try:
           #element = element_and_thickness['element']
           #thickness = element_and_thickness['thickness']
        #except:
           #element = None
           #thickness = 0.
        

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

def plot_levels():
    import pylab
    import seaborn as sns
    sns.set(color_codes=True)

    from matplotlib import rc
    rc('font', **{'family': 'serif','serif': ['Palatino'], 'size': 20})
    rc('text', usetex=True)
    
    from itertools import product
    
    energies = np.linspace(5000., 20000., 1000)

    a = attenuators()
    
    positioners = a.positioners.values()
    
    possibilities = [act.positions.keys() for act in positioners]
    print 'possibilities', possibilities
    
    configurations = product(*possibilities)
    
    #element, thickness = a.get_element_and_thickness(a.carousel)
    #transmissions = [a.get_transmission_from_element_and_thickness(element, thickness, e) for e in energies]
    levels = []
    
    for configuration in list(configurations):
        print 'configuration', configuration
        transmissions = []
        elements_and_thicknesses = []
        for k in range(len(configuration)):
            positioner = a.positioners[k+1]
            if positioner.display_name != 'filters':
                continue
            element_and_thickness = positioner.filters[positioner.positions[configuration[k]]]
            element, thickness = element_and_thickness['element'], element_and_thickness['thickness']
            elements_and_thicknesses.append((element, thickness))
            
        for energy in energies:
            transmission = 1.
            for element, thickness in elements_and_thicknesses:
                if element != None and thickness != 0.:
                    transmission *= a.get_transmission_from_element_and_thickness(element, thickness, energy)
            transmissions.append(transmission)
        levels.append(transmissions)
                
    levels = np.array(levels)
    
    pylab.figure(figsize=(16, 9))
    pylab.title('Filter transmission levels achievable on Proxima 2A', fontsize=24)
    pylab.plot(energies.T, levels.T)
    pylab.xlabel('Energy [eV]', fontsize=20)
    pylab.ylabel('Transmission', fontsize=20)
    pylab.savefig('filter_transmission_levels_PX2A.png')
    pylab.show()
    
if __name__ == '__main__':
    plot_levels()
    pass
