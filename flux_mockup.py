#!/usr/bin/env python

import pickle
from scipy.interpolate import interp1d

class flux_mockup:
    def __init__(self, flux_table='./flux_table.pickle', reference_current=500.):
        self.table = pickle.load(open(flux_table, 'rb'), encoding="bytes")
        self.flux_as_f_of_energy = interp1d(self.table[:, 0], self.table[:, 1], bounds_error=False, fill_value='extrapolate')
        self.reference_current = reference_current
        
    def get_flux(self, machine_current=500, capillary_factor=1., aperture_factor=0.95, attenuators_factor=1., transmission=1., photon_energy=12650):
        current_factor = machine_current/self.reference_current
        tabulated_flux = self.flux_as_f_of_energy(photon_energy)
        flux = tabulated_flux * current_factor * capillary_factor * aperture_factor * attenuators_factor * transmission
        return flux
    
def main():
    import argparse
    fm = flux_mockup()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--photon_energy', type=float, default=12650., help='photon energy [eV]')
    parser.add_argument('-t', '--transmission', type=float, default=1., help='transmission <0, 1>')
    parser.add_argument('-m', '--machine_current', type=float, default=500., help='machine current [mA]')
    
    args = parser.parse_args()
    print('Supplied parameters:')
    print('Machine current: %.1f mA' % args.machine_current)
    print('Transmission: %.1f' % args.transmission)
    print('Photon energy: %.1f eV' % args.photon_energy)
    flux = fm.get_flux(machine_current=args.machine_current, transmission=args.transmission, photon_energy=args.photon_energy)
    print('Proxima 2A beamline estimated flux at sample position for given parameters: %.1f (%.2e) photons per second' % (flux, flux ))
    
if __name__ == '__main__':
    main()
