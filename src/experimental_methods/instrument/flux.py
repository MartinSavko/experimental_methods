#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
from scipy.interpolate import interp1d

from .transmission import transmission
from .energy import energy, energy_mockup
from .machine_status import machine_status, machine_status_mockup
from .goniometer import goniometer
from .attenuators import attenuators

class flux_mockup:
    def __init__(
        self,
        flux_table="/usr/local/experimental_methods/flux_table.pickle",
        reference_current=500.0,
    ):
        self.table = pickle.load(open(flux_table, "rb"), encoding="bytes")
        self.flux_as_f_of_energy = interp1d(
            self.table[:, 0],
            self.table[:, 1],
            bounds_error=False,
            fill_value="extrapolate",
        )
        self.reference_current = reference_current

    def get_flux(
        self,
        machine_current=500,
        capillary_factor=1.0,
        aperture_factor=0.95,
        attenuators_factor=1.0,
        photon_energy=12650,
        slits_transmission=1.0,
    ):
        current_factor = machine_current / self.reference_current
        tabulated_flux = self.flux_as_f_of_energy(photon_energy)
        flux = (
            tabulated_flux
            * current_factor
            * capillary_factor
            * aperture_factor
            * attenuators_factor
            * slits_transmission
        )
        return flux

    def get_hypothetical_flux(self, transmission=100.0, photon_energy=12650.0):
        return self.get_flux(
            slits_transmission=transmission / 100.0, photon_energy=photon_energy
        )


class flux:
    def __init__(
        self,
        flux_table="/usr/local/experimental_methods/flux_table.pickle",
        reference_current=500.0,
    ):
        self.table = pickle.load(open(flux_table, "rb"), encoding="bytes")
        self.flux_as_f_of_energy = interp1d(
            self.table[:, 0],
            self.table[:, 1],
            bounds_error=False,
            fill_value="extrapolate",
        )
        self.reference_current = reference_current
        self.transmission = transmission()
        try:
            self.machine_status = machine_status()
        except:
            self.machine_status = machine_status_mockup()
        # self.energy = energy()
        self.energy = energy_mockup()
        self.goniometer = goniometer()
        self.attenuators = attenuators()

        self.aperture_transmission = {
            0: 0.95,
            1: 0.822,
            2: 0.287,
            3: 0.134,
            4: 0.081,
            5: 1.0,
        }
        self.capillary_transmission = 1.0

    def get_current_aperture(self):
        return self.goniometer.md.currentaperturediameterindex

    def get_aperture_transmission(self):
        return self.aperture_transmission[self.get_current_aperture()]

    def get_capillary_transmission(self):
        return self.capillary_transmission

    def get_machine_current(self):
        return round(self.machine_status.get_current(), 1)

    def get_transmission(self):
        return self.transmission.get_transmission() / 100.0

    def get_flux(
        self,
        machine_current=None,
        capillary_factor=None,
        aperture_factor=None,
        attenuators_factor=None,
        photon_energy=None,
        slits_transmission=None,
    ):
        if machine_current is None:
            machine_current = self.get_machine_current()
        current_factor = machine_current / self.reference_current
        if capillary_factor is None:
            capillary_factor = self.get_capillary_transmission()
        if aperture_factor is None:
            aperture_factor = self.get_aperture_transmission()
        if slits_transmission is None:
            slits_transmission = self.get_transmission()
        if attenuators_factor is None:
            attenuators_factor = self.attenuators.get_transmission()
        if photon_energy is None:
            photon_energy = self.energy.get_energy()
            if photon_energy < 1e3:
                photon_energy *= 1e3
        tabulated_flux = self.flux_as_f_of_energy(photon_energy)

        return (
            tabulated_flux
            * current_factor
            * capillary_factor
            * aperture_factor
            * attenuators_factor
            * slits_transmission
        )

    def get_hypothetical_flux(self, transmission=100.0, photon_energy=12650.0):
        transmission = float(transmission)
        transmission /= 100.0
        photon_energy = float(photon_energy)
        if photon_energy < 1.0e3:
            photon_energy *= 1.0e3
        tabulated_flux = self.flux_as_f_of_energy(photon_energy)

        current_factor = self.get_machine_current() / self.reference_current
        capillary_factor = self.get_capillary_transmission()
        aperture_factor = self.get_aperture_transmission()
        try:
            attenuators_factor = self.attenuators.get_transmission()
        except:
            attenuators_factor = 1.0

        return (
            tabulated_flux
            * current_factor
            * capillary_factor
            * aperture_factor
            * attenuators_factor
            * transmission
        )

    def set_flux(self, flux, wait=True):
        return


def test():
    f = flux()
    import sys

    print("current transmission", f.get_flux())


if __name__ == "__main__":
    test()
