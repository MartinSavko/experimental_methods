#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import gevent

if sys.version_info.major < 3:
    from PyTango import DeviceProxy as dp
else:
    from tango import DeviceProxy as dp
import numpy as np

# from experimental_methods.instrument.monitor import xbpm
from experimental_methods.analysis.mucal import mucal
from .energy import energy


class actuator:
    def __init__(self, tangoname):
        self.dp = dp(tangoname)
        self.display_name = str()
        self.filters = dict()
        self.positions = dict()


class attenuators:
    def __init__(self):
        self.energy = energy()

        self.carousel = actuator("i11-ma-c05/ex/att.1")
        self.imager1 = actuator("i11-ma-c02/dt/imag.1-mt_tz-pos")
        self.xbpm1 = actuator("i11-ma-c04/dt/xbpm_diode.1-pos")
        self.xbpm3 = actuator("i11-ma-c05/dt/xbpm_diode.3-pos")
        self.xbpm5 = actuator("i11-ma-c06/dt/xbpm_diode.5-pos")
        self.psd6 = actuator("i11-ma-c06/dt/xbpm_diode.6-pos")

        self.carousel.display_name = "filters"
        self.carousel.filters = {
            "00 None": {"element": None, "thickness": 0.0, "position": 0.0},
            "01 Carbon 200um": {"element": "C", "thickness": 2e-2, "position": 668.0},
            "02 Carbon 250um": {
                "element": "C",
                "thickness": 2.5e-2,
                "position": 1336.0,
            },
            "03 Carbon 300um": {"element": "C", "thickness": 3e-2, "position": 2004.0},
            "04 Carbon 500um": {"element": "C", "thickness": 5e-2, "position": 2672.0},
            "05 Carbon 1mm": {"element": "C", "thickness": 1e-1, "position": 3340.0},
            "06 Carbon 2mm": {"element": "C", "thickness": 2e-1, "position": 4008.0},
            "07 Carbon 3mm": {"element": "C", "thickness": 3e-1, "position": 4676.0},
            "08 Spare 1": {"element": None, "thickness": 0.0, "position": 5344.0},
            "09 Spare 2": {"element": None, "thickness": 0.0, "position": 6012.0},
            "10 Ref Fe 5um": {"element": "Fe", "thickness": 5e-4, "position": 6680.0},
            "11 Ref Pt 5um": {"element": "Pt", "thickness": 5e-4, "position": 7348.0},
            "UNKNOWN": {"element": None, "thickness": 0.0, "position": None},
        }
        self.carousel.positions = {
            0: "00 None",
            1: "01 Carbon 200um",
            2: "02 Carbon 250um",
            3: "03 Carbon 300um",
            4: "04 Carbon 500um",
            5: "05 Carbon 1mm",
            6: "06 Carbon 2mm",
            7: "07 Carbon 3mm",
            8: "10 Ref Fe 5um",
            9: "11 Ref Pt 5um",
        }

        self.imager1.display_name = "imager1"
        self.imager1.filters = {
            "isInserted": {"element": "C", "thickness": 500e-4},
            "isExtracted": {"element": None, "thickness": 0.0},
        }
        self.imager1.positions = {0: "isExtracted", 1: "isInserted"}

        self.xbpm1.display_name = "xbpm1"
        self.xbpm1.filters = {
            "isInserted": {"element": "Ti", "thickness": 5e-4},
            "isExtracted": {"element": None, "thickness": 0.0},
        }
        self.xbpm1.positions = {0: "isExtracted", 1: "isInserted"}

        self.xbpm3.display_name = "xbpm3"
        self.xbpm3.filters = {
            "XBPM": {"element": "Ti", "thickness": 5e-4},
            "CVD": {"element": "C", "thickness": 20e-4},
            "isExtracted": {"element": None, "thickness": 0.0},
        }
        self.xbpm3.positions = {0: "isExtracted", 1: "CVD", 2: "XBPM"}

        self.xbpm5.display_name = "xbpm5"
        self.xbpm5.filters = {
            "XBPM": {"element": "Ti", "thickness": 5e-4},
            "PSD": {"element": "C", "thickness": 20e-4},
            "isExtracted": {"element": None, "thickness": 0.0},
        }
        self.xbpm5.positions = {0: "isExtracted", 1: "PSD", 2: "XBPM"}

        self.psd6.display_name = "psd6"
        self.psd6.filters = {
            "isInserted": {"element": "C", "thickness": 20e-4},
            "isExtracted": {"element": None, "thickness": 0.0},
        }
        self.psd6.positions = {0: "isExtracted", 1: "isInserted"}

        self.positioners = {
            1: self.psd6,
            2: self.xbpm5,
            3: self.xbpm1,
            4: self.xbpm3,
            5: self.imager1,
            6: self.carousel,
        }

    def get_transmission_from_element_and_thickness(
        self, element, thickness, photon_energy=None
    ):
        # photon_energy should be given in eV
        if photon_energy == None:
            phot_e = self.energy.get_energy() * 1.0e-3
        else:
            phot_e = photon_energy * 1.0e-3
        mu = mucal(element, phot_e)[1][5]
        return np.exp(-mu * thickness)

    def get_filter(self):
        return self.carousel.dp.selectedattributename

    def set_filter(self, filter_name, wait=True):
        if sys.version_info.major < 3:
            setattr(self.carousel.dp, filter_name, True)
        else:
            getattr(self.carousel.dp, filter_name)()
        if wait:
            while self.get_filter() == filter_name:
                gevent.sleep(0.1)

    def get_element_and_thickness(self, positioner):
        element_and_thickness = positioner.filters[positioner.dp.selectedAttributeName]
        element, thickness = (
            element_and_thickness["element"],
            element_and_thickness["thickness"],
        )
        return element, thickness

    def get_transmission(self):
        transmission = 1.0
        for p in self.positioners:
            element, thickness = self.get_element_and_thickness(self.positioners[p])
            if element != None and thickness != 0.0:
                p_t = self.get_transmission_from_element_and_thickness(
                    element, thickness
                )
            else:
                p_t = 1.0
            transmission *= p_t

        return transmission


def plot_levels():
    import pylab
    import seaborn as sns

    sns.set(color_codes=True)

    from matplotlib import rc

    rc("font", **{"family": "serif", "serif": ["Palatino"], "size": 20})
    rc("text", usetex=True)

    from itertools import product

    energies = np.linspace(5000.0, 20000.0, 1000)

    a = attenuators()

    positioners = list(a.positioners.values())

    possibilities = [list(act.positions.keys()) for act in positioners]
    print("possibilities", possibilities)

    configurations = product(*possibilities)

    levels = []

    for configuration in list(configurations):
        print("configuration", configuration)
        transmissions = []
        elements_and_thicknesses = []
        for k in range(len(configuration)):
            positioner = a.positioners[k + 1]
            if positioner.display_name != "filters":
                continue
            element_and_thickness = positioner.dp.filters[
                positioner.positions[configuration[k]]
            ]
            element, thickness = (
                element_and_thickness["element"],
                element_and_thickness["thickness"],
            )
            elements_and_thicknesses.append((element, thickness))

        for energy in energies:
            transmission = 1.0
            for element, thickness in elements_and_thicknesses:
                if element != None and thickness != 0.0:
                    transmission *= a.get_transmission_from_element_and_thickness(
                        element, thickness, energy
                    )
            transmissions.append(transmission)
        levels.append(transmissions)

    levels = np.array(levels)

    pylab.figure(figsize=(16, 9))
    pylab.title("Filter transmission levels achievable on Proxima 2A", fontsize=24)
    pylab.plot(energies.T, levels.T)
    pylab.xlabel("Energy [eV]", fontsize=20)
    pylab.ylabel("Transmission", fontsize=20)
    pylab.savefig("filter_transmission_levels_PX2A.png")
    pylab.show()


if __name__ == "__main__":
    plot_levels()
    pass
