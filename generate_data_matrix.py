#!/usr/bin/env python
import numpy as np
from plot_scans import get_gap, get_slit_opening, get_ring_current
from model_scans import (
    undulator_peak_energy,
    undulator_peak_intensity,
    get_flux_vs_energy,
    get_experimental_peaks,
)
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

import seaborn as sns

sns.set(color_codes=True)

from matplotlib import rc

rc("font", **{"family": "serif", "serif": ["Palatino"]})
rc("text", usetex=True)


def main():
    import optparse
    import os
    import glob

    parser = optparse.OptionParser()
    parser.add_option(
        "-d",
        "--directory",
        default="scans/ps_4.0x4.0",
        type=str,
        help="Directory with the scan results",
    )
    parser.add_option(
        "-t",
        "--template",
        default="undulator*_step_50*pkl",
        type=str,
        help="glob template to identify the result files",
    )
    options, args = parser.parse_args()

    scans = glob.glob(os.path.join(options.directory, options.template))
    data = []
    for scan in scans:
        gap = get_gap(scan)
        # if gap > 12:
        # continue
        print("gap", gap)
        peak_energies = []
        peak_heights = []
        for harmonic in range(1, 21):
            # 2.78325828, -3.93395593,  0.71832299
            # (3.33, -5.47, 1.8)
            # k0, k1, k2 = [3.81815378, -5.53642081,  2.71692885]
            k0, k1, k2 = 2.73081755, -3.84032394, 0.60509254
            energy = undulator_peak_energy(gap, harmonic, k0=k0, k1=k1, k2=k2)
            peak_energies.append([energy, harmonic])
            peak_heights.append(0)
            peak_energies.append([energy, harmonic])
            peak_heights.append(undulator_peak_intensity(gap, harmonic))
            peak_energies.append([energy, harmonic])
            peak_heights.append(0)

        plt.figure(figsize=(16, 9))
        energies, flux, xbpm1 = get_flux_vs_energy(scan)
        experimental_peaks = get_experimental_peaks(
            peak_energies, energies, flux, gap, data
        )
        experimental_peaks_energies = []
        experimental_peaks_heights = []
        for harmonic in experimental_peaks:
            energy = harmonic[0]
            # print('energy', harmonic, energy)
            experimental_peaks_energies.append(energy)
            experimental_peaks_heights.append(0)
            experimental_peaks_energies.append(energy)
            experimental_peaks_heights.append(
                1
            )  # undulator_peak_intensity(gap, harmonic))
            experimental_peaks_energies.append(energy)
            experimental_peaks_heights.append(0)
        ep = np.array(experimental_peaks)
        exppeaks = ep[:, 0]
        print("differences")
        a = np.array([ep[k] - ep[k - 1] for k in range(1, len(ep))])
        # print(a)
        print("average difference")
        print(np.mean(a))
        # print('exp vs. theory')
        # print(ep)
        print("difference exp vs. theory")
        print((ep[:, 0] - ep[:, 1])[:])
        plt.plot(energies, flux / flux.max(), label="flux")
        # plt.plot(energy, xbpm1/xbpm1.max(), label='xbpm1')
        peak_energies = np.array(peak_energies)
        plt.plot(
            peak_energies[:, 0], peak_heights / max(peak_heights), label="model peaks"
        )
        plt.plot(
            experimental_peaks_energies,
            experimental_peaks_heights,
            label="experiment peaks",
        )
        plt.ylabel("flux [ph/s]", fontsize=18)
        plt.xlabel("energy [eV]", fontsize=18)
        plt.title("PX2 flux vs energy at %s mm undulator gap" % gap, fontsize=22)
        plt.grid(True)
        plt.ylim([-0.1, 1.1])
        plt.legend(fontsize=16)

    data = np.array(data)
    format_dictionary = {
        "slit_opening": get_slit_opening(scans[0]),
        "ring_current": get_ring_current(scans[0]),
    }
    print("format_dictionary", format_dictionary)
    f = open(
        "data_{slit_opening:s}mm_{ring_current:d}mA.pkl".format(**format_dictionary),
        "w",
    )
    pickle.dump(data, f)
    f.close()

    # plt.show()


if __name__ == "__main__":
    main()
