#!/usr/bin/env python

import pickle
import numpy as np
import glob

import re

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

import seaborn as sns

sns.set(color_codes=True)

from matplotlib import rc

rc("font", **{"family": "serif", "serif": ["Palatino"]})
rc("text", usetex=True)

from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from scipy.interpolate import interp1d
from scipy.signal import medfilt

xkcd_colors_that_i_like = [
    "pale purple",
    "coral",
    "moss green",
    "windows blue",
    "amber",
    "greyish",
    "faded green",
    "dusty purple",
    "crimson",
    "custard",
    "orangeish",
    "dusk blue",
    "ugly purple",
    "carmine",
    "faded blue",
    "dark aquamarine",
    "cool grey",
    "faded blue",
    "lilac",
    "orange",
    "pale green",
    "olive",
    "baby blue",
    "dark green",
    "magenta",
    "eggplant",
    "cornflower blue",
    "off white",
    "medium blue",
    "forrest green",
    "aqua blue",
    "sage green",
    "cobalt blue",
]

Si_peaks = [
    (1, 1, 1),
    (2, 2, 0),
    (3, 1, 1),
    (4, 0, 0),
    (3, 3, 1),
    (4, 2, 2),
    (5, 1, 1),
    (4, 4, 0),
    (5, 3, 1),
    (6, 2, 0),
    (5, 3, 3),
]

a_Si = 5.43e-10


def d(h, k, l):
    return np.sqrt(a_Si**2 / (h**2 + k**2 + l**2))


def get_gap(filename):
    return float(re.findall(".*_gap_(.*)_continuous_.*pkl", filename)[0])


def XYZ(x, y, z):
    """Go through the results and return X, Y, Z matrices for 3d plots"""
    X, Y, Z = [np.array(l) for l in [x, y, z]]
    X, Y, Z = [np.reshape(l, self.nbsteps) for l in [X, Y, Z]]
    return X, Y, Z


def get_ring_current(filename):
    print(filename)
    res = int(re.findall(".*_(.*)mA_.*pkl", filename)[0])
    print(res)
    return res


def get_slit_opening(filename):
    print(filename)
    res = re.findall(".*_ps_(.*)_gap_.*pkl", filename)[0]
    print(res)
    return res


def wavelength_from_theta(theta, d2=6.26948976):  # wavelength from angle
    return d2 * np.sin(np.radians(theta))


def theta_from_wavelength(wavelength, d2=6.26948976):
    return np.degrees(np.arcsin(wavelength / d2))


def energy_from_wavelength(wavelength):
    return 12.3984 / wavelength


def wavelength_from_energy(energy):
    return 12.3984 / energy


mystery_peaks_9p6 = [6857.17, 8490.25, 8625.0, 8764.29]
mystery_peaks_16 = []
mystery_peaks_8p7 = [6129.0, 6136.96, 6364.06, 8489.25, 8623.28, 8763.63]
mystery_peaks_10 = [6026.0]
mystery_peaks_negative = [
    6025.6,
    6786.77,
    6857.14,
    6954.62,
    7269.15,
    8490.25,
    8624.28,
    8764.0,
    11368.4,
]
mystery_peaks_positive = [9674.28, 9987.28, 10751.1]


def main():
    import optparse
    import os

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
    parser.add_option("-m", "--medfiltnum", default=11, type=int, help="medfiltnum")
    options, args = parser.parse_args()

    scans = glob.glob(os.path.join(options.directory, options.template))

    scans.sort(key=get_gap)
    print("scans")
    print(scans)

    ens = np.arange(4000, 20001, 50)
    fluxes = []
    filtered_fluxes = []
    petit_peaks = []
    gaps = []
    plt.figure(1, figsize=(16, 9))
    k = 0
    for scan in scans[:]:
        k += 1
        gap = get_gap(scan)
        print("gap", gap)
        r = np.array(pickle.load(open(scan)))
        # data structure
        # [energy, angle, f, current, xbpm1.intensity, xbpm6.intensity, gap, undulator.encoder2Position, chronos]
        energies = r[:, 0]
        angles = r[:, 1]
        chronos = r[:, -1]
        flux = r[:, 2]
        # flux = r[:, 4] #xbpm1

        train_time, test_time, train_angles, test_angles = train_test_split(
            chronos.reshape(-1, 1), angles, test_size=0.5
        )
        angle_vs_time = LinearRegression()
        angle_vs_time.fit(train_time, train_angles)

        print(
            "angle_vs_time.score test on data",
            angle_vs_time.score(test_time, test_angles),
        )
        print("angle_vs_time.coef_", angle_vs_time.coef_)
        print("angle_vs_time.intercept_", angle_vs_time.intercept_)

        sinus_of_theta = np.sin(np.radians(angles))
        wavelengths = wavelength_from_energy(energies / 1e3)
        (
            train_angles,
            test_angles,
            train_wavelengths,
            test_wavelengths,
        ) = train_test_split(sinus_of_theta.reshape(-1, 1), wavelengths, test_size=0.5)

        wavelength_vs_angle = LinearRegression()
        wavelength_vs_angle.fit(train_angles, train_wavelengths)

        print(
            "wavelength_vs_angle.score test on data",
            wavelength_vs_angle.score(test_angles, test_wavelengths),
        )
        print("wavelength_vs_angle.coef_", wavelength_vs_angle.coef_)
        print("wavelength_vs_angle.intercept_", wavelength_vs_angle.intercept_)

        fit_angles = angle_vs_time.predict(chronos.reshape(-1, 1))
        fit_wavelengths = wavelength_vs_angle.predict(
            np.sin(np.radians(fit_angles)).reshape(-1, 1)
        )
        fit_energies = 1e3 * energy_from_wavelength(fit_wavelengths)

        print("fit energies min max", fit_energies.min(), fit_energies.max())
        ef = list(zip(fit_energies, flux))
        ef.sort(key=lambda x: x[0])
        ef = np.array(ef)
        fit_energies = ef[:, 0]
        flux = ef[:, 1]
        flux_vs_energy = interp1d(
            fit_energies, flux, kind="slinear", bounds_error=False
        )

        energies_on_grid = np.linspace(4000, 20000, 16000)

        flux_on_grid = flux_vs_energy(energies_on_grid)
        plt.plot(
            energies_on_grid,
            flux_on_grid,
            "-",
            color=sns.xkcd_rgb[xkcd_colors_that_i_like[k]],
            label="gap=%s" % gap,
        )
        # if abs(energies[0] - 4000) > 2:
        # flux = flux[::-1]

        fluxes.append(flux_on_grid)
        medfiltnum = options.medfiltnum

        filtered_flux = medfilt(flux_on_grid, medfiltnum)
        filtered_fluxes.append(filtered_flux)
        normalized_difference = (flux_on_grid - filtered_flux) / flux_on_grid
        normalized_difference[flux_on_grid < 0.25e10] = 0
        petit_peaks.append(normalized_difference)

        gaps.append(gap)
        print()

    plt.xlabel("energy [eV]", fontsize=18)
    plt.ylabel("flux [ph/s]", fontsize=18)
    plt.title(
        "Photon flux vs energy and undulator gap; ring current %d mA, PS %s mm"
        % (get_ring_current(scans[0]), get_slit_opening(scans[0])),
        fontsize=22,
    )
    plt.xlim([ens[0], ens[-1]])
    plt.legend()
    plt.grid(True)
    plt.savefig(
        "photon_flux_vs_energy_%dmA_%smm.png"
        % (get_ring_current(scans[0]), get_slit_opening(scans[0]))
    )

    plt.figure(3, figsize=(16, 9))
    pp = np.array(petit_peaks)
    print("pp.shape", pp.shape)
    # plt.plot(energies_on_grid, filtered_fluxes[0], color='blue', label='median filtered flux')
    # plt.plot(energies_on_grid, fluxes[0], color='green', label='raw flux')
    wavelengths_on_grid = wavelength_from_energy(energies_on_grid / 1.0e3)
    angles = theta_from_wavelength(wavelengths_on_grid)
    print("angles", angles)
    angles = 2 * np.array(angles)
    print("angles.shape", angles.shape)

    ppm = pp.mean(axis=0)
    ppm[abs(ppm) < 0.00025] = 0
    wavs = wavelength_from_energy(energies_on_grid / 1e3)
    plt.plot(wavs, ppm, color="green", label="flux - median filtered flux")
    plt.xlabel("Wavelength [A]")
    plt.ylabel("flux [ph/s]")
    plt.legend()
    plt.grid(True)
    # plt.ylim([-0.065, 0.035])
    plt.savefig("searching_for_the_mystery_peaks.png")
    # plt.savefig('/home/smartin/figure_3_medfilt%d.png' % medfiltnum)
    # plt.figure(2, figsize=(16, 9))
    # plt.plot(angles, flux, 'd-', label='flux vs angles')
    # plt.grid(True)
    # plt.xlabel('angle [deg]')
    # plt.legend()

    # sangles = np.sin(np.radians(angles))
    # wavelengths = np.array(wavelength_from_energy(energies/1.e3))

    # print('sangles', sangles.shape)
    # print('wavelengths', wavelengths.shape)

    # from sklearn.linear_model import LinearRegression
    # from sklearn.cross_validation import train_test_split
    # train_features, test_features, train_labels, test_labels = train_test_split(sangles.reshape(-1, 1), wavelengths, test_size=0.25)

    # lm = LinearRegression()
    # lm.fit(train_features, train_labels)
    # print('lm.score test on data', lm.score(test_features, test_labels))
    # print('lm.coef_', lm.coef_)
    # print('lm.intercept_', lm.intercept_)

    # plt.figure(3, figsize=(16, 9))
    # plt.plot(sangles, wavelengths, label='wavelength vs sin(angle)')
    # plt.plot(sangles, lm.predict(sangles.reshape(-1, 1)), '--', label='fit')
    # plt.grid(True)
    # plt.xlabel('angle')
    # plt.ylabel('wavelength')
    # plt.legend()

    # plt.figure(4, figsize=(16, 9))
    # ens = wavelength_from_energy(energies)
    # ens -= ens.min()
    # stheta = np.sin(np.radians(angles))
    # stheta -= stheta.min()
    # conversion = ens/stheta
    # plt.plot(np.sin(np.radians(angles)), conversion, label='conversion factor')
    # plt.xlabel('angle')
    # plt.ylabel('energy conversion factor')
    # plt.grid(True)
    # conversion = np.mean(conversion)
    # print('average conversion factor', conversion)
    ens = energies_on_grid
    fluxes = np.array(fluxes)
    ens, gaps = np.meshgrid(ens, gaps)

    fig = plt.figure(2, figsize=(16, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_wireframe(ens, gaps, fluxes, rstride=75, cstride=75)
    ax.set_xlabel("energy [eV]", fontsize=18)
    ax.set_ylabel("gap [mm]", fontsize=18)
    ax.set_zlabel("flux [ph/s]", fontsize=18)
    plt.title(
        "Photon flux vs energy and undulator gap; ring current %d mA, PS %s mm"
        % (get_ring_current(scans[0]), get_slit_opening(scans[0])),
        fontsize=22,
    )
    ax.view_init(54, -111)
    plt.grid(True)
    plt.savefig(
        "photon_flux_vs_energy_and_undulator_gap_%dmA_%smm.png"
        % (get_ring_current(scans[0]), get_slit_opening(scans[0]))
    )

    fig = plt.figure(4, figsize=(16, 9))
    ans = angle_vs_time.predict(chronos.reshape(-1, 1))
    wavs = wavelength_vs_angle.predict(np.sin(np.radians(ans)).reshape(-1, 1))
    ens = energy_from_wavelength(wavs)
    print("ens", ens)
    # ens = energies[:]

    ens1 = np.hstack((ens, [0]))
    ens2 = np.hstack(([0], ens))
    difference_between_neighboring_points_in_eV = ens2 - ens1

    e1 = ens.min()
    e2 = ens.max()
    amin = theta_from_wavelength(wavelength_from_energy(e1 / 1e3))
    amax = theta_from_wavelength(wavelength_from_energy(e2 / 1e3))
    ans = np.linspace(ans.min(), ans.max(), len(ans))
    print("amin", ans.min())
    print("amax", ans.max())

    print("datapoints", len(energies))
    ans = ans[::-1]
    print("ans", ans)
    ens_from_ans = energy_from_wavelength(wavelength_from_theta(ans))

    ens1 = np.hstack((ens_from_ans, [0]))
    ens2 = np.hstack(([0], ens_from_ans))
    ans_difference_between_neighboring_points_in_eV = ens1 - ens2

    plt.plot(
        ens[1:], medfilt(difference_between_neighboring_points_in_eV[1:-1], 5), "bo"
    )
    print("ens_from_ans", ens_from_ans)
    print(ans_difference_between_neighboring_points_in_eV)
    plt.plot(
        ens_from_ans[1:], ans_difference_between_neighboring_points_in_eV[1:-1], "g-"
    )
    plt.ylabel("sampling [points/eV]")
    plt.xlabel("energy [eV]")
    plt.savefig("scan_speed_vs_energy.png")
    print("chronos[-1]", chronos[-1])
    plt.show()


if __name__ == "__main__":
    main()
