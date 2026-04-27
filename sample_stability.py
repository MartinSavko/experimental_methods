#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import traceback
import pylab
import numpy as np

from useful_routines import (
    get_shifts_from_images,
    get_string_from_timestamp,
    get_power_spectrum,
    get_rmsd,
)

from experiment import experiment


class sample_stability(experiment):
    specific_parameter_fields = [
        {
            "name": "start_measure",
            "type": "float",
            "description": "start measure time [s]",
        },
        {
            "name": "end_measure",
            "type": "float",
            "description": "end measure time [s]",
        },
        {
            "name": "measurement_duration",
            "type": "float",
            "description": "measurement duration [s]",
        },
        {
            "name": "calibration",
            "type": "array",
            "description": "camera pixel calibration [mm/px]",
        },
    ]

    def __init__(
        self,
        name_pattern=None,
        directory=None,
        measurement_duration=60.0,
        fresh=False,
        analysis=True,
        cameras=[
            "sample_view",
            "goniometer",
        ],
    ):
        self.measurement_duration = measurement_duration
        self.fresh = fresh

        if hasattr(self, "parameter_fields"):
            self.parameter_fields += self.specific_parameter_fields[:]
        else:
            self.parameter_fields = self.specific_parameter_fields[:]

        experiment.__init__(
            self,
            name_pattern=name_pattern,
            directory=directory,
            analysis=analysis,
            cameras=cameras,
        )

        self.start_measure = None
        self.end_measure = None

    def get_calibration(self):
        return self.camera.get_calibration()

    def run(self):
        if self.fresh:
            self.start_measure = time.time()
            time.sleep(self.measurement_duration)
            self.end_measure = time.time()
        else:
            self.end_measure = time.time()
            self.start_measure = self.end_measure - self.measurement_duration
            self.start_run_time = self.start_measure

    def analyze(self):
        times, images = self.camera.get_history(
            start=self.start_measure, end=self.end_measure
        )
        
        times = np.array(times)

        sf = len(times) / self.measurement_duration

        print(f"got history N={len(times)}, duration={self.measurement_duration:.1f} s, sampling={sf:.1f} Hz, calculating shifts")

        _start = time.time()
        shifts = get_shifts_from_images(images, reference=0) * 1.e3 * self.get_calibration()
        shifts -= shifts.mean(axis=0)
        duration = time.time() - _start
        print(f"shifts.shape", shifts.shape)
        print(f"shifts calculation took {duration/len(shifts):.4f} per image")
        self.results = shifts
        std = np.std(shifts, axis=0)
        minmax = shifts.max(axis=0) - shifts.min(axis=0)
        print(f"std {np.round(std, 4)}")
        #rmsd = get_rmsd(shifts)
        #print(f"rmsd {np.round(rmsd, 4)}")
        
        fig, axes = pylab.subplots(1, 2, figsize=(16, 9))

        fig.suptitle(
            f"{self.name_pattern}, sampling {sf:.1f} Hz",
            fontsize=20,
        )

        
        axes[0].plot(times[-len(shifts):] - times[0], shifts[:,0], label="vertical")
        axes[0].plot(times[-len(shifts):] - times[0], shifts[:,1], label="horizontal")
        axes[0].set_ylim([-1.5, 1.5])
        axes[0].set_title("shifts")
        axes[0].set_xlabel("time [s]")
        axes[0].set_ylabel("shift [microns]")
        axes[0].legend()
        axes[0].text(
            0.25,
            0.975,
            f"range (V, H): {np.round(minmax, 4)}",
            transform=axes[0].transAxes,
        )
        axes[0].text(
            0.25,
            0.945,
            f"rmsd (V, H): {np.round(std, 4)}",
            transform=axes[0].transAxes,
        )
        #axes[0].text(
            #0.25,
            #0.915,
            #f"rmsd (V, H): {np.round(rmsd, 4)}",
            #transform=axes[0].transAxes,
        #)
        
        print("calculating power spectrum")
        f, Pxx_den = get_power_spectrum(shifts, sf)

        try:
            axes[1].set_title("Power spectrum")
            axes[1].set_xlabel("Frequency [Hz]")
            #axes[1].set_ylabel("PSD [V**2/Hz]")
            axes[1].set_ylabel("Linear spectrum [RMS]")
            axes[1].semilogy(f, np.sqrt(Pxx_den))
            
            #axes[1].set_ylim([1e-7, Pxx_den.max() * 1.1])

        except:
            traceback.print_exc()
            print(f"f {f}")
            print(f"Pxx_den {Pxx_den}")

        pylab.savefig(f"{self.get_template()}_sample_stability.png")
        pylab.show()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        default="/nfs/data4/2026_Run2/com-proxima2a/Commissioning/sample_stability",
        help="directory",
    )

    parser.add_argument(
        "-n",
        "--name_pattern",
        type=str,
        default=f"sample_stability_{get_string_from_timestamp()}",
        help="name pattern",
    )
    parser.add_argument(
        "-t", "--time", default=5.0, type=float, help="duration of observation"
    )
    parser.add_argument(
        "-f", "--fresh", action="store_true", help="acquire fresh images"
    )

    args = parser.parse_args()
    print(f"args {args}")

    sampstab = sample_stability(
        name_pattern=args.name_pattern,
        directory=args.directory,
        measurement_duration=args.time,
        fresh=args.fresh,
    )

    sampstab.execute()


if __name__ == "__main__":
    main()
