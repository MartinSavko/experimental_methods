#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import pickle
from scipy.interpolate import interp1d
from skimage.transform.integral import integral_image, integrate
import numpy as np
import gevent

from slits import slits1, slits2, slits3, slits5, slits6


class slits_transmission:
    def __init__(
        self,
        name_pattern="s2f",
        directory_with_reference_scans="/nfs/data4/2024_Run4/com-proxima2a/Commissioning/slit_scans",
        ratio=1.0,
        transmission_control_slits="slits2",
        slits1_reference_scan="s1f_results.pickle",
        slits2_reference_scan="s2f_results.pickle",
        slits3_reference_scan="s3f_results.pickle",
        slits5_reference_scan="s5f_results.pickle",
        slits6_reference_scan="s6f_results.pickle",
        max_gap=4.0,
        min_gap=0.0,
        steps=4000,
        reference_position=0.0,
    ):
        self.name_pattern = name_pattern
        self.directory = directory_with_reference_scans

        self.slits1_reference_scan = slits1_reference_scan
        self.slits2_reference_scan = slits2_reference_scan
        self.slits3_reference_scan = slits3_reference_scan
        self.slits5_reference_scan = slits5_reference_scan
        self.slits6_reference_scan = slits6_reference_scan

        self.ratio = ratio
        self.max_gap = max_gap
        self.min_gap = min_gap
        self.steps = steps
        self.reference_position = reference_position

        self.slits_names = ["slits1", "slits2", "slits3", "slits5", "slits6"]

        self.slits1 = slits1()
        self.slits2 = slits2()
        self.slits3 = slits3()
        self.slits5 = slits5()
        self.slits6 = slits6()

        self.slits_names = []
        for sn in [1, 2, 3, 5, 6]:
            slits_name = f"slits{sn}"
            self.slits_names.append(slits_name)

            setattr(self, "%s_integral_image" % slits_name, None)
            setattr(
                self,
                "%s_scan_results" % slits_name,
                pickle.load(
                    open(
                        os.path.join(
                            directory_with_reference_scans,
                            getattr(self, f"{slits_name}_reference_scan"),
                        ),
                        "rb",
                    )
                ),
            )

        self.transmission_control_slits = transmission_control_slits

    def get_distribution(self, slits="slits2", steps=4000):
        slits_scan_results = getattr(self, "%s_scan_results" % slits)

        if slits == "slits1":
            hi = slits_scan_results["i11-ma-c02/ex/fent_h.1-mt_i"]
            ho = slits_scan_results["i11-ma-c02/ex/fent_h.1-mt_o"]
            vu = slits_scan_results["i11-ma-c02/ex/fent_v.1-mt_u"]
            vd = slits_scan_results["i11-ma-c02/ex/fent_v.1-mt_d"]
        elif slits == "slits2":
            hi = slits_scan_results["i11-ma-c04/ex/fent_h.2-mt_i"]
            ho = slits_scan_results["i11-ma-c04/ex/fent_h.2-mt_o"]
            vu = slits_scan_results["i11-ma-c04/ex/fent_v.2-mt_u"]
            vd = slits_scan_results["i11-ma-c04/ex/fent_v.2-mt_d"]
        elif slits == "slits3":
            h = slits_scan_results["i11-ma-c05/ex/fent_v.3-mt_tz"]
            v = slits_scan_results["i11-ma-c05/ex/fent_h.3-mt_tx"]
        elif slits == "slits5":
            h = slits_scan_results["i11-ma-c06/ex/fent_v.5-mt_tz"]
            v = slits_scan_results["i11-ma-c06/ex/fent_h.5-mt_tx"]
        elif slits == "slits6":
            h = slits_scan_results["i11-ma-c06/ex/fent_v.6-mt_tz"]
            v = slits_scan_results["i11-ma-c06/ex/fent_h.6-mt_tx"]

        blank_slate = np.ones((steps, steps))
        distribution = blank_slate[::]

        if slits in ["slits1", "slits2"]:
            for k, s in enumerate([hi, ho, vu, vd]):
                position = s["analysis"]["position"]
                transmission = s["analysis"]["transmission"]

                fill_value = np.array((0, 1))
                if k in [0, 3]:  #'mt_d' in label or 'mt_i' in label:
                    position = position[::-1]
                    fill_value = fill_value[::-1]
                ip = interp1d(
                    position,
                    transmission,
                    fill_value=tuple(fill_value),
                    bounds_error=False,
                    kind="slinear",
                )

                extended_positions = np.linspace(-2, 2, steps)
                if k in [0, 3]:  #'mt_d' in label or 'mt_i' in label:
                    extended_positions = extended_positions[::-1]

                calculated_transmission = ip(extended_positions)

                contribution = blank_slate * calculated_transmission

                if k in [0, 3]:  # 'mt_d' in label or 'mt_i' in label:
                    contribution = np.flip(contribution)

                if k in [2, 3]:  #'mt_d' in label or 'mt_u' in label:
                    contribution = contribution.T

                distribution = distribution * contribution

        elif slits in ["slits3", "slits5", "slits6"]:
            for k, s in enumerate([h, v]):
                position = s["analysis"]["position"]
                transmission = s["analysis"]["transmission"]

                fill_value = np.array((0, 0))
                ip = interp1d(
                    position,
                    transmission,
                    fill_value=tuple(fill_value),
                    bounds_error=False,
                    kind="slinear",
                )

                extended_positions = np.linspace(-2, 2, steps)

                calculated_transmission = ip(extended_positions)

                contribution = blank_slate * calculated_transmission

                if k == 1:
                    contribution = contribution.T

                distribution = distribution * contribution

        sum_distribution = distribution.sum()
        distribution /= sum_distribution

        return distribution

    def get_integral_image(self, slits="slits2", steps=4000):
        ii = getattr(self, "%s_integral_image" % slits)

        if ii is None:
            distribution = self.get_distribution(slits=slits, steps=steps)
            ii = integral_image(distribution)
            setattr(self, "%s_integral_image" % slits, ii)

        return ii

    def get_transmission_curve(
        self, slits="slits2", ratio=1.0, horizontal_position=0.0, vertical_position=0.0
    ):
        ii = self.get_integral_image(slits=slits)

        horizontal_gaps = np.linspace(self.min_gap, self.max_gap, self.steps)

        transmissions = []

        for horizontal_gap in horizontal_gaps:
            vertical_gap = horizontal_gap * ratio
            start, end = self.get_indices_for_slit_setting(
                horizontal_gap, vertical_gap, horizontal_position, vertical_position
            )
            transmission = integrate(ii, start, end)
            transmissions.append(transmission[0])

        transmissions = np.array(transmissions)

        transmission_curve = interp1d(
            transmissions, horizontal_gaps, fill_value=(0.0, 4.0)
        )
        setattr(
            self,
            "%s_ratio_%.2f_horizontal_position_%.2f_vertical_position_%.2f_transmission_curve"
            % (slits, ratio, vertical_position, horizontal_position),
            transmission_curve,
        )

        return transmission_curve

    def get_transmission(self, slits="slits2"):
        s = getattr(self, slits)
        horizontal_gap = s.get_horizontal_gap()
        vertical_gap = s.get_vertical_gap()
        horizontal_position = s.get_horizontal_position()
        vertical_position = s.get_vertical_position()

        start, end = self.get_indices_for_slit_setting(
            horizontal_gap, vertical_gap, horizontal_position, vertical_position
        )
        ii = self.get_integral_image(slits=slits)

        return min([1.0, integrate(ii, start, end)[0]])

    def get_indices_for_slit_setting(
        self,
        horizontal_gap,
        vertical_gap,
        horizontal_position=0.0,
        vertical_position=0.0,
        npixels=4000,
        extent=(-2, 2),
    ):
        e = abs(extent[1] - extent[0])
        pixels_per_mm = npixels / e
        horizontal_start = (-horizontal_gap / 2.0 - horizontal_position) - extent[0]
        horizontal_end = (horizontal_gap / 2.0 - horizontal_position) - extent[0]
        vertical_start = (-vertical_gap / 2.0 - vertical_position) - extent[0]
        vertical_end = (vertical_gap / 2.0 - vertical_position) - extent[0]

        horizontal_start *= pixels_per_mm
        horizontal_end *= pixels_per_mm
        vertical_start *= pixels_per_mm
        vertical_end *= pixels_per_mm

        horizontal_start = int(horizontal_start)
        horizontal_end = int(horizontal_end)
        vertical_start = int(vertical_start)
        vertical_end = int(vertical_end)

        if vertical_end == npixels:
            vertical_end -= 1
        if horizontal_end == npixels:
            horizontal_end -= 1
        return (int(vertical_start), int(horizontal_start)), (
            int(vertical_end),
            int(horizontal_end),
        )

    def set_auxiliary_slits(self, auxiliary_slits):
        settings_a = []
        for slits in auxiliary_slits:
            settings_a.append(
                gevent.spawn(
                    getattr(getattr(self, slits), "set_horizontal_gap"), self.max_gap
                )
            )
            settings_a.append(
                gevent.spawn(
                    getattr(getattr(self, slits), "set_vertical_gap"), self.max_gap
                )
            )

        gevent.joinall(settings_a)

        settings_b = []
        for slits in auxiliary_slits:
            settings_b.append(
                gevent.spawn(
                    getattr(getattr(self, slits), "set_horizontal_position"),
                    self.reference_position,
                )
            )
            settings_b.append(
                gevent.spawn(
                    getattr(getattr(self, slits), "set_vertical_position"),
                    self.reference_position,
                )
            )

        gevent.joinall(settings_b)

    def set_transmission(
        self,
        transmission,
        slits="slits2",
        ratio=None,
        horizontal_position=0.0,
        vertical_position=0.0,
        wait=True,
    ):
        transmission_control_slits = getattr(self, slits)

        auxiliary_slits = [
            slit_name for slit_name in self.slits_names if slit_name != slits
        ]

        self.set_auxiliary_slits(auxiliary_slits)

        if transmission_control_slits.get_horizontal_position != horizontal_position:
            transmission_control_slits.set_horizontal_position(horizontal_position)

        if transmission_control_slits.get_vertical_position() != vertical_position:
            transmission_control_slits.set_vertical_position(vertical_position)

        if ratio is None:
            ratio = self.ratio

        try:
            transmission_curve = getattr(
                self,
                "%s_ratio_%.2f_horizontal_position_%.2f_vertical_position_%.2f_transmission_curve"
                % (slits, ratio, vertical_position, horizontal_position),
            )
        except AttributeError:
            transmission_curve = self.get_transmission_curve(
                slits, ratio, horizontal_position, vertical_position
            )

        horizontal_gap = transmission_curve(transmission)
        vertical_gap = ratio * horizontal_gap

        transmission_control_slits.set_horizontal_gap(horizontal_gap)
        transmission_control_slits.set_vertical_gap(vertical_gap)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-d",
        "--directory",
        default="/nfs/data4/2024_Run4/com-proxima2a/Commissioning/slit_scans",
        type=str,
        help="directory with reference scans",
    )
    parser.add_argument(
        "-n", "--name_pattern", default="s2f", type=str, help="name pattern"
    )
    parser.add_argument("-s", default="slits2", type=str, help="slits")

    args = parser.parse_args()

    st = slits_transmission(
        name_pattern=args.name_pattern, directory_with_reference_scans=args.directory
    )

    template = os.path.join(args.directory, args.name_pattern)

    start = time.time()
    distribution = st.get_distribution(slits=args.slits)
    ii = st.get_integral_image(slits=args.slits)
    curve = st.get_transmission_curve(slits=args.slits)
    print(f"all calculation done {time.time()-start} seconds")

    start = time.time()
    np.save(f"{template}_distribution.npy", distribution)
    np.save(f"{template}_ii.npy", ii)
    np.save(f"{template}_transmission.npy", curve)

    print(f"all saved in {time.time()-start} seconds")


if __name__ == "__main__":
    main()
