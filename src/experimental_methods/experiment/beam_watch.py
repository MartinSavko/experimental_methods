#!/usr/bin/env python

import numpy as np
import time
from mirror_scan import adaptive_mirror
from experimental_methods.instrument.monitor import xbpm
from energy import energy
import traceback

e = energy()
vfm = adaptive_mirror("vfm")
hfm = adaptive_mirror("hfm")
psd5 = xbpm(device_name="i11-ma-c06/dt/xbpm_diode.psd.5-base")

name_direction = {"vfm": "vertical", "hfm": "horizontal"}


def get_vertical_difference(set_point=0.0825):
    return set_point - psd5.device.verticalPosition


def get_horizontal_difference(set_point=-0.1038):
    return set_point - psd5.device.horizontalPosition


def adjust_position(
    mirror_name="vfm",
    stop_move=0.0005,
    default_step=0.001,
    min_step=0.00025,
    small_check_time=3,
):
    mirror = adaptive_mirror(mirror_name)

    if mirror_name == "vfm":
        get_difference = get_vertical_difference
        factor = 0.385
    else:
        get_difference = get_horizontal_difference
        factor = 0.870
    print(
        time.asctime(),
        "difference between set_point and current %s beam position is %.4f. This is too large, beam watch is going to take care of it!"
        % (name_direction[mirror_name], get_difference()),
    )
    pitch_start = mirror.pitch.device.position
    print("current %s pitch position %.4f" % (mirror_name, pitch_start))
    previous_dif_sign = None
    step = default_step
    k = 0
    while abs(get_difference()) > stop_move and step > min_step:
        difference = get_difference()
        print("%d difference" % k, difference)
        step = np.abs(difference) * factor
        dif_sign = np.sign(difference)
        if dif_sign != previous_dif_sign and previous_dif_sign is not None:
            step /= 2.0
        print("step size %.6f" % step)
        print()
        try:
            mirror.pitch.device.position -= dif_sign * step
            k += 1
        except:
            traceback.print_exc()
        previous_dif_sign = dif_sign
        time.sleep(small_check_time)
    pitch_end = mirror.pitch.device.position
    print(
        "operation converged at difference %.4f after %d steps" % (get_difference(), k)
    )
    print(
        "corrected %s pitch position %.4f, delta from start %.4f"
        % (mirror_name, pitch_end, pitch_end - pitch_start)
    )


def main():
    trigger_move = 0.003
    big_check_time = 300
    small_check_time = 1
    min_trusted_intensity = 2.0
    max_trusted_intensity = 5.0
    while True:
        if (
            psd5.get_intensity() > min_trusted_intensity
            and psd5.get_intensity() < max_trusted_intensity
            and np.abs(12650 - e.get_energy()) < 100
        ):
            if (
                abs(get_vertical_difference()) > trigger_move
                or abs(get_horizontal_difference()) > trigger_move
            ):
                adjust_position(mirror_name="vfm")
                adjust_position(mirror_name="hfm")
            else:
                print(
                    time.asctime(),
                    "difference between set_point and current beam position is %.4f, %.4f (v, h), is below the threshold, nothing to do ... next check in %.1f seconds"
                    % (
                        get_vertical_difference(),
                        get_horizontal_difference(),
                        big_check_time,
                    ),
                )
            time.sleep(big_check_time)
        time.sleep(small_check_time)


if __name__ == "__main__":
    main()
