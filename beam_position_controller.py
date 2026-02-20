#!/usr/bin/env python

import logging
import time
import traceback
import pickle
import numpy as np
from pid import pid
from monitor import sai
from motor import tango_motor
from oav_camera import oav_camera as camera
from scipy.ndimage import center_of_mass

from pid import pid
import sys

sys.path.insert(0, "./")
from speech import speech, defer

from useful_routines import DEFAULT_BROKER_PORT

class position_controller(pid, speech):
    def __init__(
        self,
        port=DEFAULT_BROKER_PORT,
        kp=0,
        ki=0,
        kd=0,
        setpoint=None,
        max_output=None,
        min_output=None,
        on=False,
        reverse=False,
        ponm=True,
        period=1,
        valid_input_digits=5,
        valid_output_digits=4,
        server=None,
        service=None,
        verbose=False,
    ):
        logging.basicConfig(
            format="%(asctime)s|%(module)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            level=logging.INFO,
        )

        self.last_valid = True
        self.verbose = verbose
        self.service = service

        pid.__init__(
            self,
            kp,
            ki,
            kd,
            setpoint,
            max_output=max_output,
            min_output=min_output,
            on=on,
            reverse=reverse,
            ponm=ponm,
            period=period,
            valid_input_digits=valid_input_digits,
            valid_output_digits=valid_output_digits,
        )
        speech.__init__(self, port=port, server=server, service=self.service, verbose=self.verbose)

    def initialize(self):
        super().initialize()
        self.output = self.output_device.get_position()
        self.last_output = self.output
        self.ie = self.output

    def serve(self):
        self.initialize()

        while True:
            self.compute()

            if self.output != self.last_output:
                if self.output_valid(self.output):
                    self.output_device.set_position(
                        self.output, accuracy=self.output_accuracy
                    )
                    self.last_output = self.output
                else:
                    print(
                        f"output is not valid, we should have never gotten here, please check"
                    )
                    i = self.get_input()
                    dt = self.get_dt()

                    pe = self.get_pe(i)
                    ie = self.get_ie(pe, dt)
                    de = self.get_de(pe, dt, i)
                    print(f"i {i}")
                    print(f"dt {dt}")
                    print(f"pe {pe}")
                    print(f"ie {ie}")
                    print(f"de {de}")
                    print(f"self.output {self.output}")
                    output = self.kp * pe + ie + de
                    print(f"output = self.kp * pe + ie + de {output}")
                    output = self.reset_windup(output)
                    print(f"output = self.reset_windup(output) {output}")
                    output = round(output, self.valid_output_digits)
                    print(f"output = round(output, self.valid_output_digits) {output}")

            time.sleep(self.period)

    @defer
    def set_on(self, on=True, sleeptime=0.1):
        while self.get_on() != on:
            super().set_on(on=on)
            self.on = on

    @defer
    def get_on(self):
        on = super().get_on()
        return on

    @defer
    def get_output(self):
        output = super().get_output()
        return output

    @defer
    def get_last_output(self):
        last_output = super().get_last_output()
        return last_output

    @defer
    def print_current_settings(self):
        current_settings = super().print_current_settings()
        return current_settings

    # @defer
    # def get_pe(self, i=None):
    # pe = super().get_pe(i=i)
    # return pe

class camera_beam_position_controller(position_controller):
    def __init__(
        self,
        output_device_name="i11-ma-c05/op/mir.2-mt_tz",
        channels=(0,),
        port=DEFAULT_BROKER_PORT,
        kp=0,
        ki=0,
        kd=0,
        setpoint=0.5,
        max_output=None,
        min_output=None,
        on=False,
        reverse=False,
        ponm=True,
        period=0.1,
        server=None,
        service=None,
        verbose=True,
    ):
        self.output_device_name = output_device_name
        self.channel = channels[0]
        self.input_device = camera()
        self.output_device = tango_motor(output_device_name)

        if setpoint is None:
            setpoint = self.get_input()

        super().__init__(
            port=port,
            kp=kp,
            ki=ki,
            kd=kd,
            setpoint=setpoint,
            max_output=max_output,
            min_output=min_output,
            on=on,
            reverse=reverse,
            ponm=ponm,
            period=period,
            server=server,
            service=service,
            verbose=verbose,
        )

    def get_input(self, nsamples=3, threshold=0.50):
        _inputs = []
        for k in range(nsamples):
            img = self.input_device.get_filtered_image(color=False, threshold=threshold)
            shp = np.array(img.shape[:2])
            com = np.array(center_of_mass(img))
            com = com / shp
            _inputs.append(com[self.channel])
        _input = np.median(_inputs)
        input = round(_input, self.valid_input_digits)
        return input

    def operational_conditions_are_valid(self, min_count=2000, threshold=255.0 / 2):
        img = self.input_device.get_image(color=False)

        try:
            valid = (img > threshold).sum() > min_count
        except:
            valid = False

        if valid and not self.last_valid:
            self.initialize()

        self.last_valid = valid

        return valid

    def get_command_line(self, actuator=None, port=None):
        if port is None:
            port = self.port
        if actuator is None:
            if "tz" in self.output_device_name:
                actuator = "vertical_trans"
            elif "tx" in self.output_device_name:
                actuator = "horizontal_trans"
            elif "rz" in self.output_device_name:
                actuator = "horizontal_pitch"
            elif "rx" in self.output_device_name:
                actuator = "vertical_pitch"
        
        return f"beam_position_controller.py -m cam -a {actuator} -p {port}"
    
class sai_beam_position_controller(position_controller):
    def __init__(
        self,
        output_device_name="i11-ma-c05/op/mir.2-mt_rx",
        input_device_name="i11-ma-c00/ca/sai.4",
        channels=(0, 1),
        port=DEFAULT_BROKER_PORT,
        kp=0,
        ki=0,
        kd=0,
        setpoint=None,
        max_output=None,
        min_output=None,
        on=True,
        reverse=False,
        ponm=True,
        period=0.1,
        service=None,
        server=None,
        verbose=True,
    ):
        self.output_device_name = output_device_name
        self.channel_a = channels[0]
        self.channel_b = channels[1]

        self.input_device = sai(device_name=input_device_name)
        self.output_device = tango_motor(output_device_name)

        if setpoint is None:
            setpoint = self.get_input()

        super().__init__(
            port=port,
            kp=kp,
            ki=ki,
            kd=kd,
            setpoint=setpoint,
            max_output=max_output,
            min_output=min_output,
            on=on,
            reverse=reverse,
            ponm=ponm,
            period=period,
            server=server,
            service=service,
            verbose=verbose,
        )

    def get_input(self):
        _input = self.input_device.get_channel_difference(
            self.channel_a, self.channel_b
        )

        return _input

    def operational_conditions_are_valid(self, min_current=0.1):
        valid = self.input_device.get_total_current() >= min_current

        if valid and not self.last_valid:
            self.initialize()

        self.last_valid = valid

        return valid

    def get_command_line(self, actuator=None, port=None):
        if port is None:
            port = self.port
        if actuator is None:
            if "tz" in self.output_device_name:
                actuator = "vertical_trans"
            elif "tx" in self.output_device_name:
                actuator = "horizontal_trans"
            elif "rz" in self.output_device_name:
                actuator = "horizontal_pitch"
            elif "rx" in self.output_device_name:
                actuator = "vertical_pitch"
        
        return f"beam_position_controller.py -m sai -a {self.actuator} -p {port}"
    
# high speed, lower precision
# velocity: 0.05; acceleration/deceleration: 0.40, accuracy: 0.0003 mrad
# vfm_trans_center = -0.3433 #Run4; -0.2100 Run3
# vfm_pitch_center = +3.8713 #Run4; +3.8831 Run3
# hfm_trans_center = -2.1270 #Run4 -2.1543 #-2.1829 #Run4; -2.1821 # -2.1747 Run3
# hfm_pitch_center = -4.7061 #Run4 -4.6902 #-4.6765 #Run4; -4.6498 # -4.6750 Run3

# 2025-04-01
# In [48]: vfm.get_position()
# Out[48]: {'pitch': 4.00309281, 'translation': 0.4335}

# In [49]: hfm.get_position()
# Out[49]: {'pitch': -4.60043026, 'translation': -3.4626}

#
vfm_trans_center = (
    0.3776
    # +0.4335
    # +0.4502  # 0.2167 2025_Run1 # before MD3 -0.4465 # -0.3433 # Run5
)
vfm_pitch_center = (
    3.9963
    # +4.00309281
    # +4.0119  # 3.99099021 2025_Run1 # before MD3 +3.8234 #+3.8713 # Run5
)
hfm_trans_center = (
    -3.5572  # -3.4572 2025_Run5
)  # -3.4626 #-3.4096  # -3.5514 2025_Run1 # before MD3 -2.1316 # Run5
hfm_pitch_center = (
    -4.5975
    # -4.58810782 2025_Run5
)  # -4.60043026  #-4.6532  # -4.58365805 2025_Run1 # before MD3 -4.7035 # Run5


parameters = {
    "vertical_pitch": {
        "output_device_name": "i11-ma-c05/op/mir.2-mt_rx",
        "port": DEFAULT_BROKER_PORT,
        "min_output": vfm_pitch_center - 0.03,  # 3.855,
        "max_output": vfm_pitch_center + 0.03,  # 3.950,
        "sai": {
            "kp": 0.48925,
            "ki": 0.45613,
            "kd": 0.13119,
            "reverse": True,
            "setpoint": 0.0,
            "channels": (2, 3),
        },
        "cam": {
            "kp": 0.02086,
            "ki": 0.01933,
            "kd": 0.00562,
            "reverse": False,
            "setpoint": 0.5,
            "channels": (0,),
        },
    },
    "horizontal_pitch": {
        "output_device_name": "i11-ma-c05/op/mir.3-mt_rz",
        "port": DEFAULT_BROKER_PORT,
        "min_output": hfm_pitch_center - 0.03,  # -4.675,
        "max_output": hfm_pitch_center + 0.03,  # -4.655,
        "sai": {
            "kp": 1.37680,
            "ki": 1.31124,
            "kd": 0.36141,
            "reverse": True,
            "setpoint": 0.0,
            "channels": (0, 1),
        },
        "cam": {
            "kp": 0.05121,
            "ki": 0.04800,
            "kd": 0.01366,
            "reverse": False,
            "setpoint": 0.5,
            "channels": (1,),
        },
    },
    "vertical_trans": {
        "output_device_name": "i11-ma-c05/op/mir.2-mt_tz",
        "port": DEFAULT_BROKER_PORT,
        "min_output": vfm_trans_center - 0.2,  # -0.2084 - 0.2, #-0.3085
        "max_output": vfm_trans_center + 0.2,  # -0.2084 + 0.2,
        "sai": {
            "kp": 0.25,  # 0.48925,
            "ki": 0.45613,
            "kd": 0.13119,
            "reverse": True,
            "setpoint": 0.0,
            "channels": (2, 3),
        },
        "cam": {
            "kp": 0.02086 / 2,
            "ki": 0.01933 / 2,
            "kd": 0.00562,
            "reverse": True,
            "setpoint": 0.5,
            "channels": (0,),
        },
    },
    "horizontal_trans": {
        "output_device_name": "i11-ma-c05/op/mir.3-mt_tx",
        "port": DEFAULT_BROKER_PORT,
        "min_output": hfm_trans_center - 0.1,  # -1.9326 - 0.2, #
        "max_output": hfm_trans_center + 0.1,  # -1.9326 + 0.2,
        "sai": {
            "kp": 1.37680,
            "ki": 1.31124,
            "kd": 0.13119,  # 0.36141,
            "reverse": True,
            "setpoint": 0.0,
            "channels": (0, 1),
        },
        "cam": {
            "kp": 0.05121 / 8,
            "ki": 0.04800 / 8,
            "kd": 0.01366,
            "reverse": False,
            "setpoint": 0.5,
            "channels": (1,),
        },
    },
}

# low speed, high precision
# velocity: 0.01; acceleration/deceleration: 0.4, accuracy: 0.00025 mrad
parameters_ls = {
    "vertical_pitch": {
        "output_device_name": "i11-ma-c05/op/mir.2-mt_rx",
        "port": DEFAULT_BROKER_PORT,
        "min_output": 3.855,
        "max_output": 3.950,
        "sai": {
            "kp": 0.50559,
            "ki": 0.25101,
            "kd": 0.25459,
            "reverse": True,
            "setpoint": 0.0,
            "channels": (2, 3),
        },
        "cam": {
            "kp": 0.02249,
            "ki": 0.01132,
            "kd": 0.01116,
            "reverse": False,
            "setpoint": 0.5,
            "channels": (0,),
        },
    },
    "horizontal_pitch": {
        "output_device_name": "i11-ma-c05/op/mir.3-mt_rz",
        "port": DEFAULT_BROKER_PORT,
        "min_output": -4.685,
        "max_output": -4.635,
        "sai": {
            "kp": 1.34239,
            "ki": 0.39537,
            "kd": 1.13944,
            "reverse": True,
            "setpoint": 0.0,
            "channels": (0, 1),
        },
        "cam": {
            "kp": 0.05325,
            "ki": 0.01576,
            "kd": 0.04497,
            "reverse": False,
            "setpoint": 0.5,
            "channels": (1,),
        },
    },
}


def get_bpc(
    monitor="cam",
    actuator="vertical_trans",
    period=1.0,
    channels=(0,),
    ponm=False,
    verbose=False,
    server=None,
    service=None,
    port=DEFAULT_BROKER_PORT,
):
    if service is None:
        modifier = "h" if "hor" in actuator else "v"
        service = f"{monitor}_{modifier}bpc"
        
    params = {
        "period": period,
        "ponm": ponm,
        "verbose": verbose,
        "service": service,
        "channels": channels,
        "port": port,
        "server": server,
    }
    for parameter in ["output_device_name", "min_output", "max_output"]:
        params[parameter] = parameters[actuator][parameter]
    for parameter in ["kp", "ki", "kd", "reverse", "setpoint", "channels"]:
        params[parameter] = parameters[actuator][monitor][parameter]

    print(params)
    if monitor == "cam":
        bpc = camera_beam_position_controller(**params)
    else:
        bpc = sai_beam_position_controller(**params)

    return bpc


def autotune(
    monitor="cam",
    actuator="vertical_trans",
    D=0.01,
    periods=27,
    sleeptime=1.0,
    epsilon=0.001,
):
    bpc = get_bpc(monitor=monitor, actuator=actuator)

    durations = []
    amplitudes = []
    shifts = []
    outputs = []

    _start = time.time()

    start_position = bpc.output_device.get_position()
    signum = -1.0
    bpc.output_device.set_position(start_position + signum * D, wait=True)
    time.sleep(sleeptime)

    for k in range(periods):
        print("period %d" % k)
        signum *= -1
        _start_input = bpc.get_input()
        _start = time.time()
        bpc.output_device.set_position(start_position + signum * D, wait=True)
        _end = time.time()
        _end_input = bpc.get_input()
        duration = _end - _start
        durations.append(duration)
        shift = _end_input - _start_input
        shifts.append(shift)
        amplitudes.append(np.abs(shift))
        output = signum * D
        outputs.append(output)
        print("T: %.4f, A: %.4f, D: %.4f\n" % (duration, shift, signum * D))
        time.sleep(sleeptime)

    bpc.output_device.set_position(start_position, wait=True)

    durations = np.array(durations)
    shifts = np.array(shifts)
    amplitudes = np.array(amplitudes)
    outputs = np.array(outputs)

    A = np.median(amplitudes)
    Pu = 2 * np.median(durations)

    tuning_parameters = get_tuning_parameters(D, A, Pu)
    print("PID tuning_parameters:")
    print(tuning_parameters)
    tp = tuning_parameters["PID"]
    for key in tp:
        print("'%s': %.5f," % (key, tp[key]))

    Reversed = not np.alltrue(np.sign(shifts) == np.sign(outputs))
    print("controller direction reversed?:", Reversed)
    results = {
        "durations": durations,
        "amplitudes": amplitudes,
        "shifts": shifts,
        "outputs": outputs,
        "start_position": start_position,
        "tuning_parameters": tuning_parameters,
        "reverse": Reversed,
    }

    f = open(
        "autotune_%s_%s_D_%.4f_periods_%d_%s.pickle"
        % (monitor, actuator, D, periods, time.asctime().replace(" ", "_")),
        "wb",
    )
    pickle.dump(results, f)
    f.close()

    # last_d_sign = 1.
    # while time.time() - _start < 60:
    # t = time.time() - _start
    # print('t', t)
    # times.append(time.time()-_start)
    # pe = bpc.get_pe()
    # print('pe', pe)
    # pes.append(pe)
    # outputs.append(bpc.output_device.get_position())
    # s = np.sign(pe + signum * epsilon)
    # print('s, signum', s, signum)
    # if s != signum:
    # signum *= -1
    # last_d_sign *= -1
    # bpc.output_device.set_position(start_position + last_d_sign * D, wait=False)


def get_tuning_parameters(D, A, Pu):
    print("D: %.4f, A: %.4f, Pu: %.4f" % (D, A, Pu))
    Ku = 4 * D / (A * np.pi)

    tuning_parameters = {}
    for control in ["PID", "PI"]:
        if control == "PID":
            kp = 0.6 * Ku
            ki = 1.20 * Ku / Pu
            kd = 0.075 * Ku * Pu
            tuning_parameters[control] = {"kp": kp, "ki": ki, "kd": kd}
        elif control == "PI":
            kp = 0.4 * Ku
            ki = 0.48 * Ku / Pu
            kd = 0.0
            tuning_parameters[control] = {"kp": kp, "ki": ki, "kd": kd}

    return tuning_parameters


def test():
    vbpc = beam_position_controller(
        2,
        3,
        output_device_name="i11-ma-c05/op/mir.2-mt_rx",
        port=DEFAULT_BROKER_PORT,
        kp=0.4775,
        ki=0.4775,
        kd=0.1194,
        setpoint=0.016,
        min_output=3.863,
        max_output=3.880,
        reverse=True,
    )

    hbpc = beam_position_controller(
        0,
        1,
        output_device_name="i11-ma-c05/op/mir.3-mt_rz",
        port=DEFAULT_BROKER_PORT,
        kp=1.3263,
        ki=1.0610,
        kd=0.4145,
        setpoint=-0.0016,
        min_output=-4.6822,
        max_output=-4.6421,
        reverse=True,
    )

    # autotune
    # A = Input amplitude; D = Output shift; Ku = 4 * D / (A * pi); Pu = Peak distance [seconds]
    # PI : kp = 0.4 * Ku; ki = 0.48 * Ku / Pu
    # PID: kp = 0.6 * Ku; ki = 1.20 * Ku / Pu; kd = 0.075 * Ku * Pu

    # vertical: A = 0.008; Pu = 2., D = 0.005
    # Ku = 0.7958
    # kp = 0.4775
    # ki = 0.4775
    # kd = 0.1194

    # horizontal: A = 0.00576; Pu = 2.5; D = 0.010
    # Ku = 2.2105
    # kp = 1.3263
    # ki = 1.0610
    # kd = 0.4145
    
def speaking_bpc(
    monitor="cam", actuator="vertical_trans", channels=(0,), period=1.0, ponm=False, port=DEFAULT_BROKER_PORT, server=None,
):
    print("speaking_bpc", monitor, actuator, port, server)
    
    bpc = get_bpc(
        monitor=monitor, actuator=actuator, period=period, ponm=ponm, channels=channels, port=port, server=server,
    )
    
    return bpc

def main():
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("-m", "--monitor", default="cam", type=str, help="Monitor")
    parser.add_argument(
        "-a", "--actuator", default="vertical_trans", type=str, help="Actuator"
    )
    parser.add_argument("--ponm", default=0, type=int, help="ponm")
    parser.add_argument("-P", "--period", default=1.0, type=float, help="Period")
    parser.add_argument("-c", "--channels", default=(0,), type=tuple, help="Channels")
    parser.add_argument("-p", "--port", default=DEFAULT_BROKER_PORT, type=int, help="port")
    
    args = parser.parse_args()

    bpc = speaking_bpc(
        monitor=args.monitor,
        actuator=args.actuator,
        period=args.period,
        ponm=bool(args.ponm),
        channels=args.channels,
        port=args.port,
    )
    
    bpc.serve()

if __name__ == "__main__":
    main()
    
