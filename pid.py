#!/usr/bin/env python
# based on http://brettbeauregard.com/blog/2011/04/improving-the-beginners-pid-introduction/

import time
import numpy as np


class pid:
    def __init__(
        self,
        kp,
        ki,
        kd,
        setpoint,
        max_output=None,
        min_output=None,
        on=True,
        reverse=False,
        ponm=True,
        period=1,
        valid_input_digits=5,
        valid_output_digits=4,
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.setpoint = setpoint
        self.max_output = max_output
        self.min_output = min_output
        self.on = on
        self.reverse = reverse

        self.set_tunings(kp, ki, kd)

        self.ponm = ponm  # proportional on measurement
        self.period = period
        self.last_setpoint = setpoint
        self.last_time = None
        self.sum_error = 0.0
        self.last_error = 0.0
        self.last_input = 0.0
        self.last_output = 0.0
        self.output = None
        self.ie = 0.0
        self.de = 0.0
        self.valid_output_digits = valid_output_digits
        self.output_accuracy = 1.0 / 10**valid_output_digits
        self.valid_input_digits = valid_input_digits

    def initialize(self):
        i = self.get_input()
        self.last_input = i
        self.init_i = i
        self.ie = self.get_output()

    def get_dt(self):
        now = time.time()
        if self.last_time is None:
            dt = 0.0
        else:
            dt = now - self.last_time
        self.last_time = now
        return dt

    def get_pe(self, i=None):
        pe = 0.0
        if i is None:
            i = self.get_input()
        if self.output_valid(i):
            pe = self.get_setpoint() - i
        return pe

    def get_ds(self):
        s = self.get_setpoint()
        ds = 0.0
        if s != self.last_setpoint:
            ds = s - self.last_setpoint
            self.last_setpoint = s
        return ds

    def get_di(self, i=None, fresh=False):
        if i is None:
            i = self.get_input()
            fresh = True

        if self.output_valid(i) and self.output_valid(self.last_input):
            di = i - self.last_input
        else:
            di = 0.0

        if fresh:
            if self.output_valid(i):
                self.last_input = i
            else:
                print(f"i is not finite {i}, please check!")

        return di

    def _get_de(self, pe):
        _de = pe - self.last_error
        if self.output_valid(pe):
            self.last_error = pe
        else:
            print(f"pe is not finite {pe}, please check!")
        return _de

    def get_de(self, pe, dt, i=None, beware_of_derivative_kick=True):
        if beware_of_derivative_kick:
            de = -self.get_di(i)
        else:
            de = self._get_de(pe)

        de = de / dt if dt != 0.0 else 0.0
        if not self.output_valid(de):
            print(
                f"de is not finite {de}, pe {pe}, dt {dt}, i {i}, setting it to zero, please check!"
            )
            de = 0.0

        return self.kd * de

    def reset_windup(self, output):
        if self.max_output is not None and output > self.max_output:
            self.ie -= output - self.max_output
            output = self.max_output
        if self.min_output is not None and output < self.min_output:
            self.ie += self.min_output - output
            output = self.min_output
        return output

    def get_ie(self, pe, dt):
        self.ie += self.ki * pe * dt
        return self.ie

    def output_valid(self, output):
        invalid = output in [None, np.nan, np.inf, -np.inf] or (
            not output > 0 and not output <= 0
        )
        return not invalid

    def get_output(self):
        i = self.get_input()
        dt = self.get_dt()

        pe = self.get_pe(i)
        ie = self.get_ie(pe, dt)
        de = self.get_de(pe, dt, i)

        output = self.kp * pe + ie + de

        output = self.reset_windup(output)

        output = round(output, self.valid_output_digits)

        return output

    def get_last_output(self):
        return self.last_output

    def set_output(self, output):
        self.output = output

    def compute(self):
        if self.on and self.operational_conditions_are_valid():
            output = self.get_output()
            if self.output_valid(output):
                self.output = self.get_output()
            else:
                print(f"output is not valid, please check")
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
                print(f"last_error {self.last_error}")
                print(f"self.output {self.output}")
                output = self.kp * pe + ie + de
                print(f"output = self.kp * pe + ie + de {output}")
                output = self.reset_windup(output)
                print(f"output = self.reset_windup(output) {output}")
                output = round(output, self.valid_output_digits)
                print(f"output = round(output, self.valid_output_digits) {output}")

    def get_input(self):
        pass

    def operational_conditions_are_valid(self):
        return True

    def get_setpoint(self):
        return self.setpoint

    def set_setpoint(self, setpoint):
        self.setpoint = setpoint

    def set_tunings(self, kp=None, ki=None, kd=None):
        for name, parameter in zip(["kp", "ki", "kd"], [kp, ki, kd]):
            parameter = abs(parameter)
            if self.reverse:
                parameter *= -1
            getattr(self, "set_%s" % name)(parameter)

    def print_current_settings(self):
        a = ""
        a += "Current control parameters:"
        a += "\n\tkp: %.4f" % self.get_kp()
        a += "\n\tki: %.4f" % self.get_ki()
        a += "\n\tkd: %.4f" % self.get_kd()
        a += "\n\tperiod: %.4f" % self.get_period()
        i = self.get_input()
        dt = self.get_dt()
        pe = self.get_pe(i)
        ie = self.ie + self.ki * pe * dt
        di = i - self.last_input
        de = -di
        de = de / dt if dt != 0.0 else 0.0
        a += "\n\tp term: kp*pe %.4f (pe %.4f)" % (self.kp * pe, pe)
        a += "\n\ti term: ki*ie %.4f (ie %.4f)" % (ie, pe * dt)
        a += "\n\td term: kd*de %.4f (de %.4f)" % (self.kd * de, de)
        a += "\n\toutput: %.4f" % (self.kp * pe + ie + de)
        a += "\n\tinput: %.4f" % i
        a += "\n\tsetpoint: %.4f" % self.get_setpoint()
        a += "\n\ton: %s" % self.get_on()
        print(a)
        return a

    def set_kp(self, kp):
        self.kp = kp

    def get_kp(self):
        return self.kp

    def set_ki(self, ki):
        self.ki = ki

    def get_ki(self):
        return self.ki

    def set_kd(self, kd):
        self.kd = kd

    def get_kd(self):
        return self.kd

    def set_on(self, on=True):
        if self.on != on and on is True:
            self.initialize()
            self.on = True
        else:
            self.on = False

    def get_on(self):
        return self.on

    def set_off(self):
        self.set_on(on=False)

    def set_reverse(self, reverse=True):
        self.reverse = reverse

    def set_ponm(self, ponm=True):
        self.ponm = ponm
        self.pe = 0.0

    def get_ponm(self):
        return self.ponm

    def set_min_output(self, min_output):
        self.min_output = min_output

    def get_min_output(self):
        return self.min_output

    def set_max_output(self, max_output):
        self.max_output = max_output

    def get_max_output(self):
        return self.max_output

    def set_output_limits(self, min_output, max_output):
        self.set_min_output(min_output)
        self.set_max_output(max_output)

    def set_period(self, period):
        self.period = period

    def get_period(self):
        return self.period
