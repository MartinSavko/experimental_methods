#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import gevent
import logging

try:
    import tango
except ImportError:
    import PyTango as tango
import traceback

import numpy as np

from .monitor import tango_monitor

class machine_status(tango_monitor):
    def __init__(
        self,
        device_name="ans/ca/machinestatus",
        name="monitor",
        continuous_monitor_name="device",
        skip_attributes=[
            "functionModeTrend",
            "logs",
            "log",
            "frontEndStateColor",
            "operatorMessageHistory",
            "operatorMessage2History",
            "currentTrend",
            "currentTrendTimes",
            "defaultModesColor",
        ],
    ):
        super().__init__(
            device_name=device_name,
            name=name,
            continuous_monitor_name=continuous_monitor_name,
            skip_attributes=skip_attributes,
        )

        self.observation_fields = ["chronos", "current"]

    def get_point(self):
        return self.get_current()

    def get_current(self):
        try:
            current = self.device.current
        except:
            current = 325.0
        return current

    def get_historized_current(self, lapse=None):
        historized_current = self.device.currentTrend
        if lapse != None:
            try:
                historized_current = historized_current[-lapse:]
            except:
                print(traceback.print_exc())
        return historized_current

    def get_current_trend(self, lapse=None):
        current_trend = np.array(
            list(zip(self.device.currentTrendTimes / 1e3, self.device.currentTrend))
        )
        if lapse != None:
            try:
                current_trend = current_trend[-lapse:, :]
            except:
                print(traceback.print_exc())
        return current_trend

    def get_observations_from_history(self, start):
        current_trend = self.get_current_trend()
        timestamps = current_trend[:, 0]
        current = current_trend[:, 1]
        mask = timestamps > start
        timestamps = timestamps[mask]
        current = current[mask]
        return timestamps, current

    def get_operator_message(self):
        try:
            operator_message = (
                self.device.operatorMessage + self.device.operatorMessage2
            )
        except:
            operator_message = "operator_message"
        return operator_message

    def get_message(self):
        try:
            message = self.device.message
        except:
            message = "message"
        return message

    def get_end_of_beam(self):
        return self.device.endOfCurrentFunctionMode

    def get_vertical_emmitance(self):
        return self.device.vEmittance

    def get_horizontal_emmitance(self):
        return self.device.hEmmitance

    def get_filling_mode(self):
        return self.device.fillingMode

    def get_average_pressure(self):
        return self.device.averagePressure

    def get_function_mode(self):
        return self.device.functionMode

    def is_beam_usable(self):
        return self.device.isBeamUsable

    def get_lifetime(self):
        return self.device.lifetime * 3600.0

    def get_current_threshold(self):
        return self.device.currentThreshold

    def get_top_up_times_and_currents(
        self, filter_anomalies=True, gradient_threshold=0.1
    ):
        ct = self.get_current_trend()
        ti = ct[:, 0]
        cu = ct[:, 1]
        if filter_anomalies:
            curhold = self.get_current_threshold()
            current = self.get_current()
            good_currents = np.logical_and(
                cu > current - 2 * curhold, cu < current + 2 * curhold
            )
            ti = ti[good_currents]
            cu = cu[good_currents]
            
        gr = np.gradient(cu)

        top_up_indices = gr > gradient_threshold * gr.max()
        cuf = cu[top_up_indices]
        tif = ti[top_up_indices]
        tifd = np.diff(tif, append=tif[-1])
        min_times = tif[tifd == 1]
        min_currents = cuf[tifd == 1]
        max_times = tif[np.logical_or(tifd > 1, tifd == 0)]
        max_currents = cuf[np.logical_or(tifd > 1, tifd == 0)]
        return min_times, max_times, min_currents, max_currents, ti, cu

    def get_trigger_current(self, threshold=0.999):
        (
            min_times,
            max_times,
            min_currents,
            max_currents,
            ti,
            cu,
        ) = self.get_top_up_times_and_currents()
        cmc = min_currents[min_currents > threshold * np.median(min_currents)]
        return np.median(cmc)

    def get_top_up_period(self):
        try:
            (
                min_times,
                max_times,
                min_currents,
                max_currents,
                ti,
                cu,
            ) = self.get_top_up_times_and_currents()
            periods = np.diff(min_times)
            med = np.median(periods)
            std = np.std(periods)
            condition = np.logical_and(periods > med - std, periods < med + std)
            periods = periods[condition]
            top_up_period = np.median(periods)
        except:
            print(traceback.print_exc())
            top_up_period = np.inf
        return top_up_period

    def get_time_to_next_top_up(
        self, current=None, trigger_current=None, lifetime=None, verbose=False
    ):
        if trigger_current is None:
            trigger_current = self.get_trigger_current()
        if lifetime is None:
            lifetime = self.get_lifetime()
        if current is None:
            current = self.get_current()

        time_to_next_top_up = max(0, -lifetime * np.log(trigger_current / current))

        if verbose:
            print("trigger_current", trigger_current)
            print("lifetime", lifetime)
            print("current", current)
            print("time to go", time_to_next_top_up)

        return time_to_next_top_up

    def get_time_from_last_top_up(self, method=2):
        (
            min_times,
            max_times,
            min_currents,
            max_currents,
            ti,
            cu,
        ) = self.get_top_up_times_and_currents()
        
        assert method in [1, 2]
        
        if method == 1:
            current_time = time.time()
        elif method == 2:
            current_time = ti[-1]

        time_from_last_top_up = current_time - max_times[-1]
        if time_from_last_top_up < 0:
            print(f"Time from the last top up is coming up as a negative number ({time_from_last_top_up:.1f}).")
            print("This is likely due to a time synchronization issue between the local computer and the accelerator side.")
            print("Please get the local contact to fix the problem (excuting 'sudo ntpdate ntp' should do the trick)")
            time_from_last_top_up = 0
        return time_from_last_top_up

    def estimate_accuracy_of_top_up_prediction(self, nsamples=1000):
        (
            min_times,
            max_times,
            min_currents,
            max_currents,
            ti,
            cu,
        ) = self.get_top_up_times_and_currents()
        trigger_current = self.get_trigger_current()
        lifetime = self.get_lifetime()

        test_indices = np.random.randint(0, len(ti), nsamples)

        errors = []
        anomalies = 0
        for i in test_indices:
            time = ti[i]
            current = cu[i]
            if current < 0.999 * trigger_current:
                print("anomaly", time, current, trigger_current)
                anomalies += 1
                continue
            prediction = self.get_time_to_next_top_up(
                current=current, trigger_current=trigger_current, lifetime=lifetime
            )
            try:
                truth = self.get_closest_top_up_time(time, min_times)
            except:
                print("truth anomaly", time, current, trigger_current)
                anomalies += 1
                continue
            error = prediction - truth
            errors.append(error)

        print(
            "number of samples %d (of %d, %d anomalies)"
            % (len(errors), nsamples, anomalies)
        )
        print("mean absolute error %.3f" % np.mean(np.abs(errors)))
        print("mean error %.3f" % np.mean(errors))
        print("standard deviation %.3f" % np.std(errors))

    def get_closest_top_up_time(self, time, min_times):
        differences = min_times - time
        differences = differences[differences > 0]
        return differences.min()


    def check_top_up(self, expected_scan_duration, equilibrium_time=3., sleeptime=1.):
        logging.info("checking when the next top-up is expected to occur ...")
        try:
            trigger_current = self.get_trigger_current()
            top_up_period = self.get_top_up_period()

            time_to_next_top_up = self.get_time_to_next_top_up(
                trigger_current=trigger_current
            )
            while (
                (expected_scan_duration <= top_up_period / 4.0)
                and (time_to_next_top_up <= expected_scan_duration * 1.05)
                and time_to_next_top_up > 0
            ):
                logging.info(
                    "expected time to the next top-up %.1f seconds, waiting for it ..."
                    % time_to_next_top_up
                )
                gevent.sleep(max(sleeptime, time_to_next_top_up / 2.0))
                time_to_next_top_up = self.get_time_to_next_top_up(
                    trigger_current=trigger_current
                )

            time_from_last_top_up = self.get_time_from_last_top_up()
            if time_from_last_top_up < equilibrium_time:
                logging.info(
                    "waiting for things to settle after the last top-up (%.1f seconds ago)"
                    % time_from_last_top_up
                )
                while time_from_last_top_up < equilibrium_time and time_from_last_top_up != 0:
                    
                    gevent.sleep(max(sleeptime, time_from_last_top_up / 2))
                    time_from_last_top_up = self.get_time_from_last_top_up()
                
                if time_from_last_top_up == 0.:
                    gevent.sleep(equilibrium_time)
            
            logging.info(
                "time to next top-up %.1f seconds, expected scan duration is %.1f seconds, executing the scan ..."
                % (time_to_next_top_up, expected_scan_duration)
            )
            
        except:
            traceback.print_exc()
            
class machine_status_mockup:
    def __init__(self, default_current=450.0):
        self.default_current = default_current

    def get_current(self):
        return self.default_current

    def get_current_trend(self):
        return

    def get_operator_message(self):
        return "mockup"

    def get_message(self):
        return "mockup"

    def get_end_of_beam(self):
        return

    def get_vertical_emmitance(self):
        return 38.3e-12

    def get_horizontal_emmitance(self):
        return 4480.0e-12

    def get_filling_mode(self):
        return "Hybrid"

    def get_average_pressure(self):
        return 1e-10

    def get_function_mode(self):
        return "top-up"

    def is_beam_usable(self):
        return True

    def get_time_from_last_top_up(self):
        return "inf"

    def get_time_to_last_top_up(self):
        return "inf"


def main():
    pass
    ##from scipy.optimize import curve_fit, minimize
    ##import pylab
    # def current(time, max_current, period, offset, constant):
    # time -= offset
    # time = time % period
    # current = max_current - constant*time
    # return current

    # mac = machine_status()

    # def residual(x, measured_current_trend):
    # max_current, period, offset, constant = x
    # time = measured_current_trend[:,0]
    # measured_current = measured_current_trend[:,1]
    # diff = current(time, max_current, period, offset, constant) - measured_current
    # return np.dot(diff, diff)

    # max_current0 = 451.8
    # period0 = 210.
    # offset0 = 0.
    # constant0 = 0.0048*max_current0/period0

    # x0 = [max_current0, period0, offset0, constant0]
    # mc = mac.get_current_trend()[85140:, :]
    # fit = minimize(residual, x0, args=(mc,))
    # print(fit.x)
    # print('fit')
    # print(fit)
    ##print(mc)
    # pylab.plot(mc[:,0], mc[:,1], label='measured')
    # max_current, period, offset, constant = fit.x
    # pylab.plot(mc[:,0], current(mc[:,0], max_current, period, offset, constant), label='predicted')
    # pylab.legend()
    # pylab.grid(True)
    # pylab.show()


if __name__ == "__main__":
    main()
