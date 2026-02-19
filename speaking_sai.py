#!/usr/bin/python

import time

import numpy as np

from monitor import monitor

from speech import speech, defer

import tango

class speaking_monitor(monitor, speech):
    def __init__(
        self,
        integration_time=None,
        sleeptime=0.05,
        use_redis=True,
        name="monitor",
        port=5555,
        history_size_target=10000,
        debug_frequency=100,
        framerate_window=25,
        service=None,
        verbose=None,
        server=False,
    ):
        monitor.__init__(
            self,
            integration_time=integration_time,
            sleeptime=sleeptime,
            use_redis=use_redis,
            name=name,
        )

        speech.__init__(
            self,
            port=port,
            sleeptime=sleeptime,
            history_size_target=history_size_target,
            debug_frequency=debug_frequency,
            framerate_window=framerate_window,
            service=service,
            verbose=verbose,
            server=server,
        )

    def acquire(self):
        try:
            self.value = self.get_point()
            self.timestamp = time.time()
            self.value_id += 1
        except:
            traceback.print_exc()

        super().acquire()


class sai(speaking_monitor):
    def __init__(
        self,
        device_name="i11-ma-c00/ca/sai.2",
        number_of_channels=1,
        sleeptime=1.0,
        use_redis=True,
        channels=None,
        port=5555,
        history_size_target=45000,
        debug_frequency=100,
        framerate_window=25,
        service=None,
        verbose=None,
        server=None,
    ):
        super().__init__(
            name=device_name,
            sleeptime=sleeptime,
            use_redis=use_redis,
            port=port,
            history_size_target=history_size_target,
            debug_frequency=debug_frequency,
            framerate_window=framerate_window,
            service=service,
            verbose=verbose,
            server=server,
        )

        self.device_name = device_name
        self.device = tango.DeviceProxy(device_name)
        self.configuration_fields = [
            "configurationid",
            "samplesnumber",
            "frequency",
            "integrationtime",
            "stathistorybufferdepth",
            "datahistorybufferdepth",
        ]
        self.channels = ["channel%d" for k in range(number_of_channels)]

        self.number_of_channels = number_of_channels

        self.channel_ids = (
            list(range(self.number_of_channels)) if channels is None else channels
        )

        self.history_values = [b"" for item in self.channels]
        self.history_times = [None for item in self.channels]
        self.acquisition_timestamps = []

    def get_configuration(self):
        configuration = {}
        for parameter in self.configuration_fields:
            configuration[parameter] = self.device.read_attribute(parameter).value
        return configuration

    def get_point(self):
        return np.array(
            [
                self.get_historized_channel_values(channel)
                for channel in range(self.number_of_channels)
            ]
        )

    def get_historized_channel_values(self, channel_number):
        return self.device.read_attribute("historizedchannel%d" % channel_number).value

    def get_channel_current(self, channel_number):
        return self.device.read_attribute("averagechannel%d" % channel_number).value

    def get_channel_difference(self, channel_a, channel_b):
        a = self.get_channel_current(channel_a)
        b = self.get_channel_current(channel_b)
        channel_difference = a - b
        return channel_difference

    def get_total_current(self, absolute=True):
        current = 0
        for channel in range(self.number_of_channels):
            cc = self.get_channel_current(channel)
            if absolute:
                cc = abs(cc)
            current += cc
        return current

    def get_historized_intensity(self):
        historized_intensity = np.zeros(self.get_stathistorybufferdepth())
        historized_intensity = []
        for channel_number in range(self.number_of_channels):
            historized_intensity.append(
                self.get_historized_channel_values(channel_number)
            )
        return historized_intensity

    def get_stathistorybufferdepth(self):
        return self.device.stathistorybufferdepth

    def set_stathistorybufferdepth(self, size):
        self.device.stathistorybufferdepth = size

    def get_frequency(self):
        return self.device.frequency

    def set_frequency(self, frequency):
        self.device.frequency = frequency

    def get_integration_time(self):
        return self.device.integrationtime

    def set_integration_time(self, integration_time):
        self.device.integrationtime = integration_time

    def get_state(self):
        return self.device.state().name

    def start(self):
        return self.device.Start()

    def stop(self):
        return self.device.Stop()

    def abort(self):
        return self.device.Abort()

    def get_point(self):
        return self.get_total_current()

    def get_device_name(self):
        return self.device.dev_name()

    def get_name(self):
        return self.get_device_name()

    @defer
    def get_last_point_data(self):
        last_point_data = [
            self.get_historized_channel_values(channel) for channel in self.channel_ids
        ]

        return last_point_data

    def get_previous_timestamp(self):
        if self.acquisition_timestamps:
            previous_timestamp = self.acquisition_timestamps[-1]
        else:
            previous_timestamp = (
                self.get_timestamp() - len(self.value[0]) * self.get_integration_time()
            )
        return previous_timestamp

    def update_history(self):
        value = self.get_value()
        history_times = self.get_history_times()
        timestamp = self.get_timestamp()
        previous_timestamp = self.get_previous_timestamp()

        self.acquisition_timestamps.append(timestamp)
        for k, channel in enumerate(self.channel_ids):
            merged, new_values = self.merge_two_overlapping_buffers(
                self.history_values[k], value[k].tobytes()
            )
            self.history_values[k] = merged

            new_times = np.linspace(
                timestamp, previous_timestamp, int(new_values), endpoint=False
            )[::-1]

            self.history_times[k] = (
                new_times
                if history_times[k] is None
                else np.hstack([history_times[k], new_times])
            )

    @defer
    def acquire(self):
        self.timestamp = time.time()
        self.value_id += 1
        self.value = np.array(self.get_last_point_data())
        self.update_history()
        self.last_handled_value_id = self.value_id

    @defer
    def get_sing_value(self):
        return np.array([item[-1] for item in self.value]).tobytes()

    def get_command_line(self):
        return f"speaking_sai.py -s {self.service} -d {self.device_name} -n {self.number_of_channels}"
    
def main():
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,)

    parser.add_argument(
        "-k", "--debug_frequency", default=60, type=int, help="debug frame"
    )
    parser.add_argument(
        "-s",
        "--service",
        type=str,
        default="sipin",
        help="debug string add to the outputs",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose")
    
    parser.add_argument(
        "-d",
        "--device_name",
        default="i11-ma-c00/ca/sai.2",
        type=str,
        help="device name",
    )
    parser.add_argument(
        "-n",
        "--number_of_channels",
        default=1,
        type=int,
        help="number of channels",
    )
    
    args = parser.parse_args()
    print(args)

    mon = sai(
        device_name=args.device_name,
        service=args.service,
        number_of_channels=args.number_of_channels,
        debug_frequency=args.debug_frequency,
        verbose=False,
    )
    mon.verbose = args.verbose
    mon.set_server(True)
    mon.serve()

    sys.exit(0)


if __name__ == "__main__":
    main()
