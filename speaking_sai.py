#!/usr/bin/python

import time

import numpy as np

from monitor import sai
from speech import speech, defer


class speaking_sai(sai, speech):
    def __init__(
        self,
        device_name="i11-ma-c00/ca/sai.2",
        number_of_channels=4,
        sleeptime=1.0,
        use_redis=True,
        service="Si_PIN",
        continuous_monitor_name=None,
        channels=None,
        port=5555,
        history_size_target=45000,
        debug_frequency=100,
        framerate_window=25,
        verbose=None,
        server=None,
    ):
        sai.__init__(
            self,
            device_name=device_name,
            sleeptime=sleeptime,
            use_redis=use_redis,
            continuous_monitor_name=continuous_monitor_name,
        )

        speech.__init__(
            self,
            port=port,
            service=service,
            verbose=verbose,
            server=server,
            history_size_target=history_size_target,
            debug_frequency=debug_frequency,
            framerate_window=framerate_window,
        )

        self.channel_ids = (
            list(range(self.number_of_channels)) if channels is None else channels
        )
        self.history_values = [b"" for item in self.channels]
        self.history_times = [None for item in self.channels]
        self.acquisition_timestamps = []
        
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
            previous_timestamp = self.get_timestamp() - len(self.value[0]) * self.get_integration_time() 
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
    
    
    
    
def main():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-k", "--debug_frequency", default=10, type=int, help="debug frame"
    )
    parser.add_argument(
        "-s",
        "--service",
        type=str,
        default="Si_PIN",
        help="debug string add to the outputs",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose")
    args = parser.parse_args()
    print(args)

    mon = speaking_monitor(
        service=args.service,
        debug_frequency=args.debug_frequency,
        verbose=False,
    )
    mon.verbose = args.verbose
    mon.set_server(True)
    mon.serve()

    sys.exit(0)


if __name__ == "__main__":
    main()
