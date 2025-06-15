#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sample mount
"""

import os
import time
from experiment import experiment

class mount(experiment):
    
    specific_parameter_fields = [
        {
            "name": "puck",
            "type": "int",
            "description": "Puck number",
        },
        {
            "name": "sample", 
            "type": "int", 
            "description": "Sample number",
        },
        {
            "name": "prepare_centring", 
            "type": "bool", 
            "description": "Prepare centring",
        },
        {
            "name": "success",
            "type": "bool",
            "description": "Success",
        }
    ]
        
    def __init__(
        self,
        puck,
        sample,
        wash=False,
        prepare_centring=True,
        name_pattern=None,
        directory=None,
    ):
        
        self.timestamp = time.time()
        self.puck = puck
        self.sample = sample
        self.wash = wash
        self.prepare_centring = prepare_centring
        if name_pattern is None:
            if self.use_sample_changer():
                designation = f"mount_{puck}_{sample}"
            else:
                designation = "manually_mounted"
                
            name_pattern = f"{designation}_{time.ctime(self.timestamp).replace(' ', '_')}"
        
        if directory is None:
            directory = os.path.join(
                os.environ["HOME"], 
                "manual_optical_alignment",
            )
        
        experiment.__init__(
            self,
            name_pattern=name_pattern,
            directory=directory,
        )
            
        self.description = "Sample mount, Proxima 2A, SOLEIL, %s" % time.ctime(
            self.timestamp
        )
        
        self.cameras = [
            "sample_view",
            "goniometer",
            "cam1",
            "cam6",
            "cam8",
            "cam13",
            "cam14_quad",
            "cam14_1",
            "cam14_2",
            "cam14_3",
            "cam14_4",
        ]
        
        self.success = None
        
    def use_sample_changer(self):
        return not -1 in (self.puck, self.sample)
    
    def manually_mounted(self):
        return self.anything_mounted and -1 in self.instrument.sample_changer.get_mounted_puck_and_sample()
        
    def anything_mounted(self):
        return int(self.instrument.sample_changer.sample_mounted())
    
    def sample_mounted(self):
        mpuck, msample = self.instrument.sample_changer.get_mounted_puck_and_sample()
        return mpuck == self.puck and msample == self.sample
    
    def mount(self):
        if not self.sample_mounted():
            print('sample seems not to be mounted ...')
            self.instrument.sample_changer.mount(self.puck, self.sample, prepare_centring=False)
            if self.anything_mounted() and self.wash:
                self.instrument.sample_changer.mount(self.puck, self.sample, prepare_centring=self.prepare_centring)
        elif self.wash and not self.manually_mounted():
            print('washing sample that is not manually mounted ...')
            self.instrument.sample_changer.mount(self.puck, self.sample, prepare_centring=self.prepare_centring)
            
        return self.sample_mounted()
        
    def prepare(self):
        super().prepare()
        if self.use_sample_changer() and self.instrument.sample_changer.isoff():
            self.instrument.sample_changer.on()
            
    def run(self):
        self.success = self.mount()
        
    def get_success(self):
        return self.success

if __name__ == "__main__":
    
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--puck", default=-1, type=int, help="puck")
    parser.add_argument("-s", "--sample", default=-1, type=int, help="sample")
    parser.add_argument(
        "-d",
        "--directory",
        default="/tmp/mount25",
        help="directory",
    )
    parser.add_argument("-n", "--name_pattern", default=None, type=str, help="name_pattern")
    parser.add_argument("-w", "--wash", action="store_true", help="wash")
    parser.add_argument("-N", "--prepare_centring", action="store_false", help="prepare_centring")
    
    args = parser.parse_args()
    
    print("args", args)

    m = mount(
        args.puck,
        args.sample,
        directory = args.directory,
        wash = bool(args.wash),
        prepare_centring = bool(args.prepare_centring),
    )
    
    m.execute()
    
    
        
        
    
