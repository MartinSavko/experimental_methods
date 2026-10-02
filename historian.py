#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import time
import numpy as np

from speech import speech, defer

from useful_routines import (
    get_services, 
    DEFAULT_BROKER_PORT,
    save_pickled_file,
    get_string_from_timestamp,
    is_jpeg,
)


class historian(speech):
    def __init__(
        self,
        service="historian",
        verbose=None,
        server=None,
        port=DEFAULT_BROKER_PORT,
    ):
        self.dimensions = get_services()

        super().__init__(
            service=service,
            verbose=verbose,
            server=server,
            port=port,
        )

    def _check_dimensions(self, dimensions):
        if dimensions == []:
            dimensions = list(self.dimensions.keys())

        return dimensions
    
    def save_history(self, template, start, end, local=False, dimensions=[]):
        dimensions = self._check_dimensions(dimensions)
        for dim in dimensions:
            filename = f"{template}_{dim}.h5"
            if local:
                self.dimensions[dim].save_history_local(filename, start, end)
            else:
                self.dimensions[dim].save_history(filename, start, end)

    def get_point(self, dimensions=[]):
        dimensions = self._check_dimensions(dimensions)
        point = {"timestamp": time.time()}
        for dim in dimensions:
            if dim == "gonio":
                v = self.dimensions[dim].get_position_dictionary()
            else:
                v = self.dimensions[dim].get_value()
            point[dim] = v
        return point
    
    def save_point(self, point=None, template=None, dimensions=[]):
        if point is None:
            point = self.get_point(dimensions=dimensions)
        if template is None:
            template = f'history_point_{get_string_from_timestamp(point["timestamp"])}'
        filename = f'{template}.pickle'
        point = clean_up_point(point, template=template)
        for item in point:
            if is_jpeg(point[item]):
                imagename = f'{template}_{item}.jpg' 
                write_jpeg(imagename, point[item])
                point[item] = imagename
        save_pickled_file(filename, point)

def clean_up_point(point, template=get_string_from_timestamp()):
    for item in point:
        v = point[item]
        if isinstance(v, bytes) and is_jpeg(v):
            imagename = f'{template}_{item}.jpg'
            write_jpeg(imagename, v)
            point[item] = imagename
    return point
            
def run_server(h):
    h.verbose = True
    h.set_server = True
    h.serve()

    sys.exit(0)

def save_history(template, start=-np.inf, end=np.inf, local=False, dimensions=["oav", "gonio", "cam14_quad", "cam1"]):
    h.save_history(template, start, end, local=local, dimensions=dimensions)
    
def main():
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("-d", "--directory", default="./", type=str, help="directory")
    parser.add_argument("-n", "--name_pattern", default=get_string_from_timestamp(), type=str, help="filename template")
    parser.add_argument("-s", "--start", type=float, help="start")
    parser.add_argument("-e", "--end", type=float, help="end")
    parser.add_argument("-D", "--dimensions", type=str, default='["oav", "gonio", "cam14_quad", "cam1"]', help="dimensions")
    parser.add_argument("--remote", action="store_false", help="save under server account")
    parser.add_argument("--serve", action="store_true", help="run the service")
    parser.add_argument("--save_history", action="store_true", help="save history")
    parser.add_argument("--save_point", action="store_true", help="save point")
    
    args = parser.parse_args()

    
    template = os.path.join(args.directory, args.name_pattern)
    
    h = historian()
    
    if args.serve:
        run_server(h)
    elif args.save_history:
        h.save_history(template, start=args.start, end=args.end, local=not args.remote, dimensions=eval(args.dimensions))
    elif args.save_point:
        h.save_point(template=template)
        
    
if __name__ == "__main__":
    main()
