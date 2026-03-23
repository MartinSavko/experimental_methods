#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np

from speech import speech, defer

from useful_routines import get_services, DEFAULT_BROKER_PORT


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

    def save_history(self, filename_template, start, end, local=False, dimensions=[]):
        if dimensions == []:
            dimensions = list(self.dimensions.keys())

        for dim in dimensions:
            filename = f"{filename_template}_{dim}.h5"
            if local:
                self.dimensions[dim].save_history_local(filename, start, end)
            else:
                self.dimensions[dim].save_history(filename, start, end)


def run_server():
    h = historian()
    h.verbose = True
    h.set_server = True
    h.serve()

    sys.exit(0)

def save_history(filename_template, start=-np.inf, end=np.inf, local=False, dimensions=["oav", "gonio", "cam14_quad", "cam1"]):
    h = historian()
    h.save_history(filename_template, start, end, local=local, dimensions=dimensions)
    
    
def main():
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("-d", "--directory", type=str, help="directory")
    parser.add_argument("-n", "--name_pattern", type=str, help="filename template")
    parser.add_argument("-s", "--start", type=float, help="start")
    parser.add_argument("-e", "--end", type=float, help="end")
    parser.add_argument("-D", "--dimensions", type=str, default='["oav", "gonio", "cam14_quad", "cam1"]', help="dimensions")
    parser.add_argument("--remote", action="store_false", help="save under server account")
    parser.add_argument("--serve", action="store_true", help="run the service")
    
    args = parser.parse_args()
    
    if args.serve:
        run_server()
    else:
        filename_template = os.path.join(args.directory, args.name_pattern)
        save_history(filename_template, start=args.start, end=args.end, local=not args.remote, dimensions=eval(args.dimensions))
    
if __name__ == "__main__":
    main()
