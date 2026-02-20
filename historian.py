#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys

from speech import speech, defer

from useful_routines import get_services


class historian(speech):
    def __init__(
        self,
        service="historian",
        verbose=None,
        server=None,
    ):
        self.dimensions = get_services()

        super().__init__(
            service=service,
            verbose=verbose,
            server=server,
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


if __name__ == "__main__":
    h = historian()
    h.verbose = True
    h.set_server = True
    h.serve()

    sys.exit(0)
