#!/usr/bin/env python

import PyTango
import logging

class transmission(object):
    def __init__(self, test=False):
        self.horizontal_gap = PyTango.DeviceProxy('i11-ma-c02/ex/fent_h.1')
        self.vertical_gap = PyTango.DeviceProxy('i11-ma-c02/ex/fent_v.1')
        self.fp_parser = PyTango.DeviceProxy('i11-ma-c00/ex/fp_parser')
        self.test = test

    def get_transmission(self):
        return fp_parser.TrueTrans_FP

    def set_transmission(self, transmission):
        if self.test: return 
        try:
            truevalue = (2.0 - math.sqrt(4 - 0.04 * transmission)) / 0.02
        except ValueError:
            logging.debug('ValueError with %s' % transmission)
            truevalue = x/2.
        newGapFP_H = math.sqrt(
            (truevalue / 100.0) * Const.FP_Area_FWHM / Const.Ratio_FP_Gap)
        newGapFP_V = newGapFP_H * Const.Ratio_FP_Gap

        horizontal_gap.gap = newGapFP_H
        vertical_gap.gap = newGapFP_V

    