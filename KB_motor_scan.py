!#/usr/bin/env python
# -*- coding: utf-8 -*-

'''
VFM tz motor scan.
'''

import gevent

import traceback
import logging
import time
import os
import pickle
import numpy as np
import pylab
import sys

from xray_experiment import xray_experiment
from scipy.constants import eV, h, c, angstrom, kilo, degree
from monitor import Si_PIN_diode
import optparse

from motor import vfm_tz_motor, hfm_tx_motor
from motor import vfm_rx_motor, hfm_rz_motor

class KB_motor_scan(xray_experiment):

    def __init__(self,
                 name_pattern,
                 directory,
                 start,
                 end,
                 default_position,
                 darkcurrent_time=5.,
                 photon_energy=None,
                 diagnostic=True,
                 analysis=None,
                 conclusion=None,
                 simulation=None,
                 display=False,
                 extract=False):
                 
        xray_experiment.__init__(self, 
                                 name_pattern, 
                                 directory,
                                 photon_energy=photon_energy,
                                 diagnostic=diagnostic,
                                 analysis=analysis,
                                 conclusion=conclusion,
                                 simulation=simulation)