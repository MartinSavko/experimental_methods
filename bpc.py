#!/usr/bin/env python

import time
import pickle
import numpy as np

from beam_position_controller import get_bpc

def main(monitor='cam', actuator='vertical_trans', period=1., ponm=False):
    
    bpc = get_bpc(monitor=monitor, actuator=actuator, period=period, ponm=ponm)
    
    bpc.serve()
    
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-m', '--monitor', default='cam', type=str, help='Monitor')
    parser.add_argument('-a', '--actuator', default='vertical_trans', type=str, help='Actuator')
    parser.add_argument('--ponm', default=0, type=int, help='ponm')
    parser.add_argument('-p', '--period', default=1., type=float, help='Period')
    
    args = parser.parse_args()
    
    main(monitor=args.monitor, actuator=args.actuator, period=args.period, ponm=bool(args.ponm))
    
