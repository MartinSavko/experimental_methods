#!/usr/bin/env python

import gevent
from motor import tango_motor
from instrument import instrument
import redis
import tango
import PyTango

class focusing:
    '''
    Table 1. Mirror and experimental tables positions for all focussing modes.

    Motor           mode1   mode 1.5.v  mode 1.5.h    mode 2  mode 2.5    mode 3
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    mir1.tx          ---        ---         ---         ---     33.400    33.400
    mir1.rz          ---        ---         ---         ---      0.180     0.180
    mir1.rs          ---        ---         ---         ---      0.000     0.000
    mir2.tz        -5.327     -0.130        ---       -0.327      ---      0.000
    mir2.rx          ---       3.998        ---        3.926      ---      4.044
    mir3.tx        -6.948       ---       -1.960      -1.948    24.050    24.050
    mir3.rz          ---        ---       -4.490      -4.527     1.909     1.909
    tab1.tx         0.000      0.000       0.000      -0.250     4.500     4.500
    tab1.tz        -0.810     -0.170      -0.170      -0.810    -0.170    -0.170
    tab2.pitch     -0.100      7.960       0.000       7.951     0.000     7.960
    tab2.roll       0.000      0.000       0.000       0.000     0.000     0.000
    tab2.yaw       -0.400      0.000       8.773       8.741    -0.570    -0.570
    tab2.cZ         0.000      0.000       0.000       0.000     0.000     0.000
    tab2.cX         0.000      0.550       0.550      -0.220    14.600    14.600

    '''
    def __init__(self):
        self.redis = redis.StrictRedis()
        
        self.modes = \
            ['1',
             '2',
             '3',
             '2v',
             '2h',
             '3h']

        self.parameters = \
             ['i11-ma-c05/ex/tab.1-mt_tx/position',
              'i11-ma-c05/ex/tab.1-mt_tz/position',
              'i11-ma-c04/op/mir.1-mt_rs/position',
              'i11-ma-c04/op/mir.1-mt_rz/position',
              'i11-ma-c04/op/mir.1-mt_tx/position',
              'i11-ma-c05/op/mir.2-mt_tz/position',
              'i11-ma-c05/op/mir.2-mt_rx/position',
              'i11-ma-c05/op/mir.3-mt_rz/position',
              'i11-ma-c05/op/mir.3-mt_tx/position',
              'i11-ma-c05/ex/tab.2/pitch', 
              'i11-ma-c05/ex/tab.2/yaw', 
              'i11-ma-c05/ex/tab.2/tC', 
              'i11-ma-c05/ex/tab.2/tX',
              'robot_x', 
              'robot_y', 
              'robot_z',
              'i11-ma-c05/op/mir3-ch.00/voltage'
              'i11-ma-c05/op/mir3-ch.01/voltage'
              'i11-ma-c05/op/mir3-ch.02/voltage'
              'i11-ma-c05/op/mir3-ch.03/voltage'
              'i11-ma-c05/op/mir3-ch.04/voltage'
              'i11-ma-c05/op/mir3-ch.05/voltage'
              'i11-ma-c05/op/mir3-ch.06/voltage'
              'i11-ma-c05/op/mir3-ch.07/voltage'
              'i11-ma-c05/op/mir3-ch.08/voltage'
              'i11-ma-c05/op/mir3-ch.09/voltage'
              'i11-ma-c05/op/mir3-ch.10/voltage'
              'i11-ma-c05/op/mir3-ch.11/voltage']
    
    def get_mode_1_parameters(self):
        mode_1_parameters = {
           'i11-ma-c04/op/mir.1-mt_tx/position': 5.,
           'i11-ma-c05/ex/tab.1-mt_tx/position': 0.,
           'i11-ma-c05/ex/tab.1-mt_tz/position':-0.81,
           'i11-ma-c05/op/mir.2-mt_tz/position':-5.,
           'i11-ma-c05/op/mir.3-mt_tx/position':-6.,
           'i11-ma-c05/ex/tab.2/pitch':-0.1,
           'i11-ma-c05/ex/tab.2/yaw':-0.4,
           'i11-ma-c05/ex/tab.2/xC': 0.,
           'i11-ma-c05/ex/tab.2/zC': 0.,
           'robot_x':-8200,
           'robot_y': 12350,
           'robot_z':-5350}
        return mode_1_parameters
          
    def get_mode_2_parameters(self):
        mode_2_parameters = {
           'i11-ma-c04/op/mir.1-mt_tx/position': 5.,
           'i11-ma-c05/ex/tab.1-mt_tx/position':-0.25,
           'i11-ma-c05/ex/tab.1-mt_tz/position':-0.81,
           'i11-ma-c05/op/mir.2-mt_tz/position':-0.2308,
           'i11-ma-c05/op/mir.2-mt_rx/position': 3.9308,
           'i11-ma-c05/op/mir.3-mt_tx/position':-1.8513,
           'i11-ma-c05/op/mir.3-mt_rz/position':-4.8069,
           'i11-ma-c05/ex/tab.2/pitch': 7.949655,
           'i11-ma-c05/ex/tab.2/yaw': 8.94303,
           'i11-ma-c05/ex/tab.2/xC': -0.2998,
           'i11-ma-c05/ex/tab.2/zC': 0.,
           'i11-ma-c05/op/mir2-ch.00/voltage': 255.0,
           'i11-ma-c05/op/mir2-ch.01/voltage': 215.0,
           'i11-ma-c05/op/mir2-ch.02/voltage': 12.0,
           'i11-ma-c05/op/mir2-ch.03/voltage': 170.0,
           'i11-ma-c05/op/mir2-ch.04/voltage': 185.0,
           'i11-ma-c05/op/mir2-ch.05/voltage': 443.0,
           'i11-ma-c05/op/mir2-ch.06/voltage': 322.0,
           'i11-ma-c05/op/mir2-ch.07/voltage': 188.0,
           'i11-ma-c05/op/mir2-ch.08/voltage': 40.0,
           'i11-ma-c05/op/mir2-ch.09/voltage': -47.0,
           'i11-ma-c05/op/mir2-ch.10/voltage': -3.0,
           'i11-ma-c05/op/mir2-ch.11/voltage': 88.0,
           'i11-ma-c05/op/mir3-ch.00/voltage': 290.0,
           'i11-ma-c05/op/mir3-ch.01/voltage': 320.0,
           'i11-ma-c05/op/mir3-ch.02/voltage': 265.0,
           'i11-ma-c05/op/mir3-ch.03/voltage': 56.0,
           'i11-ma-c05/op/mir3-ch.04/voltage': 102.0,
           'i11-ma-c05/op/mir3-ch.05/voltage': 415.0,
           'i11-ma-c05/op/mir3-ch.06/voltage': 37.0,
           'i11-ma-c05/op/mir3-ch.07/voltage': -247.0,
           'i11-ma-c05/op/mir3-ch.08/voltage': -534.0,
           'i11-ma-c05/op/mir3-ch.09/voltage': -703.0,
           'i11-ma-c05/op/mir3-ch.10/voltage': -1089.0,
           'i11-ma-c05/op/mir3-ch.11/voltage': -1400.0,
           'robot_x': 0.,
           'robot_y': 0.,
           'robot_z': 0.}
           
        return mode_2_parameters

    def get_mode_3_parameters(self):
        mode_3_parameters = {
           'i11-ma-c04/op/mir.1-mt_tx/position': 33.4,
           'i11-ma-c04/op/mir.1-mt_rz/position': 0.18,
           'i11-ma-c04/op/mir.1-mt_rs/position': 0.,
           'i11-ma-c05/ex/tab.1-mt_tx/position': 4.5,
           'i11-ma-c05/ex/tab.1-mt_tz/position':-0.17,
           'i11-ma-c05/op/mir.2-mt_tz/position': 0.,
           'i11-ma-c05/op/mir.2-mt_rx/position': 4.044,
           'i11-ma-c05/op/mir.3-mt_tx/position': 24.05,
           'i11-ma-c05/op/mir.3-mt_rz/position': 1.909,
           'i11-ma-c05/ex/tab.2/pitch': 7.960,
           'i11-ma-c05/ex/tab.2/yaw': -0.57,
           'i11-ma-c05/ex/tab.2/xC': 14.6,
           'i11-ma-c05/ex/tab.2/zC': 0.,
           'i11-ma-c05/op/mir2-ch.00/voltage': 255.0,
           'i11-ma-c05/op/mir2-ch.01/voltage': 215.0,
           'i11-ma-c05/op/mir2-ch.02/voltage': 12.0,
           'i11-ma-c05/op/mir2-ch.03/voltage': 170.0,
           'i11-ma-c05/op/mir2-ch.04/voltage': 185.0,
           'i11-ma-c05/op/mir2-ch.05/voltage': 443.0,
           'i11-ma-c05/op/mir2-ch.06/voltage': 322.0,
           'i11-ma-c05/op/mir2-ch.07/voltage': 188.0,
           'i11-ma-c05/op/mir2-ch.08/voltage': 40.0,
           'i11-ma-c05/op/mir2-ch.09/voltage': -47.0,
           'i11-ma-c05/op/mir2-ch.10/voltage': -3.0,
           'i11-ma-c05/op/mir2-ch.11/voltage': 88.0,
           'i11-ma-c05/op/mir3-ch.00/voltage': 450.9,
           'i11-ma-c05/op/mir3-ch.01/voltage': 383.5,
           'i11-ma-c05/op/mir3-ch.02/voltage': 478.8,
           'i11-ma-c05/op/mir3-ch.03/voltage': 298.1,
           'i11-ma-c05/op/mir3-ch.04/voltage': 658.9,
           'i11-ma-c05/op/mir3-ch.05/voltage': 907.2,
           'i11-ma-c05/op/mir3-ch.06/voltage': 1399.0,
           'i11-ma-c05/op/mir3-ch.07/voltage': 1170.6,
           'i11-ma-c05/op/mir3-ch.08/voltage': 1185.3,
           'i11-ma-c05/op/mir3-ch.09/voltage': 942.2,
           'i11-ma-c05/op/mir3-ch.10/voltage': 1122.6,
           'i11-ma-c05/op/mir3-ch.11/voltage': 878.7,
           'robot_x': 4600.,
           'robot_y':-600.,
           'robot_z':-700.}

        return mode_3_parameters
        
    def get_mode_2v_parameters(self):
        '''Beam focusing using VFM only'''
        mode_2v_parameters = {
           'i11-ma-c04/op/mir.1-mt_tx/position': 5.,
           'i11-ma-c05/ex/tab.1-mt_tx/position': 0.,
           'i11-ma-c05/ex/tab.1-mt_tz/position':-0.170,
           'i11-ma-c05/op/mir.2-mt_tz/position':-0.130,
           'i11-ma-c05/op/mir.2-mt_rx/position': 3.998,
           'i11-ma-c05/op/mir.3-mt_tx/position':-5,
           'i11-ma-c05/ex/tab.2/pitch': 7.960,
           'i11-ma-c05/ex/tab.2/yaw': 0.,
           'i11-ma-c05/ex/tab.2/xC': -0.2998,
           'i11-ma-c05/ex/tab.2/zC': 0.,
           'i11-ma-c05/op/mir2-ch.00/voltage': 255.0,
           'i11-ma-c05/op/mir2-ch.01/voltage': 215.0,
           'i11-ma-c05/op/mir2-ch.02/voltage': 12.0,
           'i11-ma-c05/op/mir2-ch.03/voltage': 170.0,
           'i11-ma-c05/op/mir2-ch.04/voltage': 185.0,
           'i11-ma-c05/op/mir2-ch.05/voltage': 443.0,
           'i11-ma-c05/op/mir2-ch.06/voltage': 322.0,
           'i11-ma-c05/op/mir2-ch.07/voltage': 188.0,
           'i11-ma-c05/op/mir2-ch.08/voltage': 40.0,
           'i11-ma-c05/op/mir2-ch.09/voltage': -47.0,
           'i11-ma-c05/op/mir2-ch.10/voltage': -3.0,
           'i11-ma-c05/op/mir2-ch.11/voltage': 88.0,
           'robot_x': -8700,
           'robot_y': -900,
           'robot_z': -200}

        return mode_2v_parameters 

    def get_mode_2h_parameters(self):
        '''Beam focusing using HFM only'''
        mode_2h_parameters = {
           'i11-ma-c04/op/mir.1-mt_tx/position': 5.,
           'i11-ma-c05/ex/tab.1-mt_tx/position': 0.,
           'i11-ma-c05/ex/tab.1-mt_tz/position':-0.170,
           'i11-ma-c05/op/mir.2-mt_tz/position':-5.,
           'i11-ma-c05/op/mir.3-mt_tx/position':-1.960,
           'i11-ma-c05/op/mir.3-mt_rz/position':-4.490,
           'i11-ma-c05/ex/tab.2/pitch': 7.949655,
           'i11-ma-c05/ex/tab.2/yaw': 8.94303,
           'i11-ma-c05/ex/tab.2/xC': -0.2998,
           'i11-ma-c05/ex/tab.2/zC': 0.,
           'i11-ma-c05/op/mir3-ch.00/voltage': 290.0,
           'i11-ma-c05/op/mir3-ch.01/voltage': 320.0,
           'i11-ma-c05/op/mir3-ch.02/voltage': 265.0,
           'i11-ma-c05/op/mir3-ch.03/voltage': 56.0,
           'i11-ma-c05/op/mir3-ch.04/voltage': 102.0,
           'i11-ma-c05/op/mir3-ch.05/voltage': 415.0,
           'i11-ma-c05/op/mir3-ch.06/voltage': 37.0,
           'i11-ma-c05/op/mir3-ch.07/voltage': -247.0,
           'i11-ma-c05/op/mir3-ch.08/voltage': -534.0,
           'i11-ma-c05/op/mir3-ch.09/voltage': -703.0,
           'i11-ma-c05/op/mir3-ch.10/voltage': -1089.0,
           'i11-ma-c05/op/mir3-ch.11/voltage': -1400.0,
           'robot_x': -1000,
           'robot_y': 11850,
           'robot_z': -5250}
        
        return mode_2h_parameters

    def get_3h_parameters(self):
        '''Beam focusing using HFM and HPM'''
        mode_3h_parameters = {
           'i11-ma-c04/op/mir.1-mt_tx/position': 33.4,
           'i11-ma-c04/op/mir.1-mt_rz/position': 0.18,
           'i11-ma-c04/op/mir.1-mt_rs/position': 0.,
           'i11-ma-c05/ex/tab.1-mt_tx/position': 4.5,
           'i11-ma-c05/ex/tab.1-mt_tz/position':-0.17,
           'i11-ma-c05/op/mir.2-mt_tz/position':-5.,
           'i11-ma-c05/op/mir.3-mt_tx/position': 24.05,
           'i11-ma-c05/op/mir.3-mt_rz/position': 1.909,
           'i11-ma-c05/ex/tab.2/pitch': 7.960,
           'i11-ma-c05/ex/tab.2/yaw': -0.57,
           'i11-ma-c05/ex/tab.2/xC': 14.6,
           'i11-ma-c05/ex/tab.2/zC': 0.,
           'i11-ma-c05/op/mir3-ch.00/voltage': 450.9,
           'i11-ma-c05/op/mir3-ch.01/voltage': 383.5,
           'i11-ma-c05/op/mir3-ch.02/voltage': 478.8,
           'i11-ma-c05/op/mir3-ch.03/voltage': 298.1,
           'i11-ma-c05/op/mir3-ch.04/voltage': 658.9,
           'i11-ma-c05/op/mir3-ch.05/voltage': 907.2,
           'i11-ma-c05/op/mir3-ch.06/voltage': 1399.0,
           'i11-ma-c05/op/mir3-ch.07/voltage': 1170.6,
           'i11-ma-c05/op/mir3-ch.08/voltage': 1185.3,
           'i11-ma-c05/op/mir3-ch.09/voltage': 942.2,
           'i11-ma-c05/op/mir3-ch.10/voltage': 1122.6,
           'i11-ma-c05/op/mir3-ch.11/voltage': 878.7,
           'robot_x': 4400,
           'robot_y': 11850,
           'robot_z':-5250}
        
        return mode_3h_parameters

    def set_mode(self, mode='2'):
        print 'setting focusing mode %s' % mode
        parameters = getattr(self, 'get_mode_%s_parameters' % mode)()
        print 'target parameters'
        for key, value in parameters.items():
            print '%s: %.4f' % (key, value)
        
        self.mode = mode
        
        self.redis.set('focusing_mode', mode)
        
        for key, value in parameters.items():
            if 'robot' in key:
                self.redis.set(key, value)
            elif 'tab.2' not in key:
                ap = PyTango.AttributeProxy(key)
                current_value = ap.read().value
                print 'key %s, current value %.4f, target value %.4f, will execute ap.write(%s)' % (key, current_value, value, value)
                ap.write(value)
                
       for key, value in parameters.items():
           if 'tab.2' in key:
               ap = PyTango.AttributeProxy(key)
               current_value = ap.read().value
               if abs(current_value - value) > 1e-2:
                  ap.write(value)
                  k = 0
                  while abs(current_value - value) > 1e-2 and k < 10:
                     k+=1
                     gevent.sleep(0.5)

        
       
        
        
