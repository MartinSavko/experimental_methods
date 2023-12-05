#!/usr/bin/env python

import gevent
import redis
try:
    import tango
except:
    import PyTango as tango
import time

class focusing:
    '''
    Table 1. Mirror and experimental tables positions for all focussing modes.
                    mode 1    mode 1.5v   mode 1.5h   mode 2    mode 2.5h  mode 3                                                                
                    mode 1    mode 2v     mode 2h     mode 2    mode 3h    mode 3
    Motor           mode 0    mode 1v     mode 1h     mode 2    mode 2h    mode 3
    ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    mir1.tx          ---        ---         ---       0.0034    33.400    33.400
    mir1.rz          ---        ---         ---      -0.0149     0.180     0.180
    mir1.rs          ---        ---         ---        0.000     0.000     0.000
    mir2.tz        -5.327     -0.2488       ---      -0.2488      ---      0.000
    mir2.rx          ---       3.8766       ---       3.8766      ---      4.044
    mir3.tx        -6.948       ---       -1.960     -2.1187    24.050    24.050
    mir3.rz          ---        ---       -4.490     -4.6654     1.909     1.909
    tab1.tx         0.000      0.000       0.000        0.75     4.500     4.500
    tab1.tz        -0.810     -0.170      -0.170       -0.33    -0.170    -0.170
    tab2.pitch     -0.020      7.9517      0.020      7.9517     0.000     7.960
    tab2.roll       0.000      0.000       0.000       0.000     0.000     0.000
    tab2.yaw       -0.400      0.000       8.773      8.9431    -0.570    -0.570
    tab2.cZ         0.000      0.000       0.000       0.000     0.000     0.000
    tab2.cX         0.000     -0.3076    0.550     -0.3076    14.600    14.600

    '''
    def __init__(self):
        self.redis = redis.StrictRedis()
        
        self.modes = \
            ['0',
             '1v',
             '1h',
             '2',
             '2h',
             '3']

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
              'i11-ma-c05/op/mir3-ch.00/targetVoltage'
              'i11-ma-c05/op/mir3-ch.01/targetVoltage'
              'i11-ma-c05/op/mir3-ch.02/targetVoltage'
              'i11-ma-c05/op/mir3-ch.03/targetVoltage'
              'i11-ma-c05/op/mir3-ch.04/targetVoltage'
              'i11-ma-c05/op/mir3-ch.05/targetVoltage'
              'i11-ma-c05/op/mir3-ch.06/targetVoltage'
              'i11-ma-c05/op/mir3-ch.07/targetVoltage'
              'i11-ma-c05/op/mir3-ch.08/targetVoltage'
              'i11-ma-c05/op/mir3-ch.09/targetVoltage'
              'i11-ma-c05/op/mir3-ch.10/targetVoltage'
              'i11-ma-c05/op/mir3-ch.11/targetVoltage']
    
        self.mir2 = tango.DeviceProxy('i11-ma-c05/op/mir2-vfm')
        self.mir3 = tango.DeviceProxy('i11-ma-c05/op/mir3-hfm')
        self.tab2 = tango.DeviceProxy('i11-ma-c05/ex/tab.2')
        
    def get_mode_0_parameters(self):
        mode_0_auxiliary = {
            'i11-ma-cx1/dt/dtc_ccd.1-mt_tz': 14.0,
            'i11-ma-cx1/dt/dtc_ccd.1-mt_tx': 19.5,
            'i11-ma-cx1/dt/camx.1-mt_tz': 43.0
            }
        
        mode_0_parameters = {
            'i11-ma-c04/op/mir.1-mt_rs/position': +0.0000,
            'i11-ma-c04/op/mir.1-mt_rz/position': -0.0149, # 0.0
            'i11-ma-c04/op/mir.1-mt_tx/position': +0.0034, # 5.0011,
            'i11-ma-c05/ex/tab.1-mt_tx/position': +0.7500, #-0.25 5.,
            'i11-ma-c05/ex/tab.1-mt_tz/position': -0.3300, #-0.81,
            
           #'i11-ma-c04/op/mir.1-mt_tx/position': 5.,
           #'i11-ma-c05/ex/tab.1-mt_tx/position': 0.,
           #'i11-ma-c05/ex/tab.1-mt_tz/position':-0.81,
           
            'i11-ma-c05/op/mir.2-mt_tz/position':-5.,
            'i11-ma-c05/op/mir.3-mt_tx/position':-6.,
           
            'i11-ma-c05/ex/tab.2/pitch': 0.02, #-0.1,
            'i11-ma-c05/ex/tab.2/roll': 0.0000,
            'i11-ma-c05/ex/tab.2/yaw': -0.27, #-0.4,
            'i11-ma-c05/ex/tab.2/xC': 0.,
            'i11-ma-c05/ex/tab.2/zC': 0.,
            
            'robot_x': -8000, #-8400,
            'robot_y': 12450, #12350,
            'robot_z': -5000} #-5350}
        
        return mode_0_parameters
          
    def get_mode_2_parameters(self):
        mode_2_auxiliary = {
            'i11-ma-cx1/dt/dtc_ccd.1-mt_tz': 43.0,
            'i11-ma-cx1/dt/dtc_ccd.1-mt_tx': 27.5,
            'i11-ma-cx1/dt/camx.1-mt_tz': 43.0
            }
                
        mode_2_parameters = {
            'i11-ma-c04/op/mir.1-mt_rs/position': +0.0000,
            'i11-ma-c04/op/mir.1-mt_rz/position': -0.0149, # 0.0
            'i11-ma-c04/op/mir.1-mt_tx/position': +0.0034, # 5.0011,
            
            'i11-ma-c05/ex/tab.1-mt_tx/position': +0.7500, #-0.25
            'i11-ma-c05/ex/tab.1-mt_tz/position': -0.3300, #-0.81,
            
            "i11-ma-c05/op/mir.2-mt_rx/position": +3.8766, #3.9415,
            "i11-ma-c05/op/mir.2-mt_tz/position": -0.2488, #-0.2308,
           
            "i11-ma-c05/op/mir.3-mt_rz/position":  -4.6654, #-4.8089,
            "i11-ma-c05/op/mir.3-mt_tx/position":  -2.1187, #-1.8517

            "i11-ma-c05/ex/tab.2/pitch":  +7.9496, #+7.9517, #7.9497,
            "i11-ma-c05/ex/tab.2/roll": 0.0000,
            "i11-ma-c05/ex/tab.2/xC":  -0.3000, #-0.3076,  #-0.2998,
            "i11-ma-c05/ex/tab.2/yaw":  +8.9430, #+8.9431, #8.9433,
            "i11-ma-c05/ex/tab.2/zC":  0.0000, 
           
            'i11-ma-c05/op/mir2-ch.00/targetVoltage': +100.0,
            'i11-ma-c05/op/mir2-ch.01/targetVoltage': +100.0,
            'i11-ma-c05/op/mir2-ch.02/targetVoltage': -75.0,
            'i11-ma-c05/op/mir2-ch.03/targetVoltage': +150.0,
            'i11-ma-c05/op/mir2-ch.04/targetVoltage': +150.0,
            'i11-ma-c05/op/mir2-ch.05/targetVoltage': +550.0,
            'i11-ma-c05/op/mir2-ch.06/targetVoltage': +322.0,
            'i11-ma-c05/op/mir2-ch.07/targetVoltage': +188.0,
            'i11-ma-c05/op/mir2-ch.08/targetVoltage': +40.0,
            'i11-ma-c05/op/mir2-ch.09/targetVoltage': -100.0,
            'i11-ma-c05/op/mir2-ch.10/targetVoltage': -100.0,
            'i11-ma-c05/op/mir2-ch.11/targetVoltage': -200.0,
           
           #'i11-ma-c05/op/mir2-ch.00/targetVoltage': 255.0,
           #'i11-ma-c05/op/mir2-ch.01/targetVoltage': 215.0,
           #'i11-ma-c05/op/mir2-ch.02/targetVoltage': 12.0,
           #'i11-ma-c05/op/mir2-ch.03/targetVoltage': 170.0,
           #'i11-ma-c05/op/mir2-ch.04/targetVoltage': 185.0,
           #'i11-ma-c05/op/mir2-ch.05/targetVoltage': 443.0,
           #'i11-ma-c05/op/mir2-ch.06/targetVoltage': 322.0,
           #'i11-ma-c05/op/mir2-ch.07/targetVoltage': 188.0,
           #'i11-ma-c05/op/mir2-ch.08/targetVoltage': 40.0,
           #'i11-ma-c05/op/mir2-ch.09/targetVoltage': -47.0,
           #'i11-ma-c05/op/mir2-ch.10/targetVoltage': -3.0,
           #'i11-ma-c05/op/mir2-ch.11/targetVoltage': 88.0,
           
            'i11-ma-c05/op/mir3-ch.00/targetVoltage': +400.0,
            'i11-ma-c05/op/mir3-ch.01/targetVoltage': +150.0,
            'i11-ma-c05/op/mir3-ch.02/targetVoltage': +100.0,
            'i11-ma-c05/op/mir3-ch.03/targetVoltage': +50.0,
            'i11-ma-c05/op/mir3-ch.04/targetVoltage': +50.0,
            'i11-ma-c05/op/mir3-ch.05/targetVoltage': +0.0,
            'i11-ma-c05/op/mir3-ch.06/targetVoltage': -100.0,
            'i11-ma-c05/op/mir3-ch.07/targetVoltage': -500.0,
            'i11-ma-c05/op/mir3-ch.08/targetVoltage': -800.0,
            'i11-ma-c05/op/mir3-ch.09/targetVoltage': -1000.0,
            'i11-ma-c05/op/mir3-ch.10/targetVoltage': -1200.0,
            'i11-ma-c05/op/mir3-ch.11/targetVoltage': -1400.0,

           #'i11-ma-c05/op/mir3-ch.00/targetVoltage': 290.0,
           #'i11-ma-c05/op/mir3-ch.01/targetVoltage': 320.0,
           #'i11-ma-c05/op/mir3-ch.02/targetVoltage': 265.0,
           #'i11-ma-c05/op/mir3-ch.03/targetVoltage': 56.0,
           #'i11-ma-c05/op/mir3-ch.04/targetVoltage': 102.0,
           #'i11-ma-c05/op/mir3-ch.05/targetVoltage': 415.0,
           #'i11-ma-c05/op/mir3-ch.06/targetVoltage': 37.0,
           #'i11-ma-c05/op/mir3-ch.07/targetVoltage': -247.0,
           #'i11-ma-c05/op/mir3-ch.08/targetVoltage': -534.0,
           #'i11-ma-c05/op/mir3-ch.09/targetVoltage': -703.0,
           #'i11-ma-c05/op/mir3-ch.10/targetVoltage': -1089.0,
           #'i11-ma-c05/op/mir3-ch.11/targetVoltage': -1400.0,
           
            'robot_x': 0.,
            'robot_y': 0.,
            'robot_z': 0.}
           
        return mode_2_parameters

    def get_mode_3_parameters(self):
        mode_3_parameters = {
           'i11-ma-c04/op/mir.1-mt_tx/position': 24.749, #31.5, #33.4,
           'i11-ma-c04/op/mir.1-mt_rz/position': 2.25, #2.8, #3.6, #0.18,
           'i11-ma-c04/op/mir.1-mt_rs/position': 0.,
           'i11-ma-c05/ex/tab.1-mt_tx/position': 3.37, #4.2, #4.5
           'i11-ma-c05/ex/tab.1-mt_tz/position':-0.17,
           'i11-ma-c05/op/mir.2-mt_tz/position': -0.0700, #-0.0741, #0.,
           'i11-ma-c05/op/mir.2-mt_rx/position': 4.008, #4.044,
           'i11-ma-c05/op/mir.3-mt_tx/position': 24.378, #24.5, # 24.05
           'i11-ma-c05/op/mir.3-mt_rz/position': 1.666, #1.8585, #1.909,
           'i11-ma-c05/ex/tab.2/pitch': 7.960,
           'i11-ma-c05/ex/tab.2/roll': 0,
           'i11-ma-c05/ex/tab.2/yaw': -0.57,
           'i11-ma-c05/ex/tab.2/xC': 13.1, #14.6,
           'i11-ma-c05/ex/tab.2/zC': 0.,
           'i11-ma-c05/op/mir2-ch.00/targetVoltage': 255.0,
           'i11-ma-c05/op/mir2-ch.01/targetVoltage': 215.0,
           'i11-ma-c05/op/mir2-ch.02/targetVoltage': 12.0,
           'i11-ma-c05/op/mir2-ch.03/targetVoltage': 170.0,
           'i11-ma-c05/op/mir2-ch.04/targetVoltage': 185.0,
           'i11-ma-c05/op/mir2-ch.05/targetVoltage': 443.0,
           'i11-ma-c05/op/mir2-ch.06/targetVoltage': 322.0,
           'i11-ma-c05/op/mir2-ch.07/targetVoltage': 188.0,
           'i11-ma-c05/op/mir2-ch.08/targetVoltage': 40.0,
           'i11-ma-c05/op/mir2-ch.09/targetVoltage': -47.0,
           'i11-ma-c05/op/mir2-ch.10/targetVoltage': -3.0,
           'i11-ma-c05/op/mir2-ch.11/targetVoltage': 88.0,
           'i11-ma-c05/op/mir3-ch.00/targetVoltage': 450.9,
           'i11-ma-c05/op/mir3-ch.01/targetVoltage': 383.5,
           'i11-ma-c05/op/mir3-ch.02/targetVoltage': 478.8,
           'i11-ma-c05/op/mir3-ch.03/targetVoltage': 298.1,
           'i11-ma-c05/op/mir3-ch.04/targetVoltage': 658.9,
           'i11-ma-c05/op/mir3-ch.05/targetVoltage': 907.2,
           'i11-ma-c05/op/mir3-ch.06/targetVoltage': 1399.0,
           'i11-ma-c05/op/mir3-ch.07/targetVoltage': 1170.6,
           'i11-ma-c05/op/mir3-ch.08/targetVoltage': 1185.3,
           'i11-ma-c05/op/mir3-ch.09/targetVoltage': 942.2,
           'i11-ma-c05/op/mir3-ch.10/targetVoltage': 1122.6,
           'i11-ma-c05/op/mir3-ch.11/targetVoltage': 878.7,
           'robot_x': 4600.,
           'robot_y':-600.,
           'robot_z':-700.}

        return mode_3_parameters
    
    def get_mode_2h_parameters(self):
        '''Beam focusing using HFM and HPM'''
        mode_2h_parameters = {
           'i11-ma-c04/op/mir.1-mt_tx/position': 24.749, #31.5, #33.4,
           'i11-ma-c04/op/mir.1-mt_rz/position': 2.25, #2.8, #3.6, #0.18,
           'i11-ma-c04/op/mir.1-mt_rs/position': 0.,
           'i11-ma-c05/ex/tab.1-mt_tx/position': 3.37, #4.2, #4.5
           'i11-ma-c05/ex/tab.1-mt_tz/position':-0.17,
           'i11-ma-c05/op/mir.2-mt_tz/position': -5, #-0.0741, #0.,
           'i11-ma-c05/op/mir.3-mt_tx/position': 24.378, #24.5, # 24.05
           'i11-ma-c05/op/mir.3-mt_rz/position': 1.67, #1.8585, #1.909,
           'i11-ma-c05/ex/tab.2/pitch': 0.,
           'i11-ma-c05/ex/tab.2/roll': 0,
           'i11-ma-c05/ex/tab.2/yaw': -0.57,
           'i11-ma-c05/ex/tab.2/xC': 13.1,
           'i11-ma-c05/ex/tab.2/zC': 0.,
           'i11-ma-c05/op/mir3-ch.00/targetVoltage': 450.9,
           'i11-ma-c05/op/mir3-ch.01/targetVoltage': 383.5,
           'i11-ma-c05/op/mir3-ch.02/targetVoltage': 478.8,
           'i11-ma-c05/op/mir3-ch.03/targetVoltage': 298.1,
           'i11-ma-c05/op/mir3-ch.04/targetVoltage': 658.9,
           'i11-ma-c05/op/mir3-ch.05/targetVoltage': 907.2,
           'i11-ma-c05/op/mir3-ch.06/targetVoltage': 1399.0,
           'i11-ma-c05/op/mir3-ch.07/targetVoltage': 1170.6,
           'i11-ma-c05/op/mir3-ch.08/targetVoltage': 1185.3,
           'i11-ma-c05/op/mir3-ch.09/targetVoltage': 942.2,
           'i11-ma-c05/op/mir3-ch.10/targetVoltage': 1122.6,
           'i11-ma-c05/op/mir3-ch.11/targetVoltage': 878.7,
           'robot_x': 4400,
           'robot_y': 11850,
           'robot_z':-5250}
        
        return mode_2h_parameters
    
    def get_mode_1v_parameters(self):
        '''Beam focusing using VFM only'''
        mode_1v_parameters = {
           'i11-ma-c04/op/mir.1-mt_rs/position': +0.0000,
           'i11-ma-c04/op/mir.1-mt_rz/position': -0.0149, # 0.0
           'i11-ma-c04/op/mir.1-mt_tx/position': +0.0034, # 5.0011,
           
           'i11-ma-c05/ex/tab.1-mt_tx/position': +0.7500, #-0.25
           'i11-ma-c05/ex/tab.1-mt_tz/position': -0.3300, #-0.81,
            
           'i11-ma-c05/op/mir.2-mt_tz/position': -0.2488, # 2308,
           'i11-ma-c05/op/mir.2-mt_rx/position': +3.8766, # 3.9495,

           'i11-ma-c05/op/mir.3-mt_tx/position':-5,

           'i11-ma-c05/ex/tab.2/pitch': 7.9496,
           'i11-ma-c05/ex/tab.2/roll': 0,
           'i11-ma-c05/ex/tab.2/yaw': 0.,
           'i11-ma-c05/ex/tab.2/xC': -0.3000,
           'i11-ma-c05/ex/tab.2/zC': 0.,
            
           
           'i11-ma-c05/op/mir2-ch.00/targetVoltage': 255.0,
           'i11-ma-c05/op/mir2-ch.01/targetVoltage': 215.0,
           'i11-ma-c05/op/mir2-ch.02/targetVoltage': 12.0,
           'i11-ma-c05/op/mir2-ch.03/targetVoltage': 170.0,
           'i11-ma-c05/op/mir2-ch.04/targetVoltage': 185.0,
           'i11-ma-c05/op/mir2-ch.05/targetVoltage': 443.0,
           'i11-ma-c05/op/mir2-ch.06/targetVoltage': 322.0,
           'i11-ma-c05/op/mir2-ch.07/targetVoltage': 188.0,
           'i11-ma-c05/op/mir2-ch.08/targetVoltage': 40.0,
           'i11-ma-c05/op/mir2-ch.09/targetVoltage': -47.0,
           'i11-ma-c05/op/mir2-ch.10/targetVoltage': -3.0,
           'i11-ma-c05/op/mir2-ch.11/targetVoltage': 88.0,
           'robot_x': -8700,
           'robot_y': -900,
           'robot_z': -200}

        return mode_1v_parameters 

    def get_mode_1h_parameters(self):
        '''Beam focusing using HFM only'''
        mode_1h_parameters = {
           'i11-ma-c04/op/mir.1-mt_rs/position': +0.0000,
           'i11-ma-c04/op/mir.1-mt_rz/position': -0.0149, # 0.0
           'i11-ma-c04/op/mir.1-mt_tx/position': +0.0034, # 5.0011,
            
           'i11-ma-c05/ex/tab.1-mt_tx/position': +0.7500, #-0.25
           'i11-ma-c05/ex/tab.1-mt_tz/position': -0.3300, #-0.81,

           'i11-ma-c05/op/mir.2-mt_tz/position':-5.,

           'i11-ma-c05/op/mir.3-mt_tx/position': -2.0987, #-1.8047, #-1.960,
           'i11-ma-c05/op/mir.3-mt_rz/position': -4.6661, #-4.8089, #-4.490,

           'i11-ma-c05/ex/tab.2/pitch': 0.02, #7.949655,
           'i11-ma-c05/ex/tab.2/roll': 0,
           'i11-ma-c05/ex/tab.2/yaw': 8.94303,
           'i11-ma-c05/ex/tab.2/xC': -0.2998,
           'i11-ma-c05/ex/tab.2/zC': 0.,

           'i11-ma-c05/op/mir3-ch.00/targetVoltage': 290.0,
           'i11-ma-c05/op/mir3-ch.01/targetVoltage': 320.0,
           'i11-ma-c05/op/mir3-ch.02/targetVoltage': 265.0,
           'i11-ma-c05/op/mir3-ch.03/targetVoltage': 56.0,
           'i11-ma-c05/op/mir3-ch.04/targetVoltage': 102.0,
           'i11-ma-c05/op/mir3-ch.05/targetVoltage': 415.0,
           'i11-ma-c05/op/mir3-ch.06/targetVoltage': 37.0,
           'i11-ma-c05/op/mir3-ch.07/targetVoltage': -247.0,
           'i11-ma-c05/op/mir3-ch.08/targetVoltage': -534.0,
           'i11-ma-c05/op/mir3-ch.09/targetVoltage': -703.0,
           'i11-ma-c05/op/mir3-ch.10/targetVoltage': -1089.0,
           'i11-ma-c05/op/mir3-ch.11/targetVoltage': -1400.0,
           'robot_x': -1000,
           'robot_y': 11850,
           'robot_z': -5250}
        
        return mode_1h_parameters

    def get_tab2_values(self, mode='2', translation={'NominalPitch': 'pitch', 'NominalRoll': 'roll', 'NominalYaw': 'yaw', 'NominalXc': 'xC', 'NominalZc': 'zC'}):
        parameters = getattr(self, 'get_mode_%s_parameters' % mode)()
        tab2_values = {}
        for key in translation:
            tab2_values[key] = ['%.5f' % parameters['i11-ma-c05/ex/tab.2/%s' % translation[key]]]
        print(tab2_values)
        return tab2_values
            
        
    def set_tab2_values(self, tab2_values, wait=False):
        _start = time.time()
        self.tab2.put_property(tab2_values)
        self.tab2.init()
        self.wait(self.tab2)
        self.tab2.gotonominal()
        if wait:
            self.wait(self.tab2)
        print('set_tab2_values took %.3f seconds' % (time.time() - _start))
        
    def get_current_parameters(self):
        current_parameters = {}
        all_parameters = []
        for mode in self.modes:
            all_parameters += list(getattr(self, 'get_mode_%s_parameters' % mode)().keys())
        all_parameters = list(set(all_parameters))
        
        for key in all_parameters:
            if 'robot' not in key:
                ap = tango.AttributeProxy(key)
                current_parameters[key] = ap.read().value
            else:
                current_parameters[key] = float(self.redis.get(key))
        return current_parameters
        
    def get_mode(self):
        mode = self.redis.get('focusing_mode')
        if type(mode) == bytes:
            mode = mode.decode()
        return mode
    
    def write_value(self, ap, value, epsilon=1.e-2, check_time=0.5, timeout=7):
        _start = time.time()
        while abs(ap.read().value - value) > epsilon and timeout>time.time()-_start:
            self.wait(ap)
            try:
                ap.write(value)
                gevent.sleep(check_time)
                self.wait(ap)
            except:
                print('error writing %s' % ap.name)
            gevent.sleep(check_time)
                    
    def set_mode(self, mode='2', epsilon=1.e-2, timeout=40., check_time=0.5, adjust_mirror_voltage=False, interactive=False):
        start_set_mode = time.time()
        print('setting focusing mode %s' % mode)
        parameters = getattr(self, 'get_mode_%s_parameters' % mode)()
        parameters_items = list(parameters.items())
        parameters_items.sort(key=lambda x: x[0])
        print('target parameters')
        for key, value in parameters_items:
            print('%s: %.4f' % (key, value))
        
        self.mode = mode
        start_mode = self.get_mode()
        
        self.redis.set('focusing_mode', mode)
        if adjust_mirror_voltage:
            modify_vfm_voltages = True
            modify_hfm_voltages = True
        else:
            modify_vfm_voltages = False
            modify_hfm_voltages = False
        
        self.set_tab2_values(self.get_tab2_values(mode=mode), wait=False)
        
        jobs = []
        for key, value in parameters_items:
            if 'tab.2' in key:
                continue
            if 'robot' in key:
                self.redis.set(key, value)
                continue
            if 'targetVoltage' in key and not adjust_mirror_voltage:
                continue
            ap = tango.AttributeProxy(key)
            current_value = ap.read().value
            okay = False
            if abs(current_value - value) > epsilon:
                print('key %s, current value %.4f, target value %.4f, will execute ap.write(%s)' % (key, current_value, value, value))
                if interactive:
                    okay = False
                    answer = input("Can I execute? [Y/n]: ")
                    if answer in ['y', 'Y', '']:
                        okay = True
                    else:
                        print('%s skipping execution ...' % key)
                if not interactive or okay==True:
                    jobs.append(gevent.spawn(self.write_value, ap, value, epsilon=epsilon, check_time=check_time))
        gevent.joinall(jobs)
                        
        if adjust_mirror_voltage:
            for mirror_name, mirror in [('VFM', self.mir2), ('HFM', self.mir3)]:
                if interactive:
                    okay = False
                    answer = input('Should I set target voltages for %s? [Y/n]: ' % mirror_name)
                    if answer in ['y', 'Y', '']:
                        okay = True
                if (not interactive or okay==True) and ((mirror_name=='VFM' and modify_vfm_voltages==True) or (mirror_name== 'HFM' and modify_hfm_voltages==True)):
                    mirror.SetChannelsTargetVoltage()
                    print('Setting %s mirror voltages' % mirror_name)
                    print('Please wait for %s mirror tensions to settle ...' % mirror_name)
                    gevent.sleep(check_time*10)
                    while mirror.State().name != 'STANDBY':
                        gevent.sleep(check_time*10)
                        print('wait ', end=' ')
                    print()
                    print('done!')
                    print('%s mirror tensions converged' % mirror_name)
        print('tab2 state', self.tab2.state().name)
        self.wait(self.tab2)
        print('tab2 state', self.tab2.state().name)
        end_set_mode = time.time()
        
        print('Switch-over from focusing mode %s to focusing mode %s took %.2f seconds' % (start_mode, mode, end_set_mode-start_set_mode))
       
    def wait(self, proxy, sleeptime=0.5, timeout=27):
        _start = time.time()
        while proxy.state().name not in ['STANDBY', 'ALARM'] and time.time()-_start < timeout:
            gevent.sleep(sleeptime)
        
