#!/usr/bin/env python

import gevent
import redis
import PyTango
import time

class focusing:
    '''
    Table 1. Mirror and experimental tables positions for all focussing modes.

    Motor           mode1   mode 1.5.v  mode 1.5.h    mode 2   mode 3h    mode 3
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
    
        self.mir2 = PyTango.DeviceProxy('i11-ma-c05/op/mir2-vfm')
        self.mir3 = PyTango.DeviceProxy('i11-ma-c05/op/mir3-hfm')
        
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
           'i11-ma-c04/op/mir.1-mt_rz/position': 0.0,
           'i11-ma-c04/op/mir.1-mt_tx/position': 5.0011,
           'i11-ma-c05/op/mir.2-mt_tz/position':-0.2308,
           'i11-ma-c05/op/mir.2-mt_rx/position': 3.9415,
           'i11-ma-c05/op/mir.3-mt_tx/position':-1.8517,
           'i11-ma-c05/op/mir.3-mt_rz/position':-4.8089,
           'i11-ma-c05/ex/tab.2/pitch': 7.9497,
           'i11-ma-c05/ex/tab.2/yaw': 8.9433,
           'i11-ma-c05/ex/tab.2/xC': -0.2998,
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
    
    def get_mode_3h_parameters(self):
        '''Beam focusing using HFM and HPM'''
        mode_3h_parameters = {
           'i11-ma-c04/op/mir.1-mt_tx/position': 24.749, #31.5, #33.4,
           'i11-ma-c04/op/mir.1-mt_rz/position': 2.25, #2.8, #3.6, #0.18,
           'i11-ma-c04/op/mir.1-mt_rs/position': 0.,
           'i11-ma-c05/ex/tab.1-mt_tx/position': 3.37, #4.2, #4.5
           'i11-ma-c05/ex/tab.1-mt_tz/position':-0.17,
           'i11-ma-c05/op/mir.2-mt_tz/position': -5, #-0.0741, #0.,
           'i11-ma-c05/op/mir.3-mt_tx/position': 24.378, #24.5, # 24.05
           'i11-ma-c05/op/mir.3-mt_rz/position': 1.67, #1.8585, #1.909,
           'i11-ma-c05/ex/tab.2/pitch': 0.,
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
        
        return mode_3h_parameters
    
    def get_mode_2v_parameters(self):
        '''Beam focusing using VFM only'''
        mode_2v_parameters = {
           'i11-ma-c04/op/mir.1-mt_tx/position': 5.,
           'i11-ma-c05/ex/tab.1-mt_tx/position':-0.25,
           'i11-ma-c05/ex/tab.1-mt_tz/position':-0.81, #-0.170,
           'i11-ma-c04/op/mir.1-mt_rz/position': 0.0,
           'i11-ma-c04/op/mir.1-mt_tx/position': 5.0011,
           'i11-ma-c05/op/mir.2-mt_tz/position':-0.2308,
           'i11-ma-c05/op/mir.2-mt_rx/position': 3.9495,
           'i11-ma-c05/op/mir.3-mt_tx/position':-5,
           'i11-ma-c05/ex/tab.2/pitch': 7.9497,
           'i11-ma-c05/ex/tab.2/yaw': 0.,
           'i11-ma-c05/ex/tab.2/xC': -0.2998,
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

        return mode_2v_parameters 

    def get_mode_2h_parameters(self):
        '''Beam focusing using HFM only'''
        mode_2h_parameters = {
           'i11-ma-c04/op/mir.1-mt_tx/position': 5.,
           'i11-ma-c05/ex/tab.1-mt_tx/position': -0.25,
           'i11-ma-c05/ex/tab.1-mt_tz/position': -0.81, #-0.170,
           'i11-ma-c04/op/mir.1-mt_rz/position': 0.0,
           'i11-ma-c04/op/mir.1-mt_tx/position': 5.0011,
           'i11-ma-c05/op/mir.2-mt_tz/position':-5.,
           'i11-ma-c05/op/mir.3-mt_tx/position': -1.8047, #-1.960,
           'i11-ma-c05/op/mir.3-mt_rz/position': -4.8089, #-4.490,
           'i11-ma-c05/ex/tab.2/pitch': 0, #7.949655,
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
        
        return mode_2h_parameters

    def get_current_parameters(self):
        current_parameters = {}
        all_parameters = []
        for mode in self.modes:
            all_parameters += getattr(self, 'get_mode_%s_parameters' % mode)().keys()
        all_parameters = list(set(all_parameters))
        
        for key in all_parameters:
            if 'robot' not in key:
                ap = PyTango.AttributeProxy(key)
                current_parameters[key] = ap.read().value
            else:
                current_parameters[key] = self.redis.get(key)
        return current_parameters
        
    def get_mode(self):
        return self.redis.get('focusing_mode')
    
    def set_mode(self, mode='2', epsilon = 1.e-3, timeout=40., check_time=0.5, interactive=False):
        start_set_mode = time.time()
        print 'setting focusing mode %s' % mode
        parameters = getattr(self, 'get_mode_%s_parameters' % mode)()
        parameters_items = parameters.items()
        parameters_items.sort(key=lambda x: x[0])
        print 'target parameters'
        for key, value in parameters_items:
            print '%s: %.4f' % (key, value)
        
        self.mode = mode
        start_mode = self.get_mode()
        
        self.redis.set('focusing_mode', mode)
        
        modify_vfm_voltages = False
        modify_hfm_voltages = False
        
        for key, value in parameters_items:
            if 'robot' in key:
                self.redis.set(key, value)
            elif 'tab.2' not in key:
                ap = PyTango.AttributeProxy(key)
                current_value = ap.read().value
                if abs(current_value - value) > epsilon:
                    print 'key %s, current value %.4f, target value %.4f, will execute ap.write(%s)' % (key, current_value, value, value)
                    if interactive:
                        okay = False
                        answer = raw_input("Can I execute? [Y/n]: ")
                        if answer in ['y', 'Y', '']:
                            okay = True
                        else:
                            print('%s skipping execution ...' % key)
                    
                    if not interactive or okay == True:
                        ap.write(value)
                        if 'mir2' in key and 'targetVoltage' in key:
                            modify_vfm_voltages = True
                        if 'mir3' in key and 'targetVoltage' in key:
                            modify_hfm_voltages = True
    
        for key, value in parameters_items:
            if 'tab.2' in key:
                ap = PyTango.AttributeProxy(key)
                current_value = ap.read().value
                okay = False
                if abs(current_value - value) > epsilon:
                    print 'key %s, current value %.4f, target value %.4f, will execute ap.write(%s)' % (key, current_value, value, value)
                    if interactive == True:
                        answer = raw_input("Can I execute? [Y/n]: ")
                        if answer in ['y', 'Y', '']:
                            okay = True
                        else:
                            print('%s skipping execution ...' % key)
                    if not interactive or okay == True:
                        ap.write(value)
                        _start = time.time()
                        while abs(current_value - value) > epsilon and abs(time.time()-_start) < timeout:
                            gevent.sleep(check_time)
                    
        for mirror_name, mirror in [('VFM', self.mir2), ('HFM', self.mir3)]:
            if interactive:
                okay = False
                answer = raw_input('Should I set target voltages for %s? [Y/n]: ' % mirror_name)
                if answer in ['y', 'Y', '']:
                    okay = True
            if (not interactive or okay==True) and ((mirror_name=='VFM' and modify_vfm_voltages==True) or (mirror_name== 'HFM' and modify_hfm_voltages==True)):
                mirror.SetChannelsTargetVoltage()
                print 'Setting %s mirror voltages' % mirror_name
                print 'Please wait for %s mirror tensions to settle ...' % mirror_name
                gevent.sleep(check_time*10)
                while mirror.State().name != 'STANDBY':
                    gevent.sleep(check_time*10)
                    print 'wait ',
                print
                print 'done!'
                print '%s mirror tensions converged' % mirror_name
        end_set_mode = time.time()
        print 'Switch-over from focusing mode %s to focusing mode %s took %.2f seconds' % (start_mode, mode, end_set_mode-start_set_mode)
       
        
        
