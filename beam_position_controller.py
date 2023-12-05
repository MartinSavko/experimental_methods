#!/usr/bin/env python

import time
import zmq
import traceback
import pickle
import numpy as np

from pid import pid
from monitor import sai
from motor import tango_motor
from camera import camera
from scipy.ndimage import center_of_mass

class position_controller(pid):

    def __init__(self, port=9001, kp=0, ki=0, kd=0, setpoint=None, max_output=None, min_output=None, on=False, reverse=False, ponm=True, period=1, valid_input_digits=5, valid_output_digits=4):
        
        self.last_valid = True
        context = zmq.Context()
        self.master = None
        self.port = port
        try:
            self.socket = context.socket(zmq.REP)
            self.socket.bind("tcp://*:%s" % self.port)
            self.master = True
            print('controller successfully bound to port %d' % self.port)
        except:
            self.socket = context.socket(zmq.REQ)
            self.socket.connect("tcp://localhost:%s" % self.port)
            self.master = False
            print('port %d, in use' % port)

        super().__init__(kp, ki, kd, setpoint, max_output=max_output, min_output=min_output, on=on, reverse=reverse, ponm=ponm, period=period, valid_input_digits=valid_input_digits, valid_output_digits=valid_output_digits)
    
    def check_for_requests(self):
        _start = time.time()
        try:
            requests = self.socket.recv(flags=zmq.NOBLOCK)
            request = pickle.loads(requests)
            for key in request:
                print('request:', key, request[key])
                if 'set_' in key:
                    print('%s set to %s' % (key, request[key]))
                    value = getattr(self, '%s' % key)(request[key])
                else:
                    value = getattr(self, '%s' % key)()
                    print('%s returned %s' % (key, value))
                self.socket.send(pickle.dumps(value))
            print('requests processed in %.7f seconds' % (time.time() - _start))
        except (zmq.error.Again, zmq.error.ZMQError):
            pass
        except:
            print(traceback.print_exc())
            
    def serve(self):
        
        self.initialize()
        
        while True:
            self.compute()
            if self.output != self.last_output:
                self.output_device.set_position(self.output, accuracy=self.output_accuracy)
                self.last_output = self.output
            self.check_for_requests()
            time.sleep(self.period)

    def initialize(self):
        super().initialize()
        output = self.output_device.get_position()
        self.last_output = output
        self.ie = output
        
    def set_on(self, on=True):
        if self.master:
            super().set_on(on=on)
        else:
            self.communicate({'set_on': on})
            self.on = on

    def get_output(self):
        if self.master:
            return super().get_output()
        return self.communicate({'get_output': None})

    def get_last_output(self):
        if self.master:
            return super().get_last_output()
        return self.communicate({'get_last_output': None})
            
    def communicate(self, message={}):
        self.socket.send(pickle.dumps(message))
        return pickle.loads(self.socket.recv())
    
class camera_beam_position_controller(position_controller):
    
    def __init__(self, output_device_name='i11-ma-c05/op/mir.2-mt_tz', channels=(0,), port=9010, kp=0, ki=0, kd=0, setpoint=0.5, max_output=None, min_output=None, on=False, reverse=False, ponm=True, period=0.1):
        
        self.output_device_name = output_device_name
        self.channel = channels[0]
        self.input_device = camera()
        self.output_device = tango_motor(output_device_name)
        
        if setpoint is None:
            setpoint = self.get_input()
        
        super().__init__(port=port, kp=kp, ki=ki, kd=kd, setpoint=setpoint, max_output=max_output, min_output=min_output, on=on, reverse=reverse, ponm=ponm, period=period)
        
        
    def get_input(self, nsamples=11):
        
        _inputs = []
        for k in range(nsamples):
            img = self.input_device.get_image(color=False)
            
            img[img<img.max()*0.5] = 0
            com =  np.array(center_of_mass(img))/np.array(self.input_device.get_image_dimensions())[::-1]
        
            _inputs.append(com[self.channel])
        _input = np.median(_inputs)
        _input = round(_input, self.valid_input_digits)
        return _input


    def operational_conditions_are_valid(self, min_count=2000, threshold=255./2):
        
        img = self.input_device.get_image(color=False)
        
        valid = (img>threshold).sum() > min_count 
        
        if valid and not self.last_valid:
            self.initialize()
        
        self.last_valid = valid
        
        return valid
    

class sai_beam_position_controller(position_controller):
    
    def __init__(self, output_device_name='i11-ma-c05/op/mir.2-mt_rx', input_device_name='i11-ma-c00/ca/sai.4', channels=(0, 1), port=9001, kp=0, ki=0, kd=0, setpoint=None, max_output=None, min_output=None, on=True, reverse=False, ponm=True, period=0.1):
        
        self.output_device_name = output_device_name
        self.channel_a = channels[0]
        self.channel_b = channels[1]
        
        self.input_device = sai(device_name=input_device_name)
        self.output_device = tango_motor(output_device_name)
        
        if setpoint is None:
            setpoint = self.get_input()
        
        super().__init__(port=port, kp=kp, ki=ki, kd=kd, setpoint=setpoint, max_output=max_output, min_output=min_output, on=on, reverse=reverse, ponm=ponm, period=period)
        
        
    def get_input(self):
        
        _input = self.input_device.get_channel_difference(self.channel_a, self.channel_b)
        
        return _input


    def operational_conditions_are_valid(self, min_current=0.1):
        
        valid = self.input_device.get_total_current() >= min_current
        
        if valid and not self.last_valid:
            self.initialize()
        
        self.last_valid = valid
        
        return valid

# high sped, lower precision
# velocity: 0.05; acceleration/deceleration: 0.40, accuracy: 0.0003 mrad
parameters = {
    'vertical_pitch': {
        'output_device_name': 'i11-ma-c05/op/mir.2-mt_rx',
        'port': 9007,
        'min_output': 3.855,
        'max_output': 3.950,
        'sai': {'kp': 0.48925,
                'ki': 0.45613,
                'kd': 0.13119,
                'reverse': True,
                'setpoint': 0.0,
                'channels': (2, 3)},
        'cam': {'kp': 0.02086,
                'ki': 0.01933,
                'kd': 0.00562,
                'reverse': False,
                'setpoint': 0.5,
                'channels': (0,)}
        },
    'horizontal_pitch': {
        'output_device_name': 'i11-ma-c05/op/mir.3-mt_rz',
        'port': 9008,
        'min_output': -4.675,
        'max_output': -4.655,
        'sai': {'kp': 1.37680,
                'ki': 1.31124,
                'kd': 0.36141,
                'reverse': True,
                'setpoint': 0.0,
                'channels': (0, 1)},
        'cam': {'kp': 0.05121,
                'ki': 0.04800,
                'kd': 0.01366,
                'reverse': False,
                'setpoint': 0.5,
                'channels': (1,)}
        },        
    'vertical_trans': {
        'output_device_name': 'i11-ma-c05/op/mir.2-mt_tz',
        'port': 9009,
        'min_output': -0.2786 - 0.1,
        'max_output': -0.2786 + 0.1,
        'sai': {'kp': 0.48925,
                'ki': 0.45613,
                'kd': 0.13119,
                'reverse': True,
                'setpoint': 0.0,
                'channels': (2, 3)},
        'cam': {'kp': 0.02086,
                'ki': 0.01933,
                'kd': 0.00562,
                'reverse': True,
                'setpoint': 0.5,
                'channels': (0,)}
        },
    'horizontal_trans': {
        'output_device_name': 'i11-ma-c05/op/mir.3-mt_tx',
        'port': 9010,
        'min_output': -2.1153 - 0.08,
        'max_output': -2.1153 + 0.08,
        'sai': {'kp': 1.37680,
                'ki': 1.31124,
                'kd': 0.36141,
                'reverse': True,
                'setpoint': 0.0,
                'channels': (0, 1)},
        'cam': {'kp': 0.05121,
                'ki': 0.04800,
                'kd': 0.01366,
                'reverse': False,
                'setpoint': 0.5,
                'channels': (1,)}
        }
    
    }

# low speed, high precision
# velocity: 0.01; acceleration/deceleration: 0.4, accuracy: 0.00025 mrad
parameters_ls = {
    'vertical_pitch': {
        'output_device_name': 'i11-ma-c05/op/mir.2-mt_rx',
        'port': 9007,
        'min_output': 3.855,
        'max_output': 3.950,
        'sai': {'kp': 0.50559,
                'ki': 0.25101,
                'kd': 0.25459,
                'reverse': True,
                'setpoint': 0.0,
                'channels': (2, 3)},
        'cam': {'kp': 0.02249,
                'ki': 0.01132,
                'kd': 0.01116,
                'reverse': False,
                'setpoint': 0.5,
                'channels': (0,)}
        },
    'horizontal_pitch': {
        'output_device_name': 'i11-ma-c05/op/mir.3-mt_rz',
        'port': 9008,
        'min_output': -4.685,
        'max_output': -4.635,
        'sai': {'kp': 1.34239,
                'ki': 0.39537,
                'kd': 1.13944,
                'reverse': True,
                'setpoint': 0.0,
                'channels': (0, 1)},
        'cam': {'kp': 0.05325,
                'ki': 0.01576,
                'kd': 0.04497,
                'reverse': False,
                'setpoint': 0.5,
                'channels': (1,)}
        }
    
    }

def get_bpc(monitor='cam', actuator='vertical_trans', period=1., ponm=False):
    
    params = {'period': period, 'ponm': ponm}
    for parameter in ['output_device_name', 'min_output', 'max_output', 'port']:
        params[parameter] = parameters[actuator][parameter]
    for parameter in ['kp', 'ki', 'kd', 'reverse', 'setpoint', 'channels']:
        params[parameter] = parameters[actuator][monitor][parameter]
    
    print(params)
    if monitor == 'cam':
        bpc = camera_beam_position_controller(**params)
    else:
        bpc = sai_beam_position_controller(**params)
        
    return bpc


def autotune(monitor='cam', actuator='vertical_trans', D=0.01, periods=27, sleeptime=1., epsilon=0.001):
    
    bpc = get_bpc(monitor=monitor, actuator=actuator)

    durations = []
    amplitudes = []
    shifts = []
    outputs = []
    
    _start = time.time()

    start_position = bpc.output_device.get_position()
    signum = -1.
    bpc.output_device.set_position(start_position + signum*D, wait=True)
    time.sleep(sleeptime)
    
    for k in range(periods):
        print('period %d' % k)
        signum *= -1
        _start_input = bpc.get_input()
        _start = time.time()
        bpc.output_device.set_position(start_position + signum*D, wait=True)
        _end = time.time()
        _end_input = bpc.get_input()
        duration = _end - _start
        durations.append(duration)
        shift = _end_input - _start_input
        shifts.append(shift)
        amplitudes.append(np.abs(shift))
        output = signum*D
        outputs.append(output)
        print('T: %.4f, A: %.4f, D: %.4f\n' % (duration, shift, signum*D))
        time.sleep(sleeptime)
    
    bpc.output_device.set_position(start_position, wait=True)

    durations = np.array(durations)
    shifts = np.array(shifts)
    amplitudes = np.array(amplitudes)
    outputs = np.array(outputs)
    
    A = np.median(amplitudes)
    Pu = 2*np.median(durations)
    
    tuning_parameters = get_tuning_parameters(D, A, Pu)
    print('PID tuning_parameters:')
    print(tuning_parameters)
    tp = tuning_parameters['PID']
    for key in tp:
        print("'%s': %.5f," % (key, tp[key]))
    
    Reversed = not np.alltrue(np.sign(shifts) == np.sign(outputs))
    print('controller direction reversed?:', Reversed)
    results = {'durations': durations, 'amplitudes': amplitudes, 'shifts': shifts, 'outputs': outputs, 'start_position': start_position, 'tuning_parameters': tuning_parameters, 'reverse': Reversed}
    
    f = open('autotune_%s_%s_D_%.4f_periods_%d_%s.pickle' % (monitor, actuator, D, periods, time.asctime().replace(' ', '_')), 'wb')
    pickle.dump(results, f)
    f.close()

    #last_d_sign = 1.
    #while time.time() - _start < 60:
        #t = time.time() - _start
        #print('t', t)
        #times.append(time.time()-_start)
        #pe = bpc.get_pe()
        #print('pe', pe)
        #pes.append(pe)
        #outputs.append(bpc.output_device.get_position())
        #s = np.sign(pe + signum * epsilon)
        #print('s, signum', s, signum)
        #if s != signum:
            #signum *= -1
            #last_d_sign *= -1
            #bpc.output_device.set_position(start_position + last_d_sign * D, wait=False)
            
def get_tuning_parameters(D, A, Pu):

    print('D: %.4f, A: %.4f, Pu: %.4f' % (D, A, Pu))
    Ku = 4 * D / (A * np.pi)

    tuning_parameters = {}
    for control in ['PID', 'PI']:
        if control == 'PID':
            kp = 0.6 * Ku
            ki = 1.20 * Ku / Pu
            kd = 0.075 * Ku * Pu
            tuning_parameters[control] = {'kp': kp, 'ki': ki, 'kd': kd}
        elif control == 'PI':
            kp = 0.4 * Ku
            ki = 0.48 * Ku / Pu
            kd = 0.
            tuning_parameters[control] = {'kp': kp, 'ki': ki, 'kd': kd}

    return tuning_parameters

def main():
    vbpc = beam_position_controller(2, 3, output_device_name='i11-ma-c05/op/mir.2-mt_rx', port=9005, kp=0.4775, ki=0.4775, kd=0.1194, setpoint=0.016, min_output=3.863, max_output=3.880, reverse=True)

    hbpc = beam_position_controller(0, 1, output_device_name='i11-ma-c05/op/mir.3-mt_rz', port=9006, kp=1.3263, ki=1.0610, kd=0.4145, setpoint=-0.0016, min_output=-4.6822, max_output=-4.6421, reverse=True)

    # autotune
    # A = Input amplitude; D = Output shift; Ku = 4 * D / (A * pi); Pu = Peak distance [seconds]
    # PI : kp = 0.4 * Ku; ki = 0.48 * Ku / Pu
    # PID: kp = 0.6 * Ku; ki = 1.20 * Ku / Pu; kd = 0.075 * Ku * Pu  
    
    # vertical: A = 0.008; Pu = 2., D = 0.005
    # Ku = 0.7958 
    # kp = 0.4775
    # ki = 0.4775
    # kd = 0.1194

    # horizontal: A = 0.00576; Pu = 2.5; D = 0.010
    # Ku = 2.2105
    # kp = 1.3263
    # ki = 1.0610
    # kd = 0.4145
    




            

    
    
