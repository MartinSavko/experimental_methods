#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '/home/experiences/proxima2a/com-proxima2a/CATS/PyCATS_DS/pycats')
from catsapi import *
import catsapi
import numpy as np
import gevent
from goniometer import goniometer
from detector import detector
from camera import camera

class cats:
    def __init__(self, host='172.19.10.23', operator=1071, monitor=10071):
        self.connection = CS8Connection()
        self.connection.connect(host, operator, monitor)
        self._type = 1
        self._toolcal = 0
        self.goniometer = goniometer()
        self.detector = detector()
        self.camera = camera()
        
        self.state_params = catsapi.state_params
        self.di_params = catsapi.di_params
        self.do_params = catsapi.do_params
        
    def on(self):
        return self.connection.powerOn()
    def off(self):
        return self.connection.powerOff()
    def abort(self):
        return self.connection.abort()
    def pause(self):
        return self.connection.pause()
    def reset(self):
        return self.connection.reset()
    def restart(self):
        return self.connection.restart()
    def regulon(self):
        return self.connection.regulon()
    def reguloff(self):
        return self.connection.reguloff()
    def message(self):
        return self.connection.message()
    def state(self):
        return self.connection.state()
    
    def closelid(self, lid):
        if lid == 1:
            return self.connection.closelid1()
        elif lid == 2:
            return self.connection.closelid2()
        elif lid == 3:
            return self.connection.closelid3()
        else:
            print '%s is not a valid lid number' % lid
            
    def openlid(self, lid):
        if lid == 1:
            return self.connection.openlid1()
        elif lid == 2:
            return self.connection.home_openlid2()
        elif lid == 3:
            return self.connection.openlid3()
        else:
            print '%s is not a valid lid number' % lid
    
    def clear_memory(self):
        return self.connection.clear_memory()
    
    def acknowledge_missing_sample(self):
        return self.connection.ack_sample_memory()
    
    def resetmotion(self):
        return self.connection.resetmotion()
        
    def prepare_for_transfer(self, attempts=3):
        self.reset()
        
        print 'executing prepare_for_transfer'
        tried = 0
        while self.isoff() and tried<=attempts:
            self.on()
            tried += 1
            if self.message() == 'Remote Mode requested':
                print 'Please turn the robot key to the remote operation position'
                break
            
        self.goniometer.set_transfer_phase(wait=True)
        if self.detector.position.ts.get_position() < 200.:
            self.detector.position.ts.set_position(200, wait=True)
        self.detector.cover.insert()
        
    def getput(self, lid, sample, x_shift=0, y_shift=0, z_shift=0, wait=True, prepare_centring=True):
        self.prepare_for_transfer()

        if self.sample_mounted() == False:
            return self.put(lid, sample, x_shift, y_shift, z_shift, wait=wait, prepare_centring=prepare_centring)
        a = self.connection.operate('getput2(%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d)' % (1, lid, sample, 0, 0, 0, 0, self._type, self._toolcal, 0, x_shift, y_shift, z_shift))
        if 'getput2' not in self.state():
            print 'getput2 not in state' 
        gevent.sleep(1)
        if wait == True:
            while self.connection._is_trajectory_running('getput2'):
                gevent.sleep(1)
        
        if prepare_centring == True:
            self.prepare_centring()
        
        return a
    
    def prepare_centring(self, dark=False):
        if self.sample_mounted() == True:
            if dark == False:
                self.goniometer.insert_backlight()
                self.goniometer.extract_frontlight()
            else:
                self.goniometer.extract_backlight()
                self.goniometer.insert_frontlight()
            self.camera.set_zoom(1)
            if self.goniometer.has_kappa():
                self.goniometer.set_position({'AlignmentZ': 0.1017, 'AlignmentY': -1.35, 'CentringX': 0.431, 'CentringY': 0.210})
            else:
                self.goniometer.set_position({'AlignmentZ': 0.0847, 'AlignmentY': -0.84, 'CentringX': 0.041, 'CentringY': -0.579})
                
        elif not self.goniometer.sample_is_loaded():
            self.acknowledge_missing_sample()
            self.reset()
            
    def put(self, lid, sample, x_shift=0, y_shift=0, z_shift=0, wait=True, prepare_centring=True):
        if self.sample_mounted():
            lid_mounted, sample_mounted = self.get_mounted_sample_id()
            if lid == lid_mounted and sample == sample_mounted:
                print 'sample already mounted'      
                return
        self.prepare_for_transfer()
        
        if self.sample_mounted() == True:
            return self.getput(lid, sample, x_shift, y_shift, z_shift, wait=wait, prepare_centring=prepare_centring)
        a = self.connection.put(1, lid, sample, self._type, self._toolcal, x_shift, y_shift, z_shift)
        if 'put' not in self.state():
            print 'put not in state' 
        gevent.sleep(1)
        if wait == True:
            while self.connection._is_trajectory_running('put'):
                gevent.sleep(1)
        
        if prepare_centring == True:
            self.prepare_centring()
        
        return a
        
    def get(self, x_shift=0, y_shift=0, z_shift=0, wait=True):
        self.prepare_for_transfer()
        
        a = self.connection.get(self._type, self._toolcal, x_shift, y_shift, z_shift)
        if 'get' not in self.state():
            print 'get not in state' 
        gevent.sleep(1)
        if wait == True:
            while self.connection._is_trajectory_running('get'):
                gevent.sleep(1)
        return a 
        
    def get_mounted_sample_id(self):
        state_dictionary = self.get_state_dictionary()
        try:
            lid = int(state_dictionary['LID_NUM_SAMPLE_MOUNTED_ON_DIFFRACTOMETER'])
        except ValueError:
            lid = -1
        try:
            sample = int(state_dictionary['NUM_SAMPLE_MOUNTED_ON_DIFFRACTOMETER'])
        except ValueError:
            sample = -1
            
        return lid, sample
    
    def get_mounted_puck_and_sample(self, n_lids=3, n_samples=16):
       lid, sample = self.get_mounted_sample_id()
       if lid == -1:
           return -1, -1
       puck, sample = divmod(sample, n_samples)
       if sample > 0:
           puck += 1
       else:
           sample = 16
       puck += (lid-1) * n_lids
       return puck, sample
    
    def wash(self):
        if self.sample_mounted():
            lid, sample = self.get_mounted_sample_id()
            return self.put(lid, sample)
            
    def get_there(self, lid, sample, x_shift=0, y_shift=0, z_shift=0):
        self.prepare_for_transfer()
        return self.connection.operate('get2(%d, %d, %d, %d, %d, %d, %d, %d)' % (1, lid, sample, x_shift, y_shift, z_shift, 0, 1))
        #return self.connection.get(self._type, self._toolcal, x_shift, y_shift, z_shift)
    
    def sample_mounted(self):
        lid, sample = self.get_mounted_sample_id()
        if lid != -1 and sample != -1:
            print 'sample %s from lid %s mounted' % (sample, lid)
            return True
	#if self.goniometer.sample_is_loaded():
        #    return True
        return False
        
    def dry(self):
        return self.connection.dry(1)
        
    def soak(self):
        return self.connection.soak(1, 2)
    
    def soak_toolcal(self):
        return self.connection.tremp_toolcal()
    
    def tremp_toolcal(self):
        return self.soak_toolcal()

    def calibrate(self):
        return self.connection.toolcal()

    def dry_and_soak(self):
        self.connection.dry_soak(1, 2)
    
    def home(self):
        self.connection.home(1)

    def safe(self):
        self.goniometer.abort()
        self.connection.safe(1)
    
    def get_puck_presence(self):
        a = np.array([int(n) for n in self.connection.di()[15: 15+9]])
        print a
        return a
       
    def get_state(self):
        return self.state()
    
    def get_state_vector(self, verification_string='state('):
        received = False
        while not received:
            state = self.get_state()
            if verification_string == state[:len(verification_string)]:
                received = True
            else:
                gevent.sleep(0.1)
            
        print 'state', state
        state_vector = state.strip(verification_string+')').split(',')
        return state_vector
    
    def get_state_dictionary(self):
        state_vector = self.get_state_vector()    
        state_dictionary = dict(zip(self.state_params, state_vector))
        return state_dictionary
    
    def get_di(self):
        return self.connection.di()
    
    def get_di_vector(self, verification_string='di('):
        received = False
        while not received:
            di = self.get_di()
            if verification_string == di[:len(verification_string)]:
                received = True
            else:
                gevent.sleep(0.1)
        print 'di', di
        di = di.strip(verification_string+')')
        di_vector = map(int, di)
        return di_vector
    
    def get_di_dictionary(self):
        di_vector = self.get_di_vector()
        di_dictionary = dict(zip(self.di_params, di_vector))
        return di_dictionary
        
    def get_do(self):
        return self.connection.do()
    
    def get_do_vector(self, verification_string='do('):
        received = False
        while not received:
            do = self.get_do()
            if verification_string == do[:len(verification_string)]:
                received = True
            else:
                gevent.sleep(0.1)
        print 'do', do
        do = do.strip(verification_string+')')
        do_vector = map(int, do)
        return do_vector
    
    def get_do_dictionary(self):
        do_vector = self.get_do_vector()
        do_dictionary = dict(zip(self.do_params, do_vector))
        return do_dictionary
    
    def ison(self):
        return self.get_state_dictionary()['POWER_1_0'] == '1'
    
    def isoff(self):
        return self.get_state_dictionary()['POWER_1_0'] == '0'
    
def main():
    
    c = cats()
    
    import optparse
    
    parser = optparse.OptionParser()
    parser.add_option('-c', '--command', default=None, type=str, help='Specify command to execute')
    parser.add_option('-l', '--lid', default=None, type=int, help='Specify the lid')
    parser.add_option('-p', '--puck', default=None, type=int, help='Specify the puck')
    parser.add_option('-s', '--sample', default=None, type=int, help='Specify the sample')
    
    options, args = parser.parse_args()
    
    if options.command in ['get', 'getput', 'put']:
        getattr(c, options.command)(options.lid, options.sample)
    elif options.command in ['openlid', 'closelid']:
        getattr(c, options.command)(options.lid)
    elif options.command is None:
        sys.exit('No command specified, exiting ...')
    else:
        getattr(c, options.command)()
        
if __name__ == '__main__':
    main()
        
    
