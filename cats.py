#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '/927bis/ccd/gitRepos/Cats/PyCATS_DS/pycats')
from catsapi import *
import numpy as np

class cats:
    def __init__(self, host='172.19.10.2', operator=1071, monitor=10071):
        self.connection = CS8Connection()
        self.connection.connect(host, operator, monitor)
        self._type = 1
        self._toolcal = 0
    
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
        
    def getput(self, lid, sample, x_shift=0, y_shift=0, z_shift=0):
        #self.connection.getput(1, lid, sample, self._type, self._toolcal, x_shift, y_shift, z_shift)
        return self.connection.operate('getput2(%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d)' % (1, lid, sample, 0, 0, 0, 0, self._type, self._toolcal, 0, x_shift, y_shift, z_shift))
        
    def put(self, lid, sample, x_shift=0, y_shift=0, z_shift=0):
        self.connection.put(1, lid, sample, self._type, self._toolcal, x_shift, y_shift, z_shift)
    
    def get(self, x_shift=0, y_shift=0, z_shift=0):
        return self.connection.get(self._type, self._toolcal, x_shift, y_shift, z_shift)
    
    def get_there(self, lid, sample, x_shift=0, y_shift=0, z_shift=0):
        return self.connection.operate('get2(%d, %d, %d, %d, %d, %d, %d, %d)' % (1, lid, sample, x_shift, y_shift, z_shift, 0, 1))
        #return self.connection.get(self._type, self._toolcal, x_shift, y_shift, z_shift)
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
        self.connection.safe(1)
    
    def get_puck_presence(self):
        return np.array([int(n) for n in self.connection.di()[15: 15+9]])
