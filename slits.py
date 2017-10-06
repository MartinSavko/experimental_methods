#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Slits. 
'''

from motor import tango_motor
import gevent

class slits_mockup():

    def __init__(self, order=1):
        self.order = order
        
    def get_alignment_actuators(self):
        if self.order > 2:
            return None, None
        else:
            return None, None, None, None
    
    def get_horizontal_gap(self):
        return None
    
    def set_independent_mode(self):
        self.independent_mode = True
    
    def set_horizontal_gap(self, gap):
        pass
        
    def get_vertical_gap(self):
        return None
        
    def set_vertical_gap(self, gap):
        pass
        
    def get_horizontal_position(self):
        return None
    
    def set_horizontal_position(self, position):
        pass
    
    def get_vertical_position(self):
        return None
    
    def set_vertical_position(self, position):
        pass
    
        
class independent_edge_slits:
    
    slits_names={1: 'i11-ma-c02/ex/fent',
                 2: 'i11-ma-c04/ex/fent'}
                 
    def __init__(self, order=1):
        
        base_name = self.slits_names[order]
        
        self.h = tango_motor('%s_h.%d' % (base_name, order))
        self.v = tango_motor('%s_v.%d' % (base_name, order))
        self.i = tango_motor('%s_h.%d-mt_i' % (base_name, order))
        self.o = tango_motor('%s_h.%d-mt_o' % (base_name, order))
        self.u = tango_motor('%s_v.%d-mt_u' % (base_name, order))
        self.d = tango_motor('%s_v.%d-mt_d' % (base_name, order))
        
    
    def get_alignment_actuators(self):
        return self.i, self.o, self.u, self.d
        
    def get_horizontal_gap(self):
        return self.i.get_position() + self.o.get_position()
      
    def set_independent_mode(self):
        self.h.device.setindependantmode()
        self.v.device.setindependantmode()
        
    def set_horizontal_gap(self, gap):
        position = gap/2.
        self.i.wait()
        i = gevent.spawn(self.i.set_position, position)
        self.o.wait()
        o = gevent.spawn(self.o.set_position, position)
        gevent.joinall([i, o])
        
    def get_vertical_gap(self):
        return self.u.get_position() + self.d.get_position()
        
    def set_vertical_gap(self, gap):
        position = gap/2.
        self.u.wait()
        u = gevent.spawn(self.u.set_position, position)
        self.d.wait()
        d = gevent.spawn(self.d.set_position, position)
        gevent.joinall([u, d])
        
    def get_horizontal_position(self):
        return (self.o.get_position() - self.i.get_position())/2.
    
    def set_horizontal_position(self, position):
        gap = self.get_horizontal_gap()
        i_position = gap/2. - position
        o_position = gap/2. + position
        self.i.wait()
        i = gevent.spawn(self.i.set_position, i_position)
        self.o.wait()
        o = gevent.spawn(self.o.set_position, o_position)
        gevent.joinall([i, o])
        
    def get_vertical_position(self):
        return (self.u.get_position() - self.d.get_position())/2.
    
    def set_vertical_position(self, position):
        gap = self.get_vertical_gap()
        u_position = gap/2. + position
        d_position = gap/2. - position
        self.u.wait()
        u = gevent.spawn(self.u.set_position, u_position)
        self.d.wait()
        d = gevent.spawn(self.d.set_position, d_position)
        gevent.joinall([u, d])
        
        
class dependent_edge_slits:
    
    slits_names={3: 'i11-ma-c05/ex/fent',
                 5: 'i11-ma-c06/ex/fent',
                 6: 'i11-ma-c06/ex/fent'}
                 
    def __init__(self, order=3):
        
        base_name = self.slits_names[order]
        
        self.h = tango_motor('%s_h.%d' % (base_name, order))
        self.v = tango_motor('%s_v.%d' % (base_name, order))
        self.horizontal_gap = tango_motor('%s_h.%d-mt_ec' % (base_name, order))
        self.horizontal_position = tango_motor('%s_h.%d-mt_tx' % (base_name, order))
        self.vertical_gap = tango_motor('%s_v.%d-mt_ec' %  (base_name, order))
        self.vertical_position = tango_motor('%s_v.%d-mt_tz' % (base_name, order))

    def get_alignment_actuators(self):
        return self.horizontal_position, self.vertical_position
     
    def get_gap_actuators(self):
        return self.horizontal_gap, self.vertical_gap
        
    def set_pencil_scan_gap(self, k, scan_gap=0.1, wait=True):
        actuator = self.get_gap_actuators()[k]
        actuator.set_position(scan_gap, wait=wait)
        
    def get_horizontal_gap(self):
        return self.horizontal_gap.get_position()
        
    def set_horizontal_gap(self, gap):
        self.horizontal_gap.set_position(gap)
        
    def get_vertical_gap(self):
        return self.vertical_gap.get_position()
        
    def set_vertical_gap(self, gap):
        self.vertical_gap.set_position(gap)
        
    def get_horizontal_position(self):
        return self.horizontal_position.get_position()
    
    def set_horizontal_position(self, position):
        self.horizontal_position.set_position(position)
        
    def get_vertical_position(self):
        return self.vertical_position.get_position()
    
    def set_vertical_position(self, position):
        self.vertical_position.set_position(position)

class slits1(independent_edge_slits):
    
    def __init__(self):
        
        independent_edge_slits.__init__(self, 1)
        
        self.default_speed = 1.0
        self.scan_speed = 0.1
        

class slits2(independent_edge_slits):
    
    def __init__(self):
        
        independent_edge_slits.__init__(self, 2)
        
        self.default_speed = 1.0
        self.scan_speed = 0.1
        
        
class slits3(dependent_edge_slits):
    
    def __init__(self):
        
        dependent_edge_slits.__init__(self, 3)
        
        self.default_speed = 0.5
        self.scan_speed = 0.1
        

class slits5(dependent_edge_slits):
    
    def __init__(self):
        
        dependent_edge_slits.__init__(self, 5)
        
        self.default_speed = 0.2
        self.scan_speed = 0.1

class slits6(dependent_edge_slits):
    
    def __init__(self):
        
        dependent_edge_slits.__init__(self, 6)
        
        self.default_speed = 0.2
        self.scan_speed = 0.1