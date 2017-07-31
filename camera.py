# -*- coding: utf-8 -*-
import gevent

import PyTango
import time
import numpy as np

class camera(object):
    def __init__(self):
        self.md2 = PyTango.DeviceProxy('i11-ma-cx1/ex/md2')
        self.prosilica = PyTango.DeviceProxy('i11-ma-cx1/ex/imag.1')
        self.name = 'prosilica'
        self.shape = self.prosilica.image.shape
       
    def get_point(self):
        return self.get_image()
        
    def get_image(self):
        return self.prosilica.image
    
    def get_image_id(self):
        return self.prosilica.imagecounter
        
    def get_rgbimage(self):
        return self.prosilica.rgbimage.reshape((self.shape[0], self.shape[1], 3))
        
    def get_zoom(self):
        return self.md2.coaxialcamerazoomvalue
    
    def set_zoom(self, value, wait=True):
        if value is not None:
            value = int(value)
            self.md2.coaxialcamerazoomvalue = value
        if wait:
            while self.md2.getMotorState('Zoom').name == 'MOVING':
                gevent.sleep(0.1)
                
    def get_calibration(self):
        return np.array([self.md2.coaxcamscaley, self.md2.coaxcamscalex])
        
    def get_vertical_calibration(self):
        return self.md2.coaxcamscaley
        
    def get_horizontal_calibration(self):
        return self.md2.coaxcamscalex

    def set_exposure(self, exposure=0.05):
        self.prosilica.exposure = exposure
        
    def get_exposure(self):
        return self.prosilica.exposure
    