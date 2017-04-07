#!/usr/bin/env python
'''
Object calculates the position of direct beam on the detector as function of distance of the wavelength and position of the detector support translational motors
'''

try:
    import PyTango
except:
    print 'failed to import PyTango'
import logging
from detector import detector
import numpy

class beam_center_mockup:
    def __init__(self):
        self.beam_center_x = 1500
        self.beam_center_y = 1600
        self.pixel_size = 75e-6
        
    def get_beam_center(self):
        return self.beam_center_x, self.beam_center_y
    def get_beam_center_x(self):
        return self.beam_center_x
    def get_beam_center_y(self):
        return self.beam_center_y
    def get_theoric_beam_center(self, distance, wavelength, tx=36.0, tz=-19.65):
        
        coef = numpy.array([[-107.48524431,   -1.61648582,    0.63448967],
                            [   4.19204684,   -1.25690816,    2.58600155]]).T
        
        intercept = numpy.array([ 1634.36239262,  1583.7138641])
        
        q = 0.075
        
        tx -= 36.0
        tz -= -19.65
        
        X = numpy.array([distance, wavelength, wavelength**2])
        return numpy.dot(X, coef) + intercept + numpy.array([tx, tz])/q
            
    def get_detector_distance(self):
        return 100
    
class beam_center(object):
    def __init__(self):
        try:
            self.distance_motor = PyTango.DeviceProxy('i11-ma-cx1/dt/dtc_ccd.1-mt_ts')
            self.wavelength_motor = PyTango.DeviceProxy('i11-ma-c03/op/mono1')
            self.det_mt_tx = PyTango.DeviceProxy('i11-ma-cx1/dt/dtc_ccd.1-mt_tx') #.read_attribute('position').value - 30.0
            self.det_mt_tz = PyTango.DeviceProxy('i11-ma-cx1/dt/dtc_ccd.1-mt_tz') #.read_attribute('position').value + 14.3
            self.detector = detector()
        except:
            pass
        self.pixel_size = 75e-6
        
    def get_beam_center_x(self, X):
        logging.info('beam_center_x calculation')
        #theta = numpy.matrix([ 1.65113065e+03,   5.63662370e+00,   3.49706731e-03, 9.77188997e+00])
        #orgy = X * theta.T
        if self.detector.get_roi_mode() == '4M':
            #orgy -= 550
            return self.get_beam_center()[0] - 550
        return self.get_beam_center()[0] - 550
        #return float(orgy)
    
    def get_beam_center_y(self, X):
        logging.info('beam_center_y calculation')
        #theta = numpy.matrix([  1.54776707e+03,   3.65108709e-01,  -1.12769165e-01,   9.74625808e+00])
        #orgx = X * theta.T
        #return float(orgx)
        return self.get_beam_center()[1]
        
    def get_beam_center(self):
        coef = numpy.array([[-107.48524431,   -1.61648582,    0.63448967], [   4.19204684,   -1.25690816,    2.58600155]]).T
        
        intercept = numpy.array([ 1634.36239262,  1583.7138641])
        
        q = 0.075

        wavelength = self.wavelength_motor.read_attribute('lambda').value
        distance   = self.distance_motor.read_attribute('position').value / 1.e3
        tx         = self.det_mt_tx.read_attribute('position').value - 36.00
        tz         = self.det_mt_tz.read_attribute('position').value + 19.65
         
        X = numpy.array([distance, wavelength, wavelength**2])
        return numpy.dot(X, coef) + intercept + numpy.array([tx, tz])/q
    
    def get_theoric_beam_center(self, distance, wavelength, tx=36.0, tz=-19.65):
        
        coef = numpy.array([[-107.48524431,   -1.61648582,    0.63448967],
                            [   4.19204684,   -1.25690816,    2.58600155]]).T
        
        intercept = numpy.array([ 1634.36239262,  1583.7138641])
        
        q = 0.075
        
        tx -= 36.0
        tz -= -19.65
        
        X = numpy.array([distance, wavelength, wavelength**2])
        return numpy.dot(X, coef) + intercept + numpy.array([tx, tz])/q
        
    def get_old_beam_center(self):
        #Theta = numpy.matrix([[  1.54776707e+03,   1.65113065e+03], [  3.65108709e-01,   5.63662370e+00], [ -1.12769165e-01,   3.49706731e-03]])
        #X = numpy.matrix([1., self.wavelength_motor.read_attribute('lambda').value, self.distance_motor.position])
        #X = X.T
        #beam_center = Theta.T * X
        #beam_center_x = beam_center[0, 0]
        #beam_center_y = beam_center[1, 0]
        #beam_center_x -= 26.9
        #beam_center_y -= 5.7
        q = 0.075 #0.102592
        
        wavelength = self.wavelength_motor.read_attribute('lambda').value
        distance   = self.distance_motor.read_attribute('position').value
        tx         = self.det_mt_tx.read_attribute('position').value - 30.0
        tz         = self.det_mt_tz.read_attribute('position').value + 14.3
        logging.info('wavelength %s' % wavelength)
        logging.info('mt_ts %s' % distance)
        logging.info('mt_tx %s' % tx)
        logging.info('mt_tz %s' % tz)
        print('wavelength %s' % wavelength)
        print('mt_ts %s' % distance)
        print('mt_tx %s' % tx)
        print('mt_tz %s' % tz)
        #wavelength  = self.mono1.read_attribute('lambda').value
        #distance    = self.detector_mt_ts.read_attribute('position').value
        #tx          = self.detector_mt_tx.position
        #tz          = self.detector_mt_tz.position
        
        X = numpy.matrix([1., wavelength, distance, 0, 0 ]) #tx, tz])
        
        beam_center_y = self.get_beam_center_x(X[:, [0, 1, 2, 4]])
        beam_center_x = self.get_beam_center_y(X[:, [0, 1, 2, 3]])
        
        beam_center_x += tx / q
        beam_center_y += tz / q 
        
        beam_center_x += 0.58
        beam_center_y += -1.36
        
        #2016-09-06 adjusting table
        beam_center_x += -16.3
        beam_center_y += 2.0
        
        #2016-09-07 adjusting table
        #ORGX= 1534.19470215    ORGY= 1652.97814941
        #1544.05   1652.87
        beam_center_x += 10.15
        #beam_center_y += 2.0
        
        return beam_center_x, beam_center_y
        
    def get_detector_distance(self):
        return self.distance_motor.position
