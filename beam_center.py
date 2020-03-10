#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Object calculates the position of direct beam on the detector as function of distance of the wavelength and position of the detector support translational motors
'''
import logging
from detector import detector
from energy import energy
import numpy as np


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
        
        coef = np.array([[-107.48524431,   -1.61648582,    0.63448967],
                            [   4.19204684,   -1.25690816,    2.58600155]]).T
        
        intercept = np.array([ 1634.36239262,  1583.7138641])
        
        q = 0.075
        
        tx -= 36.0
        tz -= -19.65
        
        X = np.array([distance, wavelength, wavelength**2])
        return np.dot(X, coef) + intercept + np.array([tx, tz])/q
            
    def get_detector_distance(self):
        return 100
    
class beam_center(object):
    def __init__(self, pixel_size=0.075):
        try:
            self.wavelength_motor = energy()
            self.detector = detector()
        except:
            pass
        self.pixel_size = pixel_size
        
    def get_beam_center_x(self, X):
        logging.info('beam_center_x calculation')
        beam_center_vertical = self.get_beam_center()[0]
        return beam_center_vertical
    
    def get_beam_center_y(self, X):
        logging.info('beam_center_y calculation')
        beam_center_horizontal = self.get_beam_center()[1]
        return beam_center_horizontal
        
    #def get_beam_center(self):
        # 2017-07-22 After tomography experiment; Modeling tx and tz explicitly
        #coef = np.array([[ -1.10004820e-01,   1.33236212e+01,  -1.46088461e-02,  -6.30332471e+00,   2.05455735e+00],
                         #[  3.42366488e-03,   5.55270943e-03,   1.33149106e+01,  -2.28146910e+00,   2.87948678e+00]]).T

        #intercept = np.array([ 1166.84721073,  1256.11220109])
        
        #wavelength = self.wavelength_motor.read_attribute('lambda').value
        #ts         = self.distance_motor.read_attribute('position').value
        #tx         = self.det_mt_tx.read_attribute('position').value
        #tz         = self.det_mt_tz.read_attribute('position').value
        
        #X = np.array([ts, tx, tz, wavelength, wavelength**2])
        #return np.dot(X, coef) + intercept
    def get_beamstop_position(self, wavelength=None, ts=None, tx=None, tz=None, ts_offset=0, tx_offset=20.5, tz_offset=46.5, beam_center_x_reference=1432.09, beam_center_y_reference=1731.95):
    
        beam_center_x, beam_center_y = self.get_beam_center(wavelength=wavelength, ts=ts, tx=tx, tz=tz, ts_offset=ts_offset, tx_offset=tx_offset, tz_offset=tz_offset)
        
        beamstop_x = -(beam_center_x - beam_center_x_reference)*self.pixel_size
        beamstop_y = -(beam_center_y - beam_center_y_reference)*self.pixel_size
        
        if tx == None:
            tx = self.detector.position.tx.get_position()
            
        beamstop_x -=  tx - tx_offset
        
        return beamstop_x, beamstop_y
    
    def get_beam_center(self, wavelength=None, ts=None, tx=None, tz=None, ts_offset=0, tx_offset=20.5, tz_offset=46.5):
        # 2017-07-22 after tomography experiment focussing geometry changes
        # Not modeling tx and tz explicitly

        #coef = np.array([[-0.11502292, -0.89947339,  0.2325305 ],
                         #[ 0.00351967, -0.60952873,  2.22645446]]).T
        
        #intercept = np.array([ 1449.1722701,   1510.20208357]) - np.array([ 2.58, 0.31])
        
        # 2017-08-31
        # 220
        #coef = np.array([[-0.11118513, -3.68898678,  1.22657328],
                         #[ 0.00413426, -1.01159419,  2.40788137]]).T
        
        #intercept = np.array([ 1450.04096305,  1509.55992981])
        
        # 68
        #coef = np.array([[-0.1111972,  -2.96418675,  0.94843247],
                         #[ 0.00395438 -2.27778223  2.92793563]]).T
        
        #intercept = np.array([ 1449.62923794,  1510.29356759])
        
        # 118
        #coef = np.array([[-0.11119599, -3.42681679,  1.10552128],
                         #[ 0.00397335, -3.6318981,   3.42825926]]).T
        
        #intercept = np.array([1449.92271935,  1511.13875886])
        
        # 2017-08-31 beam_center3
        #coef = np.array([[ -1.07784484e-01,  -3.80411705e+00,   1.27896512e+00],
                         #[  3.14271272e-03,  -2.37131414e+00,   2.89300818e+00]]).T
        
        #intercept = np.array([ 1450.07192347,  1510.35162089])
        
        # 2017-09-13 1M prediction
        #tx_offset = 19.0
        #tz_offset = 135.0
        #coef = np.array([[-0.10702542,  3.06434418, -1.11765958],
                         #[ 0.00354367,  3.3434966,   0.78202923]]).T
        
        #intercept = np.array([ 488.95185709,  452.32962912])
        
        # 2017-09-19
        # tx_offset = 20.3
        # tz_offset = 20.5
        # coef = np.array([[-0.10779414, -2.59970090,  0.8257945 ],
        #                 [ 0.00380687, -2.07844815,  2.76835243]]).T
        #
        # intercept = np.array([ 1462.95205539,  1497.0729601 ])
        
        # 2017-11-08
        # tx_offset = 20.50
        # tz_offset = 44.50
        
        #coef = np.array([[ -1.06708151e-01,  -2.85800345e+00,   9.87774089e-01],
                         #[  2.61584487e-03,  -7.09149543e-01,   2.18903245e+00]]).T
        
        #intercept = np.array([ 1478.04730873,  1728.45302422])
        
        # 2017-11-23
        #coef = np.array([[-0.10596661, -1.72860865,  0.53923195],
                         #[ 0.00291639, -1.38650557,  2.4999531 ]]).T
                         
        #intercept = np.array([ 1477.45980118,  1728.69652014])
        
        # 2017-12-14
        #coef = np.array([[-0.1108826,  -1.06395447,  0.27716588],
                         #[ 0.00414124, -1.58808647,  2.69409456]]).T
        
        #intercept = np.array([ 1477.06896683,  1728.40462094])
        
        # 2017-12-17 ts_offset=0, tx_offset=20.5, tz_offset=44.5
        #coef = np.array([[-0.11034,    -0.85557917,  0.25766557],
                         #[ 0.00514605, -1.2018129,   2.42307962]]).T
                         
        #intercept = np.array([ 1476.81628958,  1728.71530404])

        # 2019-09-16 ts_offset=0, tx_offset=20.5, tz_offset=46.5
        coef = np.array([[-0.10803942, -1.58868791,  0.53607582],
                         [ 0.00534036, -0.95457896,  2.31875217]]).T

        intercept = np.array([1476.5555115464613, 1755.3498075722898])
        
        #print 'beam_center'
        #print 'wavelength, ts, tz, tx', wavelength, ts, tz, tx
        
        if wavelength == None:
            wavelength = self.wavelength_motor.get_wavelength()
        if ts == None:
            ts = self.detector.position.ts.get_position() 
        if tx == None:
            tx = self.detector.position.tx.get_position() 
        if tz == None:
            tz = self.detector.position.tz.get_position() 
        
        ts -= ts_offset
        tx -= tx_offset
        tz -= tz_offset
        
        X = np.array([ts, wavelength, wavelength**2])
        
        _beam_center = np.dot(X, coef) + intercept + np.array([tx, tz])/self.pixel_size
    
        try:
            if self.detector.get_roi_mode() == '4M':
                _beam_center[0] -= 550
        except:
            pass
        
        return _beam_center
    
    def get_theoric_beam_center(self, distance, wavelength, tx=36.0, tz=-19.65, tx_offset=20.5, tz_offset=46.5, q=0.075):
        
        #coef = np.array([[-110.49463429,   -3.49210741,    1.3543519],
                         #[   2.08750452,   -3.20462697,    3.61623166]]).T
        
        #intercept = np.array([ 1510.13453675,  1526.25811839])
        
        # 2019-09-16 ts_offset=0, tx_offset=20.5, tz_offset=46.5
        coef = np.array([[-0.10803942, -1.58868791,  0.53607582],
                         [ 0.00534036, -0.95457896,  2.31875217]]).T

        intercept = np.array([1476.5555115464613, 1755.3498075722898])
        
        tx -= tx_offset
        tz -= tz_offset
        
        X = np.array([distance, wavelength, wavelength**2])
        return np.dot(X, coef) + intercept + np.array([tx, tz])/q
        
    def get_old_beam_center(self):
        #Theta = np.matrix([[  1.54776707e+03,   1.65113065e+03], [  3.65108709e-01,   5.63662370e+00], [ -1.12769165e-01,   3.49706731e-03]])
        #X = np.matrix([1., self.wavelength_motor.read_attribute('lambda').value, self.distance_motor.position])
        #X = X.T
        #beam_center = Theta.T * X
        #beam_center_x = beam_center[0, 0]
        #beam_center_y = beam_center[1, 0]
        #beam_center_x -= 26.9
        #beam_center_y -= 5.7
        q = 0.075 #0.102592
        
        wavelength = self.wavelength_motor.get_wavelength()
        distance   = self.detector.position.ts.get_position() 
        tx         = self.detector.position.tx.get_position()  - 30.0
        tz         = self.detector.position.tz.get_position()  + 14.3
        #logging.info('wavelength %s' % wavelength)
        #logging.info('mt_ts %s' % distance)
        #logging.info('mt_tx %s' % tx)
        #logging.info('mt_tz %s' % tz)
        #print('wavelength %s' % wavelength)
        #print('mt_ts %s' % distance)
        #print('mt_tx %s' % tx)
        #print('mt_tz %s' % tz)
        #wavelength  = self.mono1.read_attribute('lambda').value
        #distance    = self.detector_mt_ts.read_attribute('position').value
        #tx          = self.detector_mt_tx.position
        #tz          = self.detector_mt_tz.position
        
        X = np.matrix([1., wavelength, distance, 0, 0 ]) #tx, tz])
        
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
        return self.detector.ts.get_position()
