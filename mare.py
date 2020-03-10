#!/usr/bin/env python

'''

mare in python

'''

import logging

import numpy as np
import pylab
from scipy.constants import angstrom, eV, degree
from itertools import product
from math import cos, sin, acos, sqrt

import seaborn as sns
sns.set(color_codes=True)

from matplotlib import rc
rc('font', **{'family': 'serif','serif': ['Palatino'], 'size': 20})
rc('text', usetex=True)

a_Si_NIST = 5.4311946*angstrom

Si_f0_params = 4.68669359, 2.38879059, 1.52287056, 1.07903978, 3.16354871, 0.14361992,   3.40015051, 36.83643520, 0.09557493, 114.27524168, 1.47492947

def bragg_metrictensor(a=a_Si_NIST,b=a_Si_NIST,c=a_Si_NIST,alpha=90*degree,beta=90*degree,gamma=90*degree):
    g = np.array([[a*a,            a*b*cos(gamma), a*c*cos(beta)],
                  [a*b*cos(gamma), b*b,            b*c*cos(alpha)],
                  [a*c*cos(beta),  b*c*cos(alpha), c*c]])
    
    volume2 = np.linalg.det(g)
    
    ginv = np.linalg.pinv(g)
    
    return ginv

def f0(k, params=Si_f0_params):
    a1,a2,a3,a4,a5,c,b1,b2,b3,b4,b5 = params
    # f0[k] = c + [SUM a_i*EXP(-b_i*(k^2)) ]
    # k = sin(theta)/lambda
    a = np.array([a1, a2, a3, a4, a5])
    b = np.array([b1, b2, b3, b4, b5])
    k *= 1e-10
    #print 'c' , c 
    #print 'a' , a
    #print 'b' , b
    #print 'np.exp(-b*k**2)', np.exp(-b*k**2)
    return c + np.sum(a * np.exp(-b*k**2))

class mare:
    
    def __init__(self,
                 h=1,
                 k=1,
                 l=1,
                 a=a_Si_NIST, #Si_NIST
                 h_max=3,
                 k_max=3,
                 l_max=3,
                 fh_min=1e-8,
                 wavelength_umweg=1.54*angstrom,
                 delta_wavelength=1e-2,
                 phi=-20*degree,
                 delta_phi=0.1,
                 display='spaghetti'): #umweg, glitches, spaghetti
    
        self.log = logging.getLogger()
        console = logging.StreamHandler()
        self.log.addHandler(console)
        self.log.setLevel(logging.DEBUG)
        
        self.P = np.array([h, k, l])
        self.log.debug('P %s' % self.P)
        self.pn = np.linalg.norm(self.P)
        self.log.debug('pn %s' % self.pn)
        self.p2= self.pn**2
        self.log.debug('p2 %s' % self.p2)
        
        self.h_max = h_max
        self.k_max = k_max
        self.l_max = l_max
        
        self.ginv = bragg_metrictensor()
        
        self.a = a
        
        self.log.debug('ginv %s' % self.ginv)
        mm1 = np.dot(self.ginv, self.P)
        self.log.debug('mm1 %s' % mm1)
        mm2 = np.array([mm1[1], -mm1[0], 0])
        self.log.debug('mm2 %s' % mm2)
        mm3 = min(abs(mm1[mm1!=0]))
        self.log.debug('mm3 %s' % mm3)
        self.M0 = mm2/mm3
        self.log.debug('M0 %s' % self.M0)
        
        self.alpha = np.linspace(-90., 90., 500)
        self.wavelength = np.linspace(0, 3, 500)
        
    def mare(self):
        pylab.figure(figsize=(16, 9))
        for hkl in product(np.arange(-self.h_max, self.h_max+1),
                           np.arange(-self.k_max, self.k_max+1),
                           np.arange(-self.l_max, self.l_max+1)):
            
            
            self.log.debug('hkl %s %s %s' % hkl)
            
            r = np.array(hkl)
            self.log.debug('r %s' % r)
            
            rp = np.dot(r, self.P)/self.p2*self.P
            self.log.debug('rp %s' % rp)
            
            rpn = np.linalg.norm(rp)
            self.log.debug('rpn %s' % rpn)
            
            # self.P * self.ginv * self.P.transpose
            p2new = np.dot(np.dot(self.P, self.ginv), self.P)
            self.log.debug('p2new %s' % p2new)
            # r * self.ginv * self.P.transpose/ p2new
            #self.log.debug('np.dot(r, self.ginv) %s' % np.dot(r, self.ginv))
            #self.log.debug('np.dot(np.dot(r, self.ginv), self.P.T) %s ' % np.dot(np.dot(r, self.ginv), self.P.T))
            #self.log.debug('np.dot(np.dot(r, self.ginv), self.P) %s ' % np.dot(np.dot(r, self.ginv), self.P))
            rpnew = np.dot(np.dot(r, self.ginv), self.P.T)/ p2new
            self.log.debug('rpnew %s' % rpnew)
            
            rpnew = rpnew*self.P
            self.log.debug('rpnew %s' % rpnew)
            
            cos_alpha0 = np.dot((r-rp), self.M0)/np.linalg.norm(r-rp)/np.linalg.norm(self.M0)
            self.log.debug('cos_alpha0 %s' % cos_alpha0)
            
            alpha0rad = np.arcsin(cos_alpha0)
            self.log.debug('alpha0rad %s ' % alpha0rad)
            
            alpha0 = np.degrees(alpha0rad)
            self.log.debug('alpha0 %s ' % alpha0)
            
            knew1 = 0.5*np.dot(np.dot(r, self.ginv), r.T) - np.dot(np.dot(r, self.ginv), self.P.T)
            self.log.debug('knew1 %s ' % knew1)
            knew22 = np.dot(np.dot(r, self.ginv), r.T) - np.dot(np.dot(rpnew, self.ginv), rpnew.T)
            knew2 = np.sqrt(knew22)
            self.log.debug('knew2 %s ' % knew2)
            knew = knew1/knew2
            self.log.debug('knew %s ' % knew)
            
            if abs(knew22) > 1e-8:
                goodRef = 1
            else:
                goodRef = 0
                continue
            
            # computes intensity
            # brag_calc
            # inputs:
            #         inp: structure with the input data 
                           #hmiller, kmiller, rmiller
                           #lattice
                           #f0, 
                           #f1, 
                           #f2,
                           #absorption
                           
            #     verbose: 0/1
            #   anomalous: 1/0
            
            # bragg_inp
            # inputs:
            d = self.a/np.dot(r, r)
            self.log.debug('d %s' % d)
            
            k = 1./(2*d)
            self.log.debug('k %s' % k)
            
            f0_approx = f0(k)
            self.log.debug('f0_approx %s' % f0_approx)
            
            if f0_approx <= 5:
                goodRef = 0
                
            if goodRef:
                beta = self.alpha - alpha0
                y2 = (knew/(np.cos(np.radians(beta))))**2 + p2new/4 # 1./lambda**2
                y3 = 1./np.sqrt(y2)
                pylab.plot(self.alpha+30, y3*1e10, label=str(hkl))

            self.log.debug('goodRef %s' % goodRef)
            self.log.debug('')
        pylab.title('Silicon crystal umweganregung lines and glitches', fontsize=24)
        pylab.ylim([0, 3.5])
        pylab.xlim([-60, 90])
        pylab.ylabel('Wavelength [A]', fontsize=20)
        pylab.xlabel('Azimuthal angle [deg]', fontsize=20)
        #pylab.legend()
        pylab.savefig('silicon_mare.png')
        pylab.show()
        
def main():
    m = mare()
    m.mare()

if __name__ == '__main__':
    main()
    