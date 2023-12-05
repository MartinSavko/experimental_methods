#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import itertools
import os
import time
import pickle
import pylab
import numpy as np
import sys
import shutil

def main():
    kappa_1 = np.linspace(0, 216, 7)
    phi_1 = np.linspace(0, 270, 4)

    kappa_2 = np.linspace(0, 180, 3)
    phi_2 = np.linspace(0, 324, 10)

    kappa_3 = np.linspace(60, 120, 2)
    phi_3 = np.linspace(45, 315, 4)

    kappa_4 = np.linspace(0, 240, 20)
    phi_4 = np.linspace(0, 360, 20)

    #kp = list(itertools.product(kappa_1, phi_1)) + [(0, 1)] + list(itertools.product(kappa_2, phi_2)) + [(0, 2)] + list(itertools.product(kappa_3, phi_3))
    kp = list(itertools.product(kappa_1, phi_1)) + [(0, 1)] + list(itertools.product(kappa_2, phi_2)) + [(0, 2)] + list(itertools.product(kappa_3, phi_3)) + list(itertools.product(kappa_4, phi_4))
    
    k = 0
    print('combinations')
    print(kp)
    print('number of combinations %d ' % len(kp))
   
    directory = '/nfs/ruche/proxima2a-spool/Martin/Research/MK3/2019-09-29/run2'
    
    line = 'optical_alignment.py -d %s -n {K:.0f}_{P:.0f}_{Z:d}_{id:d}_{design:s} --rightmost -K {K:.2f} -P {P:.2f} -z {Z:d} -A -C -R -g 60' % directory
    
    start = time.time()
    for kappa, phi in kp:
        k += 1
        for zoom in [1, 5, 10]:
            print(line.format(K=kappa, P=phi, Z=zoom, design='ongoing', id=k))
            os.system(line.format(K=kappa, P=phi, Z=zoom, design='ongoing', id=k))
        os.system(line.format(K=kappa, P=phi, Z=zoom, design='pre_final', id=k))
        os.system((line + ' --save_raw_images').format(K=kappa, P=phi, Z=zoom, design='final', id=k))
    end = time.time()
    
    #start = time.time()
    #table_of_results = []
    #k0p0 = True
    #k0p180 = True
    #k180p0 = True
    #k180p180 = True
    #order = 0
    #passed = []
    #for kappa, phi in kp:
        
        #k += 1
        ##print('kappa %.2f, phi %.2f' % (kappa, phi), )
        #name_pattern = '%.0f_%.0f_10_final' % (kappa, phi)
        
        #f = os.path.join(directory, '%s_report.png' % (name_pattern))
        ##print(f)
        
        #if os.path.isfile(f):
            ##print('ok')
            
            #if (kappa, phi) in passed:
                #print(order, (kappa, phi), 'already passed')
            #else:
                #passed.append((kappa, phi))
            #parameters = pickle.load(open(os.path.join(directory, '%s_parameters.pickle' % name_pattern)))
            #results = pickle.load(open(os.path.join(directory, '%s_results.pickle' % name_pattern)))
            
            #p = results['result_position']
            #position_vector = [p[key] for key in ['Kappa', 'Phi', 'AlignmentY', 'AlignmentZ', 'CentringX', 'CentringY']]
            #calibration_y, calibration_x = parameters['calibration']
            #confusion_x = abs(results['fits']['rightmost_horizontal'][0].x[1] * calibration_x)
            #confusion_y = abs(results['fits']['rightmost_vertical'][0].x[1] * calibration_y)
            #if kappa == 0. and phi == 0. and k0p0 == True:
                #k0p0 = False
                #table_of_results.append([kappa, phi] + [0.]*7)
            #elif kappa == 0. and phi == 180. and k0p180 == True:
                #k0p180 = False
                #table_of_results.append([kappa, phi] + [0.]*7)
            #elif kappa == 180. and phi == 0. and k180p0 == True:
                #k180p0 = False
                #table_of_results.append([kappa, phi] + [0.]*7)
            #elif kappa == 180. and phi == 180. and k180p180 == True:
                #k180p180 = False
                #table_of_results.append([kappa, phi] + [0.]*7)
            #else:
                #order += 1
                #table_of_results.append(position_vector + [confusion_x, confusion_y, order])
                #shutil.copy(f, '/tmp/reports/%.0f_%.0f_%d_report.png' % (kappa, phi, order))
        #else:
            #print('not ok')
        
    #table_of_results.sort(key=lambda x: (x[0], x[1]))
    #tor = np.array(table_of_results)
    #print('tor.shape', tor.shape)
    #np.savetxt(os.path.join(directory, 'mk3_aligned_positions_with_order_auto.csv'), tor, delimiter=',', fmt='%9.4f', header='66 aligned positions of the tip of a glass capillary at varying Kappa and Phi\nGoniometer: MD2 equipped with minikappa head (MK3)\nInstrument: Proxima 2A, Synchrotron SOLEIL\nDate: 2019-09-22\nUnit: mm\nFields: Kappa, Phi, AlignmentY, AlignmentZ, CentringX, CentringY, horizontal_confusion, vertical_confusion, order')
    #sys.exit()
    
    #pylab.figure(figsize=(16, 9))
    #pylab.plot(tor[:,2] - tor[:,2].min(), 'o-',label='AlignmentY')
    #pylab.plot(tor[:,3] - tor[:,3].min()-0.2, 'o-',label='AlignmentZ')
    #pylab.plot(tor[:,4] - tor[:,4].min()+0.2, 'o-', label='CentringX')
    #pylab.plot(tor[:,5] - tor[:,5].min()+0.3, 'o-', label='CentringY')
    #pylab.xlabel('Kappa and Phi combination')
    #pylab.ylabel('relative position [mm]')
    #pylab.grid()
    #pylab.legend()
    
    #pylab.figure(figsize=(16, 9))
    #pylab.plot(np.abs(tor[:,6]), 'o-', label='horizontal')
    #pylab.plot(np.abs(tor[:,7]), 'o-', label='vertical')
    
    #pylab.xlabel('Kappa and Phi combination')
    #pylab.ylabel('radius of confusion [mm]')
    #pylab.grid(True)
    #pylab.legend()
    #pylab.show()
        ##os.system(line.format(K=kappa, P=phi, Z=10, design='final'))
    #end = time.time()
    print('alignment of %d combinations of kappa and phi combinations took %.2f seconds' % (k, end-start))
    
if __name__ == '__main__':
    main()
