#!/usr/bin/python

import scipy.misc
import scipy.ndimage
from scipy.optimize import leastsq
from math import cos, sin, sqrt, radians, atan, asin, acos, pi, degrees
import optparse
import subprocess
import os
import re
import numpy
import pickle
import sys
import math
import glob
import time

from goniometer import goniometer

phiy_direction=-1.
phiz_direction=1.
    
def main():
    parser = optparse.OptionParser()

    parser.add_option('-d', '--directory', default='/nfs/ruche/proxima2a-spool/2019_Run1/x-centring', type=str, help='Directory with the scan results (default: %default)')
    parser.add_option('-n', '--name_pattern', default='excenter', type=str, help='Name pattern (default: %default)')
    parser.add_option('-c', '--calculate', action='store_true', help='Just calculate position, do not actually move the motors')
    parser.add_option('-a', '--angles', default='(90, 0, 270)', type=str, help='Specify angles for grid scans that will be used for x-centring')
    parser.add_option('-l', '--length', default=0.3, type=float, help='Specify the length of scanned area in mm (default: %default) ')
    parser.add_option('-w', '--width', default=0.01, type=float, help='Specify the widht of scanned area in mm (default: %default)')
    parser.add_option('-i', '--interpret', action='store_true', help='Just analyse the results from images already taken (do not collect anew).')
    parser.add_option('-s', '--step_size', default=0.002, type=float, help='step_size in mm (default: %default)')
    options, args = parser.parse_args()
    
    template = '%s_*_filter.png' % options.name_pattern
    
    angles = eval(options.angles)
    length = options.length
    step_size = options.step_size
    if not options.interpret:
        execute_grid_scans(angles, options.directory, options.name_pattern, length=length, step_size=step_size)
        
    imagenames = get_imagenames(options.directory, template, n_angles=len(angles))
    
    print('imagenames', imagenames)
    
    images = get_images(imagenames)
    
    angles = get_angles(imagenames, template)
    print('angles', angles)
    
    xcoms = get_xcoms(images)
    print('xcoms', xcoms)
    
    print('imagenames', imagenames)
    parameters = get_paramaters(imagenames[0])
    
    Y = xcoms[:, 1]
    Z = xcoms[:, 0]
    Phi = angles
    beam_xc, beam_yc = get_beam_center(parameters)
    print('beam_xc, beam_yc', beam_xc, beam_yc)
    move_vector = least_squares(Y, Z, Phi, beam_xc, beam_yc)
    print('move_vector px', move_vector)
    d_sampx, d_sampy, d_y, d_z = move_vector
    calibration = get_calibration(parameters)
    print('calibration', calibration)
    move_vector_mm = get_move_vector_mm(move_vector, calibration)
    print('move_vector_mm', move_vector_mm)
    reference_position = get_reference_position(parameters)
    print('reference_position', reference_position)
    aligned_position = calculate_aligned_position(reference_position, move_vector_mm)
    
    print('aligned position', aligned_position)

    if options.calculate:
        pass
    else:
        gonio = goniometer()
        aligned_position = translate_position_dictionary(aligned_position)
        gonio.set_position(aligned_position)
    
    os.system('excenter_finished_dialog.py &')
        
def execute_grid_scans(orientations, directory, name_pattern, length=0.1, step_size=0.002):
    print('orientations', orientations)
    gonio = goniometer()
    reference_position = gonio.get_position()
    for orientation in orientations:
        if not os.path.exists(os.path.join(directory, '%s_%s_master.h5' % (name_pattern, orientation))):
            gonio.set_position({'Omega': orientation}, wait=True)
            os.system('area_scan.py -d %s -p %s -n %s_%s -y %s -x 0.01 -c 1 -r %s -a vertical' % (directory, orientation, name_pattern, orientation, length, int(length/step_size)))
        else:
            print('%s already exists, skipping ...\nPlease change the destination directory or the name_pattern.' % os.path.join(directory, '%s_%s_master.h5' % (name_pattern, orientation)))
    

def translate_position_dictionary(position):
    translated_position = {}
    if 'PhiZ' in list(position.keys()):
        target_keys = ['AlignmentZ', 'AlignmentY', 'CentringX', 'CentringY', 'AlignmentX']
        source_keys = ['PhiZ', 'PhiY', 'SamX', 'SamY', 'PhiX']
    else:
        target_keys = ['PhiZ', 'PhiY', 'SamX', 'SamY', 'PhiX']
        source_keys = ['AlignmentZ', 'AlignmentY', 'CentringX', 'CentringY', 'AlignmentX']
    for source, target in zip(source_keys, target_keys):
        translated_position[target] = position[source]
        
    return translated_position
    
def calculate_aligned_position(reference_position, move_vector_mm):
    d_sampx, d_sampy, d_y, d_z = move_vector_mm
    aligned_position = translate_position_dictionary(reference_position)
    aligned_position['PhiZ'] += d_z
    aligned_position['PhiY'] += d_y
    aligned_position['SamX'] += d_sampx
    aligned_position['SamY'] += d_sampy
    return aligned_position

def get_reference_position(parameters):
    return parameters['reference_position']
    
def get_move_vector_mm(move_vector, calibration):
    d_sampx, d_sampy, d_y, d_z = move_vector
    pixels_per_mm_y = 1./calibration[1]
    pixels_per_mm_z = 1./calibration[0]
    d_sampx /= pixels_per_mm_z
    d_sampy /= pixels_per_mm_y
    d_y /= pixels_per_mm_y
    d_z /= pixels_per_mm_z
    print('d_sampx, d_sampy, d_y, d_z in mm', d_sampx, d_sampy, d_y, d_z)
    move_vector_mm = (d_sampx, d_sampy, d_y, d_z)
    return move_vector_mm
    
def get_paramaters(imagename):
    f = open(imagename.replace('filter.png', 'parameters.pickle'))
    parameters = pickle.load(f)
    f.close()
    return parameters
    
def get_beam_center(parameters):
    beam_xc, beam_yc = parameters['beam_position_horizontal'], parameters['beam_position_vertical']
    return beam_xc, beam_yc
    
def get_calibration(parameters):
    return parameters['camera_calibration_vertical'], parameters['camera_calibration_horizontal'], 
    
def get_imagenames(directory, template, n_angles=3, wait_cycles=40):
    imagenames = glob.glob(os.path.join(directory, template))
    k = 1
    while len(imagenames) < n_angles and k<wait_cycles:
        k+=1
        time.sleep(1)
        imagenames = glob.glob(os.path.join(directory, template))
        print('waiting for analysis to finish ... timeout in %s seconds' % (wait_cycles-k))
    return imagenames
    
def get_images(imagenames):
    images = [scipy.misc.imread(img) for img in imagenames]
    return images
    
def get_angles(imagenames, template):
    angles = [float(re.findall(template.replace('*', '([-|+]?[\d\.]*)'), imagename)[0]) for imagename in imagenames]
    return angles
    
def get_xcoms(xgrids):
    xcoms = numpy.array([scipy.ndimage.center_of_mass(xgrid) for xgrid in xgrids])
    return xcoms
    
def residual(varse, x, data):
    c, r, alpha = varse
    model = c + r*numpy.sin(alpha + x + pi/2)
    return data - model

def get_phi_radians(Phi):
    return [radians(phi) for phi in Phi]
    
def least_squares(Y, Z, Phi, beam_yc, beam_zc):
    varse = [10, 100, 0]
    phi = get_phi_radians(Phi)
    x = numpy.array(phi)
    data = numpy.array(Z)
    results = leastsq(residual, varse, args=(x, data))
    c, r, alpha = results[0]
    print('c', c)
    print('r', r)
    print('alpha', degrees(alpha))
    
    d_sampx = r * sin(alpha) 
    d_sampy = r * cos(alpha)
    
    yc = sum(Y)/float(len(Y))
    zc = c
    
    d_y = phiy_direction * (yc - beam_yc)
    d_z = phiz_direction * (zc - beam_zc)
    
    move_vector = (d_sampx, d_sampy, d_y, d_z)
    return move_vector 
    
if __name__ == "__main__":
    main()
