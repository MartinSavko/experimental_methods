#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial import distance_matrix
from skimage.morphology import convex_hull_image

def get_d_min_for_ddv(r_min, wavelength, detector_distance):
    d_min = get_resolution_from_distance(r_min, wavelength, detector_distance)
    return d_min

def get_resolution_from_distance(distance, wavelength, detector_distance):
    tans = distance / detector_distance
    twotheta = np.arctan(tans)
    theta = twotheta / 2.
    resolution = wavelength / (2 * np.sin(theta))

    return resolution
    
def get_distance_from_resolution(resolution, wavelength, detector_distance):
    distance = detector_distance * np.tan( 2 * np.arcsin (wavelength / ( 2 * resolution )))
    return distance

def get_ddv(spots_mm, r_min, wavelength, detector_distance):
    d_min = get_d_min_for_ddv(r_min, wavelength, detector_distance)
        
    dm = np.triu(distance_matrix(spots_mm, spots_mm, p=2))

    dm = dm[np.logical_and(dm <= d_min, dm > 0)]
    
    h = np.histogram(dm, bins=100)
    
    valu = h[0]
    reso = (h[1][1:] + h[1][:-1]) / 2.
    
    valu = np.hstack([[0], valu])
    reso = np.hstack([[0], reso])
    return valu, reso

def get_ddv_as_image(valu, offset=5):
    image = np.zeros((offset + valu.max() + offset, valu.shape[0]))
    for k, v in enumerate(valu):
        image[:v, k] = 1
    image = (image == 0).astype(int)
    
    return image


def get_baseline(valu, reso):
    
    image = get_ddv_as_image(valu)
    chi = convex_hull_image(image)
    
    bi = np.argmax(image, axis=0) == np.argmax(chi, axis=0)
    
    bvalu = valu[bi]
    breso = reso[bi]
    return bvalu, breso

def get_slope(valu, reso):
    bvalu, breso = get_baseline(valu, reso)
    A = np.expand_dims(breso, 1)
    b = np.expand_dims(bvalu, 1)
    slope, residual, rank, s = np.linalg.lstsq(A, b, rcond=None)
    print(f"slope: {slope}, residual: {residual}, rank: {rank}, s: {s}")
    return np.squeeze(slope)

    
