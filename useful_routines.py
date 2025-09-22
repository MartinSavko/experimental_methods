#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial import distance_matrix

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
    reso = h[1]
    return valu, reso
