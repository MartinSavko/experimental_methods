#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''simulating simplejpeg behaviour where it is not available'''

import traceback

try:
    from scipy.misc import imsave, imread
except ImportError:
    try:
        from skimage.io import imsave, imread
    except ImportError:
        traceback.print_exc()
    try:
        from imageio import imsave, imread
    except ImportError:
        traceback.print_exc()
        
import io
import numpy as np

def is_jpeg(data):
    return data[:2] == b'\xFF\xD8' and data[-2:] == b'\xFF\xD9'

def encode_jpeg(image_array, quality=75):
    jpeg = io.BytesIO()
    imsave(jpeg, image_array, 'jpeg')
    jpeg.seek(0)
    jpeg = jpeg.read()
    #jpeg = np.fromstring(jpeg, dtype='uint8')
    return jpeg

def decode_jpeg(data):
    jpeg = io.BytesIO(data)
    image_array = imread(jpeg)
    return image_array

def decode_jpeg_header(data):
    '''
    returns height, width, colorspace, color subsampling
    '''
