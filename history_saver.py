#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from camera import camera
import h5py
import time
import os
import numpy as np
#import logging

try:
    import simplejpeg
except ImportError:
    import complexjpeg as simplejpeg


def get_jpegs_from_arrays(images):
    jpegs = []
    for img in images:
        jpeg = simplejpeg.encode_jpeg(img)
        jpeg = np.frombuffer(jpeg, dtype='uint8')
        jpegs.append(jpeg)
    return jpegs
    
def main():
    
    import optparse
    
    parser = optparse.OptionParser()

    parser.add_option('-d', '--directory', type=str, help='directory')
    parser.add_option('-n', '--name_pattern', type=str, help='filename template')
    parser.add_option('-s', '--start', type=float, help='start')
    parser.add_option('-e', '--end', type=float, help='end')
    parser.add_option('-S', '--suffix', default='history', type=str, help='suffix')
    
    options, args = parser.parse_args()
    print('options', options)
    print('args', args)
    
    cam = camera()
    s = time.time()
    history_timestamps, history_images, history_state_vectors = cam.get_history(options.start, options.end)
    e = time.time()
    
    #print('history read in %.3f seconds' % (e-s))
    #print('history size %d' % len(history_timestamps))
    
    if not os.path.isdir(options.directory):
        os.makedirs(options.directory)
        
    s = time.time()
    filename = '%s_%s.h5' % (os.path.join(options.directory, options.name_pattern), options.suffix)
    history_file = h5py.File(filename, 'w')
    
    dt = h5py.special_dtype(vlen=np.dtype('uint8'))
    
    if len(history_images) > 0 and type(history_images[0]) is not np.ndarray and simplejpeg.is_jpeg(history_images[0]):
        history_jpegs = history_images
    elif len(history_images) > 0:
        history_jpegs = get_jpegs_from_arrays(history_images)
    else:
        sys.exit()
    history_file.create_dataset('history_images',
                                data=history_jpegs,
                                dtype=dt)
    
    #history_file.create_dataset('history_images',
                                #data=history_images, 
                                #compression='gzip',
                                #dtype=np.uint8)
    
    history_file.create_dataset('history_state_vectors',
                                data=history_state_vectors)
    
    history_file.create_dataset('history_timestamps',
                                data=history_timestamps)
    
    history_file.close()
    
    e = time.time()
    
    #print('history written in %.3f seconds' % (e-s))
    movie_line = 'movie_from_history.py -H %s &' % filename
    #print('generating movie %s' % movie_line)
    os.system(movie_line)
    
if __name__ == '__main__':
    main()
