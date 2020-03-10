#!/usr/bin/env python

from camera import camera
import h5py
import time
import os
import numpy as np

def main():
    
    import optparse
    
    parser = optparse.OptionParser()

    parser.add_option('-d', '--directory', type=str, help='directory')
    parser.add_option('-n', '--name_pattern', type=str, help='filename template')
    parser.add_option('-s', '--start', type=float, help='start')
    parser.add_option('-e', '--end', type=float, help='end')
    
    options, args = parser.parse_args()

    cam = camera()
    s = time.time()
    history_timestamps, history_images, history_state_vectors = cam.get_history(options.start, options.end)
    e = time.time()
    
    print('history read in %.3f seconds' % (e-s))
    print('history size %d' % len(history_timestamps))
    
    if not os.path.isdir(options.directory):
        os.makedirs(options.directory)
        
    s = time.time()
    history_file = h5py.File('%s_history.h5' % os.path.join(options.directory, options.name_pattern), 'w')
    
    history_file.create_dataset('history_images',
                                data=history_images, 
                                compression='gzip',
                                dtype=np.uint8)
        
    history_file.create_dataset('history_state_vectors',
                                data=history_state_vectors)
    
    history_file.create_dataset('history_timestamps',
                                data=history_timestamps)
    
    history_file.close()
    
    e = time.time()
    
    print('history written in %.3f seconds' % (e-s))
    
if __name__ == '__main__':
    main()