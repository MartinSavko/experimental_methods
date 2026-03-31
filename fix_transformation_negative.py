#!/usr/bin/env python3
import h5py
import numpy as np
import sys
import os

omega_direction = [ 0., 1., 0] # MD3Up
#omega_direction = [ 1., 0., 0.] # Cribleur
#omega_direction = [-1., 0., 0.,] # MD2

assert len(sys.argv) == 2, 'Please supply one argument, the HDF5 master file, or a .lst text file with filenames separated by newlines.'
assert os.path.exists(sys.argv[1]), 'Specified file does not exist.'

def process_file(filename):
    print(f'Processing master HDF5 file {filename}...')
    with h5py.File(filename,'r+') as f:
        vec = f['entry/instrument/detector/transformations/translation/'].attrs.get('vector')
        print(f'Old vector is {vec}')
        positivised_vec = np.abs(vec)
        print(f'New vector is {positivised_vec}')
        f['/entry/instrument/detector/transformations/translation/'].attrs.modify('vector', positivised_vec)
        f['/entry/sample/transformations/omega'].attrs.modify('vector', omega_direction)
        
    print(f'Finished processing HDF5 file {filename}.')

if sys.argv[1].endswith('master.h5'):
    process_file(sys.argv[1])

elif sys.argv[1].endswith('.lst'):
    print(f'Processing list of files {sys.argv[1]}...')
    with open(sys.argv[1]) as f:
        files_list = f.read().splitlines()
    for master_file in files_list:
        process_file(master_file)
    print(f'Finished processing list of files {sys.argv[1]}.')

else:
    print('Could not interpret specified file.')
