#!/usr/bin/env python

import h5py
import os
import shutil
import traceback
import numpy as np
    
def create_new_master(reference_master, new_name):
    shutil.copy(reference_master, '%s_master.h5' % new_name)
    
def data_block_stride_sum(datablock, n):
    for 
    
def main():
    import optparse
    import glob
    
    parser = optparse.OptionParser()
    
    parser.add_option('-m', '--master_file', type=str, default='collect_1_master.h5', help='master file to handle')
    parser.add_option('-n', '--number_of_images_to_sum', type=int, default=2, help='number of images to sum')
    
    
    options, args = parser.parse_args()
    
    new_name = options.master_file.replace('_master.h5', '%d_master.h5' % options.number_of_images_to_sum)

    create_new_master(masters[0], new_name)
    
    while not os.path.isfile(new_name):
        time.sleep(1)
    new_m = h5py.File('%s_master.h5' % options.new_name, 'a')
    
    data_keys = new_m['/entry/data'].keys()
    for key in data_keys:
        del new_m['/entry/data/%s' % key]
        
    data_file_order = 0
    
    array_accumulators = \
        {
            '/entry/sample/goniometer/omega': np.array([]),
            '/entry/sample/goniometer/omega_end': np.array([]),
            '/entry/sample/goniometer/chi': np.array([]),
            '/entry/sample/goniometer/chi_end': np.array([]),
            '/entry/sample/goniometer/phi': np.array([]), 
            '/entry/sample/goniometer/phi_end': np.array([]), 
            '/entry/sample/goniometer/kappa': np.array([]), 
            '/entry/sample/goniometer/kappa_end': np.array([]),
            '/entry/instrument/detector/goniometer/two_theta': np.array([]),
            '/entry/instrument/detector/goniometer/two_theta_end': np.array([]),
        }
    integer_accumulators = \
        {
            '/entry/instrument/detector/detectorSpecific/nimages': 0,
            '/entry/sample/goniometer/omega_range_total': 0,
            '/entry/sample/goniometer/chi_range_total': 0,
            '/entry/sample/goniometer/phi_range_total': 0,
            '/entry/sample/goniometer/kappa_range_total': 0,
            '/entry/instrument/detector/goniometer/two_theta_range_total': 0
        }
    
    for master in masters:
        print 'handling master %s' % master
        m = h5py.File(master, 'r')
        keys = m['/entry/data'].keys()
        keys.sort()
        for ac in array_accumulators:
            current_value = array_accumulators[ac]
            array_accumulators[ac] = np.hstack([current_value, m[ac].value])
        
        for ia in integer_accumulators:
            integer_accumulators[ia] +=  m[ia].value
        
        for key in keys:
            print 'key', key
            data_file_order += 1
            new_key = '/entry/data/data_%06d' % data_file_order
            print 'new key', new_key
            h5link = m['/entry/data'].get(key, getlink=True)
            print 'h5link', h5link
            filename = h5link.filename
            new_name = '%s_data_%06d.h5' % (options.new_name, data_file_order)
            print 'new_name', new_name
            if not os.path.isfile(new_name):
                os.link(filename, new_name)

            new_m[new_key] = h5py.ExternalLink(new_name, '/entry/data/data')
            new_m[new_key].attrs.modify('image_nr_low', (data_file_order-1)*100 + 1)
            new_m[new_key].attrs.modify('image_nr_high', (data_file_order)*100)
        m.close()
        
    for ac in array_accumulators:
        current_value = array_accumulators[ac]
        dtype = new_m[ac].dtype
        del new_m[ac]
        new_m.create_dataset(ac, data=current_value, shape=current_value.shape, dtype=dtype)
    for ia in integer_accumulators:
        current_value = integer_accumulators[ia]
        #dtype = new_m[ia].dtype
        #del new_m[ia]
        #new_m.create_dataset(ia, data=current_value, dtype=dtype
        new_m[ia].write_direct(np.array(current_value))
    new_m.close()
            
if __name__ == '__main__':
    main()
    