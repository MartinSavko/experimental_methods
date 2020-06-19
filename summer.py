#!/usr/bin/env python
# -*- coding: utf-8 -*-

import h5py
import os
import time
import shutil
import traceback
import numpy as np
import bitshuffle.h5
import math
import random
import logging
import sys

log = logging.getLogger()
stream_handler = logging.StreamHandler(sys.stdout)
stream_formatter = logging.Formatter('summer.py |%(asctime)s |%(levelname)-7s| %(message)s')
stream_handler.setFormatter(stream_formatter)
log.addHandler(stream_handler)
log.setLevel(logging.INFO)

def create_new_master(reference_master, new_name):
    shutil.copy(reference_master, new_name)
    
    
def get_cube(master, images_to_sum, images_per_file=None):
    cube = None
    datakeys = master['/entry/data'].keys()
    datakeys.sort()
    for key in datakeys:
        log.debug(key)
        block = master['/entry/data/%s' % key][()]
        nimages, image_height, image_width = block.shape
        
        new_nimages = int(np.ceil(nimages / images_to_sum))
        reblock = block.reshape((new_nimages, images_to_sum, image_height, image_width)).sum(1)
        if cube is not None:
            cube = np.vstack([cube, reblock]) 
        else:
            cube = reblock
        log.debug('cube.shape %s' % cube.shape)
        del reblock
    return cube

def loose_the_image(image_index, images_to_loose, indices_of_images_to_loose):
    if images_to_loose == 0:
        return False
    elif image_index in indices_of_images_to_loose:
        return True
    else:
        return False
    
def get_ill(indices_of_images_to_loose, images_to_sum):
    if indices_of_images_to_loose == 'random':
        wedge_indices = range(images_to_sum)
        iil = []
        for k in range(images_to_loose):
            iil.append(wedge_indices.pop(random.choice(wedge_indices)))
        log.debug('indices_of_images_to_loose %s' % indices_of_images_to_loose)
    elif images_to_sum == len(indices_of_images_to_loose):
        iil = indices_of_images_to_loose[:]
    else:
        iil = []
    return iil
        
def sum_and_save_not_pretending_to_be_smart(master, images_to_sum, images_per_file, images_to_loose, indices_of_images_to_loose):
    datakeys = master['/entry/data'].keys()
    datakeys.sort()
    
    datafile_template = master.filename.replace('_master', '_data_%06d')
    
    datafile_number = 0
    saved_images = 0
    total = 0
    since_last_summed = 0
    
    summed = []
    data_filenames = []
    
    for key in datakeys:
        log.debug(key)
        try:
            block = master['/entry/data/%s' % key][()]
        except KeyError:
            continue
        if key == datakeys[0]:
            dtype = block.dtype
            nimages, image_height, image_width = block.shape
            new_image = np.zeros((image_height, image_width), dtype=np.int64)
            
        for image in block:
            if images_to_loose > 0 and since_last_summed == 0:
                iil = get_iil(indices_of_images_to_loose, images_to_sum)
            else:
                iil = []
                
            since_last_summed += 1
            
            if not loose_the_image(since_last_summed, images_to_loose, iil):
                new_image = np.add(new_image, image)
            
            if since_last_summed == images_to_sum:
                new_image = new_image.reshape((1, image_height, image_width))
                if summed == []:
                    summed = new_image
                else:
                    summed = np.vstack([summed, new_image])
                log.debug('summed.shape %s' % str(summed.shape))
                new_image = np.zeros((image_height, image_width))
                since_last_summed = 0
                    
            if len(summed) == images_per_file:
                log.debug('len(summed) %s' % len(summed))
                datafile_number += 1
                data_filename = datafile_template % datafile_number
                data_filenames.append(data_filename)
                to_write = summed
                low = (datafile_number-1)*images_per_file + 1
                high = low + len(to_write) - 1
                saved_images += len(to_write)
                save_datafile(data_filename, to_write, dtype, low, high)
                summed = []
            
            del image
        del block
            
    if summed != []:
        log.debug('len(summed) %s' % len(summed))
        datafile_number += 1
        data_filename = datafile_template % datafile_number
        data_filenames.append(data_filename)
        to_write = np.array(summed)
        low = (datafile_number-1)*images_per_file + 1
        high = low + len(to_write) - 1
        saved_images += len(to_write)
        save_datafile(data_filename, to_write, dtype, low, high)
    
    return saved_images, data_filenames

def sum_and_save(master, images_to_sum, images_per_file):
    datakeys = master['/entry/data'].keys()
    datakeys.sort()
    
    datafile_template = master.filename.replace('_master', '_data_%06d')
    
    datafile_number = 0
    saved_images = 0
    nimages_in_summed_remainder = 0
    
    summed = None
    remainder = None
    
    data_filenames = []
    
    for key in datakeys:
        log.debug(key)
        
        block = master['/entry/data/%s' % key][()]
        if key == datakeys[0]:
            dtype = block.dtype
        
        if remainder is not None:
            block = np.vstack([remainder, block])
            
        nimages, image_height, image_width = block.shape
        
        if nimages + nimages_in_summed_remainder >= images_to_sum:
            to_add = images_to_sum - nimages_in_summed_remainder
            remainder += block[:to_add].sum(0)
            remainder = remainder.reshape([1, image_height, image_width])
            nimages_in_summed_remainder = 0
            if summed is None:
                summed = remainder
            else:
                summed = np.vstack([summed, remainder])
            block = block[to_add:]
            
        new_nimages, remaining_images = divmod(block.shape[0], images_to_sum)
        
        if  remaining_images > 0:
            remainder = block[-remaining_images:]
            nimages_in_summed_remainder += remaining_images
            ramainder = remainder.sum(0)
        
        if new_nimages > 0:
            block = block[:new_nimages*images_to_sum]
            new_summed = block.reshape((new_nimages, images_to_sum, image_height, image_width)).sum(1)
            del block
        
        if summed is None:
            summed = new_summed
        else:
            summed = np.vstack([summed, new_summed])
        del new_summed
        
        log.debug('summed.shape %s' % str(summed.shape))
        if summed.shape[0] >= images_per_file:
            to_dump, to_keep = divmod(summed.shape[0], images_per_file)
            log.debug('to_dump %s' % str(to_dump))
            log.debug('to_keep %s' % str(to_keep))
            for l in range(to_dump):
                datafile_number += 1
                data_filename = datafile_template % datafile_number
                data_filenames.append(data_filename)
                to_write = summed[l*images_per_file: (l+1)*images_per_file]
                low = (datafile_number-1)*images_per_file + 1
                high = low + len(to_write) - 1
                saved_images += len(to_write)
                save_datafile(data_filename, to_write, dtype, low, high)
                
            if to_keep > 0:
                summed = summed[to_dump*images_per_file:]
            else:
                summed = None
    
    if summed is not None:
        datafile_number += 1
        data_filename = datafile_template % datafile_number
        data_filenames.append(data_filename)
        to_write = summed
        low = (datafile_number-1)*images_per_file + 1
        high = low + len(to_write) - 1
        saved_images += len(to_write)
        save_datafile(data_filename, to_write, dtype, low, high)
        
    return saved_images, data_filenames
    

def save_datafile(data_filename, to_write, dtype, low, high):
    data_file = h5py.File(data_filename, 'w')
    data_file.create_dataset('/entry/data/data',
                             data=to_write, 
                             #compression=bitshuffle.h5.H5FILTER, 
                             #compression_opts=(0, bitshuffle.h5.H5_COMPRESS_LZ4), 
                             dtype=dtype)
    data_file['/entry/data/data'].attrs.create('image_nr_low', low)
    data_file['/entry/data/data'].attrs.create('image_nr_high', high)
    data_file.close()
    
def main():
    import optparse
    import glob
    
    parser = optparse.OptionParser()
    
    parser.add_option('-m', '--master_file', type=str, default='collect_1_master.h5', help='master file')
    parser.add_option('-N', '--new_master_file', type=str, default=None, help='new master filename')
    parser.add_option('-n', '--images_to_sum', type=int, default=None, help='number of images to sum')
    parser.add_option('-p', '--images_per_file', type=int, default=10, help='number of images per data file')
    parser.add_option('-l', '--images_to_loose', type=int, default=0, help='number of original images not to include in the new images -- useful for simulating increased deadtime or random loss of images')
    parser.add_option('-i', '--indices_of_images_to_loose', type=str, default='-1', help='String specifying what images not to include. Depends on the nimages_to_loose value. Either integer, string that will evaluate to python tuple or list or "random" string if images are to be chosen randomly')
    
    options, args = parser.parse_args()
    
    if options.images_to_sum != None:
        images_to_sum = options.images_to_sum
    else:
        images_to_sum = 'all'
    images_per_file = options.images_per_file
    images_to_loose = options.images_to_loose
    indices_of_images_to_loose = options.indices_of_images_to_loose
    
    if images_to_loose > 0:
        if indices_of_images_to_loose != 'random':
            indices_of_images_to_loose = eval(indices_of_images_to_loose)

    if options.new_master_file == None:
        new_name = options.master_file.replace('_master.h5', '_sum%s_master.h5' % str(images_to_sum))
    else:
        new_name = options.new_master_file

    create_new_master(options.master_file, new_name)
    
    while not os.path.isfile(new_name):
        time.sleep(1)
        
    m = h5py.File(options.master_file, 'r')
    new_m = h5py.File(new_name, 'r+')
    
    parameters_to_modify = [
        '/entry/instrument/detector/count_time',
        '/entry/instrument/detector/frame_time',
        '/entry/instrument/detector/detectorSpecific/countrate_correction_count_cutoff']
        
    angles_to_modify = {'omega':'/entry/sample/goniometer/',
                        'kappa':'/entry/sample/goniometer/',
                        'phi':'/entry/sample/goniometer/',
                        'chi':'/entry/sample/goniometer/',
                        'two_theta': '/entry/instrument/detector/goniometer/'}
                            
    #recube = get_cube(m, images_to_sum)
    #new_nimages, image_height, image_width = recube.shape
    if images_to_sum == 'all':
        images_to_sum = m['/entry/instrument/detector/detectorSpecific/nimages'][()]
        
    new_nimages, data_filenames = sum_and_save_not_pretending_to_be_smart(new_m, images_to_sum, images_per_file, images_to_loose, indices_of_images_to_loose)
    
    log.debug('new_nimages %s' % new_nimages)
    log.debug('data_filenames %s' % data_filenames)
    
    new_m['/entry/instrument/detector/detectorSpecific/nimages'].write_direct(np.array([new_nimages]))
    
    log.debug('confirm %s' % new_m['/entry/instrument/detector/detectorSpecific/nimages'][()])
    
    new_m.close()
    
    new_m = h5py.File(new_name, 'r+')
    
    for pm in parameters_to_modify:
        log.debug(pm)
        current_value = m[pm][()]
        log.debug('current_value %s' % current_value)
        new_value = np.array([current_value * images_to_sum])
        log.debug('new_value %s' % new_value)
        new_m[pm].write_direct(new_value)
        log.debug('confirm %s' % new_m[pm][()])
      
    try:
        for angle in angles_to_modify:
            log.debug('angle %s' % angle)
            path = angles_to_modify[angle]
            start = m['%s/%s_start' % (path, angle)][()]
            total = m['%s/%s_range_total' % (path, angle)][()]
            increment = m['%s/%s_increment' % (path, angle)][()]
            log.debug('start %s' % start)
            log.debug('total %s' % total)
            log.debug('increment %s' % increment)
            
            if increment != 0:
                new_increment = np.array([increment * images_to_sum])
                log.debug('new_increment %s' % new_increment)
                new_m['%s/%s_increment' % (path, angle)].write_direct(new_increment)
                range_average = m['%s/%s_range_average' % (path, angle)][()]
                new_range_average = np.array([range_average * images_to_sum])
                new_m['%s/%s_range_average' % (path, angle)].write_direct(new_range_average)
                new_starts = np.arange(start, total+start, increment*images_to_sum)
                new_ends = new_starts + new_increment
            else:
                new_starts = np.zeros(new_nimages)
                new_ends = new_starts
                
            dtype = m['%s/%s' % (path, angle)].dtype
            del new_m['%s/%s' % (path, angle)]
            
            new_m.create_dataset('%s/%s' % (path, angle), data=new_starts, dtype=dtype)
            del new_m['%s/%s_end' % (path, angle)]
            new_m.create_dataset('%s/%s_end' % (path, angle), data=new_ends, dtype=dtype)
            
            log.debug(new_m['%s/%s' % (path, angle)][()])
            log.debug(new_m['%s/%s_end' % (path, angle)][()])
    except Exception as e:
        log.debug(traceback.format_exc())
        
    m.close()
    
    
    
    data_keys = new_m['/entry/data'].keys()
    for key in data_keys:
        del new_m['/entry/data/%s' % key]
    
    #new_m.create_dataset('/entry/data/data_000001', data=recube, compression=bitshuffle.h5.H5FILTER, compression_opts=(0, bitshuffle.h5.H5_COMPRESS_LZ4), dtype=dtype)
    #new_m['/entry/data/data_000001'].attrs.create('image_nr_low', 1)
    #new_m['/entry/data/data_000001'].attrs.create('image_nr_high', new_nimages)
    
    for k, data_filename in enumerate(data_filenames):
        data_key = '/entry/data/data_%06d' % (k+1,)
        log.debug('data_filename %s' % data_filename)
        log.debug('data_key %s' % data_key)
        new_m[data_key] = h5py.ExternalLink(data_filename, '/entry/data/data')
    
    
    #time.sleep(5)
    new_m.close()
    
    
if __name__ == '__main__':
    main()
    
