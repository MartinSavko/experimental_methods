#!/usr/bin/env python

''' 
Author: Martin Savko 
Contact: savko@synchrotron-soleil.fr
Date: 2019-03-29
Version: 0.0.5

This script saves datasets stored in Eiger HDF5 format into series of CBF files.

It takes path to the master file as an argument (-m option) and generates .cbf
files with correct header information. 

It first extracts header information for individual images from the master file,
creates cbf header as a text file, uses H5ToXds binary (www.dectris.com) to
generate the cbf and and merges them together using cat GNU command.

'''

import h5py
import time 
import os
import shutil
import traceback
import multiprocessing
import sys
import glob

header_template = '''###CBF: VERSION 1.5, CBFlib v0.7.8 - SLS/DECTRIS PILATUS detectors

{filename}

_array_data.header_convention "SLS_1.0"
_array_data.header_contents
;
# Detector: {description}, {detector_number}
# {data_collection_date}
# Pixel_size {x_pixel_size} m x {y_pixel_size} m
# Exposure_time {exposure_time:.6} s
# Exposure_period {exposure_period:.6} s
# Count_cutoff {count_cutoff} counts
# Threshold_setting {threshold_setting:.5} eV
# N_excluded_pixels {n_excluded_pixels}
# Image_path {image_path}
# Beam_xy ({beam_center_x}, {beam_center_y}) pixels
# Wavelength {wavelength:.4} A
# Detector_distance {detector_distance:.6} m
# Silicon sensor, thickness {sensor_thickness:.6} m
# Omega {omega:.4} degree
# Omega_increment {omega_increment:.4} degree
# Phi {phi:.4} degree
# Phi_increment {phi_increment:.4} degree
# Kappa {kappa:.4} degree
# Kappa_increment {kappa_increment:.4} degree
# Chi {chi:.4} degree
# Chi_increment {chi_increment:.4} degree
# Start_angle {start_angle:.4} degree
# Angle_increment {angle_increment:.4} degree
# Oscillation_axis {oscillation_axis}
;

'''
#making header line endings consistent with the H5ToXds output
header_template = header_template.replace('\n', '\r\n') 

# storing hdf5 paths into mnemonic variables
sensor_thickness = "/entry/instrument/detector/sensor_thickness"
nimages = "/entry/instrument/detector/detectorSpecific/nimages"
description = "/entry/instrument/detector/description"
detector_number = "/entry/instrument/detector/detector_number"
x_pixel_size = "/entry/instrument/detector/x_pixel_size"
y_pixel_size = "/entry/instrument/detector/y_pixel_size"
data_collection_date = "/entry/instrument/detector/detectorSpecific/data_collection_date"
count_time = "/entry/instrument/detector/count_time" 
frame_time = "/entry/instrument/detector/frame_time"
countrate_correction_count_cutoff = "/entry/instrument/detector/detectorSpecific/countrate_correction_count_cutoff"
threshold_energy = '/entry/instrument/detector/threshold_energy'
number_of_excluded_pixels = '/entry/instrument/detector/detectorSpecific/number_of_excluded_pixels'
incident_wavelength = "/entry/instrument/beam/incident_wavelength"
detector_distance = "/entry/instrument/detector/detector_distance"
beam_center_x = "/entry/instrument/detector/beam_center_x"
beam_center_y = "/entry/instrument/detector/beam_center_y"
omega_range_average = "/entry/sample/goniometer/omega_range_average"
phi_range_average = "/entry/sample/goniometer/phi_range_average"
kappa_range_average = "/entry/sample/goniometer/kappa_range_average"
chi_range_average = "/entry/sample/goniometer/chi_range_average"

oscillation_axes_ranges = {'OMEGA': omega_range_average,
                           'PHI': phi_range_average,
                           'KAPPA': kappa_range_average,
                           'CHI': chi_range_average}

def get_header_information(master_file):
    h = {}
    h['description'] = master_file[description].value
    h['sensor_thickness'] = master_file[sensor_thickness].value
    h['detector_number'] = master_file[detector_number].value
    h['data_collection_date'] = master_file[data_collection_date].value
    h['x_pixel_size'] = master_file[x_pixel_size].value
    h['y_pixel_size'] = master_file[y_pixel_size].value
    h['exposure_time'] = master_file[count_time].value
    h['exposure_period'] = master_file[frame_time].value
    h['count_cutoff'] = master_file[countrate_correction_count_cutoff].value
    h['threshold_setting'] = master_file[threshold_energy].value
    h['n_excluded_pixels'] = master_file[number_of_excluded_pixels].value
    h['image_path'] = 'na'
    h['filename'] = 'na'
    h['wavelength'] = master_file[incident_wavelength].value
    h['detector_distance'] = master_file[detector_distance].value
    h['beam_center_x'] = master_file[beam_center_x].value
    h['beam_center_y'] = master_file[beam_center_y].value
    h['omega_increment'] = master_file[omega_range_average].value
    h['phi_increment'] = master_file[phi_range_average].value
    h['kappa_increment'] =master_file[kappa_range_average].value
    h['chi_increment'] = master_file[chi_range_average].value
    h['omega'] = None
    h['phi'] = None
    h['kappa'] = None
    h['chi'] = None
    oscillation_axis = get_oscillation_axis(master_file)
    h['oscillation_axis'] = oscillation_axis
    h['start_angle'] = None
    h['angle_increment'] = master_file[oscillation_axes_ranges[oscillation_axis]].value
    h['omegas'] = master_file["/entry/sample/goniometer/omega"].value
    h['phis'] = master_file["/entry/sample/goniometer/phi"].value
    h['kappas'] = master_file["/entry/sample/goniometer/kappa"].value
    h['chis'] = master_file["/entry/sample/goniometer/chi"].value
    h['oscillation_axis_values'] = master_file["/entry/sample/goniometer/%s" % oscillation_axis.lower()].value
    image_path = os.path.dirname(os.path.abspath(master_file.filename))
    h['image_path'] = image_path
    filename_template = master_file.filename.replace('_master.h5', '_#####.cbf')
    h['filename_template'] = filename_template
    h['image_numbers'] = get_image_numbers(master_file)
    return h

def get_image_numbers(master_file):
    data_items = master_file['/entry/data'].items()
    data_items.sort(key=lambda x: x[0])
    image_numbers=[]
    for key, value in data_items:
        print 'key, value', key, value
        try:
            low = value.attrs.get('image_nr_low')
            high = value.attrs.get('image_nr_high')
            image_numbers += range(low, high+1, 1)
        except:
            print traceback.print_exc()
    return image_numbers

def get_oscillation_axis(master_file):
    if master_file['/entry/sample/goniometer/omega_range_total'].value > 0:
        return 'OMEGA'
    if master_file['/entry/sample/goniometer/phi_range_total'].value > 0:
        return 'PHI'
    if master_file['/entry/sample/goniometer/kappa_range_total'].value > 0:
        return 'KAPPA'
    if master_file['/entry/sample/goniometer/chi_range_total'].value > 0:
        return 'CHI'
    return 'OMEGA'
    
def get_single_wedge(start, images_in_wedge):
    return [start + j for j in range(images_in_wedge)]
        
def get_wedges(start, nimages, n_cpu):
    print 'start, nimages, n_cpu', start, nimages, n_cpu
    n_cpu = int(n_cpu)
    iterations, rest = divmod(nimages, n_cpu)
    wedges = []
    iterations = int(iterations)
    rest = int(rest)
    for i in range(iterations):
        wedges.append(get_single_wedge(start+i*n_cpu, n_cpu))
    if rest:
        wedges.append(get_single_wedge(start+iterations*n_cpu, rest))
    return wedges

def get_nimages(master_file, first, last):
    #nimages = master_file["/entry/instrument/detector/detectorSpecific/nimages"].value
    #ntrigger = master_file["/entry/instrument/detector/detectorSpecific/ntrigger"].value
    #nimages *= ntrigger
    nimages = sum([d.shape[0] for d in master_file['/entry/data'].values() if d is not None])
    if first!=0 and last!=-1 and last>first:
        nimages = last - first
    elif first > 0:
        nimages = nimages - first
    elif last > 0 and last>first:
        nimages = last - first
    elif last < 0:
        last = nimages + last
        nimages = last - first + 1
    return nimages

def get_n_cpu():
    n_cpu = multiprocessing.cpu_count()
    if n_cpu > 4:
       n_cpu = multiprocessing.cpu_count()/2  
    return n_cpu

def extract_cbfs(master_file, master_file_absolute_path, destination_directory, first=0, last=-1, n_cpu=0, compress=None):
    nimages = get_nimages(master_file, first, last)
    
    header_dictionary = get_header_information(master_file)
    
    start = time.time()
    
    if n_cpu <= 0:
       n_cpu = get_n_cpu()
    
    wedges = get_wedges(first, nimages, n_cpu)
    k = 0
    for wedge in wedges:
        wedge_start = time.time()
        jobs = []
        for num in wedge:
            p = multiprocessing.Process(target=save_image, args=(header_dictionary, master_file_absolute_path, destination_directory, num, compress))
            jobs.append(p)
            p.start()
        for job in jobs:
            job.join()
        k += 1
        nsofar = k*len(wedge)
        wedge_end = time.time()
        print 'wedge %3d (of %d), wedge processing time %6.4f s, time per image in this wedge %6.4f s; total processing time %5.1f s, which is %6.4f s per image' % (k, len(wedges), wedge_end - wedge_start, (wedge_end-wedge_start)/len(wedge), time.time()-start, (time.time()-start)/nsofar)
    end = time.time()
    print 'total processing time %.1f s, which is %.3f s per image' % (end-start, (end-start)/nimages)
        
def save_image(header_dictionary, master_file_absolute_path, destination_directory, n, compress=None):
    filename_template = header_dictionary['filename_template']
    image_number = header_dictionary['image_numbers'][n] # n+1
    filename = os.path.basename(filename_template.replace('#####', str(image_number).zfill(5)))
    header_dictionary['filename'] = 'data_%s'  % (filename.replace('.cbf',''))
    omegas = header_dictionary['omegas']
    phis = header_dictionary['phis']
    kappas = header_dictionary['kappas']
    chis = header_dictionary['chis']
    oscillation_axis_values = header_dictionary['oscillation_axis_values']
    image_path = header_dictionary['image_path']
    try:
        if type(omegas) == float:
            header_dictionary['omega'] = omegas
            header_dictionary['phi'] = phis
            header_dictionary['kappa'] = kappas
            header_dictionary['chi'] = chis
            header_dictionary['start_angle'] = oscillation_axis_values
        else:
            header_dictionary['omega'] =  omegas[n]
            header_dictionary['phi'] = phis[n]
            header_dictionary['kappa'] = kappas[n]
            header_dictionary['chi'] = chis[n]
            header_dictionary['start_angle'] = oscillation_axis_values[n]
    except IndexError:
        print 'oscillation_axis_values', oscillation_axis_values
        header_dictionary['omega'] = oscillation_axis_values
        header_dictionary['phi'] = phis
        header_dictionary['kappa'] = kappas
        header_dictionary['chi'] = chis
        header_dictionary['start_angle'] = oscillation_axis_values

    
    header = header_template.format(**header_dictionary)
    header_filename = 'header_%s' % (str(image_number).zfill(5))
    f = open(header_filename, 'w')
    f.write(header)
    f.close()
    
    raw_cbf_filename = '%s.cbf' % str(image_number).zfill(5)
    H5ToXds_line = 'H5ToXds %s %s %s' % (master_file_absolute_path, image_number, raw_cbf_filename)
    os.system(H5ToXds_line)
    print H5ToXds_line
    os.system('cat %s | tail -n +14 >> %s' % (raw_cbf_filename, header_filename))
    
    shutil.move(header_filename, os.path.join(destination_directory, filename))
    try:
        os.remove(os.path.realpath(raw_cbf_filename))
    except:
        print '%s already discarded' % os.path.realpath(raw_cbf_filename)
    try:
        os.remove(raw_cbf_filename)
    except:
        print '%s already discarded' % raw_cbf_filename
        
    if compress == 'bzip2':
        os.system('bzip2 -f %s &' % os.path.join(destination_directory, filename))
    elif compress == 'gzip':
        os.system('gzip -f %s &' % os.path.join(stardestination_directory, filename))
    else:
        pass
   
def get_dataset_filenames(master_file_absolute_path):
    name_pattern = os.path.basename(master_file_absolute_path)[:-10]
    directory = os.path.dirname(master_file_absolute_path)
    return glob.glob('%s*.h5' % os.path.join(directory, name_pattern))
    
if __name__ == '__main__':
    import optparse
    
    parser = optparse.OptionParser()
    
    parser.add_option('-m', '--master_file', type=str, help='Path to the master_file')
    parser.add_option('-d', '--destination_directory', default=None, type=str, help='destination directory')
    parser.add_option('-t', '--treatment_directory', default='/dev/shm', type=str, help='treatment directory')
    parser.add_option('-n', '--n_cpu', default=0, type=int, help='Number of parallel extraction, by defalult it will determine the number of cores of the machine and use all of them.')
    parser.add_option('-f', '--first', default=0, type=int, help='First image to extract. Default is the first one.')
    parser.add_option('-l', '--last', default=-1, type=int, help='Last image to extract. Default is the last one.')
    parser.add_option('-b', '--bzip2', action='store_true', help='Compress cbf files using bzip2.')
    parser.add_option('-g', '--gzip', action='store_true', help='Compress cbf files using gzip.')
    parser.add_option('-O', '--overwrite', action='store_true', help='Overwrite existing files.')
    
    options, args = parser.parse_args()
    print 'options'
    print options
    if options.n_cpu <= 0:
       options.n_cpu = get_n_cpu()

    master_file_absolute_path = os.path.abspath(options.master_file)
    
    source_directory = os.path.dirname(master_file_absolute_path)
    
    dataset_filenames = get_dataset_filenames(master_file_absolute_path)
    print 'dataset_filenames', dataset_filenames
    
    for f in dataset_filenames:
        shutil.copy2(f, options.treatment_directory)
    
    if options.destination_directory == None:
        destination_directory = os.getcwd()
    else:
        destination_directory = options.destination_directory
    
    if not os.path.isdir(destination_directory):
        os.makedirs(destination_directory)
        
    #os.chdir(os.path.dirname(os.path.abspath(options.master_file)))
    
    os.chdir(options.treatment_directory)

    master_file = h5py.File(os.path.basename(options.master_file))
   
    nimages = get_nimages(master_file, options.first, options.last) 
    wedges = get_wedges(options.first, nimages, options.n_cpu)

    if options.bzip2:
        compress = 'bzip2'
    elif options.gzip:
        compress = 'gzip'
    else:
        compress = None
    
    filename_template = os.path.join(destination_directory, os.path.basename(master_file.filename.replace('_master.h5', '_?????.cbf')))
    
    cbf_files = glob.glob(filename_template)
    if nimages == len(cbf_files) and options.overwrite != True:
        print 'It seems conversion of %s to cbf has already been done. Please use option --overwrite if you wish to regenerate the files. Exiting ...'
        sys.exit()
        
    print 'Starting h5 to cbf conversion. The conversion time per image should be below 0.1 second. If it is more, there may be something wrong with the system.'
    print '%d images to extract. Using %d threads. Processing in %d wedges of %d images.\n' % (nimages, options.n_cpu, len(wedges), len(wedges[0]))
 
    extract_cbfs(master_file, master_file_absolute_path.replace(source_directory, options.treatment_directory), destination_directory, options.first, options.last, options.n_cpu, compress)
    
    os.chdir(destination_directory)
    
