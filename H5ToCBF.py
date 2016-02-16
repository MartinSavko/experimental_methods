#!/usr/bin/env python
''' 
author: Martin Savko (savko@synchrotron-soleil.fr)
license: LGPL v3
'''

import h5py
import time 
import os
import shutil

header_template = '''###CBF: VERSION 1.5, CBFlib v0.7.8 - SLS/DECTRIS PILATUS detectors

{filename}

_array_data.header_convention "SLS_1.0"
_array_data.header_contents
;
# Detector: Dectris Eiger 9M, E-18-0102
# {data_collection_date}
# Pixel_size 75e-6 m x 75e-6 m
# Silicon sensor, thickness 0.000450 m
# Exposure_time {exposure_time} s
# Exposure_period {exposure_period} s
# Tau 199.1e-09 s s
# Count_cutoff {count_cutoff}
# Threshold_setting {threshold_setting} eV
# N_excluded_pixels {n_excluded_pixels}
# Excluded_pixels /entry/instrument/detector/detectorSpecific/pixel_mask
# Flat_field /entry/instrument/detector/detectorSpecific/flatfield
# Image_path {image_path}
# Wavelength {wavelength} A
# Detector_distance {detector_distance} m
# Beam_xy ({beam_center_x}, {beam_center_y}) pixels
# Start_angle {start_angle} degree
# Angle_increment {angle_increment} degree
# Detector_2theta 0.0000 degree
# Polarization 0.990
# Omega {omega} degree
# Omega_increment {omega_increment} degree
# Oscillation_axis OMEGA
;

'''

# storing hdf5 paths into mnemonic variables
nimages = "/entry/instrument/detector/detectorSpecific/nimages"

data_collection_date = '/entry/instrument/detector/detectorSpecific/data_collection_date'
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

def get_header_information(master_file):
    h = {}
    h['data_collection_date'] = master_file[data_collection_date].value
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
    h['omega'] = 0.0
    h['start_angle'] = 0.0
    h['angle_increment'] = master_file[omega_range_average].value
    return h


def extract_cbfs(master_file, home_dir):
    nimages = master_file["/entry/instrument/detector/detectorSpecific/nimages"].value
    omegas = master_file["/entry/sample/goniometer/omega"].value
    
    header_dictionary = get_header_information(master_file)
    image_path = os.path.dirname(os.path.abspath(master_file.filename))
    
    header_dictionary['image_path'] = image_path
    filename_template = master_file.filename.replace('_master.h5', '_#####.cbf')
    
    start = time.time()
    for n in range(nimages):
        header_dictionary['filename'] = os.path.basename(filename_template.replace('#####', str(n+1).zfill(5)))
        try:
            if type(0.0) == type(omegas):
                header_dictionary['omega'] =  omegas
                header_dictionary['start_angle'] = omegas
            else:
                header_dictionary['omega'] =  omegas[n]
                header_dictionary['start_angle'] = omegas[n]
        except IndexError:
            print 'omegas', omegas
            header_dictionary['omega'] =  omegas
            header_dictionary['start_angle'] = omegas
        header = header_template.format(**header_dictionary)
        header_filename = 'header_%s' % (str(n+1).zfill(5))
        f = open(header_filename, 'w')
        f.write(header)
        f.close()
        
        raw_cbf_filename = '%s.cbf' % str(n+1).zfill(5)
        os.system('H5ToXds %s %s %s' % (os.path.join(image_path, master_file.filename), n+1, raw_cbf_filename))
        os.system('cat %s | tail -n +14 >> %s' % (raw_cbf_filename, header_filename))
        
        shutil.move(header_filename, os.path.join(home_dir, header_dictionary['filename']))
        os.remove(raw_cbf_filename)
        if n % 100 == 0:
            t = time.time() - start
            print('extracting image #%s, total time %s, time per image %s' % (str(n+1).zfill(5), t, t/(n+1)))
        if (n+1) == nimages:
            t = time.time() - start
            print('Extracted %s images, total time %s, time per image %s' % (nimages, t, t/(n+1)))
            
if __name__ == '__main__':
    import optparse
    
    parser = optparse.OptionParser()
    
    parser.add_option('-m', '--master_file', type=str, help='Path to the master_file')
    
    options, args = parser.parse_args()
    
    home_dir = os.getcwd()
    
    os.chdir(os.path.dirname(os.path.abspath(options.master_file)))
    
    master_file = h5py.File(options.master_file)
    
    extract_cbfs(master_file, home_dir)
    
    os.chdir(home_dir)
    