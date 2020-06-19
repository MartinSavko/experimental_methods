#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
The raster object allows to define and carry out a collection of series of diffraction still images on a grid specified over a rectangular area.
'''
import gevent

import time
import copy
import pickle
import scipy.misc
import numpy
import os
import numpy as np

from diffraction_experiment import diffraction_experiment
from area import area
from optical_alignment import optical_alignment

def height_model(angle, c, r, alpha, k):
    return c + r*np.cos(k*angle - alpha)
    
class raster_scan(diffraction_experiment):
    
    actuator_names = ['Omega', 'AlignmentX', 'AlignmentY', 'AlignmentZ', 'CentringX', 'CentringY']
    
    specific_parameter_fields = [{'name': 'vertical_range', 'type': 'float', 'description': ''},
                                 {'name': 'horizontal_range', 'type': 'float', 'description': ''},
                                 {'name': 'number_of_rows', 'type': 'int', 'description': ''},
                                 {'name': 'number_of_columns', 'type': 'int', 'description': ''},
                                 {'name': 'scan_start_angle', 'type': 'float', 'description': ''},
                                 {'name': 'inverse_direction', 'type': 'bool', 'description': ''},
                                 {'name': 'use_centring_table', 'type': 'bool', 'description': ''},
                                 {'name': 'focus_center', 'type': 'float', 'description': ''},
                                 {'name': 'against_gravity', 'type': 'bool', 'description': ''},
                                 {'name': 'scan_axis', 'type': 'str', 'description': ''},
                                 {'name': 'scan_range', 'type': 'float', 'description': ''},
                                 {'name': 'frame_time', 'type': 'float', 'description': ''},
                                 {'name': 'ntrigger', 'type': '', 'description': ''},
                                 {'name': 'jumps', 'type': 'array', 'description': ''},
                                 {'name': 'collect_sequence', 'type': 'array', 'description': ''},
                                 {'name': 'nframes', 'type': 'int', 'description': ''},
                                 {'name': 'beam_size', 'type': 'array', 'description': ''},
                                 {'name': 'reference_position', 'type': 'dict', 'description': ''},
                                 {'name': 'grid', 'type': 'array', 'description': ''},
                                 {'name': 'points', 'type': 'array', 'description': ''},
                                 {'name': 'shape', 'type': 'array', 'description': ''},
                                 {'name': 'nimages', 'type': 'int', 'description': ''},
                                 {'name': 'angle_per_frame', 'type': 'float', 'description': ''},
                                 {'name': 'shutterless', 'type': 'bool', 'description': ''},
                                 {'name': 'nimages_per_point', 'type': 'int', 'description': 'Number of points per grid point, only relevant in shuttered mode'},
                                 {'name': 'npasses', 'type': 'int', 'description': 'Number of passes per grid point'},
                                 {'name': 'dark_time_between_passes', 'type': 'float', 'description': 'Time in seconds between successive passes'}]
    
    def __init__(self,
                name_pattern,
                directory,
                vertical_range,
                horizontal_range,
                beam_size=np.array([0.005, 0.010]),
                number_of_rows=None,
                number_of_columns=None,
                frame_time=0.005,
                scan_start_angle=None,
                scan_range=1,
                image_nr_start=1,
                position=None, 
                kappa=None,
                phi=None,
                photon_energy=None,
                resolution=None,
                detector_distance=None,
                detector_vertical=None,
                detector_horizontal=None,
                transmission=None,
                flux=None,
                scan_axis='vertical', # 'horizontal' or 'vertical'
                shutterless=True,
                nimages_per_point=1,
                npasses=1,
                dark_time_between_passes=0.,
                use_centring_table=True,
                inverse_direction=False,
                against_gravity=False,
                zoom=None, # by default use the current zoom
                snapshot=True,
                diagnostic=None,
                analysis=None,
                simulation=None,
                conclusion=True,
                parent=None):
        
        if hasattr(self, 'parameter_fields'):
            self.parameter_fields += raster_scan.specific_parameter_fields
        else:
            self.parameter_fields = raster_scan.specific_parameter_fields[:]
            
        diffraction_experiment.__init__(self, 
                                        name_pattern, 
                                        directory,
                                        position=position,
                                        kappa=kappa,
                                        phi=phi,
                                        photon_energy=photon_energy,
                                        resolution=resolution,
                                        detector_distance=detector_distance,
                                        detector_vertical=detector_vertical,
                                        detector_horizontal=detector_horizontal,
                                        transmission=transmission,
                                        flux=flux,
                                        snapshot=snapshot,
                                        diagnostic=diagnostic,
                                        analysis=analysis,
                                        simulation=simulation,
                                        conclusion=conclusion,
                                        parent=parent)
        
        self.description = 'X-ray diffraction raster scan, Proxima 2A, SOLEIL, %s' % time.ctime(self.timestamp)
        
        self.vertical_range = vertical_range
        self.horizontal_range = horizontal_range
        if number_of_columns == None or number_of_rows == None:
            if type(beam_size) is str:
                beam_size = np.array(eval(beam_size))
            else:
                beam_size = beam_size
            shape = np.ceil(np.array((self.vertical_range, self.horizontal_range)) / beam_size).astype(np.int)
            number_of_rows, number_of_columns = shape
        
        self.beam_size = beam_size
        self.shape = numpy.array((number_of_rows, number_of_columns))
        self.number_of_rows = number_of_rows
        self.number_of_columns = number_of_columns
        self.nframes = self.number_of_rows * self.number_of_columns
        self.frame_time = frame_time
        
        print 'number_of_rows', self.number_of_rows
        print 'number_of_columns', self.number_of_columns
        print 'motor_speed', self.vertical_range/(self.number_of_rows * self.frame_time)
        print 'scan_range', scan_range
        
        if scan_start_angle == None:
            self.scan_start_angle = self.goniometer.get_omega_position()
        else:
            self.scan_start_angle = scan_start_angle
        self.scan_range = scan_range
        self.image_nr_start = image_nr_start
        if position == None:
            self.reference_position = self.goniometer.get_aligned_position()
        else:
            self.reference_position = position
        self.scan_axis = scan_axis
        self.shutterless = shutterless
        self.nimages_per_point = nimages_per_point
        self.npasses = npasses
        self.dark_time_between_passes = dark_time_between_passes
        self.inverse_direction = inverse_direction
        self.use_centring_table = use_centring_table
        self.against_gravity = against_gravity
        self.zoom = zoom
        
        if self.use_centring_table:
            self.focus_center, self.vertical_center = self.goniometer.get_focus_and_vertical_from_position(position=self.reference_position)
            #vertical_center = self.goniometer.get_centringtable_vertical_position_from_hypothetical_centringx_centringy_and_omega(self.reference_position['CentringX'], self.reference_position['CentringY'], self.scan_start_angle)
            #self.focus_center = self.goniometer.get_centringtable_focus_position_from_hypothetical_centringx_centringy_and_omega(self.reference_position['CentringX'], self.reference_position['CentringY'], self.scan_start_angle)
        else:
            self.vertical_center = self.reference_position['AlignmentZ']
            self.focus_center = None
            
        self.horizontal_center = self.reference_position['AlignmentY']
        
        self.area = area(vertical_range, horizontal_range, number_of_rows, number_of_columns, self.vertical_center, self.horizontal_center)
        grid, points = self.area.get_grid_and_points()
        self.grid = grid
        self.points = points
        if self.scan_axis == 'vertical':
            self.position_sequence = self.area.get_position_sequence(grid.T)
        else:
            self.position_sequence = self.area.get_position_sequence(grid)
        self.jumps = None
        
        if self.shutterless == True:
            if self.scan_axis == 'horizontal':
                self.line_scan_time = self.frame_time * self.number_of_columns
                self.angle_per_frame = self.scan_range / self.number_of_columns
                self.ntrigger = self.number_of_rows
                self.nimages = self.number_of_columns
                if self.inverse_direction == True:
                    raster_grid = self.area.get_horizontal_raster(grid)
                    jumps = self.area.get_jump_sequence(raster_grid)
                else:
                    jumps = self.area.get_jump_sequence(grid)
            else:
                self.line_scan_time = self.frame_time * self.number_of_rows
                self.angle_per_frame = self.scan_range / self.number_of_rows
                self.ntrigger = self.number_of_columns
                self.nimages = self.number_of_rows
                if self.inverse_direction == True:
                    raster_grid = self.area.get_vertical_raster(grid)
                    jumps = self.area.get_jump_sequence(raster_grid.T, against_gravity=self.against_gravity)
                else:
                    jumps = self.area.get_jump_sequence(grid.T, against_gravity=self.against_gravity)
            
            self.jumps = jumps
            self.collect_sequence = self.area.get_linearized_point_jumps(jumps, points)
            self.total_expected_exposure_time = self.line_scan_time * self.ntrigger

        else:
            self.angle_per_frame = self.scan_range/self.nimages_per_point
            self.ntrigger = self.number_of_columns * self.number_of_rows * self.npasses
            self.nimages = self.nimages_per_point
            if self.scan_axis == 'horizontal':
                self.line_scan_time = self.frame_time * self.nimages_per_point * self.number_of_columns
            else:
                self.line_scan_time = self.frame_time * self.nimages_per_point * self.number_of_rows
            self.total_expected_exposure_time = self.frame_time * self.nimages_per_point * self.ntrigger
            #self.collect_sequence = [(self.points[position], self.points[position]) for position in self.position_sequence]
            self.collect_sequence = []
            for position in self.position_sequence:
                for k in range(self.npasses):
                    self.collect_sequence.append((self.points[position], self.points[position]))
            
        self.total_expected_wedges = self.ntrigger
    
    def get_frame_time(self):
        return self.frame_time
    
    
    def get_vertical_step_size(self):
        return self.get_step_sizes()[0]
    
    
    def get_horizontal_step_size(self):
        return self.get_step_sizes()[1]
    
    
    def get_beam_vertical_position(self):
        return self.camera.md2.beampositionvertical
    
    
    def get_beam_horizontal_position(self):
        return self.camera.md2.beampositionhorizontal
    
        
    def get_extent(self):
        return numpy.array((self.vertical_range, self.horizontal_range))
        

    def get_step_sizes(self):
        step_sizes = self.get_extent() / numpy.array((self.shape))
        return step_sizes
    

    def get_nimages_per_file(self):
        if self.shutterless == True and self.scan_axis == 'vertical':
            nimages_per_file = self.number_of_rows
        elif self.shutterless == True:
            nimages_per_file = self.number_of_columns
        elif self.npasses > 1:
            nimages_per_file = self.npasses * self.nimages_per_point
        else:
            nimages_per_file = self.nimages
        return nimages_per_file
    
        
    def get_frames_per_second(self):
        return 1./self.get_frame_time()
    
        
    def run(self, gonio_moving_wait_time=0.5, check_wait_time=0.1, number_of_attempts=3):
        self._start = time.time()
        
        self.md2_task_info = []
        
        self.goniometer.wait()
        start_angle = '%6.4f' % self.scan_start_angle
        scan_range = '%6.4f' % self.scan_range
        if self.shutterless == True:
            exposure_time = '%6.4f' % self.line_scan_time
        else:
            exposure_time = '%6.4f' % (self.frame_time * self.nimages_per_point,)
        print 'at the start position'
        if self.use_centring_table:
            start_z = '%6.4f' % self.reference_position['AlignmentZ']
            stop_z = '%6.4f' % self.reference_position['AlignmentZ']
        else:
            start_cx = '%6.4f' % self.reference_position['CentringX']
            start_cy = '%6.4f' % self.reference_position['CentringY']
            stop_cx = '%6.4f' % self.reference_position['CentringX']
            stop_cy = '%6.4f' % self.reference_position['CentringY']
            
        k = 0
            
        previous_start, previous_stop = (None, None), (None, None)
        
        for start, stop in self.collect_sequence:
            if self._stop_flag == True:
                break
            k += 1
                
            if self.use_centring_table:
                x_start, y_start = self.goniometer.get_x_and_y(self.focus_center, start[0], self.scan_start_angle)
                x_stop, y_stop = self.goniometer.get_x_and_y(self.focus_center, stop[0], self.scan_start_angle)
                
                start_position = {'CentringX': x_start, 'CentringY': y_start, 'AlignmentY': start[1], 'AlignmentZ': self.reference_position['AlignmentZ']}
                stop_position = {'CentringX': x_stop, 'CentringY': y_stop, 'AlignmentY': stop[1], 'AlignmentZ': self.reference_position['AlignmentZ']}
                
                start_y = '%6.4f' % start_position['AlignmentY']
                start_cx = '%6.4f' % start_position['CentringX']
                start_cy = '%6.4f' % start_position['CentringY']
                stop_y = '%6.4f' % stop_position['AlignmentY']
                stop_cx = '%6.4f' % stop_position['CentringX']
                stop_cy = '%6.4f' % stop_position['CentringY']
                
            else:
                start_position = {'AlignmentZ': start[0], 'AlignmentY': start[1]}
                start_z, start_y = map(str, start)
                stop_z, stop_y = map(str, stop)
                
            if k==1 and self._stop_flag != True:
                self.goniometer.set_position(start_position, wait=True)
                
            parameters = [start_angle, scan_range, exposure_time, start_y, start_z, start_cx, start_cy, stop_y, stop_z, stop_cx, stop_cy]
            print 'helical scan parameters'
            print parameters
            self.goniometer.wait()
            
            if self.npasses == 1 or (k-1) % self.npasses == 0:
                pass
            else:
                print 'sleeping for specified dark time %f seconds' % self.dark_time_between_passes
                gevent.sleep(self.dark_time_between_passes)
            
            tried = 0
            while tried < number_of_attempts and self._stop_flag != True:
                tried += 1
                try:
                    task_id = self.goniometer.start_scan_4d_ex(parameters)
                    break
                except:
                    print 'It was not possible to start the scan. Is the MD2 still moving? Or have you specified the range in mm rather then microns ?'
                    gevent.sleep(gonio_moving_wait_time)
            while self.goniometer.is_task_running(task_id):
                gevent.sleep(check_wait_time)
            self.md2_task_info.append(self.goniometer.get_task_info(task_id))
            
        self.goniometer.set_position(self.reference_position)
        self.goniometer.wait()
    
    
    def clean(self):
        self.detector.disarm()
        self.goniometer.set_position(self.reference_position)
        self.collect_parameters()
        self.save_parameters()
        self.save_results()
        self.save_log()
        if self.diagnostic == True:
            self.save_diagnostic()
            
    
    def analyze(self):
        #spot_find_line = 'ssh process1 "source /usr/local/dials-v1-4-5/dials_env.sh; cd %s ; echo $(pwd); dials.find_spots shoebox=False per_image_statistics=True spotfinder.filter.ice_rings.filter=True nproc=80 ../%s_master.h5"' % (self.process_directory, self.name_pattern)
        #os.system(spot_find_line)
        #area_sense_line = '/927bis/ccd/gitRepos/eiger/area_sense.py -d %s -n %s &' % (self.directory, self.name_pattern)
        #command = '/home/experiences/proxima2a/com-proxima2a/mxcube_local/HardwareRepository/HardwareObjects/SOLEIL/PX2/experimental_methods/raster_scan_analysis.py'
        command = 'raster_scan_analysis.py'
        area_sense_line = '%s -d %s -n %s &' % (command, self.directory, self.name_pattern)
        os.system(area_sense_line)
        
    
    def conclude(self):
        pass
        
def main():
    import optparse
        
    parser = optparse.OptionParser()
    parser.add_option('-n', '--name_pattern', default='raster_test_$id', type=str, help='Prefix default=%default')
    parser.add_option('-d', '--directory', default='/nfs/data/default', type=str, help='Destination directory default=%default')
    parser.add_option('-y', '--vertical_range', default=0.1, type=float, help='Vertical range in mm')
    parser.add_option('-x', '--horizontal_range', default=0.2, type=float, help='Horizontal range in mm')
    parser.add_option('-r', '--number_of_rows', default=None, type=int, help='Number of rows')
    parser.add_option('-c', '--number_of_columns', default=None, type=int, help='Number of columns')
    parser.add_option('-b', '--beam_size', default='(0.005, 0.010)', type=str, help='Beam size in mm')
    parser.add_option('-a', '--scan_start_angle', default=None, type=float, help='Scan start angle [deg]')
    parser.add_option('-i', '--position', default=None, type=str, help='Gonio alignment position [dict]')
    parser.add_option('-s', '--scan_range', default=0.1, type=float, help='Scan range [deg] per helical line (-> 0)')
    parser.add_option('-e', '--frame_time', default=0.05, type=float, help='Exposure time per image [s]')
    parser.add_option('-f', '--image_nr_start', default=1, type=int, help='Start image number [int]')
    parser.add_option('-p', '--photon_energy', default=None, type=float, help='Photon energy ')
    parser.add_option('-t', '--detector_distance', default=None, type=float, help='Detector distance')
    parser.add_option('-o', '--resolution', default=None, type=float, help='Resolution [Angstroem]')
    parser.add_option('-X', '--flux', default=None, type=float, help='Flux [ph/s]')
    parser.add_option('-m', '--transmission', default=None, type=float, help='Transmission. Number in range between 0 and 1.')
    parser.add_option('-I', '--inverse_direction', action='store_true', help='Rastered acquisition')
    parser.add_option('-G', '--against_gravity', action='store_true', help='Vertical scan direction against gravity')
    parser.add_option('-T', '--do_not_use_centring_table', action='store_true', help='Do not use centring table for vertical sample movements.')
    parser.add_option('-z', '--zoom', default=None, type=int, help='Zoom to acquire optical image at.')
    parser.add_option('-A', '--analysis', action='store_true', help='If set will perform automatic analysis.')
    parser.add_option('-C', '--conclusion', action='store_true', help='If set will move the motors upon analysis.')
    parser.add_option('-D', '--diagnostic', action='store_true', help='If set will record diagnostic information.')
    parser.add_option('-S', '--simulation', action='store_true', help='If set will simulate the run.')
    parser.add_option('-O', '--optical_alignment_results', default=None, type=str, help='Use results from optical alignment analysis to specify the raster parameters')
    parser.add_option('--max', action='store_true', help='To be used with -O parameter, use parameters for max area raster.')
    parser.add_option('--min', action='store_true', help='To be used with -O parameter, use parameters for min area raster.')
    parser.add_option('-V', '--vertical_plus', default=0., type=float, help='To be used with -O parameter, specify in mm by how much the vertical scan range should be increased compared to values from optical scan analysis.')
    parser.add_option('-H', '--horizontal_plus', default=0., type=float, help='To be used with -O parameter, specify in mm by how much the horizontal scan range should be increased compared to values from optical scan analysis.')
    parser.add_option('-P', '--angle_offset', default=0., type=float, help='To be used with -O parameter, specify in degrees angle offset with respect to min (if --min specified) or max (if --max specified) orientations.')
    parser.add_option('-M', '--motor_speed', default=None, type=float, help='Motor speed [mm/s]')
    parser.add_option('-N', '--scan_duration', default=None, type=float, help='Scan duration (per line) [s]')
    parser.add_option('--scan_axis', default='vertical', type=str, help='Scan axis (vertical or horizontal) default=%default')
    parser.add_option('--shuttered', action='store_true', help='Collect in shuttered mode. The default is shutterless.')
    parser.add_option('--nimages_per_point', default=1, type=int, help='Images per point. Only relevant in shuttered mode. [int]')
    
    options, args = parser.parse_args()
    
    print 'options', options
    print 'args', args
    print
    filename = os.path.join(options.directory, options.name_pattern) + '_parameters.pickle'
    
    if options.shuttered == True:
        options.shutterless = False
        
    if options.do_not_use_centring_table == True:
        del options.do_not_use_centring_table
        options.use_centring_table = False
        
    if options.optical_alignment_results != None:
        oar = pickle.load(open(options.optical_alignment_results))
        oap = pickle.load(open(options.optical_alignment_results.replace('_results.pickle', '_parameters.pickle')))
        if oar.has_key('result_position'):
            position = oar['result_position']
        else:
            oa = optical_alignment(oap['name_pattern'], oap['directory'])
            reference_optical_scan_position = oap['position']
            optical_scan_move_vector_mm = oar['move_vector_mm']
            position = oa.get_result_position(reference_position=reference_optical_scan_position, move_vector_mm=optical_scan_move_vector_mm)
            
        if options.min:
            horizontal_range, vertical_range, scan_start_angle, zoom = oar['min_raster_parameters']
        else:
            horizontal_range, vertical_range, scan_start_angle, zoom = oar['max_raster_parameters']
        
        if options.angle_offset != 0:
            scan_start_angle += options.angle_offset
            height_fit = oar['fits']['height']
            c, r, alpha = height_fit[0].x
            k = height_fit[1]
            vertical_range = height_model(np.radians(scan_start_angle), c, r, alpha, k) * oap['calibration'][0]
            position['Omega'] = scan_start_angle
            
        if options.vertical_plus != 0.:
            vertical_range += options.vertical_plus
        if options.horizontal_plus != 0.:
            horizontal_range += options.horizontal_plus
            position['AlignmentY'] += options.horizontal_plus/2.
            
        options.position = position
        options.vertical_range = vertical_range
        options.horizontal_range = horizontal_range
        options.scan_start_angle = scan_start_angle
        options.zoom = zoom
        
    if options.inverse_direction != True:
        options.inverse_direction = False
    if options.against_gravity != True:
        options.against_gravity = False
    
    if options.scan_duration != None:
        motor_speed = options.vertical_range/options.scan_duration
        nimages = int(options.scan_duration/options.frame_time)
        vertical_step = options.vertical_range/nimages
        if type(options.beam_size) is str:
            beam_size = np.array(eval(options.beam_size))
        else:
            beam_size = options.beam_size
        beam_size[0] = vertical_step
        options.beam_size = beam_size
        
    elif options.motor_speed != None:
        scan_duration = options.vertical_range/ options.motor_speed
        nimages = int(scan_duration/options.frame_time)
        vertical_step = options.vertical_range/nimages
        if type(options.beam_size) is str:
            beam_size = np.array(eval(options.beam_size))
        else:
            beam_size = options.beam_size
        beam_size[0] = vertical_step
        options.beam_size = beam_size
    
    print
    print 'options after update from optical scan analysis results and priority options', options
    print 
    del options.min
    del options.max
    del options.optical_alignment_results
    del options.vertical_plus
    del options.horizontal_plus
    del options.angle_offset
    del options.motor_speed
    del options.scan_duration
    del options.shuttered
    
    r = raster_scan(**vars(options))
    
    if not os.path.isfile(filename):
        r.execute()
    elif options.analysis == True:
        r.analyze()
        if options.conclusion == True:
            r.conclude()
    
if __name__ == '__main__':
    main()
