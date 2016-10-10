import PyTango
import logging
import traceback
    
class goniometer(object):
    motorsNames = ['AlignmentX', 
                   'AlignmentY', 
                   'AlignmentZ',
                   'CentringX', 
                   'CentringY']
                   
    motorShortNames = ['PhiX', 'PhiY', 'PhiZ', 'SamX', 'SamY']
    mxcubeShortNames = ['phix', 'phiy', 'phiz', 'sampx', 'sampy']
    
    shortFull = dict(zip(motorShortNames, motorsNames))
    phiy_direction=-1.
    phiz_direction=1.
    
    def __init__(self):
        self.md2 = PyTango.DeviceProxy('i11-ma-cx1/ex/md2')
      
    def set_scan_start_angle(self, scan_start_angle):
        self.scan_start_angle = scan_start_angle
        self.md2.scanstartangle = scan_start_angle
    
    def get_scan_start_angle(self):
        return self.md2.scanstartangle
       
    def set_scan_range(self, scan_range):
        self.scan_range = scan_range
        self.md2.scanrange = scan_range
        
    def get_scan_range(self):
        return self.md2.scanrange
        
    def set_scan_exposure_time(self, scan_exposure_time):
        self.scan_exposure_time = scan_exposure_time
        self.md2.scanexposuretime = scan_exposure_time
    
    def get_scan_exposure_time(self):
        return self.md2.scanexposuretime
    
    def set_scan_number_of_frames(self, scan_number_of_frames):
        self.scan_number_of_frames = scan_number_of_frames
        self.md2.scannumberofframes = scan_number_of_frames
       
    def get_scan_number_of_frames(self):
        return self.md2.scannumberofframes
        
    def set_collect_phase(self):
        return self.md2.startsetphase('DataCollection')
        
    def abort(self):
        return self.md2.abort()

    def start_scan(self, wait=False):
        tried = 0
        while tried < 3:
            tried += 1
            try:
                scan_id = self.md2.startscan()
                break
            except:
                print 'Not possible to start the scan. Is the MD2 still moving ?'
                time.sleep(0.5)
        
        if wait:
            while self.md2.istaskrunning(scan_id):
                time.sleep(0.1)
            print self.md2.gettaskinfo(scan_id)
        else:
            return scan_id

    def helical_scan(self, start, stop, scan_start_angle, scan_range, scan_exposure_time, wait=True):
        scan_start_angle = '%6.4f' % scan_start_angle
        scan_range = '%6.4f' % scan_range
        scan_exposure_time = '%6.4f' % scan_exposure_time
        start_z = '%6.4f' % start['AlignmentZ']
        start_y = '%6.4f' % start['AlignmentY']
        stop_z = '%6.4f' % stop['AlignmentZ']
        stop_y = '%6.4f' % stop['AlignmentY']
        start_cx = '%6.4f' % start['CentringX']
        start_cy = '%6.4f' % start['CentringY']
        stop_cx = '%6.4f' % stop['CentringX']
        stop_cy = '%6.4f' % stop['CentringY']
        parameters = [scan_start_angle, scan_range, scan_exposure_time, start_y,start_z, start_cx, start_cy, stop_y, stop_z, stop_cx, stop_cy]
        print 'helical scan parameters'
        print parameters
        tried = 0
        while tried < 3:
            tried += 1
            try:
                scan_id = self.md2.startScan4DEx(parameters)
                break
            except:
                print 'Not possible to start the scan. Is the MD2 still moving or have you specified the range in mm rather then microns ?'
                time.sleep(0.5)
        if wait:
            while self.md2.istaskrunning(scan_id):
                time.sleep(0.1)
            print self.md2.gettaskinfo(scan_id)
        else:
            return scan_id
                    
    def start_helical_scan(self):
        return self.md2.startscan4d()
        
    def set_helical_start(self):
        return self.md2.setstartscan4d()
    
    def set_helical_stop(self):
        return self.md2.setstopscan4d()
        
    def get_motor_state(self, motor_name):
        return self.md2.getMotorState(motor_name).name
        
    def get_state(self):
        motors = ['Omega', 'AlignmentX', 'AlignmentY', 'AlignmentZ', 'CentringX', 'CentringY', 'ApertureHorizontal', 'ApertureVertical', 'CapillaryHorizontal', 'CapillaryVertical', 'ScintillatorHorizontal', 'ScintillatorVertical', 'Zoom']
        state = set([self.get_motor_state(m) for m in motors])
        if len(state) == 1 and 'STANDBY' in state:
            return 'STANDBY'
        else:
            return 'MOVING'
            
    def wait(self, device=None):
        green_light = False
        while green_light is False:
            try:
                if device is None:
                    if self.getState() in ['MOVING', 'RUNNING']:
                        logging.info("MiniDiffPX2 wait" )
                    else:
                        green_light = True
                        return
                else:   
                    if device.state().name not in ['STANDBY']:
                        logging.info("Device %s wait" % device)
                    else:
                        green_light = True
                        return
            except:
                traceback.print_exc()
                logging.info('Problem occured in wait %s ' % device)
                logging.info(traceback.print_exc())
            time.sleep(.1)
        

    def move_to_position(self, position={}, epsilon = 0.0002):
        print 'position %s' % position
        if position != {}:
            for motor in position:
                while abs(self.md2.read_attribute('%sPosition' % self.shortFull[motor]).value - position[motor]) > epsilon:
                    self.wait()
                    time.sleep(0.5)
                    self.md2.write_attribute('%sPosition' % self.shortFull[motor], position[motor])
                
            self.wait()
        self.wait()
        return
    
    def set_position(self, position, motors=['AlignmentX', 'AlignmentY', 'AlignmentZ', 'CentringY', 'CentringX']):
        motor_name_value_list = ['%s=%6.4f' % (motor, position[motor]) for motor in motors]
        command_string = ','.join(motor_name_value_list)
        print 'command string', command_string
        k=0
        while k < 3:
            k+=1
            try:
                return self.md2.startSimultaneousMoveMotors(command_string)
            except:
                time.sleep(1)
             
    def set_omega_position(self, omega_position):
        self.md2.OmegaPosition = omega_position
    
    def get_omega_positions(self):
        return self.md2.OmegaPosition
    
    def get_position(self):
        return dict([(m.split('=')[0], float(m.split('=')[1])) for m in self.md2.motorpositions])

    def insert_backlight(self):
        self.md2.backligtison = True

    def remove_backlight(self):
        if self.md2.backlightison == True:
            self.md2.backlightison = False
    
    def insert_frontlight(self):
        self.md2.frontlightison = True
    
    def extract_frontlight(self):
        self.md2.frontlightison = False

    def backlight_is_on(self):
        return self.md2.backlightison

    def insert_fluorescence_detector(self):
        self.md2.fluoisback = False
    
    def extract_fluorescence_detector(self):
        self.md2.fluoisback = True

    def start_raster_scan(self, vertical_range, horizontal_range, number_of_rows, number_of_columns, direction_inversion):
        return self.md2.startRasterScan([vertical_range, horizontal_range, number_of_rows, number_of_columns, direction_inversion])

    def start_scan_4d_ex(self, parameters):
        return self.md2.startScan4DEx(parameters)

    def insert_cryostream(self):
        self.md2.cryoisback = False
    
    def extract_cryostream(self):
        self.md2.cryoisback = True

    def is_task_running(self, task_id):
        return self.md2.istaskrunning(task_id)

    def get_task_info(self, task_id):
        return self.md2.gettaskinfo(task_id)
        
    def set_detector_gate_pulse_enabled(self, value):
        return self.md2.DetectorGatePulseEnabled = value

    def set_data_collection_phase(self):
        return self.md2.startsetphase('DataCollection')

    def set_transfer_phase(self):
        return self.md2.startsetphase('Transfer')

    def set_beam_location_phase(self):
        return self.md2.startsetphase('BeamLocation')

    def set_centrig_phase(self):
        return self.md2.startsetphase('Centring')

