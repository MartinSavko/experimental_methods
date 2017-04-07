#!/usr/bin/env python
from experiment import experiment
from PyTango import DeviceProxy as dp

class intensity_scan(experiment):
    
    def __init__(self,
                 name_pattern, 
                 directory,
                 photon_energy=None):
        
        experiment.__init__(self, 
                            name_pattern=name_pattern, 
                            directory=directory)
        
        self.photon_energy = photon_energy
        
        # Necessary equipment
        self.instrument = instrument()
        try:
            self.energy_motor = energy_motor()
        except:
            from energy import energy_mockup
            self.energy_motor = energy_mockup()
    
    def initialize_actuators(self):
        actuators = {}
        actuators['md2'] = dp('i11-ma-cx1/ex/md2')
        actuators['mono'] = dp('i11-ma-c03/op/mono1')
        actuators['mono_fine'] = dp('i11-ma-c03/op/mono1-mt_rx')
        actuators['ble'] = dp('i11-ma-c00/ex/beamlineenergy')
        actuators['undulator'] = dp('ans-c11/ei/m-u24')
        return actuators
    
    def initialize_monitors(self):
        monitors = {}
        monitors['sai'] = dp('i11-ma-c00/ca/sai.2')
        monitors['xbpm1'] = dp('i11-ma-c04/dt/xbpm_diode.1')
        monitors['cvd1'] = dp('i11-ma-c05/dt/xbpm-cvd.1')
        monitors['xbpm3'] = dp('i11-ma-c05/dt/xbpm_diode.3')
        monitors['xbpm5'] = dp('i11-ma-c06/dt/xbpm_diode.5')
        monitors['psd5'] = dp('i11-ma-c06/dt/xbpm_diode.psd.5')
        monitors['xbpm6'] = dp('i11-ma-c06/dt/xbpm_diode.6')
        monitors['machine'] = dp('ans/ca/machinestatus')
        return monitors
    
    def set_photon_energy(self, photon_energy=None):
        if photon_energy is not None:
            self.photon_energy = photon_energy
            self.energy_motor.set_energy(photon_energy)

    def get_photon_energy(self):
        return self.photon_energy

    def prepare(self):
        self.check_directory(self.process_directory)
        self.set_photon_energy(self.photon_energy)
        self.monitors = self.initialize_monitors()
        self.actuators = self.initialize_actuators()
        self.write_destination_namepattern(image_path=self.directory, name_pattern=self.name_pattern)
        self.energy_motor.turn_off()
        
    def collect(self):
        return self.run()
    def measure(self):
        return self.run()
    def run():
        pass
    def set_operational_speed(self, actuator):
        pass
    def set_scan_speed(self, actuator):
        pass
    
    def step_scan(actuator, positions):
        results = []
        for point in positions:
            self.move(actuator, point, wait=True)
            results.append(self.measure())
        return results
    
    
    def continuous_scan(actuator, start, stop):
        results = []
        self.move(actuator, start, wait=True)
        self.set_scan_speed(actuator)
        self.move(actuator, stop, wait=False)
        while self.is_moving(actuator):
            results.append(self.measure())
        self.set_operational_speed(actuator)
    
    def collect(self):
        return self.measure()
    def observe(self):
        return self.measure()
    def measure(self):
        
    
    def clean(self):
        pass
    
    def stop(self):
        pass
        
    def analyze(self):
        pass
    