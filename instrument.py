#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Instrument object. Gives access to all of the beamline and machine parameters relevant to the experiment.
'''

from PyTango import DeviceProxy as dp
import time
import sqlite3

class instrument(object):
    def __init__(self):
        #source
        self.machine = machine() # done
        self.undulator = undulator() # done
        self.monochromator = monochromator() # done 
        self.beamlineenergy = beamlineenergy() # done
        # slits
        self.primary_slits = primary_slits() # done
        self.secondary_slits = secondary_slits() # done
        self.slits3 = slits3() # done
        self.slits5 = slits5()
        self.experimental_slits = experimental_slits() # done
        # filters
        self.filters = filters()# done
        # mirrors
        self.hpm = hpm()# done
        self.vfm = vfm()# done
        self.hfm = hfm()# done
        # tables
        self.hpm_table = hpm_table()# done
        self.experimental_table = experimental_table()# done
        self.detector_table = detector_table()# done
        # apertures
        self.apertures = apertures()# done
        self.beamstop = beamstop() # done
        #self.movable_beamstop = movable_beamstop()
        # beam position and intensity monitors
        self.xbpm1 = xbpm1() # done
        self.xbpm3 = xbpm3() # done
        self.xbpm5 = xbpm5() # done
        self.cvd1 = cvd1() # done
        self.beam_position = beam_position() # done
        # thermomethers
        self.thermometers = thermometers() # done
        # presure gauges
        self.vacuum = vacuum() # done
        # shutters
        self.safety_shutter = safety_shutter()
        
    def get_state(self):
        p = {}
        # source
        p['time'] = time.time()
        p['time_stamp'] = time.asctime()
        p['machine_current'] = self.machine.get_current()
        p['undulator_gap'] = self.undulator.get_gap()
        p['undulator_computed_gap'] = self.undulator.get_computed_gap()
        p['undulator_energy'] = self.undulator.get_energy()
        p['monochromator_energy'] = self.monochromator.get_energy()
        p['monochromator_wavelength'] = self.monochromator.get_wavelength()
        p['monochromator_position'] = self.monochromator.get_position()
        p['monochromator_rx'] = self.monochromator.get_rx()
        p['monochromator_rx_fine'] = self.monochromator.get_rx_fine()
        p['beamlineenergy_energy'] = self.beamlineenergy.get_energy()
        p['beamlineenergy_coupling'] = self.beamlineenergy.get_coupling()
        p['hu640_state'] = self.machine.get_hu640_status()
        # slits
        p['primary_slits_horizontal_gap'] = self.primary_slits.get_horizontal_gap()
        p['primary_slits_vertical_gap'] = self.primary_slits.get_vertical_gap()
        p['primary_slits_horizotal_position'] = self.primary_slits.get_horizontal_position()
        p['primary_slits_vertical_position'] = self.primary_slits.get_vertical_position()

        p['secondary_slits_horizontal_gap'] = self.secondary_slits.get_horizontal_gap()
        p['secondary_slits_vertical_gap'] = self.secondary_slits.get_vertical_gap()
        p['secondary_slits_horizotal_position'] = self.secondary_slits.get_horizontal_position()
        p['secondary_slits_vertical_position'] = self.secondary_slits.get_vertical_position()

        p['slits3_horizontal_gap'] = self.slits3.get_horizontal_gap()
        p['slits3_vertical_gap'] = self.slits3.get_vertical_gap()
        p['slits3_horizotal_position'] = self.slits3.get_horizontal_position()
        p['slits3_vertical_position'] = self.slits3.get_vertical_position()
        
        p['slits5_horizontal_gap'] = self.slits5.get_horizontal_gap()
        p['slits5_vertical_gap'] = self.slits5.get_vertical_gap()
        p['slits5_horizotal_position'] = self.slits5.get_horizontal_position()
        p['slits5_vertical_position'] = self.slits5.get_vertical_position()

        p['experimental_slits_horizontal_gap'] = self.experimental_slits.get_horizontal_gap()
        p['experimental_slits_vertical_gap'] = self.experimental_slits.get_vertical_gap()
        p['experimental_slits_horizotal_position'] = self.experimental_slits.get_horizontal_position()
        p['experimental_slits_vertical_position'] = self.experimental_slits.get_vertical_position()

        p['filters'] = self.filters.get_filter()

        # mirror tables
        p['hpm_table_X'] = self.hpm_table.get_x()
        p['hpm_table_Z'] = self.hpm_table.get_z()

        p['experimental_table_pitch'] = self.experimental_table.get_pitch()
        p['experimental_table_roll'] = self.experimental_table.get_roll()
        p['experimental_table_yaw'] = self.experimental_table.get_yaw()
        p['experimental_table_z'] = self.experimental_table.get_z()
        p['experimental_table_x'] = self.experimental_table.get_x()

        # mirrors
        p['hpm_tx'] = self.hpm.get_x()
        p['hpm_rz'] = self.hpm.get_rz()
        p['hpm_rs'] = self.hpm.get_rs()

        p['vfm_tz'] = self.vfm.get_z()
        p['vfm_rx'] = self.vfm.get_rx()

        p['hfm_tx'] = self.hfm.get_x()
        p['hfm_rz'] = self.hfm.get_rz()

        p['vfm_tensions'] = self.hfm.get_voltages()
        p['hfm_tensions'] = self.hfm.get_voltages()

        p['aperture_diameter'] = self.apertures.get_diameter()
        p['aperture_x'] = self.apertures.get_x()
        p['aperture_z'] = self.apertures.get_z()

        p['beamstop_x'] = self.beamstop.get_x()
        p['beamstop_z'] = self.beamstop.get_z()

        #p['movable_beamstop_x'] = self.movable_beamstop.get_x()
        #p['movable_beamstop_z'] = self.movable_beamstop.get_z()
        #p['movable_beamstop_s'] = self.movable_beamstop.get_s()
        p['detector_table_x'] = self.detector_table.get_x()
        p['detector_table_z'] = self.detector_table.get_z()
        p['detector_table_s'] = self.detector_table.get_s()

        # beam intensity and position monitors
        p['xbpm1_intensity'] = self.xbpm1.get_intensity()
        p['xbpm1_x'] = self.xbpm1.get_x()
        p['xbpm1_z'] = self.xbpm1.get_z()

        p['xbpm3_intensity'] = self.xbpm3.get_intensity()
        p['xbpm3_x'] = self.xbpm3.get_x()
        p['xbpm3_z'] = self.xbpm3.get_z()

        p['cvd1_intensity'] = self.cvd1.get_intensity()
        p['cvd1_x'] = self.cvd1.get_x()
        p['cvd1_z'] = self.cvd1.get_z()

        p['xbpm5_intensity'] = self.xbpm5.get_intensity()
        p['xbpm5_x'] = self.xbpm5.get_x()
        p['xbpm5_z'] = self.xbpm5.get_z()

        p['beam_position_x'] = self.beam_position.get_x()
        p['beam_position_z'] = self.beam_position.get_z()

        # thermomethers
        p['temperatures'] = self.thermometers.get_temperatures()
        # vacuum
        p['vacuum'] = self.vacuum.get_pressures()

        return p

class machine:
    def __init__(self):
        self.machine = dp('ans/ca/machinestatus')
        self.hu640 = dp('ans-c05/ei/l-hu640')

    def get_current(self):
        return self.machine.current

    def get_current_trend(self):
        return (self.machine.currenttrendtimes, self.machine.currenttrend)

    def get_filling_mode(self):
        return self.machine.fillingmode

    def get_hu640_status(self):
        return self.hu640.status()
    
    def get_hu640_currents(self):
        timesteps = [time.time()] + [self.hu640.read_attribute('steptimeps%d' % k).value for k in range(1, 4)]
        currents = []
        for k in range(3):
            currents.append([time.time()] + [self.hu640.read_attribute('currentps%d' % k).value for k in range(1, 4)])
            time.sleep(timesteps[-1])
        return currents, timesteps
        

class undulator:
    def __init__(self):
        self.undulator = dp('ans-c11/ei/m-u24')
        self.undulator_energy = dp('ans-c11/ei/m-u24_energy')

    def get_energy(self):
        return self.undulator_energy.energy

    def get_gap(self):
        return self.undulator.gap

    def get_computed_gap(self):
        return self.undulator_energy.computedgap
    
    def get_harmonic(self):
        return self.undulator_energy.harmonic
    
    def get_mode(self):
        return self.undulator_energy.mode
    
    def get_current_mode(self):
        return self.undulator_energy.currentmode
    


class monochromator:
    def __init__(self):
        self.monochromator = dp('i11-ma-c03/op/mono1')
        self.monochromator_rx = dp('i11-ma-c03/op/mono1-mt_rx')
        self.monochromator_rx_fine = dp('i11-ma-c03/op/mono1-mt_rx_fine')
        
    def get_energy(self):
        return self.monochromator.read_attribute('energy').value

    def get_wavelength(self):
        return self.monochromator.read_attribute('lambda').value

    def get_position(self):
        return self.monochromator.read_attribute('thetabragg').value

    def get_crystal_temperature(self):
        return self.monochromator.read_attribute('crystaltemperature').value

    def get_beam_height(self):
        return self.monochromator.beamheight

    def get_rx(self):
        return self.monochromator_rx.position

    def get_rx_fine(self):
        return self.monochromator_rx_fine.position
    
    def turn_off(self):
        self.monochromator_rx.Off()
        self.monochromator_rx_fine.Off()
        
    def turn_on(self):
        self.monochromator_rx.On()
        self.monochromator_rx_fine.On()

class beamlineenergy:
    def __init__(self):
        self.beamlineenergy = dp('i11-ma-c00/ex/beamlineenergy')
        self.coupling = dp('i11-ma-c00/ex/ble-coupling')
        
    def get_energy(self):
        return self.beamlineenergy.energy

    def get_coupling(self):
        return self.beamlineenergy.currentcouplingname
    
    def set_coupling(self, coupling):
        return

class xbpm:
    def __init__(self, device_id):
        self.device = dp(device_id)

    def get_intensity(self):
        return self.device.intensity

    def get_x(self):
        return self.device.horizontalposition

    def get_z(self):
        return self.device.verticalposition

class xbpm1(xbpm):
    def __init__(self):
        xbpm.__init__(self, 'i11-ma-c04/dt/xbpm_diode.1')
        
class xbpm3(xbpm):
    def __init__(self):
        xbpm.__init__(self, 'i11-ma-c05/dt/xbpm_diode.3')
        
class xbpm5(xbpm):
    def __init__(self):
        xbpm.__init__(self, 'i11-ma-c06/dt/xbpm_diode.5')
        
class cvd1(xbpm):
    def __init__(self):
        xbpm.__init__(self, 'i11-ma-c05/dt/xbpm-cvd.1')

class beam_position:
    def __init__(self):
        self.md2 = dp('i11-ma-cx1/ex/md2')

    def get_x(self):
        return self.md2.beampositionhorizontal

    def get_z(self):
        return self.md2.beampositionvertical

class slits:
    def get_horizontal_gap(self):
        return self.h.gap

    def get_vertical_gap(self):
        return self.v.gap

    def get_horizontal_position(self):
        return self.h.position

    def get_vertical_position(self):
        return self.v.position
    
    def get_i_position(self):
        return self.i.position
    
    def set_i_position(self, position):
        self.i.position = position
        
    def get_i_offset(self):
        return self.i.offset
    
    def set_i_offset(self, offset):
        self.i.offset = offset
        
    def get_o_position(self):
        return self.o.position
    
    def set_o_position(self, position):
        self.o.position = position
        
    def get_o_offset(self):
        return self.o.offset
    
    def set_o_offset(self, offset):
        self.o.offset = offset
        
    def get_d_position(self):
        return self.d.position
    
    def set_d_position(self, position):
        self.d.position = position
        
    def get_d_offset(self):
        return self.d.offset
    
    def set_d_offset(self, offset):
        self.d.offset = offset
        
    def get_u_offset(self):
        return self.i.offset
    
    def get_u_position(self):
        return self.u.position
    
    def set_u_position(self, position):
        self.u.position = position
        
    def set_u_offset(self, offset):
        self.u.offset = offset
    
    def get_transmission(self):
        return
    
    def set_transmission(self, transmission):
        pass

class other_slits(slits):
    def get_horizontal_gap(self):
        return self.h_ec.position
    
    def get_vertical_gap(self):
        return self.v_ec.position
        
    def get_horizontal_position(self):
        return self.h_tx.position
        
    def get_vertical_position(self):
        return self.v_tz.position
        
class primary_slits(slits):
    def __init__(self):
        self.h = dp('i11-ma-c02/ex/fent_h.1')
        self.v = dp('i11-ma-c02/ex/fent_v.1')
        self.i = dp('i11-ma-c02/ex/fent_h.1-mt_i')
        self.o = dp('i11-ma-c02/ex/fent_h.1-mt_o')
        self.d = dp('i11-ma-c02/ex/fent_v.1-mt_d')
        self.u = dp('i11-ma-c02/ex/fent_v.1-mt_u')
        
class secondary_slits(slits):
    def __init__(self):
        self.h = dp('i11-ma-c04/ex/fent_h.2')
        self.v = dp('i11-ma-c04/ex/fent_v.2')
        self.i = dp('i11-ma-c04/ex/fent_h.2-mt_i')
        self.o = dp('i11-ma-c04/ex/fent_h.2-mt_o')
        self.d = dp('i11-ma-c04/ex/fent_v.2-mt_d')
        self.u = dp('i11-ma-c04/ex/fent_v.2-mt_u')

class slits3(other_slits):
    def __init__(self):
        self.h = dp('i11-ma-c05/ex/fent_h.3')
        self.v = dp('i11-ma-c05/ex/fent_v.3')
        self.h_ec = dp('i11-ma-c05/ex/fent_h.3-mt_ec')
        self.h_tx = dp('i11-ma-c05/ex/fent_h.3-mt_tx')
        self.v_ec = dp('i11-ma-c05/ex/fent_v.3-mt_ec')
        self.v_tz = dp('i11-ma-c05/ex/fent_v.3-mt_tz')

class slits5(other_slits):
    def __init__(self):
        self.h = dp('i11-ma-c06/ex/fent_h.5')
        self.v = dp('i11-ma-c06/ex/fent_v.5')
        self.h_ec = dp('i11-ma-c06/ex/fent_h.5-mt_ec')
        self.h_tx = dp('i11-ma-c06/ex/fent_h.5-mt_tx')
        self.v_ec = dp('i11-ma-c06/ex/fent_v.5-mt_ec')
        self.v_tz = dp('i11-ma-c06/ex/fent_v.5-mt_tz')
        
class experimental_slits(other_slits):
    def __init__(self):
        self.h = dp('i11-ma-c06/ex/fent_h.6')
        self.v = dp('i11-ma-c06/ex/fent_v.6')
        self.h_ec = dp('i11-ma-c06/ex/fent_h.6-mt_ec')
        self.h_tx = dp('i11-ma-c06/ex/fent_h.6-mt_tx')
        self.v_ec = dp('i11-ma-c06/ex/fent_v.6-mt_ec')
        self.v_tz = dp('i11-ma-c06/ex/fent_v.6-mt_tz')

class thermometers:
    def __init__(self):
        self.tc1 = dp('i11-ma-cx1/ex/tc.1')
        #self.tc2 = dp('i11-ma-cx1/ex/tc.2')
        self.tc3 = dp('i11-ma-c06/ex/tc.1')
        self.tc4 = dp('i11-ma-c05/ex/tc.1')
        self.tc5 = dp('i11-ma-c04/ex/tc.1')
        self.tc6 = dp('i11-ma-c04/ex/tc.1')
        self.tc7 = dp('i11-ma-c03/ex/tc.2')
        self.tc8 = dp('i11-ma-c02/ex/tc.1')
        #self.tc9 = dp('i11-ma-c02/ex/tc.2')
        self.tc10 = dp('i11-ma-c02/ex/tc.3')
        self.tc11 = dp('i11-ma-c00/ex/tc.1')

    def get_temperatures(self):
        return [(getattr(self, 'tc%d' % k).dev_name(), getattr(self, 'tc%d' % k).temperature) for k in range(1, 12)]

class vacuum:
    def __init__(self):
        self.v1 = dp('i11-ma/vi/c01-pi.1-dmz')
        self.v2 = dp('i11-ma-c01/vi/jpen.1')
        self.v3 = dp('i11-ma-c01/vi/jpir.1')
        self.v4 = dp('i11-ma-c01/vi/pi.1')
        self.v5 = dp('i11-ma-c01/vi/pi.2')
        self.v6 = dp('i11-ma-c02/vi/jpen.1')
        self.v7 = dp('i11-ma-c02/vi/jpir.1')
        self.v8 = dp('i11-ma-c02/vi/pi.1')
        self.v9 = dp('i11-ma-c03/vi/jpen.1')
        self.v10 = dp('i11-ma-c03/vi/jpen.2')
        self.v11 = dp('i11-ma-c03/vi/jpir.1')
        self.v12 = dp('i11-ma-c03/vi/jpir.2')
        self.v13 = dp('i11-ma-c03/vi/pi.1')
        self.v14 = dp('i11-ma-c04/vi/jaull.1')
        self.v15 = dp('i11-ma-c04/vi/jpen.1')
        self.v16 = dp('i11-ma-c04/vi/jpir.1')
        self.v17 = dp('i11-ma-c04/vi/pi.1')
        self.v18 = dp('i11-ma-c04/vi/pi.2')
        self.v19 = dp('i11-ma-c05/vi/jaull.1')
        self.v20 = dp('i11-ma-c05/vi/jaull.2')
        self.v21 = dp('i11-ma-c05/vi/pi.1')
        self.v22 = dp('i11-ma-c05/vi/pi.2')
        self.v23 = dp('i11-ma-c05/vi/pi.3')
        self.v24 = dp('i11-ma-c06/vi/jaull.1')
        
    def get_pressures(self):
        return [(getattr(self, 'v%d' % k).dev_name(), getattr(self, 'v%d' % k).pressure) for k in range(1, 25)]
    
class hpm_table:
    def __init__(self):
        self.tx = dp('i11-ma-c05/ex/tab.1-mt_tx')
        self.tz = dp('i11-ma-c05/ex/tab.1-mt_tz')
        
    def set_x(self, position):
        self.tx.position = position
        
    def get_x(self):
        return self.tx.position
    
    def set_z(self, position):
        self.tz.position = position
    
    def get_z(self):
        return self.tz.position

class experimental_table:
    def __init__(self):
        self.table = dp('i11-ma-c05/ex/tab.2')
    
    def get_pitch(self):
        return self.table.pitch
    
    def get_roll(self):
        return self.table.roll
    
    def get_yaw(self):
        return self.table.yaw
    
    def get_x(self):
        return self.table.xC
    
    def get_z(self):
        return self.table.zC
    
    def set_position(self, position):
        for key in position:
            setattr(self.table, key, position)
            
    def get_position(self):
        return {'pitch': self.get_pitch(), 'roll': self.get_roll(), 'yaw': self.get_yaw(), 'x': self.get_x(), 'z': self.get_z()}
    
class detector_table:
    def __init__(self):
        self.s = dp('i11-ma-cx1/dt/dtc_ccd.1-mt_ts')
        self.x = dp('i11-ma-cx1/dt/dtc_ccd.1-mt_tx')
        self.z = dp('i11-ma-cx1/dt/dtc_ccd.1-mt_tz')
    
    def set_s(self, position):
        self.s.position = position
    def get_s(self):
        return self.s.position
        
    def set_x(self, position):
        self.x.position = position
    def get_x(self):
        return self.x.position
        
    def set_z(self, position):
        self.z.position = position
    def get_z(self):
        return self.z.position
    
    def set_position(self, position):
        for key in position:
            getattr(self, key).position = position[key]
    def get_position(self):
        return {'s': self.get_s(), 'x': self.get_x(), 'z': self.get_z()}
    
class hpm:
    def __init__(self):
        self.rs = dp('i11-ma-c04/op/mir.1-mt_rs')
        self.rz = dp('i11-ma-c04/op/mir.1-mt_rz')
        self.tx = dp('i11-ma-c04/op/mir.1-mt_tx')
    
    def get_rs(self):
        return self.rs.position
    
    def get_rz(self):
        return self.rz.position
    
    def get_x(self):
        return self.tx.position
    
    def get_position(self):
        return {'rs': self.get_rs(), 'rz': self.get_rz(), 'x': self.get_x()}
    
class vfm:
    def __init__(self):
        self.rx = dp('i11-ma-c05/op/mir.2-mt_rx')
        self.tz = dp('i11-ma-c05/op/mir.2-mt_tz')
        self.voltages = [dp('i11-ma-c05/op/mir2-ch.%02d' % k) for k in range(12)]
    
    def get_rx(self):
        return self.rx.position
   
    def get_z(self):
        return self.tz.position
    
    def get_voltages(self):
        return [(channel.name(), channel.voltage) for channel in self.voltages]
    
    def get_position(self):
        return dict([(channel.name(), channel.voltage) for channel in self.voltages])
    
class hfm:
    def __init__(self):
        self.rz = dp('i11-ma-c05/op/mir.3-mt_rz')
        self.tx = dp('i11-ma-c05/op/mir.3-mt_tx')
        self.voltages = [dp('i11-ma-c05/op/mir3-ch.%02d' % k) for k in range(12)]
    
    def get_rz(self):
        return self.rz.position
   
    def get_x(self):
        return self.tx.position
    
    def get_voltages(self):
        return [(channel.name(), channel.voltage) for channel in self.voltages]
    
    def get_position(self):
        return dict([(channel.name(), channel.voltage) for channel in self.voltages])
    
    
class filters:
    def __init__(self):
        self.filters = dp('i11-ma-c05/ex/att.1')
    
    def get_filter(self):
        return self.filters.selectedattributename
    
    def set_filter(self, filter_name):
        return setattr(self.filters, filter_name, True)
    
class apertures:
    def __init__(self):
        self.md2 = dp('i11-ma-cx1/ex/md2')
    
    def get_diameter(self):
        return self.md2.aperturediameters[self.md2.currentaperturediameterindex]
    
    def set_position(self, position_name):
        self.md2.apertureposition = position_name
    def get_position(self):
        return self.md2.apertureposition
    
    def set_x(self, position):
        self.md2.aperturehorizontalposition = position
    def get_x(self):
        return self.md2.aperturehorizontalposition
    
    def set_z(self, position):
        self.md2.apertureverticalposition = position
    def get_z(self):
        return self.md2.apertureverticalposition
    
    def set_position_mm(self, x, z):
        return self.set_x(x), self.set_z(z)
    def get_position_mm(self):
        return self.get_x(), self.get_z()
    
class beamstop:
    def __init__(self):
        self.md2 = dp('i11-ma-cx1/ex/md2')
           
    def get_position(self):
        return self.md2.capillaryposition
    def set_position(self, position_name):
        self.md2.capillaryposition = position_name
    
    def set_x(self, position):
        self.md2.capillaryhorizontalposition = position
    def get_x(self):
        return self.md2.capillaryhorizontalposition

    def set_z(self, position):
        self.md2.capillaryverticalposition = position
    def get_z(self):
        return self.md2.capillaryverticalposition
    
    def set_position_mm(self, x, z):
        return self.set_x(x), self.set_z(z)
    def get_position_mm(self):
        return self.get_x(), self.get_z()

class safety_shutter:
    def __init__(self):
        self.device = dp('i11-ma-c04/ex/obx.1')
    
    def open(self):
        return self.device.Open()
        
    def close(self):
        return self.device.Close()
    
    def state(self):
        return self.device.State()
        
class frontend:
    def __init__(self):
        self.device = dp('tdl-i11-ma/vi/tdl.1')
    
    def open(self):
        return self.device.Open()
        
    def close(self):
        return self.device.Close()

    def state(self):
        return self.device.State()
        
class fast_shutter:
    def __init__(self):
        self.device = dp('i11-ma-cx1/ex/md2')
        self.motor_x = dp('i11-ma-c06/ex/shutter-mt_tx')
        self.motor_z = dp('i11-ma-c06/ex/shutter-mt_tz')
        
    def open(self):
        self.device.fastshutterisopen = True
    
    def close(self):
        self.device.fastshutterisopen = False
        
    def get_x(self):
        return self.motor_x.position
        
    def set_x(self):
        self.motor_x.position = position
    
    def get_z(self):
        return self.motor_z.position
        
    def set_z(self):
        self.motor_z.position = position
        
    def state(self):
        return self.device.fastshutterisopen
        