#!/usr/bin/env python
'''
Instrument object. Gives access to all of the beamline and machine parameters relevant to the experiment.
'''

from PyTango import DeviceProxy as dp
import time
import sqlite3

class instrument(object):

	def __init__(self):
		#source
		self.machine = machine() 
		self.undulator = undulator()
		self.monochoromator = monochromator()
		self.beamlineenergy = beamlineenergy()
		# slits
		self.primary_slits = primary_slits()
		self.secondary_slits = secondary_slits()
		self.tertiary_slits = tertiary_slits()
		self.experimental_slits = experimetal_slits()
		# filters
		self.filters = filters()
		# mirrors
		self.hpm = hpm()
		self.vfm = vfm()
		self.hfm = hfm()
		# tables
		self.hpm_table = hpm_table()
		self.experimental_table = experimental_table()
		self.detector_table = detector_table()
		# apertures
		self.apertures = apertures()
		self.beamstop = beamstop()
		self.movable_beamstop = movable_beamstop()
		# beam position and intensity monitors
		self.xbpm1 = xbpm1()
		self.xbpm3 = xbpm3()
		self.xbpm5 = xbpm5()
		self.cvd1 = cvd1()
		self.beam_position = beam_postion()
		# thermomethers
		self.thermometers = thermometers()
		# presure gauges
		self.vacuum = vacuum()
		
	def get_state(self):
		p = {}
		# source
		p['time'] = time.time()
		p['time_stamp'] = time.asctime()
		p['machine_current'] = self.machine.get_current()
		p['undulator_gap'] = self.undulator.get_gap()
		p['undulator_calculated_gap'] = self.undulator.get_calculated_gap()
		p['undulator_energy'] = self.undulator.get_energy()
		p['monochromator_energy'] = self.monochromator.get_energy()
		p['monochromator_wavelength'] = self.monochromator.get_wavelength()
		p['monochromator_position'] = self.monochromator.get_position()
		p['monochromator_rx'] = self.monochromator.get_rx()
		p['monochromator_rx_fine'] = self.monochromator.get_rx_fine()
		p['beamlineenergy_energy'] = self.beamlineenergy.get_energy()
		p['beamlineenergy_mode'] = self.beamlineenergy.get_mode()
		p['hu640_state'] = self.machine.get_hu640_state()
		# slits
		p['primary_slits_horizontal_gap'] = self.primary_slits.get_horizontal_gap()
		p['primary_slits_vertical_gap'] = self.primary_slits.get_vertical_gap()
		p['primary_slits_horizotal_position'] = self.primary_slits.get_horizontal_position()
		p['primary_slits_vertical_position'] = self.primary_slits.get_vertical_position()

		p['secondary_slits_horizontal_gap'] = self.secondary_slits.get_horizontal_gap()
		p['secondary_slits_vertical_gap'] = self.secondary_slits.get_vertical_gap()
		p['secondary_slits_horizotal_position'] = self.secondary_slits.get_horizontal_position()
		p['secondary_slits_vertical_position'] = self.secondary_slits.get_vertical_position()

		p['tertiary_slits_horizontal_gap'] = self.tertiary_slits.get_horizontal_gap()
		p['tertiary_slits_vertical_gap'] = self.tertiary_slits.get_vertical_gap()
		p['tertiary_slits_horizotal_position'] = self.tertiary_slits.get_horizontal_position()
		p['tertiary_slits_vertical_position'] = self.tertiary_slits.get_vertical_position()

		p['experimental_slits_horizontal_gap'] = self.experimental_slits.get_horizontal_gap()
		p['experimental_slits_vertical_gap'] = self.experimental_slits.get_vertical_gap()
		p['experimental_slits_horizotal_position'] = self.experimental_slits.get_horizontal_position()
		p['experimental_slits_vertical_position'] = self.experimental_slits.get_vertical_position()

		p['filters'] = self.filters.get_filters()

		# mirror tables
		p['hpm_table_X'] = self.hpm_table.get_x()
		p['hpm_table_Z'] = self.hpm_table.get_z()

		p['experimental_table_pitch'] = self.experimental_table.get_pitch()
		p['experimental_table_roll'] = self.experimental_table.get_roll()
		p['experimental_table_yaw'] = self.experimental_table.get_yaw()
		p['experimental_table_z'] = self.experimental_table.get_z()
		p['experimental_table_x'] = self.experimental_table.get_x()

		# mirrors
		p['hpm_tx'] = self.hpm.get_tx()
		p['hpm_rz'] = self.hpm.get_rz()
		p['hpm_rs'] = self.hpm.get_rs()

		p['vfm_tz'] = self.vfm.get_tz()
		p['vfm_rx'] = self.vfm.get_rx()

		p['hfm_tx'] = self.hfm.get_tx()
		p['hfm_rz'] = self.hfm.get_rz()

		p['vfm_tensions'] = self.hfm.get_tensions()
		p['hfm_tensions'] = self.hfm.get_tensions()

		p['aperture_diameter'] = self.apertures.get_aperture_diameter()
		p['aperture_x'] = self.apertures.get_aperture_x()
		p['aperture_z'] = self.apertures.get_aperture_z()

		p['beamstop_x'] = self.beamstop.get_x()
		p['beamstop_z'] = self.beamstop.get_z()

		p['movable_beamstop_x'] = self.movable_beamstop.get_x()
		p['movable_beamstop_z'] = self.movable_beamstop.get_z()
		p['movable_beamstop_s'] = self.movable_beamstop.get_s()
		p['detector_table_x'] = self.detector_table.get_x()
		p['detector_table_z'] = self.detector_table.get_z()
		p['detector_table_s'] = self.detecotr_table.get_s()

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
		self.machine = dp('ans')
		self.hu640 = dp('hu640')
	
	def get_current(self):
		return self.machine.current
	
	def get_mode(self):
		return self.machine.mode
	
	def get_message(self):
		return self.machine.message

	def get_hu640_state(self):
		return self.hu640.state

class undulator:
	
	def __init__(self):
		self.undulator_energy = dp('undulator_energy')
		self.undulator_state = dp('undulator_state')
	
	def get_energy(self):
		return self.undulator_energy.energy
	
	def get_gap(self):
		return self.undulator_energy.gap
	
	def get_calculated(sefl):
		return self.undulator_energy.calculated_gap

class monochromator:
	
	def __init__(self):
		self.monochromator = dp('i11-ma-c03/op/mono1')
	
	def get_energy(self):
		return self.monochromator.read_attribute('energy').value
	
	def get_wavelength(self):
		return self.monochromator.read_attribute('lambda').value
	
	def get_position(self):
		return self.monochromator.read_attribute('position').value
	
	def get_rx(self):
		return self.monochromator.read_attribute('rx_fine').value
	
	def get_rx_fine(self):
		return self.monochromator.read_attribute('rx_fine').value

class beamlineenergy:
	
	def __init__(self):
		self.beamlineenergy('i11-ma-c00/ex/beamlineenergy')
	
	def get_energy(self):
		return self.beamlineenergy.energy
	
	def get_mode(self):
		return self.beamlineenergy.mode

class xbpm:
	
	def __init__(self, device_id):
		self.device = dp(device_id)
	
	def get_intensity(self):
		return self.device.intensity
	
	def get_x(self):
		return self.device.position_x
	
	def get_z(self):
		return self.device.position_z

class beam_position:

	def __init__(self):
		self.md2 = dp('i11-ma-cx1/ex/md2')
	
	def get_x(self):
		return self.md2.beampostionhorizontal
	
	def get_z(self):
		return self.md2.beampositionvertical

class primary_slits:
	
	def __init__(self):
		self.horizontal_gap = dp('i11-ma-c02/ex/fent.1_h')
		self.vertical_gap = dp('i11-ma-c02/ex/fent.1_v')
		self.position = dp('i11-ma-c02/ex/fent.1')
	
	def get_horizontal_gap(self):
		return self.horizontal_gap.position
	
	def get_vertical_gap(self):
		return self.vertical_gap.position
	
	def get_position(self):
		return self.position.position

class secondary_slits:
	
	def __init__(self):
		self.horizontal_gap = dp('i11-ma-c03/ex/fent.2_h')
		self.vertical_gap = dp('i11-ma-c03/ex/fent.2_v')
		self.position = dp('i11-ma-c03/ex/fent.2')
	
	def get_horizontal_gap(self):
		return self.horizontal_gap.position
	
	def get_vertical_gap(self):
		return self.vertical_gap.position
	
	def get_position(self):
		return self.position.position

class tertiary_slits:
	
	def __init__(self):
		self.horizontal_gap = dp('i11-ma-c04/ex/fent.3_h')
		self.vertical_gap = dp('i11-ma-c04/ex/fent.3_v')
		self.position = dp('i11-ma-c04/ex/fent.3')
	
	def get_horizontal_gap(self):
		return self.horizontal_gap.position
	
	def get_vertical_gap(self):
		return self.vertical_gap.position
	
	def get_position(self):
		return self.position.position

class experimental_slits:
	
	def __init__(self):
		self.horizontal_gap = dp('i11-ma-c06/ex/fent.6_h')
		self.vertical_gap = dp('i11-ma-c06/ex/fent.6_v')
		self.position = dp('i11-ma-c06/ex/fent.6')
	
	def get_horizontal_gap(self):
		return self.horizontal_gap.position
	
	def get_vertical_gap(self):
		return self.vertical_gap.position
	
	def get_position(self):
		return self.position.position

class thermometers:
	
	def __init__(self):
		self.tc1 = dp(tc1)
		self.tc2 = dp(tc2)
		self.tc3 = dp(tc3)
		self.tc4 = dp(tc4)
		self.tc5 = dp(tc5)
		self.tc6 = dp(tc6)
	
	def get_temperatures(self)
	 	tcs = ['tc1', 'tc2', 'tc3', 'tc4', 'tc5', 'tc6']
		return [getattr(self, t) for t in tcs]

class vacuum:
	
	def __init__(self):
		self.v1 = dp(v1)
		self.v2 = dp(v2)
		self.v3 = dp(v3)
		self.v4 = dp(v4)
		self.v5 = dp(v5)
	
	def get_pressures(self):
		vs = ['v1', 'v2', 'v3', 'v4', 'v5']
		return [gettattr(self, v) for v in vs]
