"""
  This code is provided AS IS for example purpose and testing MD Device Server
  ARINAX Sep. 2021
"""
# import tango
# import ipdb
#import epics
import os
import time
import sys
from ClientFactory import ClientFactory
import Logger


#########################
# Tests selection
#########################

USE_EVENTS = True
TEST_PREDEFINED_POSITIONS = False
TEST_SMALL_MOVES = False
TEST_SCANS = False
TEST_IO = False
TEST_STATES = False


######################################
# Device names (see devices in MD app)
######################################

SCINTILLATOR_DEVICE     = "Scintillator"
BEAMSTOP_DEVICE         = "Beamstop"
APERTURE_DEVICE         = "Aperture"
CAPILLARY_DEVICE        = "Capillary"
ALIGNMENT_TABLE_DEVICE  = "Alignment Table"
CENTRING_TABLE_DEVICE   = "Centring Table"


###############
# Phase names #
###############

PHASE_CENTRING          = "Centring"
PHASE_BEAM_LOCATION     = "BeamLocation"
PHASE_DATA_COLLECTION   = "DataCollection"
PHASE_TRANSFER          = "Transfer"
PHASE_UNKNOWN           = "Unknown"


########################
# Predefined positions #
########################

POSITION_BEAM           = "BEAM"
POSITION_OFF            = "OFF"
POSITION_PARK           = "PARK"
POSITION_TRANSFER       = "TRANSFER"
POSITION_STORED         = "STORED"
POSITION_REF_1          = "REF_1"
POSITION_REF_2          = "REF_2"
POSITION_SCINTILLATOR   = "SCINTILLATOR"
POSITION_PHOTODIODE     = "PHOTODIODE"

POSITION_DEFAULT            = "DEFAULT"
POSITION_CENTRED            = "CENTRED"
POSITION_ALIGNED            = "ALIGNED"
POSITION_CLEAR_SCINTILLATOR = "CLEAR_SCINTILLATOR"
POSITION_CLEARED            = "CLEARED"
POSITION_UNKNOWN            = "UNKNOWN"


#########################
# Connection to server
#########################

def get_server(address="172.19.10.181", port=9001):
    server = ClientFactory.instantiate(type="exporter", args={'address': address , 'port': port})
    return server

server = get_server()

# server = ClientFactory.instantiate(type="epics", args={'prefix': 'MD3'})
# server = ClientFactory.instantiate(type="tango", args={'address':"localhost:18001/embl/md/1#dbase=no"})
# _event_id = server.subscribe_event("MotorStates", tango.EventType.CHANGE_EVENT, eventCallback)

if server is None:
    Logger.log("Failed to connect to server", success=Logger.FAILED)
    exit(-1)


#########################
# Tests
#########################

Logger.log(server['State'], Logger.OK)


if TEST_STATES:
    Logger.log("==================== TESTING ATTRIBUTES SECTION =========================", Logger.DEBUG)
    presentRWAttributes = server.readAllAttributesAndGetWritables()


if TEST_SMALL_MOVES:

    Logger.log("==================== TESTING SMALL MOVES SECTION =========================", Logger.DEBUG)
    server.waitReady(10000)
    #
    # Simple motors
    pos = float(server.getMotorPosition("Omega"))   # auto-generated member function of the GenericClient. Calls "getMotorPosition" from DeviceServer
    Logger.log('Omega Limits = %s' % server.getMotorLimits("Omega"))
    Logger.log('Omega Dynamics Limits = %s' % server.getMotorDynamicLimits("Omega"))
    Logger.log('Omega Max speed = %s' % server.getMotorMaxSpeed("Omega"))
    Logger.log('Omega Position = %s' % pos)
    server.syncMoveMotor("Omega", pos + 0.1)
    pos = float(server.getMotorPosition("Kappa"))
    server.syncMoveMotor("Kappa", pos + 0.1)
    pos = float(server.getMotorPosition("Phi"))
    server.syncMoveMotor("Phi", pos + 0.1)
    pos = float(server.getMotorPosition("Centring X"))  # syntax for reading DeviceServer's method
    pos2 = server["CentringXPosition"]                  # syntax for reading DeviceServer's attribute
    server.syncMoveMotor("Centring X", pos + 0.1)
    pos = float(server.getMotorPosition("Centring Y"))
    server.syncMoveMotor("Centring Y", pos + 0.1)
    pos = float(server.getMotorPosition("Alignment X"))
    server.syncMoveMotor("Alignment X", pos + 0.1)
    pos = float(server.getMotorPosition("Alignment Y"))
    server.syncMoveMotor("Alignment Y", pos + 0.1)
    # pos = float(server.getMotorPosition("Alignment Z"))
    pos = server["AlignmentZPosition"]
    server.syncMoveMotor("Alignment Z", pos + 0.1)
    pos = float(server.getMotorPosition("Aperture Vertical"))
    server.syncMoveMotor("Aperture Vertical", pos + 0.1)
    pos = float(server.getMotorPosition("Aperture Horizontal"))
    server.syncMoveMotor("Aperture Horizontal", pos + 0.1)
    pos = float(server.getMotorPosition("Capillary Vertical"))
    server.syncMoveMotor("Capillary Vertical", pos + 0.1)
    pos = float(server.getMotorPosition("Capillary Horizontal"))
    server.syncMoveMotor("Capillary Horizontal", pos + 0.1)
    pos = float(server.getMotorPosition("Scintillator Vertical"))
    server.syncMoveMotor("Scintillator Vertical", pos + 0.1)
    pos = float(server.getMotorPosition("Beamstop X"))
    server.syncMoveMotor("Beamstop X", pos + 0.1)
    pos = float(server.getMotorPosition("Beamstop Y"))
    server.syncMoveMotor("Beamstop Y", pos + 0.1)
    pos = float(server.getMotorPosition("Beamstop Z"))
    server.syncMoveMotor("Beamstop Z", pos + 0.1)
    #
    # Meta-motors (composites)
    pos = float(server.getMotorPosition("CentringTableFocus"))
    server.syncMoveMotor("CentringTableFocus", pos + 0.1)
    pos = float(server.getMotorPosition("CentringTableVertical"))
    server.syncMoveMotor("CentringTableVertical", pos + 0.1)
    pos = float(server.getMotorPosition("BeamstopDistance"))
    server.syncMoveMotor("BeamstopDistance", pos + 0.1)


if TEST_PREDEFINED_POSITIONS:

    Logger.log("==================== TESTING PREDEFINED POSITIONS SECTION =========================", Logger.DEBUG)
    server.waitReady(10000)
    #
    # First move devices to safe positions
    server.setPredefinedPosition(BEAMSTOP_DEVICE, POSITION_PARK, USE_EVENTS)
    server.setPredefinedPosition(ALIGNMENT_TABLE_DEVICE, POSITION_CLEARED, USE_EVENTS)
    server.setPredefinedPosition(CAPILLARY_DEVICE, POSITION_PARK, USE_EVENTS)
    server.setPredefinedPosition(APERTURE_DEVICE, POSITION_PARK, USE_EVENTS, timeout=60)        # a specific timeout is used here because aperture moves in/out very slowly
    server.setPredefinedPosition(SCINTILLATOR_DEVICE, POSITION_PARK, USE_EVENTS)

    server.setPredefinedPosition(ALIGNMENT_TABLE_DEVICE, POSITION_DEFAULT, USE_EVENTS)
    server.setPredefinedPosition(CENTRING_TABLE_DEVICE, POSITION_DEFAULT, USE_EVENTS)
    server.setPredefinedPosition(ALIGNMENT_TABLE_DEVICE, POSITION_ALIGNED, USE_EVENTS)
    server.setPredefinedPosition(CENTRING_TABLE_DEVICE, POSITION_CENTRED, USE_EVENTS)
    # Move alignment table device back to a safe position (clear scintillator to move scintillator out safely)
    server.setPredefinedPosition(ALIGNMENT_TABLE_DEVICE, POSITION_CLEAR_SCINTILLATOR, USE_EVENTS)
    # Alignment table and centring table have already been tested, now test other devices
    # Return to a safe place
    server.setPredefinedPosition(BEAMSTOP_DEVICE, POSITION_PARK, USE_EVENTS)
    server.setPredefinedPosition(CAPILLARY_DEVICE, POSITION_PARK, USE_EVENTS)
    server.setPredefinedPosition(APERTURE_DEVICE, POSITION_PARK, USE_EVENTS, timeout=60)
    server.setPredefinedPosition(SCINTILLATOR_DEVICE, POSITION_PARK, USE_EVENTS)
    server.setPredefinedPosition(ALIGNMENT_TABLE_DEVICE, POSITION_DEFAULT,  USE_EVENTS)


if TEST_SCANS:
    # Scan parameters
    npasses = 1             # number of passes inside a scan
    expo = 0.5               # exposure in seconds
    rasterexpo = 0.2
    scanrange = 4           # omega range in deg per scan
    ndetframes = 0          # 10         # number of detector triggers per scan or per line
    rasterrange = 1         # height and width in mm of the raster scan
    nlines = 5
    nscans = 1
    timeout = 200*expo

    # Message to user
    Logger.log("==================== TESTING SCANS SECTION =========================", Logger.DEBUG)
    server.waitReady(10000)

    server.startSetPhase('Transfer')
    server.waitReady(timeout=60000)
    server.startSetPhase('Centring')
    server.waitReady(timeout=60000)
    server.saveCentringPositions()  # build a ALIGNED + CENTRED positions used later in Phase positioning
    server.startSetPhase('BeamLocation')
    server.waitReady(timeout=60000)
    server.startSetPhase('DataCollection')
    server.waitReady(timeout=60000)

    startangle = 0
    scan_idx = 1
    # Inputs : int frame_idx, "frame_number", "start_angle", "scan_range", "exposure_time", "number_of_passes"
    server.startScanEx2(scan_idx, ndetframes, startangle, scanrange, expo, npasses)
    server.waitReady(30000)

    # Inputs : "start_angle", "scan_range", "exposure_time", "start_y", "start_z", "start_cx", "start_cy",
    # "stop_y", "stop_z", "stop_cx", "stop_cy"
    server.startScan4DEx(startangle, scanrange, expo, 0, 0, -0.4, -0.4, 0.1, 0.1, 0.1, 0.1)
    server.waitReady(30000)

    # Inputs : "omega_range", "line_range", "total_uturn_range", "start_omega", "start_y", "start_z",
    # "start_cx", "start_cy", "number_of_lines", "frames_per_lines", "exposure_time",
    # "invert_direction", "use_centring_table", "shutterless"
    server.startRasterScanEx(scanrange, rasterrange, rasterrange, startangle, 0, 0, 0, 0, nlines, ndetframes,
                             rasterexpo+1, True, True, True)
    server.waitReady(30000)


if TEST_IO:
    Logger.log("==================== TESTING IOs SECTION =========================", Logger.DEBUG)
    server.waitReady(10000)

    server["FastShutterIsOpen"] = True
    server["FastShutterIsOpen"] = False
    server["FluoDetectorIsBack"] = False
    time.sleep(2)           # allow fluo table to move front
    server["FluoDetectorIsBack"] = True
    # server["CryoIsBack"] = True
    # server["CryoIsBack"] = False
    server["BeamstopPosition"] = POSITION_PARK
    server.waitReady(30000)
    server["BackLightIsOn"] = True
    server["BackLightIsOn"] = False

Logger.log("This is the END !", Logger.OK)
