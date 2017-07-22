class md2_mockup:
    positions = ['OmegaPosition', 'AlignmentXPosition', 'AlignmentYPosition', 'AlignmentZPosition', 'CentringXPosition', 'CentringYPosition', 'CoaxialCameraZoomValue', 'ZoomPosition', 'BackLightLevel', 'FrontLightLevel', 'BackLightFactor', 'FrontLightFactor']
    motors = ['Omega', 'AlignmentX', 'AlignmentY', 'AlignmentZ', 'CentringX', 'CentringY', 'ApertureHorizontal', 'ApertureVertical', 'CapillaryHorizontal', 'CapillaryVertical', 'ScintillatorHorizontal', 'ScintillatorVertical', 'Zoom']
    booleans = ['DetectorGatePulseEnabled', 'CryoIsBack', 'FluoIsBack', 'SampleIsOn', 'BackLightIsOn', 'FrontLightIsOn', 'FastShutterIsEnabled', 'FastShutterIsOpen']
    attributes = [('ScanRange', 180), ('ScanExposureTime', 1), ('ScanStartAngle', 0), ('ScanSpeed', 1), ('ScanNumberOfFrames', 1), ('MotorPositions', ('Omega=0','AlignmentX=1','AlignmentY=1','AlignmentZ=1','CentringX=1','CentringY=1'))]
    def __init__(self):
        for attribute in self.positions:
            setattr(self, attribute, 0.)
            setattr(self, attribute.lower(), 0.)
        for motor in self.motors:
            setattr(self, motor, None)
            setattr(self, motor.lower(), None)
        for b in self.booleans:
            setattr(self, b, False)
            setattr(self, b.lower(), False)
        for attribute in self.attributes:
            setattr(self, attribute[0], attribute[1])
            setattr(self, attribute[0].lower(), attribute[1])
    def getMotorState(self, motor_name):
        return
    def startscan(self):
        return
    def startscanex(self, parameters):
        return
    def startsetphase(self, phase_name):
        return
    def gettaskinfo(self, task_id):
        return
    def istaskrunning(self, task_id):
        return False
    def setStartScan4DEx(self, parameters):
        return
    def get_motor_state(self, motor_name):
        return
    def setstartscan4d(self, start):
        return
    def setstopscan4d(self, start):
        return
    def savecentringpositions(self):
        return

