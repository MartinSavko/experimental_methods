import PyTango

class camera(object):
    def __init__(self):
        self.md2 = PyTango.DeviceProxy('i11-ma-cx1/ex/md2')
        self.prosilica = PyTango.DeviceProxy('i11-ma-cx1/ex/imag.1')
        
    def get_image(self):
        return self.prosilica.image
        
    def get_rgbimage(self):
        return self.prosilica.rgbimage.reshape((493, 659, 3))
        
    def get_zoom(self):
        return self.md2.coaxialcamerazoomvalue
    
    def set_zoom(self, value):
        if value is not None:
            value = int(value)
            self.md2.coaxialcamerazoomvalue = value
    
    def get_calibration(self):
        return numpy.array(self.md2.coaxcamscaley, self.md2.coaxcamscalex)
        
    def get_vertical_calibration(self):
        return self.md2.coaxcamscaley
        
    def get_horizontal_calibration(self):
        return self.md2.coaxcamscalex

    def set_exposure(self, exposure):
        self.prosilica.exposure = exposure
        
    def get_exposure(self):
        return self.prosilica.exposure