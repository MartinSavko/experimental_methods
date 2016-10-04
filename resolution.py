import PyTango

class resolution(object):
    def __init__(self, x_pixels_in_detector=3110, y_pixels_in_detector=3269, x_pixel_size=75e-6, y_pixel_size=75e-6):
        self.distance_motor = PyTango.DeviceProxy('i11-ma-cx1/dt/dtc_ccd.1-mt_ts')
        self.wavelength_motor = PyTango.DeviceProxy('i11-ma-c03/op/mono1')
        self.x_pixel_size = x_pixel_size
        self.y_pixel_size = y_pixel_size
        self.x_pixels_in_detector = x_pixels_in_detector
        self.y_pixels_in_detector = y_pixels_in_detector
        self.bc = beam_center()
        
    def get_detector_radii(self):
        beam_center_x, beam_center_y = self.bc.get_beam_center()
        detector_size_x = self.x_pixel_size * self.x_pixels_in_detector
        detector_size_y = self.y_pixel_size * self.y_pixels_in_detector
        
        beam_center_distance_x = self.x_pixel_size * beam_center_x
        beam_center_distance_y = self.y_pixel_size * beam_center_y
        
        distances_x = numpy.array([detector_size_x - beam_center_distance_x, beam_center_distance_x])
        distances_y = numpy.array([detector_size_y - beam_center_distance_y, beam_center_distance_y])
        
        edge_distances = numpy.hstack([distances_x, distances_y])
        corner_distances = numpy.array([(x**2 + y**2)**0.5 for x in distances_x for y in distances_y])
        
        distances = numpy.hstack([edge_distances, corner_distances]) * 1000.
        return distances
        
    def get_detector_min_radius(self):
        distances = self.get_detector_radii()
        return distances.min()
        
    def get_detector_max_radius(self):
        distances = self.get_detector_radii()
        return distances.max()
        
    def get_distance(self):
        return self.distance_motor.position
        
    def get_wavelength(self):
        return self.wavelength_motor.read_attribute('lambda').value
        
    def get_resolution(self, distance=None, wavelength=None, radius=None):
        if distance is None:
            distance = self.get_distance()
        if radius is None:
            detector_radius = self.get_detector_min_radius()
        if wavelength is None:
            wavelength = self.get_wavelength()
        
        two_theta = numpy.math.atan(detector_radius/distance)
        resolution = 0.5 * wavelength / numpy.sin(0.5*two_theta)
        
        return resolution
        
    def get_resolution_from_distance(self, distance, wavelength=None):
        return self.get_resolution(distance=distance, wavelength=wavelength)
        
    def get_distance_from_resolution(self, resolution, wavelength=None):
        if wavelength is None:
            wavelength = self.get_wavelength()
        two_theta = 2*numpy.math.asin(0.5*wavelength/resolution)
        detector_radius = self.get_detector_min_radius()
        distance = detector_radius/numpy.math.tan(two_theta)
        return distance