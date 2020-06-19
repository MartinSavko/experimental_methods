#!/usr/bin/env python

import time
import PyTango
import pickle
import numpy
import scipy.misc

def acquire():
    last_image = None
    images = []
    
    md2.backlightison = True
    md2.startscan()
    while md2.fastshutterisopen==False:
        pass
    
    while md2.fastshutterisopen==True:
        if imag.imagecounter != last_image:
            last_image = imag.imagecounter
            images.append([imag.imagecounter, md2.OmegaPosition, imag.image])
    return images

def save(images, name, zoom):
    f = open('images_%s_zoom%s.pck' % ( name, zoom), 'w')
    pickle.dump(images, f)
    f.close()
    
    #for k, img in enumerate(images):
        #scipy.misc.imsave('image_FL_H10um_%s.png' % str(k).zfill(5), img[-1])
    
if __name__ == '__main__':
    import optparse

    parser = optparse.OptionParser()

    parser.add_option('-r', '--range', default=360, type=float, help='Duration of acquisition')
    parser.add_option('-n', '--name', default='sample', type=str, help='filename')
    parser.add_option('-z', '--zoom', default=None, help='zoom')
    options, args = parser.parse_args()
    
    imag = PyTango.DeviceProxy('i11-ma-cx1/ex/imag.1')
    md2 = PyTango.DeviceProxy('i11-ma-cx1/ex/md2')
    
    md2.scanrange = options.range
    md2.scanexposuretime = options.range/36.
    md2.backlightison = True
    
    if options.zoom != None:
        zoom = int(options.zoom)
        md2.coaxialcamerazoomvalue = zoom
        while md2.getMotorState('Zoom').name == 'MOVING':
            time.sleep(0.1)
        
    images = acquire()    
    
    save(images, options.name, md2.coaxialcamerazoomvalue)

