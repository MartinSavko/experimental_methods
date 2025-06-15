#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import pylab
from camera import camera
from goniometer import goniometer
import numpy as np

def main():
    
    cam = camera()
    g = goniometer()
    
    x = np.linspace(-0.2, 0.6, 50)
    #x = x[::-1]
    contrasts = []
    
    for p in x:
        g.set_position({'AlignmentX': p}, wait=True)
        time.sleep(0.01)
        img = cam.get_image()
        contrasts.append(cam.get_contrast(image=img[300:-300, 400:-400]))
        
    pylab.plot(x, contrasts)
    pylab.show()
    
    
    
if __name__ == '__main__':
    main()
