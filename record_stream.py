#!/usr/bin/env python

import os
import time
import multiprocessing
import sys

from camera import camera as prosilica
from scipy.misc import imsave
from axis_camera import axis_camera

dewar = {'pan': 4, 'tilt':-83.7, 'zoom': 5900.0}

lid1 = {'pan': -9.4, 'tilt': -86.775, 'zoom': 8700.0}
lid2 = {'pan': -22.65, 'tilt': -79.125, 'zoom': 8700.0}
lid3 = {'pan': 22.9, 'tilt': -79.35, 'zoom': 8700.0}

puck1 = {'pan': -16.8725, 'tilt': -87.75, 'zoom': 12700.0}
puck2 = {'pan': 11., 'tilt': -85.8, 'zoom': 12700.0}
puck3 = {'pan': -23.9, 'tilt': -85.8, 'zoom': 12700.0}
puck4 = {'pan': -24.9, 'tilt': -78.2, 'zoom': 12700.0}
puck5 = {'pan': -24.6, 'tilt': -80.325, 'zoom': 12700.0}
puck6 = {'pan': -15.2, 'tilt': -79.0, 'zoom': 12700.0}
puck7 = {'pan': 27.1, 'tilt': -78.35, 'zoom': 12700.0}
puck8 = {'pan': 16.05, 'tilt': -79.125, 'zoom': 12700.0}
puck9 = {'pan': 25.35, 'tilt': -80.4, 'zoom': 12700.0}

positions = {'dewar': dewar,
             'lid1': lid1,
             'lid2': lid2,
             'lid3': lid3,
             'puck1': puck1,
             'puck2': puck2,
             'puck3': puck3,
             'puck4': puck4,
             'puck5': puck5,
             'puck6': puck6,
             'puck7': puck7,
             'puck8': puck8,
             'puck9': puck9}

def record(camera, output, duration):
    print('camera', camera)
    start = time.time()
    k = 0
    image_id = None
    if not os.path.isdir(output):
        try:
            os.makedirs(output)
        except OSError:
            pass
    if camera == 'prosilica':
        cam = prosilica()
    if duration is not None:
        while time.time() < start + duration:
            k += 1
            if camera != 'prosilica':
                os.system('wget http://%s/jpg/image.jpg -O %s/%s_%s.jpg > /dev/null' % (camera, output, camera, str(k).zfill(5)))
            else:
                new_image_id = cam.get_image_id()
                if image_id != new_image_id:
                    image_id = new_image_id
                    image = cam.get_rgbimage()
                    imsave('%s/%s_%s.jpg' % (output, camera, str(k).zfill(5)), image)
    else:
        while True:
            k += 1
            if camera != 'prosilica':
                os.system('wget http://%s/jpg/image.jpg -O %s/%s_%s.jpg > /dev/null' % (camera, output, camera, str(k).zfill(5)))
            else:
                new_image_id = cam.get_image_id()
                if image_id != new_image_id:
                    image_id = new_image_id
                    image = cam.get_rgbimage()
                    imsave('%s/%s_%s.jpg' % (output, camera, str(k).zfill(5)), image)
                    
def record_all(output, duration):
    recorders = []
    for cam in ['cam6', 'cam8', 'cam1']:
        p = multiprocessing.Process(target=record, args=(cam, output, duration))
        recorders.append(p)
        p.start()
    for r in recorders:
        r.join()
  
def record_prosilica(output, duration):
    print('camera', prosilica)
    start = time.time()
    k = 0
    if not os.path.isdir(output):
        try:
            os.makedirs(output)
        except OSError:
            pass
    if duration is not None:
        while time.time() < start + duration:
            k += 1
            os.system('wget http://%s/jpg/image.jpg -O %s/%s_%s.jpg' % (camera, output, camera, str(k).zfill(5)))
    else:
        while True:
            k += 1
            os.system('wget http://%s/jpg/image.jpg -O %s/%s_%s.jpg' % (camera, output, camera, str(k).zfill(5)))
def main():
    import optparse
    parser = optparse.OptionParser()
    parser.add_option('-c', '--camera', type=str, default='172.19.10.149', help='IP or hostname of camera to record stream from')
    parser.add_option('-d', '--duration', type=float, default=None, help='Duration of recording')
    parser.add_option('-o', '--output', type=str, default=None, help='Output')
    parser.add_option('-a', '--all', action='store_true', default=False, help='Record from all cameras')
    parser.add_option('-w', '--what', type=str, default='', help='Camera position and zoom setting optimized for either whole dewar, particular lid (lid1 .. lid3) or puck (puck1 .. puck9) default=%default')
    options, args = parser.parse_args()
    print('options', options)
    print('args', args)
    if options.camera == 'cam8' and options.what in ['dewar', 'lid1', 'lid2', 'lid3', 'puck1', 'puck2', 'puck3', 'puck4', 'puck5','puck6', 'puck7', 'puck8', 'puck9']:
        ac = axis_camera(host=options.camera)
        ac.set_position(positions[options.what])
        print('waiting 5 sec for movement to complete')
        time.sleep(5)
        
    if options.all is True:
        record_all(options.output, options.duration)
    else:    
        record(options.camera, options.output, options.duration)
    
if __name__ == '__main__':
    main()