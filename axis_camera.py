#!/usr/bin/env python

import httplib
import urllib3
import re
import time
import os
import numpy as np
import glob
from scipy.misc import imshow, imread, imsave

class axis_camera:
    def __init__(self, host='cam1', timeout=3600):
        self.host = host
        self.timeout = timeout
        self.connection = httplib.HTTPConnection(host, timeout=self.timeout)
        self.command_base = '/axis-cgi/com/ptz.cgi?camera=1&'
        # http://cam14/mjpg/quad/video.mjpg
        self.url = 'http://%s/mjpg/video.mjpg' % self.host
        self.imagewidth = 765
        self.imageheight = 576
        
    def create_command_string(self, parameter_dictionary):
        pass
        
    def execute_command(self, command_string):
        self.connection.request('GET', command_string)
        response = self.connection.getresponse()
        return response.read()
        
    def get_pan(self):
        position = self.get_position()
        return position['pan']
        
    def set_pan(self, pan):
        command_string = self.command_base + 'pan=%f' % pan
        return self.execute_command(command_string)
        
    def get_tilt(self):
        position = self.get_position()
        return position['tilt']
        
    def set_tilt(self, tilt):
        command_string = self.command_base + 'tilt=%f' % tilt
        return self.execute_command(command_string)
        
    def get_zoom(self):
        position = self.get_position()
        return position['zoom']
        
    def set_zoom(self, zoom):
        command_string = self.command_base + 'zoom=%f' % zoom
        return self.execute_command(command_string)
    
    def get_position(self):
        self.connection.request('GET', '/axis-cgi/com/ptz.cgi?camera=1&query=position')
        response = self.connection.getresponse()
        position_string = response.read()
        pan = float(re.findall('pan=([\d\-\.]*).*', position_string)[0])
        tilt = float(re.findall('tilt=([\d\-\.]*).*', position_string)[0])
        zoom = float(re.findall('zoom=([\d\-\.]*).*', position_string)[0])
        autofocus = re.findall('autoiris=(.*)\r\n.*', position_string)[0]
        autoiris = re.findall('autoiris=(.*)\r\n.*', position_string)[0]
        position = {'pan': pan, 'tilt': tilt, 'zoom': zoom, 'autofocus': autofocus, 'autoiris': autoiris}
        return position
        
    def set_position(self, position):
        pan, tilt, zoom = position['pan'], position['tilt'], position['zoom']
        self.set_pan(pan)
        time.sleep(2)
        self.set_tilt(tilt)
        time.sleep(2)
        self.set_zoom(zoom)
        
    def get_header(self):
        self.header = {}
        while(True):
            line = self.fp.readline()
            if line == "\r\n":
                break
            line = line.strip()
            parts = line.split(": ", 1)
            try:
                self.header[parts[0]] = parts[1]
            except:
                import traceback
                print(traceback.print_exc())
                print('Problem encountered with image header.  Setting content_length to zero')
                self.header['Content-Length'] = 0 # set content_length to zero if there is a problem reading header
        self.content_length = int(self.header['Content-Length'])
        
    def get_image(self):
        self.fp = urllib3.urlopen(self.url, timeout=self.timeout)
        self.get_header()
        if self.content_length > 0:
            self.img = self.fp.read(self.content_length)
            self.fp.readline()
        return self.img
    
    def save_image(self, image_name=None):
        position = self.get_position()
        if image_name is None:
            pan, tilt, zoom = position['pan'], position['tilt'], position['zoom']
            image_name = '/927bis/ccd/Snapshots/%s_%s_%s_%s.jpg' % (self.host, pan, tilt, zoom, time.time())
        os.system('wget http://%s/jpg/image.jpg -O %s' % (self.host, image_name))
        
    def combine(self, template='lid3_empty_*jpg'):
        i = np.zeros((576, 768, 3), dtype=np.uint32)
        images = glob.glob(template)
        imgs = [imread(img) for img in images]
        for img in imgs:
            i += img
        imshow(i)

if __name__ == "__main__":
    #dewar = {'pan': -6.15, 'tilt': 23.775, 'zoom': 4300.0}
    #lid1 =  {'pan': -12.0, 'tilt': 19.0, 'zoom': 7501.0}
    #lid2 = {'pan': -5.0, 'tilt': 29.0, 'zoom': 7501.0}
    #lid3 = {'pan': 0.0, 'tilt': 18.5, 'zoom': 7501.0}
    #puck1 = {'pan': -13.35, 'tilt': 17.7, 'zoom': 9600.0}
    #puck2 = {'pan': -9.97, 'tilt': 17.625, 'zoom': 9600.0}
    #puck3 = {'pan': -11.7, 'tilt': 20.25, 'zoom': 9600.0}
    #puck4 = {'pan': -5.0, 'tilt': 29.625, 'zoom': 9600.0}
    #puck5 = {'pan': -6.6, 'tilt': 27.6, 'zoom': 9600.0}
    #puck6 = {'pan': -3.225, 'tilt': 27.07, 'zoom': 9600.0}
    #puck7 = {'pan': 2.17, 'tilt': 17.1, 'zoom': 9600.0}
    #puck8 = {'pan': 0.9, 'tilt': 19.725, 'zoom': 9600.0}
    #puck9 = {'pan': -1.125, 'tilt': 17.25, 'zoom': 9600.0}
    #2016-11-06
    #dewar = {'pan': -6.674, 'tilt': 24.3, 'zoom': 4100.0}
    #lid1 = {'pan': -12.973, 'tilt': 20.25, 'zoom': 7501.0}
    #lid2 = {'pan': -5.024, 'tilt': 29.02, 'zoom': 7501.0}
    #lid3 = {'pan': -0.675, 'tilt': 18.675, 'zoom': 7501.0}
    #puck1 = {'pan': -14.84, 'tilt': 18.6, 'zoom': 9600.0}
    #puck2 = {'pan': -11.6, 'tilt': 18.2, 'zoom': 9600.0}
    #puck3 = {'pan': -12.75, 'tilt': 21.1, 'zoom': 9600.0}
    #puck4 = {'pan': -5.324, 'tilt': 30.675, 'zoom': 9600.0}
    #puck5 = {'pan': -7.5, 'tilt': 28.1, 'zoom': 9600.0}
    #puck6 = {'pan': -3.97, 'tilt': 27.52, 'zoom': 9600.0}
    #puck7 = {'pan': 0.6, 'tilt': 16.95, 'zoom': 9600.0}
    #puck8 = {'pan': -0.52, 'tilt': 20.0, 'zoom': 9600.0}
    #puck9 = {'pan': -2.625, 'tilt': 17.55, 'zoom': 9600.0}
    #2016-12-21
    #dewar = {'pan': 4, 'tilt':-83.7, 'zoom': 5900.0}
    #2017-05-27
    dewar = {'pan': -17.4724, 'tilt': -87.525000000000006, 'zoom': 5900.0}
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
    
    cam = axis_camera(host='cam8')
    print('current position is')
    print(cam.get_position())
    #
    #cam6 sample environment 
    cam6_samp_env = {'pan': -63.890599999999999, 'tilt': 3.2000000000000002, 'zoom': 5447.0}
    #cam8 detector
    cam8_detector =  {'pan': -28.195799999999998, 'tilt': -55.424999999999997, 'zoom': 900.0}

