#!/usr/bin/env python
# -*- coding: utf-8 -*-

from slits import slits1, slits2
import pickle
from scipy.interpolate import interp1d, RectBivariateSpline
import numpy as np
import math
import logging
import time
import redis
import random
import sys
import os
import gevent
import subprocess

log = logging.getLogger('HWR')
#log.setLevel(logging.INFO)
#stream_handler = logging.StreamHandler(sys.stdout)
#stream_formatter = logging.Formatter('%(asctime)s || %(message)s')
#stream_handler.setFormatter(stream_formatter)
#log.addHandler(stream_handler)


def integrate(distribution, start, end, use_skimage=False):
    transmission = np.abs(distribution[start[0]:end[0]+1,start[1]:end[1]+1].sum())
    return transmission

class transmission_mockup:
    def __init__(self):
        self.transmission = 1
    def get_transmission(self):
        return self.transmission
    def set_transmission(self, transmission):
        self.transmission = transmission
    

class transmission:
    
    def __init__(self,
                 slits2_reference_distribution='/usr/local/slits_reference/distribution_s2_observe.npy',
                 slits2_reference_distribution_key = 'slits2_reference_distribution',
                 #/nfs/data3/2020_Run5/Commissioning/2020-12-01/slits/distribution_s2_observe.npy',
                 slits2_reference_ii = '/usr/local/slits_reference/ii_s2_observe.npy',
                 slits2_reference_ii_key = 'slits2_reference_ii',
                 #'/nfs/data3/2020_Run5/Commissioning/2020-12-01/slits/ii_s2_observe.npy',
                 slits2_reference_transmissions = '/usr/local/slits_reference/s2_transmissions.npy',
                 slits2_reference_transmissions_key = 's2_transmissions',
                 master_is_alive_question_key = 'transmission_master_is_alive_question',
                 master_is_alive_answer_key = 'transmission_master_is_alive_answer',
                 current_transmission_question_key = 'current_transmission_question',
                 current_transmission_answer_key = 'current_transmission_answer',
                 set_transmission_key = 'set_transmission',
                 reference_gap=4.,
                 reference_position=0.,
                 percent_factor=100.,
                 redis_host='172.19.10.125',
                 master=False):
        
        self.master = master
        self.redis_host = redis_host
        self.redis = redis.StrictRedis(self.redis_host)
        self.pubsub = self.redis.pubsub()
        self.reference_gap = reference_gap
        self.reference_position = reference_position
        self.s2 = slits2()
        self.percent_factor = percent_factor
        self.slits2_reference_distribution = slits2_reference_distribution
        self.slits2_reference_distribution_key = slits2_reference_distribution_key
        self.slits2_reference_ii = slits2_reference_ii
        self.slits2_reference_ii_key = slits2_reference_ii_key
        self.slits2_reference_transmissions = slits2_reference_transmissions
        self.slits2_reference_transmissions_key = slits2_reference_transmissions_key
        self.distribution = None
        self.ii = None
        self.transmissions = None
        self.predict_gap_from_transmission = None
        self.master_is_alive_question_key = master_is_alive_question_key
        self.master_is_alive_answer_key = master_is_alive_answer_key
        self.current_transmission_question_key = current_transmission_question_key
        self.current_transmission_answer_key = current_transmission_answer_key
        self.set_transmission_key = set_transmission_key
        self.id = str(random.random())
        self.run()
        
    def run(self):
        if not self.is_master_alive() or self.master == True:
            message = 'master is not alive, starting one'
            log.debug(message)
            #gevent.spawn(self.become_master)
            self.become_master()
        else:
            message = 'master is alive, relying on it...'
        log.debug(message)
        self.unbecome_master()
            
    def unbecome_master(self):
        self.master = False
        self.pubsub.unsubscribe(self.master_is_alive_question_key)
        log.debug('there is instance already alive, will rely on it for now')
        
    def become_master(self):
        self.master = True
        self.pubsub.subscribe(self.master_is_alive_question_key)
        self.pubsub.subscribe(self.current_transmission_question_key)
        self.pubsub.subscribe(self.set_transmission_key)
        for item in self.pubsub.listen():
            if self.master == True:
                log.debug(item)
                log.debug(item)
                if item['type'] != 'message':
                    continue
                if item['channel'].decode() == self.master_is_alive_question_key:
                    log.debug('answering question from id %s ' % item["data"])
                    self.redis.publish(self.master_is_alive_answer_key, item["data"])
                if item['channel'].decode() == self.set_transmission_key:
                    transmission = float(item["data"])
                    self.set_transmission(transmission)
                if item['channel'].decode() == self.current_transmission_question_key:
                    tr = self.get_transmission()
                    log.debug('freshly determined transmission %.4f' % tr)
                    self.redis.publish(self.current_transmission_answer_key, str(tr))
                log.debug(self.id, 'became the master')
            else:
                break
            
    def get_fresh_answer(self, question, answer, threshold=10):
        print('get_fresh_answer')
        message = 'Asking %s' % question
        log.debug(message)
        pubsub = self.redis.pubsub()
        pubsub.subscribe(answer)
        self.redis.publish(question, self.id)
        to_report = None
        k = 1
        for item in pubsub.listen():
            message = 'item %s' % str(item)
            log.debug(message)
            k += 1
            print("item['channel'].decode('utf-8')", item['channel'].decode('utf-8'))
            if item['channel'].decode('utf-8') == answer and item['type'] == 'message':
                to_report = item['data']
                print('to_report', to_report)
                break
            if k > threshold:
                break
        pubsub.unsubscribe(answer)
        return to_report
    
    def is_master_alive(self, threshold=1000):
        is_alive = False
        self.pubsub.subscribe(self.master_is_alive_answer_key)
        question_id = self.id
        log.debug('my question_id is %s' % question_id)
        self.redis.publish(self.master_is_alive_question_key, question_id)
        return True
        k = 0
        while True and k<threshold:
            k += 1
            message = self.pubsub.get_message(self.master_is_alive_answer_key)
            log.debug('is_master_alive message %s' % str(message))
            log.debug('%d is_master_alive message %s' % (k, str(message)))

            if message == None or message['channel'].decode('utf-8') != self.master_is_alive_answer_key or message['type'] != 'message':
                continue
            if 'transmission is running' in subprocess.getoutput('transmission status'):
                is_alive = True
                break
            if str(message['data']) == question_id:
                is_alive = True
                break
            gevent.sleep(0.1)
            
        self.pubsub.unsubscribe(self.master_is_alive_answer_key)
        return is_alive
    
    
    def load_reference(self):
        self.distribution = np.load(self.slits2_reference_distribution)
        self.ii = np.load(self.slits2_reference_ii)
        self.transmissions = np.load(self.slits2_reference_transmissions)
        self.gaps = np.linspace(0, 4, 4000)
        self.transmissions = [self.get_hypothetical_transmission(gap, self.distribution) for gap in self.gaps]
        self.predict_gap_from_transmission = interp1d(self.transmissions, self.gaps, fill_value=tuple([0, 4]), bounds_error=False, kind='slinear')
        
    def get_transmission(self):
        message = 'transmission instance %s received request for current transmission' % self.id
        
        log.debug(message)
        
        if not self.master:
            current_transmission = float(self.get_fresh_answer(self.current_transmission_question_key, self.current_transmission_answer_key))
            return current_transmission
        
        if self.distribution is None or self.ii is None:
            self.load_reference()
            
        start, end = self.get_indices_for_slit_setting()
        transmission = integrate(self.distribution, start, end)
        transmission *= self.percent_factor
        message = 'transmission master id %s received request to get transmission %.2f' % (self.id, transmission)
        log.debug(message)
        
        return transmission
    
    def get_hypothetical_transmission(self, gap, distribution):
        if self.master and (self.ii is None or self.distribution is None):
            self.load_reference()
        start, end = self.get_indices_for_slit_setting(horizontal_gap=gap, vertical_gap=gap, vertical_center=0., horizontal_center=0.)
        
        hypothetical_transmission = integrate(distribution, start, end) * self.percent_factor
        return hypothetical_transmission
    
    def set_transmission(self, transmission, factor=1, epsilon=1e-3):
        start = time.time()
        if transmission > 100:
            logging.info('transmission specified is larger then 100 percent %s setting to 100' % transmission)
            transmission = 100
        if not self.master:
            self.redis.publish(self.set_transmission_key, str(transmission))
            return
        message = 'transmission master received request to set transmission %.2f' % transmission
        log.debug(message)
        
        if self.predict_gap_from_transmission is None:
            self.load_reference()
        k=0
        gap = self.predict_gap_from_transmission(transmission)
        self.s2.set_horizontal_gap(gap, wait=True)
        self.s2.set_vertical_gap(gap, wait=True)
        log.debug('set_transmission took %.4f' % (time.time()-start))
    
    def get_indices_for_slit_setting(self, horizontal_gap=None, vertical_gap=None, horizontal_center=None, vertical_center=None, npixels=4000, extent=(-2, 2)):
        e = extent[1] - extent[0]
        pixels_per_mm = npixels/e
        if horizontal_gap is None:
            horizontal_gap = self.s2.get_horizontal_gap()
        if vertical_gap is None:
            vertical_gap = self.s2.get_vertical_gap()
        if horizontal_center is None:
            horizontal_center = self.s2.get_horizontal_position()
        if vertical_center is None:
            vertical_center = self.s2.get_vertical_position()
            
        horizontal_start = (-horizontal_gap/2. - horizontal_center) - extent[0]
        horizontal_end = (horizontal_gap/2. - horizontal_center) - extent[0]
        vertical_start = (-vertical_gap/2. - vertical_center) - extent[0]
        vertical_end = (vertical_gap/2. - vertical_center) - extent[0]

        horizontal_start *= pixels_per_mm
        horizontal_end *= pixels_per_mm
        vertical_start *= pixels_per_mm
        vertical_end *= pixels_per_mm
        
        return (int(vertical_start), int(horizontal_start)), (int(min(npixels-1, vertical_end)), int(min(npixels-1, horizontal_end)))
    
def test():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--master', type=int, default=0, help='master')
    args = parser.parse_args()
    t = transmission(master=bool(args.master))
    print(args)
    print('current transmission', t.get_transmission())
    
    gevent.sleep(1)
    print('current transmission', t.get_transmission())
    
if __name__ == '__main__':
    test()
    
