#!/usr/bin/env python

import logging
import pickle
import time
import threading
import traceback

import sys
sys.path.insert(0, './')

import zmq
import MDP
from mdworker import MajorDomoWorker
from mdclient2 import MajorDomoClient

def defer(func):
    def consider(*args, **kwargs):
        arg0 = args[0]
        args = args[1:]
        #print({func.__name__: {"args": args, "kwargs": kwargs}})
        #print('arg0', arg0)
        if getattr(arg0, "server"):
            try:
                considered = func(arg0, *args, **kwargs)
            except:
                traceback.print_exc()
                considered = -1
        else:
            params = {}
            if args:
                params["args"] = args
            if kwargs:
                params["kwargs"] = kwargs
            if not params:
                params = None
            #print('params', params)
            considered = getattr(arg0, "talk")({func.__name__: params})
        return considered
    
    return consider

       
class speech:
    
    server = None
    service = None
    listen_thread = None
    giver = None
    talker = None
    singer = None
    verbose = False    
    singer_port = None
    hear_hear = None
    value = None
    value_id = None
    last_sung_id = None
    sung = None
    timestamp = None
    
    def __init__(self, port=5555, service=None, verbose=True, server=None, ctx=None):
        
        logging.basicConfig(format="%(asctime)s|%(module)s %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S",
                            level=logging.INFO)
        
        self.broker_address = "tcp://localhost:%s" % port
        self.service = service
        self.hear_hear = False
        if self.service is None:
            self.service = self.__class__.__name__
        
        self.service_name = ('%s' % self.service).encode()
        
        self.ctx = zmq.Context() # ms 2024-09-18 is that what made zmq issue not appear during the UDC tests in July?
        
        #if ctx is None:
            #self.ctx = zmq.Context()
        #else:
            #self.ctx = ctx
            
        self.talker = MajorDomoClient(self.broker_address, ctx=self.ctx)
        self.value_id = 0
        self.last_sung_id = 0
        self.sung = 0
        self.timestamp = time.time()
        
        if server is None:
            if not self.service_already_registered():
                self.set_up_listen_thread()
                self.start_listen()
                self.server = True
                self.singer = self.ctx.socket(zmq.PUB)
                self.singer_port = self.singer.bind_to_random_port("tcp://*")
                self.singer.setsockopt(zmq.SNDHWM, 1)
                
                #https://stackoverflow.com/questions/58663965/pyzmq-req-socket-hang-on-context-term
                #self.singer.setsockopt(zmq.IMMEDIATE, 1)
                logging.info(f"singer_port {self.singer_port}")
                logging.info(f'serving {self.service_name}')
            else:
                self.server = False
                logging.debug('not serving')
        else:
            self.server = server
        
    def service_already_registered(self):
        self.talker.send(b"mmi.service", self.service_name)
        reply = self.talker.recv()
        return reply[0] == b'200'
    
    def make_sense_of_request(self, request):
        logging.info('make_sense_of_request')
        logging.info('reqest received %s ' % request)
        _start = time.time()

        request = pickle.loads(request[0])
        logging.info('request decoded %s' % request)
        value = None
        for key in request:
            
            try:
                method = getattr(self, '%s' % key)
                arguments = request[key]
                args = ()
                kwargs = {}
                if type(arguments) is dict:
                    if 'args' in request[key]:
                        args = arguments['args']
                    if 'kwargs' in request[key]:
                        kwargs = arguments['kwargs']
                elif arguments is not None:
                    args = (arguments,)
                value = method(*args, **kwargs)
            except:
                logging.exception('%s' % traceback.format_exc())
                                
        logging.info('requests processed in %.7f seconds' % (time.time() - _start))
        return pickle.dumps(value)
    
    
    def sing(self):
        if self.value_id != self.last_sung_id and self.value_id > 0:
            self.singer.send_multipart(
                [
                    self.service_name,
                    b"%f" % self.value,
                    b"%f" % self.timestamp,
                    b"%d" % self.value_id,
                ]
            )
            self.last_sung_id = self.value_id
            self.sung += 1
            
    def set_up_listen_singing_thread(self):
        self.listen_singing_event = threading.Event()
        self.listen_singing_thread = threading.Thread(target=self.listen_singing)
        self.listen_singing_thread.daemon = True

    def listen_singing_start(self):
        self.set_up_listen_singing_thread()
        self.listen_singing_thread.start()
        self.hear_hear = True
        
    def listen_singing_stop(self):
        self.listen_singing_event.set()
        self.hear_hear = False
        
    def listen_singing(self):
        self.song_listener = self.ctx.socket(zmq.SUB)
        self.song_listener.connect("tcp://localhost:%d" % self.get_singer_port())
        self.song_listener.setsockopt(zmq.SUBSCRIBE, b"%s" % self.service_name)
    
        self.id_message = -1
        while not self.listen_singing_event.is_set():
            message = self.song_listener.recv_multipart()
            self.id_message += 1
            self.value = float(message[1])
            timestamp = message[2]
            value_id = message[3]
            print(f"received message {self.id_message}: value: {self.value}, timestamp: {timestamp}, value_id: {value_id}")
            
    def set_up_listen_thread(self):
        self.listen_thread = threading.Thread(target=self.listen)
        self.listen_thread.daemon = True

    def start_listen(self):
        if self.giver is None:
            self.giver = MajorDomoWorker(self.broker_address, self.service_name, verbose=self.verbose, ctx=self.ctx)
        self.set_up_listen_thread()
        self.listen_event = threading.Event()
        self.listen_thread.start()
        self.server = True
        
    def stop_listen(self):
        logging.info('set listen_event')
        self.listen_event.set()
        self.server = False

    def listen(self):
        reply = None
        if self.listen_event is None:
            self.listen_event = threading.Event()
        while not self.listen_event.is_set():
            logging.info('listening')
            request = self.giver.recv(reply)
            reply = self.make_sense_of_request(request)
            if not isinstance(reply, list):
                reply = [reply]
        reply = [pickle.dumps('stopping to listen')]
        reply = [self.giver.reply_to, b''] + reply
        self.giver.send_to_broker(MDP.W_REPLY, msg=reply)
        logging.info('stop listen')
 
    def talk(self, request):
        encoded_request = pickle.dumps(request)
        self.talker.send(self.service_name, encoded_request)
        reply = self.talker.recv()
        logging.debug('reply %s' % reply)
        decoded_reply = None
        if reply is not None:
            decoded_reply = pickle.loads(reply[0])
        return decoded_reply
    
    @defer
    def get_singer_port(self):
        return self.singer_port
    
    def destroy(self):
        self.ctx.destroy()
        
    #def make_sense_of_request(self, request):
        #logging.info('make_sense_of_request')
        #logging.info('reqest received %s ' % request)
        #_start = time.time()

        #request = pickle.loads(request[0])
        #logging.info('request decoded %s' % request)
        
        #for key in request:
            
            ##if 'set_transmission' in key:
                ##logging.info('%s set to %s' % (key, request[key]))
                ##value = getattr(self, '%s' % key)(*request[key]['args'], **request[key]['kwargs'])
            #if 'set_' in key:
                #logging.info('%s set to %s' % (key, request[key]))
                #value = getattr(self, '%s' % key)(request[key])
            #elif 'ask' == key:
                #logging.info('%s with parameters %s' % (key, request[key]))
                #value = getattr(self, '%s' % key)(request[key])
            #elif request[key] is not None:
                #arg = request[key]
                #value = getattr(self, '%s' % key)(arg)
                #logging.info('%s(%s) returned %s' % (key, arg, value))
            #else:
                #value = getattr(self, '%s' % key)()
                #logging.info('%s returned %s' % (key, value))
        #logging.info('requests processed in %.7f seconds' % (time.time() - _start))
        #return pickle.dumps(value)
