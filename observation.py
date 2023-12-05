#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gevent

import traceback
import time
import os
import pylab


class observation(object):
    
    def __init__(self,
                 monitors,
                 duration=None,
                 observe=True,
                 plotsleeptime=0.1):
                     
        self.timestamp = time.time()
        self.monitors = monitors
        self.plotsleeptime = plotsleeptime
        self.observe = observe
        self.description = 'Observation'
        self.duration = duration
        print('self.duration', self.duration)
        self.colors = ['green', 'yellow']
        
    def execute(self):
        try:
            self.prepare()
            self.run()
        except:
            print('problem during the run')
            print(traceback.print_exc())
        finally:
            self.clean()
    
    def prepare(self):
        for monitor in self.monitors:
            monitor.observe = True
        
        self.create_plot()
            
    def run(self):
        self._start = time.time()
        
        observers = [gevent.spawn(monitor.monitor, self._start) for monitor in self.monitors]
        if self.observe == True:
            self.monitors = [self] + self.monitors
        
        observers = [gevent.spawn(monitor.monitor, self._start) for monitor in self.monitors]
        
        gevent.joinall(observers)
        
    def clean(self):
        self.stop()
        self.save_plot()
        
    def stop(self):
        for monitor in self.monitors:
            monitor.observe = False
            
    def monitor(self, start_time):
        while self.observe == True:
            self.ax.lines = []
            for k, monitor in enumerate(self.monitors[1:]):
                chronos = monitor.get_chronos()
                points = monitor.get_points()
                if len(points) >= 1:                    
                    pylab.plot(chronos, points, 'o-', color=self.colors[k], label=monitor.get_name())
            pylab.legend(loc=5)
            pylab.draw()
            if self.duration != None and (time.time() - start_time) > self.duration:
                self.stop()
            gevent.sleep(self.plotsleeptime)
                
    def create_plot(self, figsize=(16, 9)):
        
        pylab.interactive(True)
        pylab.figure(figsize=figsize)
        
        pylab.title(self.description)
        
        pylab.xlabel('time')
        pylab.ylabel('points')
        
        pylab.grid(True)
        
        self.figure = pylab.gcf()
        self.ax = pylab.gca()
        
        
        
    def save_plot(self):
        pylab.legend(loc=5)
        self.figure.savefig('%s_%.3f.png' % (self.__module__, self.timestamp))
        pylab.show()
        

def main():
    
    import optparse
    
    parser = optparse.OptionParser()
    parser.add_option('-d', '--duration', default=5, type=float, help='Specify how long to observe (default=%default). If not specified observe without a limit.')
    
    options, args = parser.parse_args()
    
    from monitor import xbpm, sai, peltier, thermometer
    from machine_status import machine_status
    
    monitors = [peltier(), thermometer(), machine_status()]
    
    o = observation(monitors, duration=options.duration)
    
    o.execute()
    
    
if __name__ == '__main__':
    main()