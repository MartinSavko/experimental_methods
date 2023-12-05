#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
x-ray beam observation analysis
'''

import os
import pickle
import pylab
import numpy as np
import traceback
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter
#from xray_observation import xray_observation
from experiment import experiment

def get_observations(r, key):
    #print('key', key)
    o = r[key]['observations']
    f = r[key]['observation_fields']
    if hasattr(o[0][0], '__len__') and len(o[0][0])>1:
        _o = []
        for k in range(len(o[0])):
            _o.append(np.array([o[0][k], o[1][k]]))
        o = _o
    elif type(o[0][1]) not in [float, type(None)] and hasattr(o[0][1], '__len__') and len(o[0][1]) > 1:
        o = np.array([[item[0]] + list(item[1]) for item in o])
    else:
        o = np.array(o)
    return o, f

def normalize(lien, divide=True):
    try:
        lien -= lien.mean()
        if divide:
            lien /= lien.std()
    except:
        traceback.print_exc()
    return lien

def get_predictor(timeline, observation, kind='linear', fill_value='extrapolate'):
    predictor = interp1d(timeline, observation, kind=kind, fill_value=fill_value, bounds_error=False)
    #fit = np.polyfit(timeline, observation, deg)
    #predictor = np.poly1d(fit)
    return predictor

def get_sai_predictions(s4):
    s4_p = []
    n_min_max = []
    for k in range(len(s4)):
        t = s4[k][0]
        o = s4[k][1]
        lt = len(t)
        lo = len(o)
        mltlo = min(lt, lo)
        t = t[:mltlo]
        o = o[:mltlo]
        n_min_max.append([len(t), t.min(), t.max()])
        p = get_predictor(t, o)
        s4_p.append(p)
    n_min_max = np.array(n_min_max)
    
    n = np.mean(n_min_max[:, 0]).astype(int)
    t_min = n_min_max[:, 1].min()
    t_max = n_min_max[:, 2].max()
    
    tline = np.linspace(t_min, t_max, n)
    s4_e = []
    total = 0.
    for p in s4_p:
        current = p(tline)
        s4_e.append(current)
    
    currents = np.array(s4_e)
    total = np.abs(currents).sum(axis=0)
    currents /= total
    #c01 = (s4_e[0] - s4_e[1]) 
    #c23 = (s4_e[2] - s4_e[3]) 
    c01 = currents[0, :] - currents[1, :]
    c23 = currents[2, :] - currents[3, :]
    

    return c01, c23, tline

def get_observation_results(name_pattern, directory, sensors=['sai4', 'sai3', 'oav_camera', 'vfm_pitch', 'hfm_pitch', 'jaull05', 'vfm_trans', 'hfm_trans', 'tab2_tx1', 'tab2_tx2', 'tab2_tz1', 'tab2_tz2', 'tab2_tz3'], epoch=True):
    
    xo = experiment(name_pattern, directory)

    print(xo.get_template())
    
    p = xo.get_parameters()
    r = xo.get_diagnostics()
    
    start_time = 0
    if epoch:
        try:
            start_time = p['start_time']
        except:
            print(xo.get_template(), 'problem getting start_time')
            print(p.keys())
            
    results = {}
    results['start_time'] = start_time
    
    for sensor in r.keys():
        if sensor not in sensors:
            continue
        results[sensor] = {}
        so, f = get_observations(r, sensor)
        if 'sai' in sensor:
            c01, c23, tline = get_sai_predictions(so)
            results[sensor]['observation'] = np.array([tline+start_time, c01, c23]).T
            results[sensor]['fields'] = ['chronos', 'c01', 'c02']
        elif sensor == 'machine_status':
            print('so.T shape', so.T.shape)
            results[sensor]['observation'] = so.T

            results[sensor]['fields'] = f
        else:
            so[:, 0] += start_time
            results[sensor]['observation'] = so
            if sensor == 'oav_camera':
                f = ['chronos', 'vertical', 'horizontal']
            elif sensor in ['tdl_xbpm1', 'tdl_xbpm2']:
                f = ['chronos', 'z', 'x']
            elif 'jaull' in sensor:
                f = ['chronos', 'pressure']
            results[sensor]['fields'] = f

    return results

    #vp = o[:, 1]
    #hp = o[:, 2]
    #pitch_v = v[:, 1]
    #pitch_h = h[:, 1]
    
    #nm = 27
    
    #vp = median_filter(vp, nm)
    #hp = median_filter(hp, nm)
    
    #pitch_v = median_filter(pitch_v, nm)
    #pitch_h = median_filter(pitch_h, nm)
    
    #s4_c01 = median_filter(s4_c01, nm*21)
    #s4_c23 = median_filter(s4_c23, nm*21)
    
    #results = {}
    #results[
    
def plot_single_observation(name_pattern, directory):
    
    r = get_observation_results(name_pattern, directory)
        
    for sensor in r.keys():
        observation = r[sensor]['observation']
        fields = r[sensor]['fields']
        chronos = observation[:, 0]
        for k, f in enumerate(fields[1:]):
            fo = normalize(observation[:, k+1])
            if 'sai' in sensor:
                fo = median_filter(fo, 21*27)
            else:
                fo = median_filter(fo, 27)
            pylab.plot(chronos, fo, color=colors[sensor][k], label='%s %s' % (sensor, f))


def main():
    import argparse
    import glob
    import os
    import matplotlib.dates as mdates
    import datetime
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name_pattern', default='en_12650_d_3600_t_100_3', type=str, help='name pattern')
    parser.add_argument('-d', '--directory', default='/nfs/data4/2023_Run4/com-proxima2a/2023-09-23/Commissioning', type=str, help='directory')
    #parser.add_argument('-s', '--sensors', default=['sai4', 'oav_camera', 'vfm_pitch', 'hfm_pitch'], type=list, help='sensors')
    
    colors = {'sai4': ['orange', 'blue'], 'sai3': ['green', 'red'], 'oav_camera': ['red', 'green'], 'vfm_pitch': ['magenta'], 'hfm_pitch': ['cyan'], 'jaull05': ['purple'], 'jaull06': ['brown'], 'vfm_trans': ['blue'], 'hfm_trans': ['red'], 'tab2_tx1': ['orange'], 'tab2_tx2': ['cyan'], 'tab2_tz1': ['red'], 'tab2_tz2': ['orange'], 'tab2_tz3': ['magenta'], 'mono_mt_rx': ['brown'], 'mono_mt_rx_fine': ['black']}
    
    args = parser.parse_args()
    
    pylab.figure(figsize=(16, 9))
    
    observations = glob.glob(os.path.join(args.directory, args.name_pattern))
    observations.sort()

    labels = []
    results = []
    #sensors = ['tdl_xbpm1', 'tdl_xbpm2', 'mono_mt_rx', 'vfm_pitch', 'hfm_pitch', 'oav_camera','tab2_tz1', 'tab2_tz2', 'tab2_tz3', 'tab2_tx1', 'tab2_tx2', 'jaull05']
    sensors = ['tdl_xbpm2', 'vfm_trans', 'hfm_trans', 'vfm_pitch', 'hfm_pitch', 'sai4', 'sai3', 'oav_camera', 'jaull05', 'jaull06', 'fast_shutter', 'xbpm1', 'cvd1', 'machine_status']
    #sensors = ['hfm_pitch', 'vfm_pitch']
    #sensors = [ 'sai4', 'oav_camera', 'vfm_pitch', 'hfm_pitch', 'tab2_tz1', 'tab2_tz2', 'tab2_tz3', 'tab2_tx1', 'tab2_tx2', 'jaull05'] #, 'vfm_trans', 'hfm_trans'] #sensors=['sai4', 'sai3', 'oav_camera', 'vfm_pitch', 'hfm_pitch', 'jaull05', 'vfm_trans', 'hfm_trans', 'tab2_tx1', 'tab2_tx2', 'tab2_tz1', 'tab2_tz2', 'tab2_tz3']
    for obs in observations:
        print(obs)
        directory = os.path.dirname(obs)
        name_pattern = os.path.basename(obs).replace('_diagnostics.pickle', '')
        try:
            r = get_observation_results(name_pattern, directory, sensors=sensors)
            results.append(r)
        except:
            traceback.print_exc()
        
    results.sort(key=lambda r: r['start_time'])
    
    aligned_results = {}
    for r in results:
        del r['start_time']
        for sensor in r:
            if sensor not in aligned_results:
                aligned_results[sensor] = {}
                
            o = r[sensor]['observation']
            for k, f in enumerate(r[sensor]['fields']):
                ok = o[:, k]
                if f not in aligned_results[sensor]:
                    aligned_results[sensor][f] = ok
                else:
                    aligned_results[sensor][f] = np.hstack([aligned_results[sensor][f], ok])
            
    print(aligned_results.keys())
    l = 0
    for sensor in sensors: #aligned_results.keys():
        #if sensor == 'oav_camera':
            #continue
        try:
            chronos = aligned_results[sensor]['chronos']
            chronos = list(map(datetime.datetime.fromtimestamp, chronos))
        except:
            continue
        k = 0
        for f in aligned_results[sensor]:
            if f != 'chronos':
                k += 1
                l += 1
                label = '%s %s' % (sensor, f)
                ##if 'oav' in sensor or 'pitch' in sensor:
                    ##p = normalize(aligned_results[sensor][f], divide=False)
                ##else:
                try:
                    p = normalize(aligned_results[sensor][f])
                except:
                    p = aligned_results[sensor][f]
                #try:
                    #color = colors[sensor][k-1]
                    #pylab.plot(chronos, 2*(l-1) + p, color=color, label=label)
                #except:
                try:
                    p = 3*(l-1) + p
                except:
                    pass
                pylab.plot(chronos, p, label=label)
        #try:
            ##plot_single_observation(name_pattern, directory)
            #r = get_observation_results(name_pattern, directory)
            #results.append(r)
            #for sensor in r.keys():
                #observation = r[sensor]['observation']
                #fields = r[sensor]['fields']
                #chronos = observation[:, 0]
                #for k, f in enumerate(fields[1:]):
                    #fo = observation[:, k+1]
                    ##fo = normalize(observation[:, k+1])
                    #if 'sai' in sensor:
                        #fo = median_filter(fo, 21*27)
                    #else:
                        #fo = median_filter(fo, 27)
                    #label = '%s %s' % (sensor, f)
                    #try:
                        #color = colors[sensor][k]
                    #except:
                        #color = 'gray'
                    #if label not in labels:
                        ##chronos = list(map(datetime.datetime.fromtimestamp, chronos))
                        #pylab.plot(chronos, fo, color=color, label=label)
                        #labels.append(label)
                    #else:
                        #pylab.plot(chronos, fo, color=color)
        #except:
            #traceback.print_exc()
            #print('problem plotting %s' % os.path.join(directory, name_pattern))
    
    #ax = pylab.gca()
    #ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d %H:%M:%S'))
    #try:
        #for label in ax.get_xticklabels(which='major'):
            #label.set(rotation=30, horizontalalignment='right')                        
    #except:
        #pass
    pylab.ylim(-10, 45)
    pylab.legend(loc=1)
    pylab.show()
               
if __name__ == '__main__':
    main()


    #s4 = get_observations(r, 'sai4')
    #s4_c01, s4_c23, s4_tline = get_sai_predictions(s4)
    #o = get_observations(r, 'oav_camera')
    #v = get_observations(r, 'vfm_pitch')
    #h = get_observations(r, 'hfm_pitch')
    
    #s5 = get_observations(r, 'sai5')
    #s5_c01, s5_c23, s5_tline = get_sai_predictions(s5)
    
    #psd5 = get_observations(r, 'psd5')
    #t5 = psd5[:, 0]
    #h5 = psd5[:, 2]
    #v5 = psd5[:, 3]
              
    #s5_c01 = median_filter(s5_c01, 327*21)
    #s5_c23 = median_filter(s5_c23, 327*21)
    #v5 = median_filter(v5, 107*21)
    #h5 = median_filter(h5, 107*21)
    #pylab.plot(s5_tline, normalize(s5_c01), label='sai5_c01')
    #pylab.plot(s5_tline, normalize(s5_c23), label='sai5_c23')
    #pylab.plot(t5, normalize(v5), label='psd5 vertical')
    #pylab.plot(t5, normalize(h5), label='psd5 horizontal')
