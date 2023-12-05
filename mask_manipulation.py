#!/usr/bin/env python
import json    
import numpy   
import requests    
    
from base64 import b64encode, b64decode   
import pickle
    
IP = '172.19.10.26'   
PORT =  '80'    
    
def get_mask(ip,    port):  
                """ 
                Return  the pixel   mask    of  host    EIGER   system  as  numpy.ndarray   
                """ 
                url =   'http://%s:%s/detector/api/1.6.0/config/pixel_mask' %   (ip, port)   
                reply   =   requests.get(url)   
                darray  =   reply.json()['value']   
                return  numpy.fromstring(b64decode(darray['data']), 
                                dtype=numpy.dtype(str(darray['type']))).reshape(darray['shape'])    

def set_mask(ndarray, ip, port):  
        """ 
        Put a   pixel   mask    as  ndarray on  host    EIGER   system  and return  its reply   
        """ 
        url =   'http://%s:%s/detector/api/1.6.0/config/pixel_mask' %   (ip,    port)   
        data_json   =   json.dumps({'value':    {
                                    '__darray__':   (1,0,0),    
                                    'type':    ndarray.dtype.str,  
                                    'shape':    ndarray.shape,
                                    'filters':  ['base64'], 
                                    'data': b64encode(ndarray.data) }                                                                              })  
        headers =   {'Content-Type':    'application/json'} 
        return  requests.put(url, data=data_json, headers=headers)    
    
if  __name__    ==  '__main__': 
    #   get the mask    
    mask = get_mask(ip=IP, port=PORT)
    
    #2022-05-31
    print(' modifying mask')
    #mask[514:551, 0:1030] = 1
    ##codes = {'new_dead':  2**1, 'new_cold': 2**2, 'new_hot': 2**3}
    codes = {'dead': 2**1, 'cold': 2**2, 'hot': 2**3, 'noisy': 2**4}
   
    #2022-09-13
    mask[1864, 1580] = codes['hot']
    mask[607, 880] = codes['hot']
    mask[608, 879] = codes['hot']
    mask[606, 99] = codes['hot']
    
    #mask[1508: 1517, 1553: 1557] = 0
    #mask[1756: 1768, 1553: 1557] = 0
    #mask[1768, 1555: 1557] = 0
    
    #pixels_to_mask = pickle.load(open('/927bis/ccd/gitRepos/eiger/pixels_to_mask.pickle'))
    #pixels_to_mask = pickle.load(open('/nfs/ruche/proxima2a-spool/2019_Run2/com-proxima2a/2019-04-04/RAW_DATA/Commissioning/pixels/photon_energy_12000_nimages_1000_frame_time_0.01s_bad_pixels.pickle'))
    #pixels_to_mask = pickle.load(open('/nfs/data2/2022_Run3/com-proxima2a/2022-07-15/RAW_DATA/Commissioning/8p1keV_0p1s_1200_1_bad_pixels.pickle', 'r'))
    
    #for key in pixels_to_mask:
        
        #for pixel in pixels_to_mask[key]:
            #v, h = pixel
            #mask[v, h] = codes[key]
            
            
    # half module after firmware upgrade 2022-05-24
    #import h5py
    #m = h5py.File('/nfs/data2/s1_1_h5/s1_1_master.h5', 'r')
    #mask = m['/entry/instrument/detector/detectorSpecific/pixel_mask'][()]
    #mask[257:514 , 1040:2071] = codes['dead']
    
    #mask[mask!=1] = 0
    
    #mask[1468: 1472, 1522: 1527] += 6
    #mask[1469, 1520:1522] += 6
    #mask[1469:1471, 1528: 1533] += 6
    #mask[1468, 1528] += 6
    #mask[1471, 1528] += 6
    
    # 2019-04-04
    #pixels_to_mask = pickle.load(open('/nfs/ruche/proxima2a-spool/2019_Run2/com-proxima2a/2019-04-04/RAW_DATA/Commissioning/pixels/additional_cold.pickle'))
    #for pixel in pixels_to_mask:
        #v, h = pixel
        #mask[v, h] = codes['cold']
        
    #mask[1590: 1680, 1524: 1594] = 0
    #mask[1616: 1653, 1524: 1594] = 1
    #mask[1730: 1734, 1433: 1441] = 6
    #mask[1733, 1441] = 6
    #mask[1730: 1733, 1453: 1457] = 6
    #mask[1663, 1525] = codes['hot']
    #mask[1139, 1600] = codes['cold']
    #mask[1539, 1083] = codes['cold']
    #mask[842, 2337] = codes['hot']
    #mask[1517, 2336] = codes['hot']
    # 2019-03-20
    #mask[1102: 1359, 0: 1030] = 0
    #mask[1102: 1616, 0: 1030] = 0

    #mask[1730:1734, 1434:1441] = 2**6
    #   set a   new dead    pixel   
    #mask[123,   234]    =   0   
    #   set a   new noisy   pixel   
    #mask[234,   123]    =   0   
    
    #mask[895, 1162] = 3
    #mask[1903, 801] = 3
    #mask[2311, 1065] = 3
    #mask[401, 140] = 3
    #mask[1187, 2943] = 3
    #mask[3118, 3073] = 3
    # 2017-07-28
    #mask[1921, 2910] = 3
    #mask[1921, 2865] = 3
    #mask[1921, 2416] = 3
    
    # masking pixels in the centre (possible direct beam damaged)
    #mask[1658, 1507:1509] = codes['hot']
    #mask[1657, 1507:1509] = codes['hot']
    #mask[1656, 1508] = 6 # dead, part of a problematic group
    #mask[1658, 1509] = 6 # hot, part of a problematic group
    #mask[1657, 1509] = 6 # hot, part of a problematic group
    #mask[1657, 1510] = codes['hot']
    #mask[1657, 1512] = codes['hot']
    #mask[895, 1162] = 3
    #mask[1257, 2535] = 3 
    #mask[1187, 2943] = 3
    #mask[1210:1212, 2983] = 3
    #mask[1903, 801] = 3
    #mask[2311, 1065] = 3
    #mask[1141, 2713] = 3
    #mask[844, 764] = 3
    #mask[3118, 3073] = 3
    #mask[2896, 2168] = 3
    #mask[2779:2781, 1499] = 3
    #mask[1467, 3013:3015] = 3
    # observed at Cu K alpha
    #mask[1251, 2535] = 3
    #mask[2361, 2371] = 3
    #mask[2264, 1202] = 3
    # observed at 12650 eV
    #mask[847, 1150] = 3
    #mask[847, 1190] = 3
    
    # masking hot regions in the interchip connections of the last module (south-east)
    # middle
    #mask[3010:3014, 2335:2339] = 3 #hot
    #mask[3010:3014, 2593:2597] = 3 #hot
    #mask[3010:3014, 2851:2855] = 3 #hot
    # lowest row
    #mask[3268, 2335:2339] = 3 #hot
    #mask[3268, 2593:2597] = 3 #hot
    #mask[3268, 2851:2855] = 3 #ho
    
    #module 51
    #mask[2461:2463, 1295:1299] = 3
    #mask[2461:2463, 1553:1557] = 3
    #mask[2461:2463, 1811:1815] = 3
    #2016-11-19 disabling the top left half-module
    #mask[0:257, 0:1030] = 3
    #mask[0:257, 0:1030] = 0
    #2016-11-21 reactivating back on 
    #mask[0:257, 0:1030] = 0
    #mask[2339, 1412] = 2**3 #blinking pixel
   # #2017-04-28 masking two half modules
    #2017-09-18
    #mask[0:257, 0:1030] = 0 #3 # 3
    #mask[514:551, 0:1030] = 1
    #mask[1065:1102, 0:1030] = 1
    #mask[551:808, 0:1030] = 0 #3 # 3
    #mask[1102:1359, 0:1030] = 0 #3
    #mask[2204:2461, 0:1030] = 0 
    #2017-08-29 line south-east bottom halfmodule
    #mask[3012:3014, 2080:3110] = 0 # 6
    #   upload  the new mask    
    # 2017-10-25
    #mask[3014, 2134] = 3
    #mask[3019, 2090] = 3
    #mask[3010:3013, 2081] = 3
    #mask[3012:3014, 2080] = 3
    #mask[3012:3014, 3109] = 3
    #mask[3010:3013, 3108] = 3
    #mask[3268, 2080] = 3
    #mask[3268, 3109] = 3
    
    reply   =   set_mask(mask,  ip=IP,  port=PORT)  
    print('reply', reply)
    #   reply.status_code   should  be  200,    then    arm and disarm  to  store   the mask    
    if  reply.status_code   ==  200:    
        for command in  ('arm', 'disarm'):  
                        url =   'http://%s:%s/detector/api/1.6.0/command/%s'    %   (IP,    PORT,   command)    
                        requests.put(url)   
    else:   
        print(reply.content)   
