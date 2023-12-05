#!/usr/bin/env python
import h5py
import numpy as np

def main():
    m = h5py.File('merged_master.h5')
    data_keys = list(m['/entry/data/'].keys())
    data_keys.sort()
    pm = m['/entry/instrument/detector/detectorSpecific/pixel_mask'].value
    cutoff = m['/entry/instrument/detector/detectorSpecific/countrate_correction_count_cutoff'].value
    problematic = {}
    for l, key in enumerate(data_keys):
        print('data file %d' % l)
        d = m['/entry/data/'][key]
        for k, i in enumerate(d):
            b = i[::]
            b[pm!=0] = 0
            mxa = b.max()
            if mxa > cutoff:
                ind = np.unravel_index(b.argmax(), b.shape)
                print(k, mxa, ind)
                if ind not in problematic:
                    problematic[ind] = 1
                else:
                    problematic[ind] += 1
        print(list(problematic.keys()))
        print(list(problematic.values()))
    print('problematic pixels')
    print(problematic)
    print(list(problematic.keys()))
    print(list(problematic.values()))
    
if __name__ == '__main__':
    main()
    
#(2339, 1412)
#(1561, 1634)
#(1604, 1653)
#(1554, 1601)
#(1547, 1607)
#(1615, 1597)
#(1537, 1606)
#(1552, 1638)
#(1549, 1616)
#(1562, 1635)
#(1552, 1604)
#(1615, 1601)
#(1554, 1597)
#(1549, 1626)
#(1537, 1606)
#(1558, 1619)
#(1615, 1641)
#(1549, 1616)
#(1552, 1604)
#(1558, 1623)
#(1554, 1641)
#(1615, 1645)
#(1554, 1597)
#(1561, 1634)
#(1558, 1623)
#(1554, 1601)
#(1554, 1645)
#(1562, 1635)
#(1558, 1623)
#(1615, 1601)
#(1615, 1645)
