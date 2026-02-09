#!/bin/bash
n=35
for e in 36 18 9 4.5 3.6 1.8 1 1.8 3.6 4.5 9 18 36; do
    echo ${e} 
    ./goniometer.py -n ${n} --scan_exposure_time ${e}
done
