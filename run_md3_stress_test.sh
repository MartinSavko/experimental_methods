#!/bin/bash
n=100
base_path=/nfs/data4/2026_Run1/com-proxima2a/md3_stress_test/20260116_a
for e in 36 18 9 4.5 3.6 1.8 1 1.8 3.6 4.5 9 18 36; do
    echo ${e} 
    ./goniometer.py -n ${n} --random --scan_exposure_time ${e} --base_path ${base_path}
done
