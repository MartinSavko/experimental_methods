#!/bin/bash
n=5
#base_path=/nfs/data4/2026_Run1/com-proxima2a/md3_stress_test/20260117_a
base_path=/nfs/data4/2026_Run1/com-proxima2a/md3_stress_test/20260306_c
for e in $(cat stress_test_speeds.txt); do
    goniometer.py -n ${n} --random --scan_exposure_time ${e} --base_path ${base_path}
done
