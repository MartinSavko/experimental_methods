#!/bin/bash
directory=/nfs/data4/2025_Run3/com-proxima2a/Commissioning/mk3_calibration/$(date +%Y-%m-%d_%H%M%S)
for k in {0..240..15}; do 
    for p in {0..360..15}; do
	z=1;
	r=0;
        optical_alignment.py -d ${directory} -n ${k}_${p}_${r}_zoom_${z}_eager -K ${k} -P ${p} -r ${r} -z ${z} -b -A -C --extreme;
	r=360;
        for z in 1 4 5; do 
            echo ${k} ${p} ${z}; 
	    optical_alignment.py -d ${directory} -n ${k}_${p}_${r}_zoom_${z} -K ${k} -P ${p} -r ${r} -z ${z} -b -A -C --extreme; echo; 
        done; 
    done; 
done

