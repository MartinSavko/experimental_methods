#!/bin/bash
target="tungsten_pin_42um"
directory=/nfs/data4/2025_Run5/com-proxima2a/Commissioning/mk3_calibration/${target}_$(date +%Y%m%d_%H%M%S)
angles="(1, 97, 179, 223, 313)"
k=0

function align {
    k=${1}
    p=${2}
    r=0;
    for z in 1 4 5; do
        optical_alignment.py -d ${directory} -n ${k}_${p}_${r}_zoom_${z}_eager -K ${k} -P ${p} -r ${r} -z ${z} -b -A -C --extreme -a "${angles}";
    done;
    r=360;
    for z in 5; do 
        echo ${k} ${p} ${z}; 
        optical_alignment.py -d ${directory} -n ${k}_${p}_${r}_zoom_${z} -K ${k} -P ${p} -r ${r} -z ${z} -b -A -C --extreme; echo; 
}

k=0
for p in {0..360..10}; do
    align ${k} ${p}
done

p=0
for k in {0..240..10}; do 
    align ${k} ${p}
done

for k in 23 37 47 67 97 127 181; do
    for p in 89 181 227 317; do 
        align ${k} ${p}
    done; 
done

