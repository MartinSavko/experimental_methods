#!/bin/bash
# target="tungsten_pin_42um"
target="hampton_loop_with_sharp_crystal"
directory=/nfs/data4/2025_Run5/com-proxima2a/Commissioning/mk3_calibration/$(date +%Y%m%d_%H%M%S)_${target}
angles="(0,90,180,270)" #,223,313)"

function my_align() {
    k=${1};
    p=${2};
    echo ${k} ${p};
    r=0;
    for z in 1 4 5; do
        optical_alignment.py -d ${directory} -n ${k}_${p}_${r}_${target}_zoom_${z}_eager -K ${k} -P ${p} -r ${r} -z ${z} -b -A -C --extreme -a ${angles};
    done;
#     r=360;
#     for z in 5; do 
#         echo ${k} ${p} ${z}; 
#         optical_alignment.py -d ${directory} -n ${k}_${p}_${r}_zoom_${z} -K ${k} -P ${p} -r ${r} -z ${z} -b -A -C --extreme; 
#     done;
    echo; echo; echo; echo; echo; 
}

p=0;
for k in {0..240..15}; do 
    time my_align ${k} ${p}
done

k=0;
for p in {1..361..30}; do
    time my_align ${k} ${p}
done
# 
# p=90;
# for k in {0..240..10}; do 
#     time my_align ${k} ${p}
# done

#for k in 23 37 47 67 97 127 181; do
#    for p in 89 179 227 317; do 
#        time my_align ${k} ${p}
#    done
#done

