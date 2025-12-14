#!/bin/bash
# target="tungsten_pin_42um"
# target="hampton_loop_with_sharp_crystal"
target=$1
echo target is ${target}
directory=/nfs/data4/2025_Run5/com-proxima2a/Commissioning/mk3_calibration/$(date +%Y%m%d_%H%M%S)_${target}
angles="(0,90,180,270,360)" #225,315)" #,223,313)"

# time for s in 8 9 1 2 3 4 5 7; do mount.py -p 10 -s ${s}; optical_alignment.py -r 0 -A -C --extreme -z 1; time run_mk3_calibration.sh sample_${s}; done; cats.py -c umount

function my_align() {
    k=${1};
    p=${2};
    echo ${k} ${p};
    r=0;
    for z in 4 5; do
        optical_alignment.py -d ${directory} -n ${k}_${p}_${r}_${target}_zoom_${z}_eager -K ${k} -P ${p} -r ${r} -z ${z} -b -A -C --extreme -a ${angles};
    done;
    r=360;
    for z in 5; do 
        echo ${k} ${p} ${z}; 
        optical_alignment.py -d ${directory} -n ${k}_${p}_${r}_zoom_${z}_careful -K ${k} -P ${p} -r ${r} -z ${z} -f -b -A -C --extreme; 
    done;
    echo; echo; echo; echo; echo; 
}

for k in 0 45 90 135; do
    for p in 0 90 180 225 315 360; do
        time my_align ${k} ${p}
    done
done
# for p in 0 90 180 225 315; do
#     for k in {0..250..40}; do 
#         time my_align ${k} ${p}
#     done
# done

# for k in 0 23 37 47 67 97 127 181; do
#    for p in {23..360..60}; do 
#        time my_align ${k} ${p}
#    done
# done

# 
# k=0;
# for p in {5..360..45}; do
#     time my_align ${k} ${p}
# done
# 
# p=90;
# for k in {0..240..10}; do 
#     time my_align ${k} ${p}
# done
# 

