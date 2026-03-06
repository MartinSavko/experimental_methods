#!/bin/bash
echo ${1}
a=${1};
#start=20260216_144154; 
start=$(ls -tr1 | head -n 1)
echo ${start}
end=$(ls -tr1 | tail -n 1); 
#end=20260216_140227; #220260209_16552; #10260209_164003;
echo ${end}; 
first=${start}/before_Omega_at_${a}_1.jpg; 
echo ${first}; 
last=$(ls ${end}/after_Omega_at_${a}_*.jpg -tr1 | tail -n 1); 
echo ${last}; 
echo eog -n ${first} ${last}; 
eog -n ${first} ${last}

