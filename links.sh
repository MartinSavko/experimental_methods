#!/bin/bash

for c in $(find ./ -iname "process" | grep main); do 
	echo ${c};
	d=$(echo ${c/\.\//});
    echo ${d};
	e=${d/main\/process/main}; 
	echo ${e}; 
	f=../../PROCESSED_DATA/L20F5/${e}; 
	echo ${f};
        mkdir -p ${f}	
        so=$(realpath ${c})	
	for o in $(ls ${so}); do
	    rsync -rv --exclude-from=/nfs/data/exclude.tmp $(realpath ${so}/${o}) ${f}/;
	done
	#echo ln -s $(realpath ${c}) ${f}; 
done

