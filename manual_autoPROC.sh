#!/bin/bash

# specify directory with the RAW_DATA
#raw_data=/nfs/data4/2022_Run5/20211717/2022-11-11/RAW_DATA

#raw_data=/nfs/data4/2025_Run3/20231333/cristynonato/2025-07-04/RAW_DATA
raw_data=/nfs/data4/2025_Run3/com-proxima2a/2025-07-04/RAW_DATA

if [ X${1} != X ]; then
   raw_data=$(realpath ${1})
fi
echo raw_data variable is set to ${raw_data}

# we are going to use existence of summary.tar.gz as test for a successful processing
check_file=summary.tar.gz

for collect in $(find ${raw_data} -iname "*_master.h5" | grep -v ref); do
    raw_directory=$(dirname ${collect}) 
    processed_directory=${raw_directory//RAW_DATA/PROCESSED_DATA}
    filename=$(basename ${collect})
    template=${filename::-10}
    file_exists_if_processing_finished_okay=$(echo ${processed_directory}/autoPROC_${template}/${check_file})

    echo checking ${file_exists_if_processing_finished_okay}
    if [ -f ${file_exists_if_processing_finished_okay} ]; then
       echo exists!
       echo
    else
       echo does not exist!
       workspace=$(dirname ${file_exists_if_processing_finished_okay})
       mkdir -p ${workspace}
       cd ${workspace}
       process -B -xml -nthreads 12 autoPROC_XdsKeyword_LIB="/data2/bioxsoft/progs/AUTOPROC/AUTOPROC/autoPROC/bin/linux64/plugins-x86_64/durin-plugin.so" autoPROC_XdsKeyword_MAXIMUM_NUMBER_OF_JOBS="12" -h5 ${collect}
    fi
done
 
