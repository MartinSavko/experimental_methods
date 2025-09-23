
#!/bin/bash
curdir=$(pwd);
lowres=65
proc=12
jobs=12
max_iterations=3
min_cc_half=50
for x in $(find ./ -iname "*parameters.pickle")
do
    template=$(basename ${x::-18})
    a=$(grep "description" ${x::-18}.log | grep "Omega scan")
    test_string=${a::23}
    if [ "description: Omega scan" = "${test_string}" ]
    then
        echo x ${x};
        processing_directory=$(dirname ${x})/process/xdsme_auto_${template}
        if [ ! -d ${processing_directory} ]
        then
            echo ${processing_directory} not yet present
            cd $(dirname ${processing_directory})
            xdsme --brute -p auto_${template} ../${template}_master.h5
            cd ${curdir}
        fi
        cd ${processing_directory};
        if [ -f CORRECT.LP ]
        then
            grep -B 25 'WILSON STATISTICS OF DATA SET' CORRECT.LP | head -n 22 > correct.ods
            max_line=$(grep -B1 'total' correct.ods | grep -v total | sed 's/  */:/g')
            highres=$(echo ${max_line} | cut -d : -f 2)
            i_over_sigma=$(echo ${max_line} | cut -d : -f 10)
            cc_half=$(echo ${max_line} | cut -d : -f 12)
            echo highres i_over_sigma cc_half: ${highres} ${i_over_sigma} ${cc_half}
            k=0
            while (( $(bc -l <<< "${cc_half::-1} > ${min_cc_half} && ${k} < ${max_iterations}") ))
            do
                highres=$(bc -l <<< "scale=2; ${highres}/1.05")
                echo new highres ${highres}
                runxds.py -a -i "MAXIMUM_NUMBER_OF_PROCESSORS= ${proc}\n MAXIMUM_NUMBER_OF_JOBS= ${jobs}\n NUMBER_OF_IMAGES_IN_CACHE= 400\n INCLUDE_RESOLUTION_RANGE= ${lowres} ${highres}"
                grep -B 25 'WILSON STATISTICS OF DATA SET' CORRECT.LP | head -n 22 > correct.ods
                max_line=$(grep -B1 'total' correct.ods | grep -v total | sed 's/  */:/g')
                highres=$(echo ${max_line} | cut -d : -f 2)
                i_over_sigma=$(echo ${max_line} | cut -d : -f 10)
                cc_half=$(echo ${max_line} | cut -d : -f 12)
                echo highres i_over_sigma cc_half: ${highres} ${i_over_sigma} ${cc_half}
                k=$((k+1))
                echo ${k}
            done
            xdsconv.py XDS_ASCII.HKL ccp4if
        fi
        cd ${curdir}
        echo
    fi
done

