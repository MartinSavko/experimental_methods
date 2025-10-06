#!/bin/bash
sed -n $((RANDOM % $(wc -l ${1} | cut -d " " -f 1)))p ${1}

