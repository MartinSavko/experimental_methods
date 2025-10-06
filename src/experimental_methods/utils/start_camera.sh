#!/bin/bash
for k in 1 6 8 ; do axis_stream.py -c cam${k} -s cam${k} -o h264 & done
axis_stream.py -c cam13 -s cam13 -o hevc &
for m in 1 2 3 4 quad; do axis_stream.py -c cam14 -m ${m} -s cam14_${m} -o hevc & done
