#!/bin/bash

for epoch in {1..50}; do
    stress_test_series.sh
    echo sleep
    sleep 10
done
