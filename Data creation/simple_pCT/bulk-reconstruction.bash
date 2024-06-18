#!/bin/bash

NUM_ANGLE_START=90
NUM_ANGLE_END=180

DENUMINATOR=16
START=$((DENUMINATOR/2))

AVAILABLE_FILTERS=( "ramp" "shepp-logan" "cosine" "hamming" "hann" )

for NUMERATOR in $(seq $START $DENUMINATOR); do
    NUM_ANGLE=$(bc -l <<< "180 * ${NUMERATOR}/${DENUMINATOR}")
    NUM_ANGLE=${NUM_ANGLE%.*}
    
    for FILTER in ${AVAILABLE_FILTERS[@]}; do
        python one-shot-creation.py $NUM_ANGLE 1 130 $FILTER
    done
done
