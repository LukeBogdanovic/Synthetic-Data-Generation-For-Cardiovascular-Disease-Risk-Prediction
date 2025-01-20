#!/bin/bash

while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    nvidia-smi --query-gpu=index,power.draw --format=csv,noheader,nounits | while IFS=',' read -r INDEX POWER; do
        echo "$TIMESTAMP - GPU $INDEX: $POWER W"
    done
    sleep 5
done
