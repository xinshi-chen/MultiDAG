#!/bin/bash

epoch0=0
num_epochs=1000

python3 main.py \
    -mmd True \
    -start_epoch $epoch0 \
    -num_epochs $num_epochs \
    $@

