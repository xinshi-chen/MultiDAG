#!/bin/bash

epoch0=1000
num_epochs=1000

python3 main.py \
    -baseline True \
    -start_epoch $epoch0 \
    -num_epochs $num_epochs \
    $@

