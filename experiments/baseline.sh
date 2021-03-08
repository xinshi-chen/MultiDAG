#!/bin/bash

epoch0=0
num_epochs=2000

python3 main.py \
    -baseline True \
    -start_epoch $epoch0 \
    -num_epochs $num_epochs \
    $@

