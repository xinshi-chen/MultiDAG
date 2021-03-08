#!/bin/bash

epoch0=1000
num_epochs=1000
num_gen_sample=128

python3 main.py \
    -num_sample_gen $num_gen_sample \
    -start_epoch $epoch0 \
    -num_epochs $num_epochs \
    $@

