#!/bin/bash

num_gen_sample=128

python3 main.py \
    -num_sample_gen $num_gen_sample \
    $@

