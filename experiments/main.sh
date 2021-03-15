#!/bin/bash

export CUDA_VISIBLE_DEVICES=0


python3 main.py \
    -gpu 0 \
    $@

