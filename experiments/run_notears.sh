#!/bin/bash

p=32
K=32
n=20
s=12
s0=8
d=4

python baseline_notears.py \
-p ${p} \
-K ${K} \
-n ${n} \
-s ${s} \
-s0 ${s0} \
-d ${d} &
