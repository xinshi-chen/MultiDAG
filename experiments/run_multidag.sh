#!/bin/bash

p=(32 128 512)
K=64
n=(20 40 80 160 320 640)
s=(160 1000 6000)
s0=(64 384 2048)
d=(4 6 8)
gpu=(5 6 1 4 0 7)
num_epoches=4000

for i in ${!p[*]}; do
  for j in 5; do
    python main_linear.py \
    -p ${p[i]} \
    -K ${K} \
    -n ${n[j]} \
    -s ${s[i]} \
    -s0 ${s0[i]} \
    -d ${d[i]} \
    -num_epochs ${num_epoches} \
    -gpu ${gpu[j]}
  done
done