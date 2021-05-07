#!/bin/bash

p=(32 64 128 256 512 1024)
K=32
n=(20 40 80 160 320 640)
s=(120 288 672 1536 3456 7680)
s0=(40 96 224 512 1152 2560)
d=(5 6 7 8 9 10)
gpu=(5 6 1 4 0 7)
num_epoches=(6000 7000 8000 9000 10000 11000 12000 13000 14000 15000)

for i in ${!p[*]}; do
  for j in $1; do
    python main_linear.py \
    -p ${p[i]} \
    -K ${K} \
    -n ${n[j]} \
    -s ${s[i]} \
    -s0 ${s0[i]} \
    -d ${d[i]} \
    -num_epochs ${num_epoches[$((i+j))]} \
    -gpu ${gpu[j]}
  done
done