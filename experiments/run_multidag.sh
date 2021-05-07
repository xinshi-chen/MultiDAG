#!/bin/bash

p=(32 128 512)
K=64
n=(20 40 80 160 320 640)
s=(12 18 24)
s0=(8 12 16)
d=4
num_epoches=4000

for i in ${!p[*]}; do
  for j in ${!n[*]}; do
    python main_linear.py \
    -p ${p[i]} \
    -K ${K} \
    -n ${n[j]} \
    -s ${s[i]} \
    -s0 ${s0[i]} \
    -d ${d} \
    -num_epoches ${num_epoches}
  done
done