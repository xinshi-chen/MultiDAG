#!/bin/bash

p=(32 128 512)
K=64
n=(20 40 80 160 320 640)
s=(160 1000 6000)
s0=(64 384 2048)
d=(8 12 16)

cd ../multidag || exit
for i in ${!p[*]}; do
  for j in ${!n[*]}; do
    python data_generator.py \
    -p ${p[i]} \
    -K ${K} \
    -n ${n[j]} \
    -s ${s[i]} \
    -s0 ${s0[i]} \
    -d ${d[i]}
  done
done