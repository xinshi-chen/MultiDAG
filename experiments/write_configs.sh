#!/bin/bash

p=(32 64 128 256 512 1024)
K=32
k=($(for i in {1..32}; do echo 1; done) $(for i in {1..16}; do echo 2; done) $(for i in {1..8}; do echo 4; done) 8 8 8 8 16 16 32)
k0=({0..31} $(seq 0 2 30) $(seq 0 4 28) $(seq 0 8 24) 0 16 0)
n=(20 40 80 160 320 640)
s=(120 288 672 1536 3456 7680)
s0=(40 96 224 512 1152 2560)
d=(5 6 7 8 9 10)
num_epochs=(5000 6000 7000 8000 9000 10000)

for i in ${!p[*]}; do
  for j in ${!n[*]}; do
    for l in ${!k[*]}; do
      echo -K 32 >> configs/file_$(($i * 378 + $j * 63 + $l + 1)).txt
      echo -p ${p[i]} >> configs/file_$(($i * 378 + $j * 63 + $l +1)).txt
      echo --group_size ${k[l]} >> configs/file_$(($i * 378 + $j * 63 + $l +1)).txt
      echo --group_start ${k0[l]} >> configs/file_$(($i * 378 + $j * 63 + $l + 1)).txt
      echo -n_sample ${n[j]} >> configs/file_$(($i * 378 + $j * 63 + $l +1)).txt
      echo -s ${s[i]} >> configs/file_$(($i * 378 + $j * 63 + $l +1)).txt
      echo -s0 ${s0[i]} >> configs/file_$(($i * 378 + $j * 63 + $l +1)).txt
      echo -d ${d[i]} >> configs/file_$(($i * 378 + $j * 63 + $l +1)).txt
      echo -num_epochs ${num_epochs[i]} >> configs/file_$(($i * 378 + $j * 63 + $l +1)).txt
    done
  done
done
