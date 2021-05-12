#!/bin/bash

p=(32 64 128 256 512 1024)
K=32
k=(1 2 4 8 16 32)
n=(20 40 80 160 320 640)
s=(120 288 672 1536 3456 7680)
s0=(40 96 224 512 1152 2560)
d=(5 6 7 8 9 10)

for i in ${!p[*]}; do
  for j in ${!n[*]}; do
      echo -K 32 >> data_configs/file_$(($i * 6 + $j + 1)).txt
      echo -p ${p[i]} >> data_configs/file_$(($i * 6 + $j +1)).txt
      echo -n_sample ${n[j]} >> data_configs/file_$(($i * 6 + $j +1)).txt
      echo -s ${s[i]} >> data_configs/file_$(($i * 6 + $j +1)).txt
      echo -s0 ${s0[i]} >> data_configs/file_$(($i * 6 + $j +1)).txt
      echo -d ${d[i]} >> data_configs/file_$(($i * 6 + $j +1)).txt
  done
done
