#!/bin/bash

p=(32 64 128 256)
K=256
n=(10 20 40 80 160 320)
s=(120 288 672 1536)
s0=(40 96 224 512)
d=(5 6 7 8)
# group_start=($(seq 0 8 255) $(seq 0 16 255) $(seq 0 32 255) $(seq 0 64 255) 0 128 0)
# group_end=($(seq 8 8 256) $(seq 16 16 256) $(seq 32 32 256) $(seq 64 64 256) 128 256 256)
# group_size=($(for i in {1..32}; do echo 1; done) $(for i in {1..16}; do echo 2; done) $(for i in {1..8}; do echo 4; done) $(for i in {1..4}; do echo 8; done) 16 16 32)
group_start=($(seq 0 8 255))
group_end=($(seq 8 8 256))
group_size=($(for i in {1..32}; do echo 1; done))
num_epochs=(20000 20000 20000 20000)
eta=0.125
dual=50

output_root=GOD_configs

if [ ! -e $output_root ];
then
    mkdir -p $output_root
fi

for idx in ${!p[*]}; do
  for j in ${!n[*]}; do
    for l in ${!group_size[*]}; do
      echo "-K ${K} " > ${output_root}/file_$(($idx * 192 + $j * 32 + $l + 1)).txt
      echo "-p ${p[idx]} " >> ${output_root}/file_$(($idx * 192 + $j * 32 + $l +1)).txt
      echo "--group_size ${group_size[l]} " >> ${output_root}/file_$(($idx * 192 + $j * 32 + $l + 1)).txt
      echo "--group_start ${group_start[l]} " >> ${output_root}/file_$(($idx * 192 + $j * 32 + $l + 1)).txt
      echo "--group_end ${group_end[l]} " >> ${output_root}/file_$(($idx * 192 + $j * 32 + $l + 1)).txt
      echo "-n_sample ${n[j]} " >> ${output_root}/file_$(($idx * 192 + $j * 32 + $l +1)).txt
      echo "-s ${s[idx]} " >> ${output_root}/file_$(($idx * 192 + $j * 32 + $l +1)).txt
      echo "-s0 ${s0[idx]} " >> ${output_root}/file_$(($idx * 192 + $j * 32 + $l +1)).txt
      echo "-d ${d[idx]} " >> ${output_root}/file_$(($idx * 192 + $j * 32 + $l +1)).txt
    done
  done
done
