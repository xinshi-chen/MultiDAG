#!/bin/bash


K=1
p=100
e=10
n=100
nh=2.0
rho=(1)

base_dir="../data/"
sim_file="sergio_K-${K}_p-${p}_e-${e}_n-${n}_nh-${nh}.npz"
real_dir="${base_dir}${sim_file}"

for i in ${!rho[*]}; do
  python main_linear.py \
  --real_dir ${real_dir} \
  --group_size ${K} \
  -rho ${rho[i]} \
  -K ${K} \
  -p ${p} \
  -n_sample ${n} \
  -num_epochs 100 \
  -gpu 1
done
